use rustc_hash::{FxHashMap, FxHashSet};

use super::doubly::closest_on_triangle;
use super::fit::Final;
use super::linalg::{V3, add, angle_deg, cross, norm, scale, sub, unit};
use super::refine::{lsq_point, off_surfaces, onto_crossing};
use super::topology::NONE;
use crate::ir::{Kind, VertexRole};
use crate::mesh::Point;

pub(super) struct EdgeRec {
    ra: u32,
    rb: u32,
    u: u32,
    v: u32,
    fa: u32,
    fb: u32,
}

pub struct RawBoundary {
    pub kind: Kind,
    pub dihedral_deg: f64,
    pub closed: bool,
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub points: Vec<Point>,
}

pub struct RawAdjacency {
    pub regions: [u32; 2],
    pub boundaries: Vec<RawBoundary>,
}

pub struct VertexEntry {
    pub mesh: u32,
    pub position: Point,
    pub role: VertexRole,
    pub regions: Vec<u32>,
}

pub const KIND_HYSTERESIS_DEG: f64 = 0.5;
const MIN_RUN_SEGMENTS: usize = 2;

/// A run of polyline nodes `[lo, hi]`; on a closed polyline of `n` nodes it
/// may wrap past the end.
#[derive(Clone, Copy)]
struct Span {
    lo: usize,
    hi: usize,
}

impl Span {
    fn len(&self, n: usize, closed: bool) -> usize {
        if closed {
            (self.hi + n - self.lo) % n
        } else {
            self.hi - self.lo
        }
    }

    fn nodes(&self, n: usize, closed: bool) -> Vec<usize> {
        (0..=self.len(n, closed))
            .map(|k| (self.lo + k) % n)
            .collect()
    }
}

fn spans(joints: &[usize], n: usize, closed: bool) -> Vec<Span> {
    if closed {
        (0..joints.len())
            .map(|j| Span {
                lo: joints[j],
                hi: joints[(j + 1) % joints.len()],
            })
            .collect()
    } else {
        let mut bounds = vec![0];
        bounds.extend_from_slice(joints);
        bounds.push(n - 1);
        bounds
            .windows(2)
            .map(|w| Span { lo: w[0], hi: w[1] })
            .collect()
    }
}

fn lex_points(a: &[Point], b: &[Point]) -> std::cmp::Ordering {
    for (p, q) in a.iter().zip(b) {
        let o = lex(p, q);
        if o.is_ne() {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

/// The nodes where a polyline's kind changes, with hysteresis: a change
/// counts only between samples below `threshold - KIND_HYSTERESIS_DEG` and
/// above `threshold + KIND_HYSTERESIS_DEG`, each run spans at least
/// `MIN_RUN_SEGMENTS` segments, and neighbouring runs of the same kind merge.
/// Every choice is ordered by node position, so the result does not depend on
/// walk direction or on where a closed loop starts. `allowed` excludes nodes
/// that cannot become a vertex. Returns the joints in node order.
pub(super) fn kind_joints(
    pts: &[Point],
    samples: &[f64],
    closed: bool,
    threshold: f64,
    allowed: &dyn Fn(usize) -> bool,
) -> Vec<usize> {
    let n = pts.len();
    if samples.len() != n || n < 2 {
        return Vec::new();
    }
    let lo = threshold - KIND_HYSTERESIS_DEG;
    let hi = threshold + KIND_HYSTERESIS_DEG;
    let tangent: Vec<bool> = samples.iter().map(|&s| s < lo).collect();
    let transversal: Vec<bool> = samples.iter().map(|&s| s > hi).collect();
    if !tangent.iter().any(|&t| t) || !transversal.iter().any(|&t| t) {
        return Vec::new();
    }
    let out: Vec<usize> = (0..n).filter(|&i| tangent[i] || transversal[i]).collect();
    let mut pairs: Vec<(usize, usize)> = out.windows(2).map(|w| (w[0], w[1])).collect();
    if closed && out.len() > 1 {
        pairs.push((out[out.len() - 1], out[0]));
    }
    let mut joints: Vec<usize> = Vec::new();
    for (a, b) in pairs {
        if tangent[a] == tangent[b] {
            continue;
        }
        let between: Vec<usize> = if a < b {
            (a + 1..b).collect()
        } else {
            (a + 1..n).chain(0..b).collect()
        };
        let between: Vec<usize> = between.into_iter().filter(|&i| allowed(i)).collect();
        let cands = if between.is_empty() {
            vec![a, b]
        } else {
            between
        };
        let pick = cands
            .into_iter()
            .min_by(|&x, &y| {
                (samples[x] - threshold)
                    .abs()
                    .total_cmp(&(samples[y] - threshold).abs())
                    .then(lex(&pts[x], &pts[y]))
                    .then(x.cmp(&y))
            })
            .unwrap();
        if !joints.contains(&pick) {
            joints.push(pick);
        }
    }
    joints.sort_unstable();
    let drop_key = |j: usize| (pts[j], samples[j]);
    let drop_cmp = |x: &usize, y: &usize| {
        let (px, sx) = drop_key(*x);
        let (py, sy) = drop_key(*y);
        lex(&px, &py).then(sx.total_cmp(&sy))
    };
    let enough = |joints: &[usize]| !joints.is_empty() && (!closed || joints.len() >= 2);
    loop {
        let all = spans(&joints, n, closed);
        let bad: Vec<Span> = all
            .into_iter()
            .filter(|s| s.len(n, closed) < MIN_RUN_SEGMENTS)
            .collect();
        if bad.is_empty() {
            break;
        }
        if joints.len() < if closed { 2 } else { 1 } {
            return Vec::new();
        }
        let key = |s: &Span| {
            let mut v: Vec<Point> = s.nodes(n, closed).iter().map(|&i| pts[i]).collect();
            v.sort_by(lex);
            (s.len(n, closed), v)
        };
        let worst = bad
            .iter()
            .min_by(|x, y| {
                let (lx, vx) = key(x);
                let (ly, vy) = key(y);
                lx.cmp(&ly).then(lex_points(&vx, &vy))
            })
            .unwrap();
        let drop = [worst.lo, worst.hi]
            .into_iter()
            .filter(|b| joints.contains(b))
            .min_by(drop_cmp)
            .unwrap();
        joints.retain(|&j| j != drop);
    }
    if !enough(&joints) {
        return Vec::new();
    }
    loop {
        let all = spans(&joints, n, closed);
        let kinds: Vec<bool> = all
            .iter()
            .map(|s| median(s.nodes(n, closed).iter().map(|&i| samples[i]).collect()) < threshold)
            .collect();
        let shared: Vec<usize> = if closed {
            (0..all.len())
                .filter(|&j| kinds[j] == kinds[(j + 1) % all.len()])
                .map(|j| joints[(j + 1) % joints.len()])
                .collect()
        } else {
            (0..all.len() - 1)
                .filter(|&j| kinds[j] == kinds[j + 1])
                .map(|j| joints[j])
                .collect()
        };
        let Some(drop) = shared.into_iter().min_by(drop_cmp) else {
            break;
        };
        joints.retain(|&j| j != drop);
        if !enough(&joints) {
            return Vec::new();
        }
    }
    joints
}

pub(super) fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub(super) fn lex(a: &Point, b: &Point) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

pub struct BuildArgs<'a> {
    pub nbr: &'a [[u32; 3]],
    pub faces: &'a [[u32; 3]],
    pub flabel: &'a [u32],
    pub finals: &'a [Final],
    pub pos: &'a [Point],
    pub orig: &'a [Point],
    pub center: V3,
    pub tangent_threshold_deg: f64,
    pub reach: f64,
}

pub struct Built {
    pub adjacencies: Vec<RawAdjacency>,
    pub vertices: Vec<VertexEntry>,
    pub moved_deviation: f64,
}

pub fn build(args: BuildArgs<'_>) -> Built {
    let BuildArgs {
        nbr,
        faces,
        flabel,
        finals,
        pos,
        orig,
        center,
        tangent_threshold_deg,
        reach,
    } = args;
    let nv = pos.len();
    let mut vr: Vec<u64> = Vec::new();
    for (f, t) in faces.iter().enumerate() {
        if flabel[f] == NONE {
            continue;
        }
        for &v in t {
            vr.push(((v as u64) << 32) | flabel[f] as u64);
        }
    }
    vr.sort_unstable();
    vr.dedup();
    let mut vstart = vec![0usize; nv + 1];
    for k in &vr {
        vstart[(k >> 32) as usize + 1] += 1;
    }
    for i in 0..nv {
        vstart[i + 1] += vstart[i];
    }
    let vregs = |v: u32| -> Vec<u32> {
        vr[vstart[v as usize]..vstart[v as usize + 1]]
            .iter()
            .map(|k| (k & 0xffff_ffff) as u32)
            .collect()
    };
    let is_junction = |v: u32| vstart[v as usize + 1] - vstart[v as usize] >= 3;

    let mut vfaces: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (f, t) in faces.iter().enumerate() {
        if flabel[f] == NONE {
            continue;
        }
        for &v in t {
            vfaces[v as usize].push(f as u32);
        }
    }

    let tri_normal = |f: u32| -> V3 {
        let t = faces[f as usize];
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        unit(cross(sub(b, a), sub(c, a)))
    };
    let normal_of = |r: u32, mid: V3, f: u32| -> V3 {
        finals[r as usize]
            .surface
            .normal_at(mid)
            .unwrap_or_else(|| tri_normal(f))
    };

    let mut edges: Vec<EdgeRec> = Vec::new();
    for (f, nb) in nbr.iter().enumerate() {
        let ra = flabel[f];
        if ra == NONE {
            continue;
        }
        for (k, &g) in nb.iter().enumerate() {
            if g == NONE {
                continue;
            }
            let rb = flabel[g as usize];
            if rb != NONE && ra < rb {
                edges.push(EdgeRec {
                    ra,
                    rb,
                    u: faces[f][k],
                    v: faces[f][(k + 1) % 3],
                    fa: f as u32,
                    fb: g,
                });
            }
        }
    }
    edges.sort_by_key(|e| (e.ra, e.rb, e.u, e.v));

    let thr = tangent_threshold_deg;
    let near_mesh = |x: Point, v: u32| -> f64 {
        let mut best = f64::INFINITY;
        for &f in &vfaces[v as usize] {
            for &k in &faces[f as usize] {
                for &g in &vfaces[k as usize] {
                    let t = faces[g as usize].map(|m| orig[m as usize]);
                    best = best.min(norm(sub(x, closest_on_triangle(x, t))));
                }
            }
        }
        best
    };
    let mut moved_deviation: f64 = 0.0;
    let mut reg: Vec<VertexEntry> = Vec::new();
    let mut ids: FxHashMap<u32, usize> = FxHashMap::default();
    let mut adjacencies: Vec<RawAdjacency> = Vec::new();
    let mut s = 0;
    while s < edges.len() {
        let mut e = s + 1;
        while e < edges.len() && edges[e].ra == edges[s].ra && edges[e].rb == edges[s].rb {
            e += 1;
        }
        let group = &edges[s..e];
        let (ra, rb) = (group[0].ra, group[0].rb);
        let (sa, sb) = (&finals[ra as usize].surface, &finals[rb as usize].surface);
        let both_analytic = sa.is_analytic() && sb.is_analytic();
        let sample = |i: usize, p: Point| -> f64 {
            let q = sub(p, center);
            angle_deg(normal_of(ra, q, group[i].fa), normal_of(rb, q, group[i].fb))
        };
        let mut out: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        let mut in_edge: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (i, g) in group.iter().enumerate() {
            out.entry(g.u).or_default().push(i);
            in_edge.entry(g.v).or_default().push(i);
        }
        let simple = |v: u32| -> bool {
            matches!((out.get(&v), in_edge.get(&v)), (Some(o), Some(i)) if o.len() == 1 && i.len() == 1)
        };
        let is_break = |v: u32| -> bool {
            is_junction(v)
                || !matches!((out.get(&v), in_edge.get(&v)), (Some(o), Some(i)) if o.len() == i.len())
        };
        let fan_of = |head: u32, t0: u32| -> FxHashSet<u32> {
            let region = flabel[t0 as usize];
            let mut seen: FxHashSet<u32> = FxHashSet::default();
            let mut stack = vec![t0];
            seen.insert(t0);
            while let Some(t) = stack.pop() {
                for &x in &faces[t as usize] {
                    if x == head {
                        continue;
                    }
                    for &f2 in &vfaces[head as usize] {
                        if flabel[f2 as usize] != region || !faces[f2 as usize].contains(&x) {
                            continue;
                        }
                        if seen.insert(f2) {
                            stack.push(f2);
                        }
                    }
                }
            }
            seen
        };
        let next = |head: u32, incoming: usize, visited: &[bool]| -> Option<usize> {
            let cands = out.get(&head)?;
            let mut unvis = cands.iter().filter(|&&j| !visited[j]);
            let first = *unvis.next()?;
            let mut rest: Vec<usize> = unvis.copied().collect();
            if rest.is_empty() {
                return Some(first);
            }
            rest.insert(0, first);
            let fan = fan_of(head, group[incoming].fa);
            rest.iter()
                .find(|&&j| fan.contains(&group[j].fa))
                .copied()
                .or(Some(first))
        };
        let mut visited = vec![false; group.len()];
        let mut boundaries: Vec<RawBoundary> = Vec::new();
        let mut make = |chain: &[usize],
                        closed: bool,
                        reg: &mut Vec<VertexEntry>,
                        ids: &mut FxHashMap<u32, usize>,
                        moved_deviation: &mut f64| {
            let mesh: Vec<u32> = if closed {
                chain.iter().map(|&i| group[i].u).collect()
            } else {
                std::iter::once(group[chain[0]].u)
                    .chain(chain.iter().map(|&i| group[i].v))
                    .collect()
            };
            let n = mesh.len();
            let edge_at = |k: usize| chain[k.min(chain.len() - 1)];
            let mut points: Vec<Point> = mesh.iter().map(|&m| pos[m as usize]).collect();
            let samples: Vec<f64> = (0..n).map(|k| sample(edge_at(k), points[k])).collect();
            let allowed = |k: usize| (closed || (k > 0 && k + 1 < n)) && simple(mesh[k]);
            let joints = kind_joints(&points, &samples, closed, thr, &allowed);
            let mut vid = |v: u32, at: Point, reg: &mut Vec<VertexEntry>| -> u32 {
                let len = reg.len();
                let id = *ids.entry(v).or_insert(len);
                if id == len {
                    let (role, regions) = if is_junction(v) {
                        (VertexRole::Junction, vregs(v))
                    } else {
                        (VertexRole::KindChange, vec![ra, rb])
                    };
                    reg.push(VertexEntry {
                        mesh: v,
                        position: at,
                        role,
                        regions,
                    });
                }
                id as u32
            };
            for &j in &joints {
                if !both_analytic {
                    continue;
                }
                let at = |k: usize| sub(points[k], center);
                let pj = at(j);
                let prev = if j > 0 {
                    Some(j - 1)
                } else {
                    closed.then(|| n - 1)
                };
                let after = if j + 1 < n {
                    Some(j + 1)
                } else {
                    closed.then_some(0)
                };
                let x = [prev.map(|k| (k, j)), after.map(|k| (j, k))]
                    .into_iter()
                    .flatten()
                    .find_map(|(a, b)| onto_crossing(sa, sb, at(a), at(b), thr))
                    .unwrap_or_else(|| lsq_point(pj, pj, &[sa, sb]));
                let w = add(x, center);
                let dev = near_mesh(w, mesh[j]) + off_surfaces(x, &[sa, sb]);
                if dev <= reach {
                    points[j] = w;
                    *moved_deviation = moved_deviation.max(dev);
                }
            }
            let kind_of = |d: f64| {
                if d < thr {
                    Kind::Tangent
                } else {
                    Kind::Transversal
                }
            };
            if joints.is_empty() {
                let d = if !closed && n == 2 {
                    sample(chain[0], scale(add(points[0], points[1]), 0.5))
                } else {
                    median(samples.clone())
                };
                let (sv, ev) = if closed {
                    (None, None)
                } else {
                    (
                        Some(vid(mesh[0], points[0], reg)),
                        Some(vid(mesh[n - 1], points[n - 1], reg)),
                    )
                };
                if closed {
                    let first = (0..n).min_by(|&a, &b| lex(&points[a], &points[b])).unwrap();
                    points.rotate_left(first);
                }
                boundaries.push(RawBoundary {
                    kind: kind_of(d),
                    dihedral_deg: d,
                    closed,
                    start: sv,
                    end: ev,
                    points,
                });
                return;
            }
            for span in spans(&joints, n, closed) {
                let nodes = span.nodes(n, closed);
                let d = median(nodes.iter().map(|&k| samples[k]).collect());
                let (a, b) = (nodes[0], nodes[nodes.len() - 1]);
                boundaries.push(RawBoundary {
                    kind: kind_of(d),
                    dihedral_deg: d,
                    closed: false,
                    start: Some(vid(mesh[a], points[a], reg)),
                    end: Some(vid(mesh[b], points[b], reg)),
                    points: nodes.iter().map(|&k| points[k]).collect(),
                });
            }
        };
        for i in 0..group.len() {
            if visited[i] || !is_break(group[i].u) {
                continue;
            }
            let mut chain = vec![i];
            visited[i] = true;
            let mut head = group[i].v;
            let mut incoming = i;
            while !is_break(head) {
                match next(head, incoming, &visited) {
                    Some(nxt) => {
                        visited[nxt] = true;
                        chain.push(nxt);
                        head = group[nxt].v;
                        incoming = nxt;
                    }
                    None => break,
                }
            }
            make(&chain, false, &mut reg, &mut ids, &mut moved_deviation);
        }
        for i in 0..group.len() {
            if visited[i] {
                continue;
            }
            let mut chain = vec![i];
            visited[i] = true;
            let mut head = group[i].v;
            let mut incoming = i;
            while head != group[i].u {
                match next(head, incoming, &visited) {
                    Some(nxt) => {
                        visited[nxt] = true;
                        chain.push(nxt);
                        head = group[nxt].v;
                        incoming = nxt;
                    }
                    None => break,
                }
            }
            make(&chain, true, &mut reg, &mut ids, &mut moved_deviation);
        }
        boundaries.sort_by(|a, b| lex(&a.points[0], &b.points[0]));
        adjacencies.push(RawAdjacency {
            regions: [ra, rb],
            boundaries,
        });
        s = e;
    }
    Built {
        adjacencies,
        vertices: reg,
        moved_deviation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> Vec<Point> {
        (0..n).map(|i| [i as f64, 0.0, 0.0]).collect()
    }

    fn joints(samples: &[f64], closed: bool) -> Vec<usize> {
        let pts = line(samples.len());
        kind_joints(&pts, samples, closed, 3.0, &|_| true)
    }

    #[test]
    fn one_crossing_splits_at_the_node_nearest_the_threshold() {
        assert_eq!(joints(&[1.0, 1.0, 1.0, 2.9, 5.0, 5.0, 5.0], false), vec![3]);
        assert_eq!(joints(&[1.0, 1.0, 1.0, 3.6, 5.0, 5.0, 5.0], false), vec![3]);
    }

    #[test]
    fn a_single_segment_spike_is_merged_away() {
        assert!(joints(&[1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0], false).is_empty());
        assert!(joints(&[1.0, 5.0, 5.0, 5.0, 5.0], false).is_empty());
    }

    #[test]
    fn samples_inside_the_band_never_split() {
        assert!(joints(&[2.6, 3.4, 2.6, 3.4, 2.6, 3.4], false).is_empty());
        assert!(joints(&[1.0, 1.0, 3.2, 3.4, 3.3, 3.4], false).is_empty());
    }

    #[test]
    fn split_does_not_depend_on_walk_direction() {
        let s = [1.0, 1.0, 2.0, 3.1, 3.2, 4.0, 4.0, 4.0];
        let pts = line(s.len());
        let fwd = kind_joints(&pts, &s, false, 3.0, &|_| true);
        let mut rp = pts.clone();
        rp.reverse();
        let mut rs = s.to_vec();
        rs.reverse();
        let back: Vec<usize> = kind_joints(&rp, &rs, false, 3.0, &|_| true)
            .into_iter()
            .map(|j| s.len() - 1 - j)
            .collect();
        assert_eq!(fwd, back);
        assert_eq!(fwd, vec![3]);
    }

    #[test]
    fn closed_loop_splits_twice_and_ignores_its_start() {
        let s = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let pts: Vec<Point> = (0..8)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 8.0;
                [t.cos(), t.sin(), 0.0]
            })
            .collect();
        let base = kind_joints(&pts, &s, true, 3.0, &|_| true);
        assert_eq!(base.len(), 2);
        for k in 1..8 {
            let rp: Vec<Point> = (0..8).map(|i| pts[(i + k) % 8]).collect();
            let rs: Vec<f64> = (0..8).map(|i| s[(i + k) % 8]).collect();
            let mut got: Vec<usize> = kind_joints(&rp, &rs, true, 3.0, &|_| true)
                .into_iter()
                .map(|j| (j + k) % 8)
                .collect();
            got.sort_unstable();
            assert_eq!(got, base, "start {k}");
        }
    }

    #[test]
    fn disallowed_nodes_are_never_joints() {
        let s = [1.0, 1.0, 1.0, 2.9, 3.0, 5.0, 5.0, 5.0];
        let pts = line(s.len());
        assert_eq!(kind_joints(&pts, &s, false, 3.0, &|k| k != 4), vec![3]);
    }
}
