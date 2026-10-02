use rustc_hash::FxHashMap;

use super::fit::Final;
use super::linalg::{V3, angle_deg, cross, sub, unit};
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
    pub role: VertexRole,
    pub regions: Vec<u32>,
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

pub fn build(
    nbr: &[[u32; 3]],
    faces: &[[u32; 3]],
    flabel: &[u32],
    finals: &[Final],
    pos: &[Point],
    tangent_threshold_deg: f64,
) -> (Vec<RawAdjacency>, Vec<VertexEntry>) {
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

    let tri_normal = |f: u32| -> V3 {
        let t = faces[f as usize];
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        unit(cross(sub(b, a), sub(c, a)))
    };
    let normal_of =
        |r: u32, f: u32| -> V3 { finals[r as usize].surface.outward_normal(tri_normal(f)) };

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
        let dih: Vec<f64> = group
            .iter()
            .map(|g| angle_deg(normal_of(ra, g.fa), normal_of(rb, g.fb)))
            .collect();
        let tangent: Vec<bool> = dih.iter().map(|d| *d < thr).collect();
        let mut out: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        let mut in_edge: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (i, g) in group.iter().enumerate() {
            out.entry(g.u).or_default().push(i);
            in_edge.entry(g.v).or_default().push(i);
        }
        let is_break = |v: u32| -> bool {
            if is_junction(v) {
                return true;
            }
            match (out.get(&v), in_edge.get(&v)) {
                (Some(o), Some(i)) if o.len() == 1 && i.len() == 1 => {
                    tangent[o[0]] != tangent[i[0]]
                }
                _ => false,
            }
        };
        let next = |head: u32, incoming: usize, visited: &[bool]| -> Option<usize> {
            let cands = out.get(&head)?;
            cands
                .iter()
                .find(|&&j| !visited[j] && tangent[j] == tangent[incoming])
                .copied()
                .or_else(|| cands.iter().find(|&&j| !visited[j]).copied())
        };
        let mut visited = vec![false; group.len()];
        let mut boundaries: Vec<RawBoundary> = Vec::new();
        let mut make = |chain: &[usize],
                        closed: bool,
                        reg: &mut Vec<VertexEntry>,
                        ids: &mut FxHashMap<u32, usize>| {
            let mut points: Vec<Point> = Vec::new();
            if closed {
                points.extend(chain.iter().map(|&i| pos[group[i].u as usize]));
                let first = (0..points.len())
                    .min_by(|&a, &b| lex(&points[a], &points[b]))
                    .unwrap();
                points.rotate_left(first);
            } else {
                points.push(pos[group[chain[0]].u as usize]);
                points.extend(chain.iter().map(|&i| pos[group[i].v as usize]));
            }
            let d = median(chain.iter().map(|&i| dih[i]).collect());
            let mut vid = |v: u32| -> u32 {
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
                        role,
                        regions,
                    });
                }
                id as u32
            };
            let (sv, ev) = if closed {
                (None, None)
            } else {
                (
                    Some(vid(group[chain[0]].u)),
                    Some(vid(group[*chain.last().unwrap()].v)),
                )
            };
            boundaries.push(RawBoundary {
                kind: if d < thr {
                    Kind::Tangent
                } else {
                    Kind::Transversal
                },
                dihedral_deg: d,
                closed,
                start: sv,
                end: ev,
                points,
            });
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
            make(&chain, false, &mut reg, &mut ids);
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
            make(&chain, true, &mut reg, &mut ids);
        }
        boundaries.sort_by(|a, b| lex(&a.points[0], &b.points[0]));
        adjacencies.push(RawAdjacency {
            regions: [ra, rb],
            boundaries,
        });
        s = e;
    }
    (adjacencies, reg)
}
