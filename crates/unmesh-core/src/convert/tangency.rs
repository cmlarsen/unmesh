use super::adjacency::median;
use super::curved::{basis, sagitta, solve_dense};
use super::dsu::Dsu;
use super::fit::Final;
use super::linalg::{V3, add, cross, dot, norm, scale, sub, unit};
use super::refine::dihedral;
use super::segment::TriInfo;
use super::surface::Surface;
use super::topology::NONE;

const DEFECT_REL: f64 = 1e-4;
const REACH: f64 = 5.0;
const EXACT_REL: f64 = 1e-10;
const RANK_EPS: f64 = 1e-9;
const ITERS: usize = 30;
const MAX_PARAMS: usize = 600;
const SSR_SLACK: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rel {
    PlaneCyl { plane: usize, cyl: usize },
    PlaneTorus { plane: usize, torus: usize },
    CylTorus { cyl: usize, torus: usize },
    CylSphere { cyl: usize, sphere: usize },
}

impl Rel {
    fn ends(self) -> (usize, usize) {
        match self {
            Rel::PlaneCyl { plane, cyl } => (plane, cyl),
            Rel::PlaneTorus { plane, torus } => (plane, torus),
            Rel::CylTorus { cyl, torus } => (cyl, torus),
            Rel::CylSphere { cyl, sphere } => (cyl, sphere),
        }
    }

    fn map(self, f: impl Fn(usize) -> usize) -> Rel {
        match self {
            Rel::PlaneCyl { plane, cyl } => Rel::PlaneCyl {
                plane: f(plane),
                cyl: f(cyl),
            },
            Rel::PlaneTorus { plane, torus } => Rel::PlaneTorus {
                plane: f(plane),
                torus: f(torus),
            },
            Rel::CylTorus { cyl, torus } => Rel::CylTorus {
                cyl: f(cyl),
                torus: f(torus),
            },
            Rel::CylSphere { cyl, sphere } => Rel::CylSphere {
                cyl: f(cyl),
                sphere: f(sphere),
            },
        }
    }

    fn direction_constraints(self) -> usize {
        match self {
            Rel::PlaneCyl { .. } => 1,
            Rel::PlaneTorus { .. } | Rel::CylTorus { .. } => 2,
            Rel::CylSphere { .. } => 0,
        }
    }
}

fn sign(reversed: bool) -> f64 {
    if reversed { -1.0 } else { 1.0 }
}

fn relation(a: usize, sa: &Surface, b: usize, sb: &Surface) -> Option<Rel> {
    use Surface::*;
    match (sa, sb) {
        (Plane { .. }, Cylinder { .. }) => Some(Rel::PlaneCyl { plane: a, cyl: b }),
        (Cylinder { .. }, Plane { .. }) => Some(Rel::PlaneCyl { plane: b, cyl: a }),
        (Plane { .. }, Torus { .. }) => Some(Rel::PlaneTorus { plane: a, torus: b }),
        (Torus { .. }, Plane { .. }) => Some(Rel::PlaneTorus { plane: b, torus: a }),
        (Cylinder { .. }, Torus { .. }) => Some(Rel::CylTorus { cyl: a, torus: b }),
        (Torus { .. }, Cylinder { .. }) => Some(Rel::CylTorus { cyl: b, torus: a }),
        (Cylinder { reversed: r1, .. }, Sphere { reversed: r2, .. }) if r1 == r2 => {
            Some(Rel::CylSphere { cyl: a, sphere: b })
        }
        (Sphere { reversed: r1, .. }, Cylinder { reversed: r2, .. }) if r1 == r2 => {
            Some(Rel::CylSphere { cyl: b, sphere: a })
        }
        _ => None,
    }
}

fn perp(v: V3, a: V3) -> V3 {
    sub(v, scale(a, dot(v, a)))
}

/// How far `rel` is from exact tangency: the sine of the angle its axes
/// are off, and the largest positional mismatch (offset, axis distance or
/// radius), in length units.
fn defects(rel: Rel, s: &[Surface]) -> Option<(f64, f64)> {
    match rel {
        Rel::PlaneCyl { plane, cyl } => {
            let (n, d) = s[plane].as_plane()?;
            let Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed,
            } = s[cyl]
            else {
                return None;
            };
            Some((
                dot(n, axis).abs(),
                (dot(n, origin) + sign(reversed) * radius - d).abs(),
            ))
        }
        Rel::PlaneTorus { plane, torus } => {
            let (n, d) = s[plane].as_plane()?;
            let Surface::Torus {
                center,
                axis,
                minor,
                reversed,
                ..
            } = s[torus]
            else {
                return None;
            };
            Some((
                norm(cross(n, axis)),
                (dot(n, center) + sign(reversed) * minor - d).abs(),
            ))
        }
        Rel::CylTorus { cyl, torus } => {
            let Surface::Cylinder {
                origin,
                axis: a,
                radius,
                reversed: rc,
            } = s[cyl]
            else {
                return None;
            };
            let Surface::Torus {
                center,
                axis,
                major,
                minor,
                reversed: rt,
            } = s[torus]
            else {
                return None;
            };
            let eps = sign(rc) * sign(rt);
            Some((
                norm(cross(a, axis)),
                norm(perp(sub(center, origin), a)).max((radius - major - eps * minor).abs()),
            ))
        }
        Rel::CylSphere { cyl, sphere } => {
            let Surface::Cylinder {
                origin,
                axis: a,
                radius,
                ..
            } = s[cyl]
            else {
                return None;
            };
            let Surface::Sphere {
                center, radius: rs, ..
            } = s[sphere]
            else {
                return None;
            };
            Some((
                0.0,
                norm(perp(sub(center, origin), a)).max((radius - rs).abs()),
            ))
        }
    }
}

fn axis_of(s: &Surface) -> Option<V3> {
    match *s {
        Surface::Cylinder { axis, .. } | Surface::Torus { axis, .. } => Some(axis),
        _ => None,
    }
}

fn with_axis(s: Surface, a: V3) -> Surface {
    match s {
        Surface::Cylinder {
            origin,
            radius,
            reversed,
            ..
        } => Surface::Cylinder {
            origin,
            axis: a,
            radius,
            reversed,
        },
        Surface::Torus {
            center,
            major,
            minor,
            reversed,
            ..
        } => Surface::Torus {
            center,
            axis: a,
            major,
            minor,
            reversed,
        },
        s => s,
    }
}

/// Snaps the axes of the curved members to the directions their tangencies
/// demand: perpendicular to the normal of every plane a cylinder touches
/// along a line, parallel to the normal of a plane a torus touches along a
/// circle, and shared by a cylinder and the torus it touches. Plane normals
/// stay fixed. `None` when the demands contradict each other by more than
/// `max_sin`.
fn snap_axes(s: &mut [Surface], rels: &[Rel], max_sin: f64) -> Option<()> {
    let n = s.len();
    let mut dsu = Dsu::new(n);
    for &r in rels {
        if let Rel::CylTorus { cyl, torus } = r {
            dsu.union(cyl as u32, torus as u32);
        }
    }
    let mut par: Vec<Vec<V3>> = vec![Vec::new(); n];
    let mut ort: Vec<Vec<V3>> = vec![Vec::new(); n];
    for &r in rels {
        match r {
            Rel::PlaneCyl { plane, cyl } => {
                ort[dsu.find(cyl as u32) as usize].push(s[plane].as_plane()?.0);
            }
            Rel::PlaneTorus { plane, torus } => {
                par[dsu.find(torus as u32) as usize].push(s[plane].as_plane()?.0);
            }
            _ => {}
        }
    }
    let max_cos = (1.0 - max_sin * max_sin).sqrt();
    for root in 0..n {
        if dsu.find(root as u32) as usize != root {
            continue;
        }
        let Some(reference) = axis_of(&s[root]) else {
            continue;
        };
        let g = if let Some(&first) = par[root].first() {
            first
        } else if let Some(&first) = ort[root].first() {
            let mut best = (0.0, first);
            for (i, &u) in ort[root].iter().enumerate() {
                for &v in &ort[root][i + 1..] {
                    let c = cross(u, v);
                    if norm(c) > best.0 {
                        best = (norm(c), c);
                    }
                }
            }
            if best.0 > max_sin {
                unit(best.1)
            } else {
                let p = perp(reference, first);
                if norm(p) == 0.0 {
                    return None;
                }
                if p == reference { reference } else { unit(p) }
            }
        } else {
            reference
        };
        let g = if dot(g, reference) < 0.0 {
            scale(g, -1.0)
        } else {
            g
        };
        if dot(g, reference) < max_cos {
            return None;
        }
        if par[root].iter().any(|&u| dot(u, g).abs() < max_cos)
            || ort[root].iter().any(|&u| dot(u, g).abs() > max_sin)
        {
            return None;
        }
        for m in 0..n {
            if dsu.find(m as u32) as usize != root {
                continue;
            }
            if let Some(a) = axis_of(&s[m]) {
                let new = if dot(a, g) < 0.0 { scale(g, -1.0) } else { g };
                if new != a {
                    s[m] = with_axis(s[m], new);
                }
            }
        }
    }
    Some(())
}

#[derive(Clone, Copy)]
struct Block {
    start: usize,
    len: usize,
    base: Surface,
    e1: V3,
    e2: V3,
}

fn block_len(s: &Surface) -> Option<usize> {
    match s {
        Surface::Plane { .. } => Some(1),
        Surface::Cylinder { .. } => Some(3),
        Surface::Torus { .. } => Some(5),
        Surface::Sphere { .. } => Some(4),
        _ => None,
    }
}

impl Block {
    fn surface(&self, t: &[f64]) -> Surface {
        let t = &t[self.start..self.start + self.len];
        match self.base {
            Surface::Plane { normal, offset } => Surface::Plane {
                normal,
                offset: offset + t[0],
            },
            Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed,
            } => Surface::Cylinder {
                origin: add(origin, add(scale(self.e1, t[0]), scale(self.e2, t[1]))),
                axis,
                radius: radius + t[2],
                reversed,
            },
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                reversed,
            } => Surface::Torus {
                center: add(center, [t[0], t[1], t[2]]),
                axis,
                major: major + t[3],
                minor: minor + t[4],
                reversed,
            },
            Surface::Sphere {
                center,
                radius,
                reversed,
            } => Surface::Sphere {
                center: add(center, [t[0], t[1], t[2]]),
                radius: radius + t[3],
                reversed,
            },
            s => s,
        }
    }

    fn residual(&self, s: &Surface, p: V3, jac: &mut [f64; 5]) -> f64 {
        match *s {
            Surface::Plane { normal, offset } => {
                jac[0] = -1.0;
                dot(normal, p) - offset
            }
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                let w = perp(sub(p, origin), axis);
                let rho = norm(w);
                let w = if rho > 0.0 {
                    scale(w, 1.0 / rho)
                } else {
                    [0.0; 3]
                };
                jac[0] = -dot(w, self.e1);
                jac[1] = -dot(w, self.e2);
                jac[2] = -1.0;
                rho - radius
            }
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => {
                let q = sub(p, center);
                let h = dot(q, axis);
                let rv = sub(q, scale(axis, h));
                let rho = norm(rv);
                let u = if rho > 0.0 {
                    scale(rv, 1.0 / rho)
                } else {
                    [0.0; 3]
                };
                let gx = rho - major;
                let gn = gx.hypot(h);
                let (m, k) = if gn > 0.0 {
                    (gx / gn, h / gn)
                } else {
                    (0.0, 0.0)
                };
                let nn = add(scale(u, m), scale(axis, k));
                jac[0] = -nn[0];
                jac[1] = -nn[1];
                jac[2] = -nn[2];
                jac[3] = -m;
                jac[4] = -1.0;
                gn - minor
            }
            Surface::Sphere { center, radius, .. } => {
                let q = sub(p, center);
                let l = norm(q);
                let w = if l > 0.0 { scale(q, 1.0 / l) } else { [0.0; 3] };
                jac[0] = -w[0];
                jac[1] = -w[1];
                jac[2] = -w[2];
                jac[3] = -1.0;
                l - radius
            }
            _ => 0.0,
        }
    }
}

fn constraint_rows(rel: Rel, b: &[Block], n: usize) -> Vec<(Vec<f64>, f64)> {
    let mut rows = Vec::new();
    let row = || vec![0.0; n];
    match rel {
        Rel::PlaneCyl { plane, cyl } => {
            let (p, c) = (&b[plane], &b[cyl]);
            let (
                Surface::Plane { normal, offset },
                Surface::Cylinder {
                    origin,
                    radius,
                    reversed,
                    ..
                },
            ) = (p.base, c.base)
            else {
                return rows;
            };
            let sg = sign(reversed);
            let mut r = row();
            r[c.start] = dot(normal, c.e1);
            r[c.start + 1] = dot(normal, c.e2);
            r[c.start + 2] = sg;
            r[p.start] = -1.0;
            rows.push((r, -(dot(normal, origin) + sg * radius - offset)));
        }
        Rel::PlaneTorus { plane, torus } => {
            let (p, t) = (&b[plane], &b[torus]);
            let (
                Surface::Plane { normal, offset },
                Surface::Torus {
                    center,
                    minor,
                    reversed,
                    ..
                },
            ) = (p.base, t.base)
            else {
                return rows;
            };
            let sg = sign(reversed);
            let mut r = row();
            for k in 0..3 {
                r[t.start + k] = normal[k];
            }
            r[t.start + 4] = sg;
            r[p.start] = -1.0;
            rows.push((r, -(dot(normal, center) + sg * minor - offset)));
        }
        Rel::CylTorus { cyl, torus } => {
            let (c, t) = (&b[cyl], &b[torus]);
            let (
                Surface::Cylinder {
                    origin,
                    radius,
                    reversed: rc,
                    ..
                },
                Surface::Torus {
                    center,
                    major,
                    minor,
                    reversed: rt,
                    ..
                },
            ) = (c.base, t.base)
            else {
                return rows;
            };
            for (i, e) in [c.e1, c.e2].into_iter().enumerate() {
                let mut r = row();
                for k in 0..3 {
                    r[t.start + k] = e[k];
                }
                r[c.start + i] = -1.0;
                rows.push((r, -dot(e, sub(center, origin))));
            }
            let eps = sign(rc) * sign(rt);
            let mut r = row();
            r[c.start + 2] = 1.0;
            r[t.start + 3] = -1.0;
            r[t.start + 4] = -eps;
            rows.push((r, -(radius - major - eps * minor)));
        }
        Rel::CylSphere { cyl, sphere } => {
            let (c, s) = (&b[cyl], &b[sphere]);
            let (
                Surface::Cylinder { origin, radius, .. },
                Surface::Sphere {
                    center, radius: rs, ..
                },
            ) = (c.base, s.base)
            else {
                return rows;
            };
            for (i, e) in [c.e1, c.e2].into_iter().enumerate() {
                let mut r = row();
                for k in 0..3 {
                    r[s.start + k] = e[k];
                }
                r[c.start + i] = -1.0;
                rows.push((r, -dot(e, sub(center, origin))));
            }
            let mut r = row();
            r[c.start + 2] = 1.0;
            r[s.start + 3] = -1.0;
            rows.push((r, -(radius - rs)));
        }
    }
    rows
}

/// The affine parametrisation `t0 + T z` of every solution of the linear
/// constraints, by Gauss-Jordan elimination with full pivoting (the rows
/// of a cluster are often redundant: a corner sphere sits on three axes).
/// `None` when the redundant rows disagree by more than `eps`.
fn null_space(rows: Vec<(Vec<f64>, f64)>, n: usize, eps: f64) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let m = rows.len();
    let (mut a, mut b): (Vec<Vec<f64>>, Vec<f64>) = rows.into_iter().unzip();
    let mut pivots: Vec<usize> = Vec::new();
    let mut used = vec![false; n];
    for r in 0..m {
        let mut best = (0.0, 0, 0);
        for (i, row) in a.iter().enumerate().skip(r) {
            for (j, &v) in row.iter().enumerate() {
                if !used[j] && v.abs() > best.0 {
                    best = (v.abs(), i, j);
                }
            }
        }
        if best.0 <= RANK_EPS {
            break;
        }
        let (_, i, j) = best;
        a.swap(r, i);
        b.swap(r, i);
        let p = a[r][j];
        for v in a[r].iter_mut() {
            *v /= p;
        }
        b[r] /= p;
        for k in 0..m {
            if k == r {
                continue;
            }
            let f = a[k][j];
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                a[k][c] -= f * a[r][c];
            }
            b[k] -= f * b[r];
        }
        used[j] = true;
        pivots.push(j);
    }
    if b[pivots.len()..].iter().any(|v| v.abs() > eps) {
        return None;
    }
    let free: Vec<usize> = (0..n).filter(|&j| !used[j]).collect();
    let mut t0 = vec![0.0; n];
    let mut t = vec![vec![0.0; free.len()]; n];
    for (k, &j) in free.iter().enumerate() {
        t[j][k] = 1.0;
    }
    for (r, &pj) in pivots.iter().enumerate() {
        t0[pj] = b[r];
        for (k, &j) in free.iter().enumerate() {
            t[pj][k] = -a[r][j];
        }
    }
    Some((t0, t))
}

pub struct Member {
    pub surface: Surface,
    pub pts: Vec<(V3, f64)>,
}

fn ssr(s: &Surface, pts: &[(V3, f64)]) -> (f64, f64, f64) {
    let mut s2 = 0.0;
    let mut sw = 0.0;
    let mut mx: f64 = 0.0;
    for &(p, w) in pts {
        let d = s.distance(p);
        s2 += w * d * d;
        sw += w;
        mx = mx.max(d.abs());
    }
    (s2, sw, mx)
}

fn valid(s: &Surface) -> bool {
    match *s {
        Surface::Cylinder { radius, .. } | Surface::Sphere { radius, .. } => radius > 0.0,
        Surface::Torus { major, minor, .. } => minor > 0.0 && major > minor,
        _ => true,
    }
}

/// Refits the members jointly so that every relation in `rels` holds
/// exactly: least squares over the members' weighted points, with the
/// plane normals fixed, the axes snapped first (`snap_axes`), and the
/// remaining offsets, axis positions, centres and radii constrained by the
/// linear tangency conditions. Returns the new surfaces when every member
/// still fits its points within `tol` and the total weighted squared
/// residual grows by no more than the constraints allow for the noise the
/// unconstrained fit shows.
pub fn solve(members: &[Member], rels: &[Rel], tol: f64, max_sin: f64) -> Option<Vec<Surface>> {
    let mut s: Vec<Surface> = members.iter().map(|m| m.surface).collect();
    snap_axes(&mut s, rels, max_sin)?;
    let mut blocks = Vec::with_capacity(s.len());
    let mut n = 0;
    let mut extent: f64 = 0.0;
    for m in members {
        for &(p, _) in &m.pts {
            extent = extent.max(norm(p));
        }
    }
    for &surface in &s {
        let len = block_len(&surface)?;
        let (e1, e2) = match surface {
            Surface::Cylinder { axis, .. } => basis(axis),
            _ => ([0.0; 3], [0.0; 3]),
        };
        blocks.push(Block {
            start: n,
            len,
            base: surface,
            e1,
            e2,
        });
        n += len;
    }
    if n > MAX_PARAMS {
        return None;
    }
    let exact = EXACT_REL * (1.0 + extent);
    let rows: Vec<(Vec<f64>, f64)> = rels
        .iter()
        .flat_map(|&r| constraint_rows(r, &blocks, n))
        .collect();
    let k_pos = rows.len();
    let (t0, t) = null_space(rows, n, exact)?;
    let nf = t.first().map_or(0, |r| r.len());
    let theta = |z: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| t0[i] + t[i].iter().zip(z).map(|(a, b)| a * b).sum::<f64>())
            .collect()
    };
    let cost = |th: &[f64]| -> f64 {
        blocks
            .iter()
            .zip(members)
            .map(|(b, m)| ssr(&b.surface(th), &m.pts).0)
            .sum()
    };
    let mut z = vec![0.0; nf];
    let mut th = theta(&z);
    let mut c = cost(&th);
    let mut lambda = 1e-9;
    for _ in 0..ITERS {
        if nf == 0 {
            break;
        }
        let mut ht = vec![vec![0.0; nf]; n];
        let mut g = vec![0.0; n];
        for (b, m) in blocks.iter().zip(members) {
            let surf = b.surface(&th);
            let mut h = [[0.0; 5]; 5];
            for &(p, w) in &m.pts {
                let mut jac = [0.0; 5];
                let r = b.residual(&surf, p, &mut jac);
                for i in 0..b.len {
                    g[b.start + i] += w * jac[i] * r;
                    for j in 0..b.len {
                        h[i][j] += w * jac[i] * jac[j];
                    }
                }
            }
            for i in 0..b.len {
                for k in 0..nf {
                    ht[b.start + i][k] = (0..b.len).map(|j| h[i][j] * t[b.start + j][k]).sum();
                }
            }
        }
        let mut a = vec![vec![0.0; nf]; nf];
        let mut rhs = vec![0.0; nf];
        for k in 0..nf {
            for l in 0..nf {
                a[k][l] = (0..n).map(|i| t[i][k] * ht[i][l]).sum();
            }
            rhs[k] = -(0..n).map(|i| t[i][k] * g[i]).sum::<f64>();
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut damped = a.clone();
            for k in 0..nf {
                damped[k][k] += lambda * a[k][k].max(1e-12);
            }
            let Some(step) = solve_dense(damped, rhs.clone()) else {
                lambda *= 10.0;
                continue;
            };
            let z2: Vec<f64> = z.iter().zip(&step).map(|(a, b)| a + b).collect();
            let th2 = theta(&z2);
            let c2 = cost(&th2);
            if c2 <= c {
                let moved = th2
                    .iter()
                    .zip(&th)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                z = z2;
                th = th2;
                c = c2;
                lambda = (lambda * 0.1).max(1e-12);
                improved = moved > 1e-13 * (1.0 + extent);
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let out: Vec<Surface> = blocks.iter().map(|b| b.surface(&th)).collect();
    if !out.iter().all(valid) {
        return None;
    }
    for &r in rels {
        let (dir, pos) = defects(r, &out)?;
        if pos > exact || dir > 1e-12 {
            return None;
        }
    }
    let mut old = 0.0;
    let mut new = 0.0;
    let mut count = 0usize;
    for (m, s) in members.iter().zip(&out) {
        let (o, _, _) = ssr(&m.surface, &m.pts);
        let (nw, _, mx) = ssr(s, &m.pts);
        if mx > tol {
            return None;
        }
        old += o;
        new += nw;
        count += m.pts.len();
    }
    let k = k_pos
        + rels
            .iter()
            .map(|r| r.direction_constraints())
            .sum::<usize>();
    let dof = count.saturating_sub(n + 2 * members.len()).max(1);
    if new > old + old / dof as f64 * (2.0 * k as f64 + SSR_SLACK) {
        return None;
    }
    Some(out)
}

pub struct Args<'a> {
    pub vc: &'a [V3],
    pub faces: &'a [[u32; 3]],
    pub nbr: &'a [[u32; 3]],
    pub info: &'a [TriInfo],
    pub flabel: &'a [u32],
    pub tol: f64,
    pub threshold_deg: f64,
}

fn tangent_relations(args: &Args<'_>, finals: &[Final]) -> Vec<Rel> {
    let Args {
        vc,
        faces,
        nbr,
        flabel,
        tol,
        threshold_deg,
        ..
    } = *args;
    let mut shared: Vec<(u32, u32, u32)> = Vec::new();
    for (f, t) in faces.iter().enumerate() {
        let a = flabel[f];
        if a == NONE || !finals[a as usize].surface.is_analytic() {
            continue;
        }
        for k in 0..3 {
            let g = nbr[f][k];
            if g == NONE {
                continue;
            }
            let b = flabel[g as usize];
            if b == NONE || b <= a || !finals[b as usize].surface.is_analytic() {
                continue;
            }
            shared.push((a, b, t[k]));
            shared.push((a, b, t[(k + 1) % 3]));
        }
    }
    shared.sort_unstable();
    shared.dedup();
    let surfaces: Vec<Surface> = finals.iter().map(|f| f.surface).collect();
    let max_sin = threshold_deg.to_radians().sin();
    let mut rels = Vec::new();
    let mut i = 0;
    while i < shared.len() {
        let (a, b, _) = shared[i];
        let mut j = i;
        while j < shared.len() && shared[j].0 == a && shared[j].1 == b {
            j += 1;
        }
        let (sa, sb) = (&surfaces[a as usize], &surfaces[b as usize]);
        if let Some(rel) = relation(a as usize, sa, b as usize, sb)
            && let Some((dir, pos)) = defects(rel, &surfaces)
            && dir <= max_sin
            && pos <= REACH * tol
        {
            let angles: Vec<f64> = shared[i..j]
                .iter()
                .filter_map(|&(_, _, v)| dihedral(sa, sb, vc[v as usize]))
                .collect();
            if !angles.is_empty() && median(angles) < threshold_deg {
                rels.push(rel);
            }
        }
        i = j;
    }
    rels
}

fn member_points(faces: &[[u32; 3]], info: &[TriInfo], vc: &[V3], list: &[u32]) -> Vec<(V3, f64)> {
    let mut v: Vec<(u32, f64)> = list
        .iter()
        .flat_map(|&f| {
            let w = info[f as usize].area / 3.0;
            faces[f as usize].map(|k| (k, w))
        })
        .collect();
    v.sort_by_key(|x| x.0);
    let mut out: Vec<(V3, f64)> = Vec::with_capacity(v.len());
    let mut last = NONE;
    for (k, w) in v {
        if k == last {
            out.last_mut().unwrap().1 += w;
        } else {
            out.push((vc[k as usize], w));
            last = k;
        }
    }
    out
}

/// Makes every `tangent` pair of analytic regions (plane-cylinder,
/// plane-torus, cylinder-torus, cylinder-sphere) exactly tangent by a
/// joint constrained refit of each connected cluster of such pairs (see
/// `solve`). Clusters already tangent to within `DEFECT_REL * tol` are
/// left untouched, so clean input keeps its fit bit for bit. Each changed
/// region's residual and chord sagitta are recomputed on its new surface,
/// so the reported deviation covers the move.
pub fn snap(args: Args<'_>, finals: &mut [Final]) {
    let rels = tangent_relations(&args, finals);
    if rels.is_empty() {
        return;
    }
    let Args {
        vc,
        faces,
        info,
        tol,
        threshold_deg,
        ..
    } = args;
    let max_sin = threshold_deg.to_radians().sin();
    let mut dsu = Dsu::new(finals.len());
    for r in &rels {
        let (a, b) = r.ends();
        dsu.union(a as u32, b as u32);
    }
    let mut clusters: Vec<(u32, Vec<Rel>)> = Vec::new();
    for &r in &rels {
        let root = dsu.find(r.ends().0 as u32);
        match clusters.iter_mut().find(|c| c.0 == root) {
            Some(c) => c.1.push(r),
            None => clusters.push((root, vec![r])),
        }
    }
    clusters.sort_by_key(|c| c.0);
    let surfaces: Vec<Surface> = finals.iter().map(|f| f.surface).collect();
    let extent = vc.iter().map(|&p| norm(p)).fold(0.0, f64::max);
    for (_, crels) in clusters {
        let worst = crels
            .iter()
            .filter_map(|&r| defects(r, &surfaces))
            .map(|(dir, pos)| (dir * extent).max(pos))
            .fold(0.0, f64::max);
        if worst <= DEFECT_REL * tol {
            continue;
        }
        let mut ids: Vec<usize> = crels
            .iter()
            .flat_map(|r| {
                let (a, b) = r.ends();
                [a, b]
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let local: Vec<Rel> = crels
            .iter()
            .map(|r| r.map(|g| ids.binary_search(&g).unwrap()))
            .collect();
        let members: Vec<Member> = ids
            .iter()
            .map(|&g| Member {
                surface: finals[g].surface,
                pts: member_points(faces, info, vc, &finals[g].faces),
            })
            .collect();
        let Some(out) = solve(&members, &local, tol, max_sin) else {
            continue;
        };
        for ((&g, m), s) in ids.iter().zip(&members).zip(out) {
            let (s2, sw, mx) = ssr(&s, &m.pts);
            let fr = &mut finals[g];
            fr.surface = s;
            fr.rms = (s2 / sw.max(f64::MIN_POSITIVE)).sqrt();
            fr.max = mx;
            if s.as_plane().is_none() {
                fr.sag = fr
                    .faces
                    .iter()
                    .map(|&f| sagitta(&s, faces[f as usize].map(|v| vc[v as usize])))
                    .fold(0.0, f64::max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noise(u64);

    impl Noise {
        fn next(&mut self, amp: f64) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * amp
        }
    }

    const AMP: f64 = 1e-3;
    const TOL: f64 = 3e-3;

    fn noisy(s: &Surface, p: V3, noise: &mut Noise) -> (V3, f64) {
        let n = s.normal_at(p).unwrap();
        (add(p, scale(n, noise.next(AMP))), 1.0)
    }

    fn grid(n: usize, f: impl Fn(f64, f64) -> V3) -> Vec<V3> {
        let mut out = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                out.push(f(i as f64 / n as f64, j as f64 / n as f64));
            }
        }
        out
    }

    fn member(truth: Surface, start: Surface, pts: Vec<V3>, noise: &mut Noise) -> Member {
        Member {
            surface: start,
            pts: pts.into_iter().map(|p| noisy(&truth, p, noise)).collect(),
        }
    }

    fn free_fit(members: &mut [Member]) {
        let out = solve(members, &[], TOL, 0.05).expect("unconstrained fit");
        for (m, s) in members.iter_mut().zip(out) {
            m.surface = s;
        }
    }

    fn total_ssr(members: &[Member], surfaces: &[Surface]) -> f64 {
        members
            .iter()
            .zip(surfaces)
            .map(|(m, s)| ssr(s, &m.pts).0)
            .sum()
    }

    fn check(members: &[Member], rels: &[Rel]) -> Vec<Surface> {
        let before: Vec<Surface> = members.iter().map(|m| m.surface).collect();
        let worst = rels
            .iter()
            .map(|&r| defects(r, &before).unwrap().1)
            .fold(0.0, f64::max);
        assert!(
            worst > 1e-5,
            "the free fit should not already be tangent: {worst}"
        );
        let out = solve(members, rels, TOL, 0.05).expect("tangent fit accepted");
        for &r in rels {
            let (dir, pos) = defects(r, &out).unwrap();
            assert!(dir < 1e-14 && pos < 1e-12, "{r:?}: {dir} {pos}");
        }
        for (m, s) in members.iter().zip(&out) {
            assert!(ssr(s, &m.pts).2 <= AMP * 1.5);
        }
        let (old, new) = (total_ssr(members, &before), total_ssr(members, &out));
        assert!(new <= old * 1.2, "{old} {new}");
        out
    }

    fn plane(n: V3, d: f64) -> Surface {
        Surface::Plane {
            normal: n,
            offset: d,
        }
    }

    fn cyl(origin: V3, axis: V3, radius: f64) -> Surface {
        Surface::Cylinder {
            origin,
            axis,
            radius,
            reversed: false,
        }
    }

    fn torus(center: V3, major: f64, minor: f64, reversed: bool) -> Surface {
        Surface::Torus {
            center,
            axis: [0.0, 0.0, 1.0],
            major,
            minor,
            reversed,
        }
    }

    const R: f64 = 2.0;

    fn rounded_edge(noise: &mut Noise, tilt: f64) -> Vec<Member> {
        let top = plane([0.0, 0.0, 1.0], 0.0);
        let side = plane([0.0, 1.0, 0.0], 0.0);
        let fillet = cyl([0.0, -R, -R], [1.0, 0.0, 0.0], R);
        let axis = unit([1.0, tilt, 0.0]);
        let mut m = vec![
            member(
                top,
                plane([0.0, 0.0, 1.0], 3e-4),
                grid(12, |u, v| [10.0 * u, -R - 8.0 * v, 0.0]),
                noise,
            ),
            member(
                fillet,
                cyl([0.0, -R + 2e-4, -R - 3e-4], axis, R - 5e-4),
                grid(12, |u, v| {
                    let t = v * std::f64::consts::FRAC_PI_2;
                    [10.0 * u, -R + R * t.sin(), -R + R * t.cos()]
                }),
                noise,
            ),
            member(
                side,
                plane([0.0, 1.0, 0.0], -2e-4),
                grid(12, |u, v| [10.0 * u, 0.0, -R - 8.0 * v]),
                noise,
            ),
        ];
        free_fit(&mut m);
        m
    }

    #[test]
    fn plane_cylinder_pair_becomes_exactly_tangent() {
        let mut noise = Noise(1);
        let m = rounded_edge(&mut noise, 0.0);
        let out = check(&m[..2], &[Rel::PlaneCyl { plane: 0, cyl: 1 }]);
        let Surface::Cylinder { radius, .. } = out[1] else {
            panic!()
        };
        assert!((radius - R).abs() < 1e-3);
    }

    #[test]
    fn fillet_between_two_planes_is_solved_jointly() {
        let mut noise = Noise(2);
        let m = rounded_edge(&mut noise, 5e-5);
        let out = check(
            &m,
            &[
                Rel::PlaneCyl { plane: 0, cyl: 1 },
                Rel::PlaneCyl { plane: 2, cyl: 1 },
            ],
        );
        let Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } = out[1]
        else {
            panic!()
        };
        assert_eq!(axis, [1.0, 0.0, 0.0]);
        assert!((radius - R).abs() < 5e-4);
        assert!(norm(perp(sub(origin, [0.0, -R, -R]), axis)) < 5e-4);
        assert_eq!(out[0].as_plane().unwrap().0, [0.0, 0.0, 1.0]);
    }

    fn rounded_disc(noise: &mut Noise) -> Vec<Member> {
        let (rd, r) = (10.0, 1.5);
        let top = plane([0.0, 0.0, 1.0], 0.0);
        let edge = torus([0.0, 0.0, -r], rd - r, r, false);
        let rim = cyl([0.0; 3], [0.0, 0.0, 1.0], rd);
        let polar = |rho: f64, phi: f64, z: f64| [rho * phi.cos(), rho * phi.sin(), z];
        let tau = std::f64::consts::TAU;
        let mut m = vec![
            member(
                top,
                plane([0.0, 0.0, 1.0], -4e-4),
                grid(16, |u, v| polar((rd - r) * u, tau * v, 0.0)),
                noise,
            ),
            member(
                edge,
                torus([2e-4, 0.0, -r + 3e-4], rd - r + 3e-4, r - 4e-4, false),
                grid(16, |u, v| {
                    let t = v * std::f64::consts::FRAC_PI_2;
                    polar(rd - r + r * t.sin(), tau * u, -r + r * t.cos())
                }),
                noise,
            ),
            member(
                rim,
                cyl([0.0, 3e-4, 0.0], [0.0, 0.0, 1.0], rd + 5e-4),
                grid(16, |u, v| polar(rd, tau * u, -r - 4.0 * v)),
                noise,
            ),
        ];
        free_fit(&mut m);
        m
    }

    #[test]
    fn plane_torus_pair_becomes_exactly_tangent() {
        let mut noise = Noise(3);
        let m = rounded_disc(&mut noise);
        check(&m[..2], &[Rel::PlaneTorus { plane: 0, torus: 1 }]);
    }

    #[test]
    fn cylinder_torus_pair_becomes_exactly_tangent() {
        let mut noise = Noise(4);
        let m = rounded_disc(&mut noise);
        let pair = [
            Member {
                surface: m[2].surface,
                pts: m[2].pts.clone(),
            },
            Member {
                surface: m[1].surface,
                pts: m[1].pts.clone(),
            },
        ];
        check(&pair, &[Rel::CylTorus { cyl: 0, torus: 1 }]);
    }

    #[test]
    fn plane_torus_cylinder_chain_is_solved_jointly() {
        let mut noise = Noise(5);
        let m = rounded_disc(&mut noise);
        let out = check(
            &m,
            &[
                Rel::PlaneTorus { plane: 0, torus: 1 },
                Rel::CylTorus { cyl: 2, torus: 1 },
            ],
        );
        let Surface::Torus { major, minor, .. } = out[1] else {
            panic!()
        };
        assert!((major - 8.5).abs() < 1e-3 && (minor - 1.5).abs() < 1e-3);
    }

    #[test]
    fn cylinder_sphere_pair_becomes_exactly_tangent() {
        let mut noise = Noise(6);
        let wall = cyl([0.0; 3], [0.0, 0.0, 1.0], 3.0);
        let dome = Surface::Sphere {
            center: [0.0; 3],
            radius: 3.0,
            reversed: false,
        };
        let tau = std::f64::consts::TAU;
        let mut m = vec![
            member(
                wall,
                cyl([2e-4, 0.0, 0.0], [0.0, 0.0, 1.0], 3.0 - 4e-4),
                grid(16, |u, v| {
                    [3.0 * (tau * u).cos(), 3.0 * (tau * u).sin(), -5.0 * v]
                }),
                &mut noise,
            ),
            member(
                dome,
                Surface::Sphere {
                    center: [0.0, -2e-4, 3e-4],
                    radius: 3.0 + 3e-4,
                    reversed: false,
                },
                grid(16, |u, v| {
                    let t = v * std::f64::consts::FRAC_PI_2;
                    let rho = 3.0 * t.cos();
                    [rho * (tau * u).cos(), rho * (tau * u).sin(), 3.0 * t.sin()]
                }),
                &mut noise,
            ),
        ];
        free_fit(&mut m);
        check(&m, &[Rel::CylSphere { cyl: 0, sphere: 1 }]);
    }

    #[test]
    fn a_tangency_the_points_contradict_is_refused() {
        let mut noise = Noise(7);
        let mut m = rounded_edge(&mut noise, 0.0);
        for p in m[0].pts.iter_mut() {
            p.0[2] += 0.01;
        }
        free_fit(&mut m);
        assert!(solve(&m[..2], &[Rel::PlaneCyl { plane: 0, cyl: 1 }], TOL, 0.05).is_none());
    }

    #[test]
    fn redundant_constraints_are_eliminated() {
        let rows = vec![
            (vec![1.0, 1.0, 0.0], 2.0),
            (vec![2.0, 2.0, 0.0], 4.0),
            (vec![0.0, 1.0, -1.0], 0.5),
        ];
        let (t0, t) = null_space(rows, 3, 1e-12).unwrap();
        assert_eq!(t[0].len(), 1);
        for z in [-1.0, 0.0, 2.5] {
            let x: Vec<f64> = (0..3).map(|i| t0[i] + t[i][0] * z).collect();
            assert!((x[0] + x[1] - 2.0).abs() < 1e-12);
            assert!((x[1] - x[2] - 0.5).abs() < 1e-12);
        }
        let bad = vec![(vec![1.0, 0.0], 1.0), (vec![1.0, 0.0], 1.5)];
        assert!(null_space(bad, 2, 1e-12).is_none());
    }
}
