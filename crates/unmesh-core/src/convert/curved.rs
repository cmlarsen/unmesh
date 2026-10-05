use super::fit::Region;
use super::linalg::{V3, add, cross, dot, norm, outer_add, scale, sub, sym_eigen, unit};
use super::segment::TriInfo;
use super::surface::{Surface, radial};

const AXIS_STEP: f64 = 1e-7;
const LM_ITERS: usize = 40;
const IRLS_ROUNDS: usize = 3;
pub const MAX_GROUP: usize = 16;
const LAW_GAP_SPREAD: f64 = 0.02;
const LAW_N_REL: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Cylinder,
    Cone { sign: f64 },
}

#[derive(Debug, Clone)]
pub struct Member {
    pub pts: Vec<(V3, f64)>,
    pub kind: Kind,
    pub slot: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Axis {
    pub a: V3,
    pub c: V3,
}

#[derive(Debug, Clone)]
pub struct Joint {
    pub axis: Axis,
    pub shape: Vec<f64>,
    pub fit: Vec<(f64, f64)>,
}

fn basis(a: V3) -> (V3, V3) {
    let e = if a[0].abs() < 0.6 {
        [1.0, 0.0, 0.0]
    } else if a[1].abs() < 0.6 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let u = unit(cross(a, e));
    (u, cross(a, u))
}

fn huber(r: f64, k: f64) -> f64 {
    let r = r.abs();
    if r <= k || k <= 0.0 { 1.0 } else { k / r }
}

fn point_residual(axis: &Axis, shape: &[f64], m: &Member, p: V3) -> f64 {
    let (h, rho, _) = radial(p, axis.c, axis.a);
    match m.kind {
        Kind::Cylinder => rho - shape[m.slot],
        Kind::Cone { sign } => {
            let (s, c) = shape[m.slot + 1].sin_cos();
            rho * c - sign * (h - shape[m.slot]) * s
        }
    }
}

pub fn member_residual(axis: &Axis, shape: &[f64], m: &Member) -> (f64, f64) {
    let mut sw = 0.0;
    let mut s2 = 0.0;
    let mut mx: f64 = 0.0;
    for &(p, w) in &m.pts {
        let r = point_residual(axis, shape, m, p);
        sw += w;
        s2 += w * r * r;
        mx = mx.max(r.abs());
    }
    ((s2 / sw.max(f64::MIN_POSITIVE)).sqrt(), mx)
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

struct Problem<'a> {
    members: &'a [Member],
    fix_dir: bool,
    len_step: f64,
}

impl Problem<'_> {
    fn n_axis(&self) -> usize {
        if self.fix_dir { 2 } else { 4 }
    }

    fn apply(&self, axis: &Axis, shape: &[f64], x: &[f64]) -> (Axis, Vec<f64>) {
        let (u, v) = basis(axis.a);
        let (a, k) = if self.fix_dir {
            (axis.a, 0)
        } else {
            (unit(add(axis.a, add(scale(u, x[0]), scale(v, x[1])))), 2)
        };
        let c = add(axis.c, add(scale(u, x[k]), scale(v, x[k + 1])));
        let k = self.n_axis();
        let s: Vec<f64> = shape.iter().zip(&x[k..]).map(|(s, d)| s + d).collect();
        (Axis { a, c }, s)
    }

    fn residuals(&self, axis: &Axis, shape: &[f64], weights: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let mut i = 0;
        for m in self.members {
            for &(p, _) in &m.pts {
                out.push(weights[i] * point_residual(axis, shape, m, p));
                i += 1;
            }
        }
    }

    fn step_size(&self, j: usize) -> f64 {
        if !self.fix_dir && j < 2 {
            AXIS_STEP
        } else {
            self.len_step
        }
    }
}

fn cost(r: &[f64]) -> f64 {
    r.iter().map(|x| x * x).sum()
}

fn lm(problem: &Problem<'_>, axis: &mut Axis, shape: &mut Vec<f64>, weights: &[f64]) {
    let n = problem.n_axis() + shape.len();
    let mut r0 = Vec::new();
    let mut rp = Vec::new();
    let mut rm = Vec::new();
    let mut lambda = 1e-6;
    problem.residuals(axis, shape, weights, &mut r0);
    let mut c0 = cost(&r0);
    for _ in 0..LM_ITERS {
        if c0 == 0.0 {
            break;
        }
        let m = r0.len();
        let mut jac = vec![vec![0.0; m]; n];
        for j in 0..n {
            let h = if j >= problem.n_axis() && j - problem.n_axis() < shape.len() {
                let s = shape[j - problem.n_axis()];
                problem.len_step.min(1e-7 * s.abs().max(1e-3))
            } else {
                problem.step_size(j)
            };
            let mut x = vec![0.0; n];
            x[j] = h;
            let (ap, sp) = problem.apply(axis, shape, &x);
            problem.residuals(&ap, &sp, weights, &mut rp);
            x[j] = -h;
            let (am, sm) = problem.apply(axis, shape, &x);
            problem.residuals(&am, &sm, weights, &mut rm);
            for i in 0..m {
                jac[j][i] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for a in 0..n {
            for b in a..n {
                let s: f64 = jac[a].iter().zip(&jac[b]).map(|(x, y)| x * y).sum();
                jtj[a][b] = s;
                jtj[b][a] = s;
            }
            jtr[a] = -jac[a].iter().zip(&r0).map(|(x, y)| x * y).sum::<f64>();
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut aug = jtj.clone();
            for d in 0..n {
                aug[d][d] += lambda * jtj[d][d].max(1e-30);
            }
            let Some(x) = solve_dense(aug, jtr.clone()) else {
                lambda *= 10.0;
                continue;
            };
            let (na, ns) = problem.apply(axis, shape, &x);
            problem.residuals(&na, &ns, weights, &mut rp);
            let c1 = cost(&rp);
            if c1 < c0 {
                let rel = (c0 - c1) / c0;
                *axis = na;
                *shape = ns;
                std::mem::swap(&mut r0, &mut rp);
                c0 = c1;
                lambda = (lambda / 4.0).max(1e-12);
                improved = rel > 1e-14;
                break;
            }
            lambda *= 8.0;
        }
        if !improved {
            break;
        }
    }
}

pub fn fit_joint(
    members: &[Member],
    mut axis: Axis,
    mut shape: Vec<f64>,
    fix_dir: bool,
    tol: f64,
) -> Joint {
    let extent = members
        .iter()
        .flat_map(|m| m.pts.iter())
        .map(|(p, _)| norm(sub(*p, axis.c)))
        .fold(0.0, f64::max)
        .max(1e-9);
    let problem = Problem {
        members,
        fix_dir,
        len_step: 1e-7 * extent,
    };
    let base: Vec<f64> = members
        .iter()
        .flat_map(|m| m.pts.iter().map(|&(_, w)| w.sqrt()))
        .collect();
    let mut weights = base.clone();
    for round in 0..IRLS_ROUNDS {
        if round > 0 {
            let mut i = 0;
            for m in members {
                for &(p, w) in &m.pts {
                    let r = point_residual(&axis, &shape, m, p);
                    weights[i] = (w * huber(r, 0.25 * tol)).sqrt();
                    i += 1;
                }
            }
        }
        lm(&problem, &mut axis, &mut shape, &weights);
    }
    let fit = members
        .iter()
        .map(|m| member_residual(&axis, &shape, m))
        .collect();
    Joint { axis, shape, fit }
}

pub struct Tri {
    pub normal: V3,
    pub area: f64,
    pub centroid: V3,
}

fn normal_scatter(tris: &[Tri], centred: bool) -> Option<([f64; 3], [V3; 3], V3)> {
    let mut sw = 0.0;
    let mut mean = [0.0; 3];
    for t in tris {
        sw += t.area;
        mean = add(mean, scale(t.normal, t.area));
    }
    if sw <= 0.0 {
        return None;
    }
    mean = scale(mean, 1.0 / sw);
    let mut m = [[0.0; 3]; 3];
    for t in tris {
        let n = if centred {
            sub(t.normal, mean)
        } else {
            t.normal
        };
        outer_add(&mut m, n, n, t.area / sw);
    }
    let (vals, vecs) = sym_eigen(m);
    Some((vals, vecs, mean))
}

fn kasa(pts: &[(V3, f64)], a: V3) -> Option<(V3, f64)> {
    let (u, v) = basis(a);
    let mut sw = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for &(p, w) in pts {
        sw += w;
        cx += w * dot(p, u);
        cy += w * dot(p, v);
    }
    if sw <= 0.0 {
        return None;
    }
    cx /= sw;
    cy /= sw;
    let mut m = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for &(p, w) in pts {
        let x = dot(p, u) - cx;
        let y = dot(p, v) - cy;
        let row = [x, y, 1.0];
        outer_add(&mut m, row, row, w);
        let rhs = -(x * x + y * y);
        for k in 0..3 {
            b[k] += w * rhs * row[k];
        }
    }
    let sol = solve_dense(m.iter().map(|r| r.to_vec()).collect(), b.to_vec())?;
    let (ox, oy) = (-sol[0] / 2.0, -sol[1] / 2.0);
    let r2 = ox * ox + oy * oy - sol[2];
    if r2 <= 0.0 {
        return None;
    }
    let c = add(scale(u, ox + cx), scale(v, oy + cy));
    Some((c, r2.sqrt()))
}

pub fn init_cylinder(pts: &[(V3, f64)], tris: &[Tri]) -> Option<(Axis, f64)> {
    let (vals, vecs, _) = normal_scatter(tris, false)?;
    if vals[1].is_nan() || vals[1] <= 1e-10 * vals[2] {
        return None;
    }
    let a = vecs[0];
    let (c, r) = kasa(pts, a)?;
    Some((Axis { a, c }, r))
}

pub fn init_cone(pts: &[(V3, f64)], tris: &[Tri]) -> Option<(Axis, f64)> {
    let (vals, vecs, _) = normal_scatter(tris, true)?;
    if vals[1].is_nan() || vals[1] <= 1e-10 * vals[2].max(f64::MIN_POSITIVE) {
        return None;
    }
    let mut a = vecs[0];
    let mut m = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for t in tris {
        outer_add(&mut m, t.normal, t.normal, t.area);
        b = add(b, scale(t.normal, t.area * dot(t.normal, t.centroid)));
    }
    let (ev, _) = sym_eigen(m);
    if ev[0].is_nan() || ev[0] <= 1e-8 * ev[2] {
        return None;
    }
    let apex = solve_dense(m.iter().map(|r| r.to_vec()).collect(), b.to_vec())?;
    let apex = [apex[0], apex[1], apex[2]];
    let mut sw = 0.0;
    let mut hm = 0.0;
    for &(p, w) in pts {
        sw += w;
        hm += w * dot(sub(p, apex), a);
    }
    if hm < 0.0 {
        a = scale(a, -1.0);
    }
    let mut ang = 0.0;
    for &(p, w) in pts {
        let (h, rho, _) = radial(p, apex, a);
        ang += w * rho.atan2(h);
    }
    let alpha = ang / sw.max(f64::MIN_POSITIVE);
    if !(alpha > 0.0 && alpha < std::f64::consts::FRAC_PI_2) {
        return None;
    }
    Some((Axis { a, c: apex }, alpha))
}

/// Radius from the tessellation law: a circle cut into `N` equal chords `c`
/// turns by `2π/N` at every ruling, so `r = c / (2 sin(π/N))`. Chords and
/// turning angles are measured between rulings and do not depend on the
/// fitted centre; only the ruling order does. `None` unless the chords and
/// turning angles are equal and the turning angle divides the full turn.
pub fn tessellation_radius(pts: &[(V3, f64)], axis: &Axis, radius: f64, tol: f64) -> Option<f64> {
    let (u, v) = basis(axis.a);
    let mut polar: Vec<(f64, f64, f64)> = pts
        .iter()
        .map(|&(p, _)| {
            let q = sub(p, axis.c);
            let (x, y) = (dot(q, u), dot(q, v));
            (y.atan2(x), x, y)
        })
        .collect();
    polar.sort_by(|a, b| a.0.total_cmp(&b.0));
    let tau = std::f64::consts::TAU;
    let eps = (8.0 * tol / radius).max(1e-9);
    let mut rulings: Vec<[f64; 4]> = Vec::new();
    for &(phi, x, y) in &polar {
        match rulings.last_mut() {
            Some(last) if phi - last[0] / last[3] <= eps => {
                *last = [last[0] + phi, last[1] + x, last[2] + y, last[3] + 1.0];
            }
            _ => rulings.push([phi, x, y, 1.0]),
        }
    }
    if rulings.len() >= 2 {
        let (first, last) = (rulings[0], rulings[rulings.len() - 1]);
        if first[0] / first[3] + tau - last[0] / last[3] <= eps {
            rulings.pop();
            rulings[0] = [
                last[0] - tau * last[3] + first[0],
                last[1] + first[1],
                last[2] + first[2],
                last[3] + first[3],
            ];
            rulings.sort_by(|a, b| (a[0] / a[3]).total_cmp(&(b[0] / b[3])));
        }
    }
    let k = rulings.len();
    if k < 3 {
        return None;
    }
    let xy: Vec<(f64, f64)> = rulings.iter().map(|r| (r[1] / r[3], r[2] / r[3])).collect();
    let chord = |i: usize| {
        let (a, b) = (xy[i], xy[(i + 1) % k]);
        (b.0 - a.0, b.1 - a.1)
    };
    let len = |d: (f64, f64)| (d.0 * d.0 + d.1 * d.1).sqrt();
    let mut chords: Vec<(f64, f64)> = (0..k).map(chord).collect();
    let widest = (0..k)
        .max_by(|&i, &j| len(chords[i]).total_cmp(&len(chords[j])))
        .unwrap();
    let rest = (0..k)
        .filter(|&i| i != widest)
        .map(|i| len(chords[i]))
        .sum::<f64>()
        / (k - 1) as f64;
    let closed = len(chords[widest]) <= rest * (1.0 + LAW_GAP_SPREAD);
    if !closed {
        chords.rotate_left(widest + 1);
        chords.pop();
    }
    let lens: Vec<f64> = chords.iter().map(|&d| len(d)).collect();
    let c = lens.iter().sum::<f64>() / lens.len() as f64;
    let turns: Vec<f64> = (0..chords.len())
        .filter(|&i| closed || i + 1 < chords.len())
        .map(|i| {
            let (a, b) = (chords[i], chords[(i + 1) % chords.len()]);
            (a.0 * b.1 - a.1 * b.0).atan2(a.0 * b.0 + a.1 * b.1)
        })
        .collect();
    if turns.is_empty() {
        return None;
    }
    let theta = turns.iter().sum::<f64>() / turns.len() as f64;
    let spread = |xs: &[f64], m: f64| xs.iter().any(|x| (x - m).abs() > LAW_GAP_SPREAD * m.abs());
    if theta <= 0.0 || spread(&lens, c) || spread(&turns, theta) {
        return None;
    }
    let n = (tau / theta).round();
    if n < 3.0 || (tau / n - theta).abs() > LAW_N_REL * theta {
        return None;
    }
    Some(c / (2.0 * (std::f64::consts::PI / n).sin()))
}

pub fn orientation(surface: &Surface, tris: &[Tri]) -> Surface {
    let natural = match *surface {
        Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } => Surface::Cylinder {
            origin,
            axis,
            radius,
            reversed: false,
        },
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } => Surface::Cone {
            apex,
            axis,
            half_angle,
            reversed: false,
        },
        s => return s,
    };
    let mut s = 0.0;
    for t in tris {
        if let Some(n) = natural.normal_at(t.centroid) {
            s += t.area * dot(n, t.normal);
        }
    }
    match natural {
        Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } => Surface::Cylinder {
            origin,
            axis,
            radius,
            reversed: s < 0.0,
        },
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } => Surface::Cone {
            apex,
            axis,
            half_angle,
            reversed: s < 0.0,
        },
        other => other,
    }
}

pub fn surface_of(axis: &Axis, shape: &[f64], m: &Member) -> Surface {
    match m.kind {
        Kind::Cylinder => {
            let mut a = axis.a;
            let big = (0..3)
                .max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs()))
                .unwrap();
            if a[big] < 0.0 {
                a = scale(a, -1.0);
            }
            Surface::Cylinder {
                origin: sub(axis.c, scale(a, dot(axis.c, a))),
                axis: a,
                radius: shape[m.slot],
                reversed: false,
            }
        }
        Kind::Cone { sign } => Surface::Cone {
            apex: add(axis.c, scale(axis.a, shape[m.slot])),
            axis: scale(axis.a, sign),
            half_angle: shape[m.slot + 1],
            reversed: false,
        },
    }
}

/// Upper bound on the distance from any point of the triangle to the surface
/// when its corners lie on it: the chord sagitta. Exact for a cylinder (the
/// distance to the axis is convex, so its minimum over the triangle is the
/// 2D point-triangle distance in the plane normal to the axis); sampled for a
/// cone.
pub fn sagitta(surface: &Surface, t: [V3; 3]) -> f64 {
    match *surface {
        Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } => {
            let (u, v) = basis(axis);
            let p: Vec<(f64, f64)> = t
                .iter()
                .map(|x| {
                    let q = sub(*x, origin);
                    (dot(q, u), dot(q, v))
                })
                .collect();
            let dmin = dist_to_triangle_2d(&p);
            let dmax = p
                .iter()
                .map(|(x, y)| (x * x + y * y).sqrt())
                .fold(0.0, f64::max);
            (radius - dmin).max(dmax - radius).max(0.0)
        }
        Surface::Cone { .. } => {
            let k = 8;
            let mut worst: f64 = 0.0;
            for i in 0..=k {
                for j in 0..=k - i {
                    let (a, b) = (i as f64 / k as f64, j as f64 / k as f64);
                    let c = 1.0 - a - b;
                    let p = add(add(scale(t[0], a), scale(t[1], b)), scale(t[2], c));
                    worst = worst.max(surface.distance(p).abs());
                }
            }
            worst
        }
        _ => 0.0,
    }
}

fn dist_to_triangle_2d(p: &[(f64, f64)]) -> f64 {
    let cross2 = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    let o = (0.0, 0.0);
    let s0 = cross2(p[0], p[1], o);
    let s1 = cross2(p[1], p[2], o);
    let s2 = cross2(p[2], p[0], o);
    let area = cross2(p[0], p[1], p[2]);
    if area != 0.0
        && ((s0 >= 0.0 && s1 >= 0.0 && s2 >= 0.0) || (s0 <= 0.0 && s1 <= 0.0 && s2 <= 0.0))
    {
        return 0.0;
    }
    (0..3)
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % 3]);
            let ab = (b.0 - a.0, b.1 - a.1);
            let l2 = ab.0 * ab.0 + ab.1 * ab.1;
            let t = if l2 > 0.0 {
                ((-a.0) * ab.0 + (-a.1) * ab.1) / l2
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            let q = (a.0 + t * ab.0, a.1 + t * ab.1);
            (q.0 * q.0 + q.1 * q.1).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

struct Fitted {
    region: usize,
    kind: Kind,
    axis: Axis,
    shape: Vec<f64>,
    rms: f64,
    max: f64,
}

fn region_tris(vc: &[V3], faces: &[[u32; 3]], info: &[TriInfo], r: &Region) -> Vec<Tri> {
    r.faces
        .iter()
        .filter(|&&f| info[f as usize].area > 0.0)
        .map(|&f| {
            let t = faces[f as usize];
            let c = scale(
                add(add(vc[t[0] as usize], vc[t[1] as usize]), vc[t[2] as usize]),
                1.0 / 3.0,
            );
            Tri {
                normal: info[f as usize].normal,
                area: info[f as usize].area,
                centroid: c,
            }
        })
        .collect()
}

fn fit_single(
    pts: &[(V3, f64)],
    tris: &[Tri],
    tol: f64,
) -> Option<(Kind, Axis, Vec<f64>, f64, f64)> {
    if let Some((axis, r)) = init_cylinder(pts, tris) {
        let m = [Member {
            pts: pts.to_vec(),
            kind: Kind::Cylinder,
            slot: 0,
        }];
        let j = fit_joint(&m, axis, vec![r], false, tol);
        let (mut rms, mut max) = j.fit[0];
        let mut shape = j.shape;
        if max <= tol && shape[0] > 0.0 {
            if let Some(rl) = tessellation_radius(pts, &j.axis, shape[0], tol)
                && (rl - shape[0]).abs() <= tol
            {
                let (r2, m2) = member_residual(&j.axis, &[rl], &m[0]);
                if m2 <= tol {
                    shape[0] = rl;
                    rms = r2;
                    max = m2;
                }
            }
            return Some((Kind::Cylinder, j.axis, shape, rms, max));
        }
    }
    let (axis, alpha) = init_cone(pts, tris)?;
    let m = [Member {
        pts: pts.to_vec(),
        kind: Kind::Cone { sign: 1.0 },
        slot: 0,
    }];
    let j = fit_joint(&m, axis, vec![0.0, alpha], false, tol);
    let (rms, max) = j.fit[0];
    let a = j.shape[1];
    (max <= tol && a > 0.0 && a < std::f64::consts::FRAC_PI_2).then_some((
        Kind::Cone { sign: 1.0 },
        j.axis,
        j.shape,
        rms,
        max,
    ))
}

fn line_distance(p: V3, axis: &Axis) -> f64 {
    radial(p, axis.c, axis.a).1
}

fn coaxial(a: &Fitted, b: &Fitted, cos_lim: f64, dist: f64) -> bool {
    dot(a.axis.a, b.axis.a).abs() >= cos_lim
        && line_distance(b.axis.c, &a.axis) <= dist
        && line_distance(a.axis.c, &b.axis) <= dist
}

struct GroupFit {
    members: Vec<Member>,
    axis: Axis,
    shape: Vec<f64>,
}

fn build_group(fits: &[&Fitted], pts: &[Vec<(V3, f64)>], tol: f64) -> GroupFit {
    let lead = fits
        .iter()
        .max_by(|x, y| pts[x.region].len().cmp(&pts[y.region].len()))
        .unwrap();
    let axis = lead.axis;
    let mut shape: Vec<f64> = Vec::new();
    let mut members = Vec::new();
    let mut radii: Vec<(f64, usize)> = Vec::new();
    for f in fits {
        let ps = pts[f.region].clone();
        match f.kind {
            Kind::Cylinder => {
                let r = f.shape[0];
                let slot = match radii.iter().find(|(x, _)| (x - r).abs() <= tol) {
                    Some(&(_, s)) => s,
                    None => {
                        shape.push(r);
                        radii.push((r, shape.len() - 1));
                        shape.len() - 1
                    }
                };
                members.push(Member {
                    pts: ps,
                    kind: Kind::Cylinder,
                    slot,
                });
            }
            Kind::Cone { .. } => {
                let sign = if dot(f.axis.a, axis.a) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let s = dot(
                    sub(add(f.axis.c, scale(f.axis.a, f.shape[0])), axis.c),
                    axis.a,
                );
                shape.push(s);
                shape.push(f.shape[1]);
                members.push(Member {
                    pts: ps,
                    kind: Kind::Cone { sign },
                    slot: shape.len() - 2,
                });
            }
        }
    }
    GroupFit {
        members,
        axis,
        shape,
    }
}

fn world_axis(a: V3, cos_lim: f64) -> Option<V3> {
    let k = (0..3).max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs()))?;
    if a[k].abs() < cos_lim {
        return None;
    }
    let mut e = [0.0; 3];
    e[k] = a[k].signum();
    (e != a).then_some(e)
}

fn accept(fit: &Joint, tol: f64) -> bool {
    fit.fit
        .iter()
        .all(|&(rms, max)| max <= tol && rms <= 0.5 * tol)
        && fit.shape.iter().all(|s| s.is_finite())
}

pub fn refine_regions(
    vc: &[V3],
    faces: &[[u32; 3]],
    info: &[TriInfo],
    regions: &mut [Region],
    tol: f64,
    snap_deg: f64,
) {
    let mut fits: Vec<Fitted> = Vec::new();
    let mut pts: Vec<Vec<(V3, f64)>> = vec![Vec::new(); regions.len()];
    let mut tris: Vec<Vec<Tri>> = (0..regions.len()).map(|_| Vec::new()).collect();
    for (ri, r) in regions.iter().enumerate() {
        if r.area <= 0.0 || r.max <= tol {
            continue;
        }
        let p: Vec<(V3, f64)> = r.verts.iter().map(|&(v, w)| (vc[v as usize], w)).collect();
        let t = region_tris(vc, faces, info, r);
        if let Some((kind, axis, shape, rms, max)) = fit_single(&p, &t, tol) {
            fits.push(Fitted {
                region: ri,
                kind,
                axis,
                shape,
                rms,
                max,
            });
        }
        pts[ri] = p;
        tris[ri] = t;
    }

    let cos_lim = snap_deg.to_radians().cos();
    let mut dsu = super::dsu::Dsu::new(fits.len());
    for i in 0..fits.len() {
        for j in i + 1..fits.len() {
            if coaxial(&fits[i], &fits[j], cos_lim, 3.0 * tol) {
                dsu.union(i as u32, j as u32);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot_of: Vec<Option<usize>> = vec![None; fits.len()];
    for i in 0..fits.len() {
        let root = dsu.find(i as u32) as usize;
        let g = *slot_of[root].get_or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }

    for g in groups {
        let members: Vec<&Fitted> = g.iter().map(|&i| &fits[i]).collect();
        let mut gf = build_group(&members, &pts, tol);
        let mut joint: Option<Joint> = None;
        if g.len() >= 2 && g.len() <= MAX_GROUP {
            let j = fit_joint(&gf.members, gf.axis, gf.shape.clone(), false, tol);
            if accept(&j, tol) {
                gf.axis = j.axis;
                gf.shape = j.shape.clone();
                joint = Some(j);
            }
        }
        if (g.len() == 1 || joint.is_some())
            && let Some(e) = world_axis(gf.axis.a, cos_lim)
        {
            let axis = Axis { a: e, c: gf.axis.c };
            let j = fit_joint(&gf.members, axis, gf.shape.clone(), true, tol);
            if accept(&j, tol) {
                gf.axis = j.axis;
                gf.shape = j.shape.clone();
                joint = Some(j);
            }
        }
        for (k, &fi) in g.iter().enumerate() {
            let f = &fits[fi];
            let (surface, rms, max) = match &joint {
                Some(j) => (
                    surface_of(&j.axis, &j.shape, &gf.members[k]),
                    j.fit[k].0,
                    j.fit[k].1,
                ),
                None => (
                    surface_of(
                        &f.axis,
                        &f.shape,
                        &Member {
                            pts: Vec::new(),
                            kind: f.kind,
                            slot: 0,
                        },
                    ),
                    f.rms,
                    f.max,
                ),
            };
            let r = &mut regions[f.region];
            let surface = orientation(&surface, &tris[f.region]);
            let sag = r
                .faces
                .iter()
                .map(|&fc| {
                    let t = faces[fc as usize];
                    sagitta(&surface, t.map(|v| vc[v as usize]))
                })
                .fold(0.0, f64::max);
            r.surface = surface;
            r.rms = rms;
            r.max = max;
            r.sag = sag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Frame;
    use super::*;

    fn strip(
        frame: &Frame,
        radius: impl Fn(f64) -> f64,
        angles: &[f64],
        zs: &[f64],
    ) -> (Vec<(V3, f64)>, Vec<Tri>) {
        let at = |t: f64, z: f64| frame.map([radius(z) * t.cos(), radius(z) * t.sin(), z]);
        let mut pts = Vec::new();
        for &t in angles {
            for &z in zs {
                pts.push((at(t, z), 1.0));
            }
        }
        let mut tris = Vec::new();
        for w in angles.windows(2) {
            for zz in zs.windows(2) {
                let (a, b, c, d) = (
                    at(w[0], zz[0]),
                    at(w[1], zz[0]),
                    at(w[1], zz[1]),
                    at(w[0], zz[1]),
                );
                for t in [[a, b, c], [a, c, d]] {
                    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
                    tris.push(Tri {
                        normal: unit(n),
                        area: norm(n) / 2.0,
                        centroid: scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0),
                    });
                }
            }
        }
        (pts, tris)
    }

    fn arc(n: usize, segments: usize, start: f64) -> Vec<f64> {
        (0..=segments)
            .map(|k| start + std::f64::consts::TAU * k as f64 / n as f64)
            .collect()
    }

    #[test]
    fn tessellation_law_recovers_radius_for_low_n_and_partial_arcs() {
        let frame = Frame::tilted();
        let a = frame.dir([0.0, 0.0, 1.0]);
        for (n, segments) in [(6, 5), (6, 3), (8, 2), (32, 8), (5, 4), (360, 12)] {
            let r = 3.7;
            let (pts, _) = strip(&frame, |_| r, &arc(n, segments, 0.3), &[0.0, 2.0]);
            let off = scale(cross(a, [1.0, 0.0, 0.0]), 0.004 * r);
            let axis = Axis {
                a,
                c: add(frame.off, off),
            };
            let law = tessellation_radius(&pts, &axis, r, 1e-9).unwrap();
            assert!((law - r).abs() < 1e-12 * r, "n={n}: {law}");
        }
    }

    #[test]
    fn tessellation_law_refuses_irregular_turning() {
        let frame = Frame::identity();
        let axis = Axis {
            a: [0.0, 0.0, 1.0],
            c: [0.0; 3],
        };
        let step = 73f64.to_radians() / 7.0;
        let odd: Vec<f64> = (0..=7).map(|k| k as f64 * step).collect();
        let (pts, _) = strip(&frame, |_| 2.0, &odd, &[0.0, 1.0]);
        assert_eq!(tessellation_radius(&pts, &axis, 2.0, 1e-9), None);
        let uneven = [0.0, 0.1, 0.25, 0.3, 0.5];
        let (pts, _) = strip(&frame, |_| 2.0, &uneven, &[0.0, 1.0]);
        assert_eq!(tessellation_radius(&pts, &axis, 2.0, 1e-9), None);
    }

    #[test]
    fn partial_arc_cylinder_fits_exactly() {
        let frame = Frame::tilted();
        for (n, segments) in [(32, 8), (6, 2), (40, 3)] {
            let (pts, tris) = strip(&frame, |_| 1.25, &arc(n, segments, 1.0), &[0.0, 0.5, 4.0]);
            let (kind, axis, shape, _, max) = fit_single(&pts, &tris, 1e-7).unwrap();
            assert_eq!(kind, Kind::Cylinder);
            assert!((shape[0] - 1.25).abs() < 1e-10, "n={n}: {}", shape[0]);
            assert!(dot(axis.a, frame.dir([0.0, 0.0, 1.0])).abs() > 1.0 - 1e-13);
            assert!(max < 1e-9);
        }
    }

    #[test]
    fn partial_arc_cone_fits_exactly() {
        let frame = Frame::tilted();
        for (n, segments) in [(48, 12), (8, 3)] {
            let (pts, tris) = strip(
                &frame,
                |z| 2.0 + 0.6 * z,
                &arc(n, segments, -0.4),
                &[0.0, 1.5, 3.0],
            );
            let (kind, axis, shape, _, max) = fit_single(&pts, &tris, 1e-7).unwrap();
            assert_eq!(kind, Kind::Cone { sign: 1.0 });
            assert!((shape[1] - 0.6f64.atan()).abs() < 1e-10, "n={n}");
            let apex = add(axis.c, scale(axis.a, shape[0]));
            assert!(norm(sub(apex, frame.map([0.0, 0.0, -2.0 / 0.6]))) < 1e-8);
            assert!(dot(axis.a, frame.dir([0.0, 0.0, 1.0])) > 1.0 - 1e-13);
            assert!(max < 1e-9);
        }
    }

    #[test]
    fn noisy_cylinder_snaps_to_the_world_axis() {
        let frame = Frame::identity();
        let (mut pts, tris) = strip(&frame, |_| 5.0, &arc(24, 24, 0.0), &[0.0, 3.0]);
        for (i, (p, _)) in pts.iter_mut().enumerate() {
            let e = 1e-5 * ((i * 7919 % 13) as f64 / 6.0 - 1.0);
            p[0] += e;
            p[1] -= 0.5 * e;
        }
        let (kind, axis, shape, _, _) = fit_single(&pts, &tris, 1e-4).unwrap();
        assert_eq!(kind, Kind::Cylinder);
        assert_ne!(axis.a.map(f64::abs), [0.0, 0.0, 1.0]);
        let member = Member {
            pts: pts.clone(),
            kind,
            slot: 0,
        };
        let snapped = fit_joint(
            &[member],
            Axis {
                a: [0.0, 0.0, 1.0],
                c: axis.c,
            },
            shape,
            true,
            1e-4,
        );
        assert!(accept(&snapped, 1e-4));
        assert_eq!(snapped.axis.a, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn cylinder_sagitta_is_exact() {
        let s = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 2.0,
            reversed: false,
        };
        let t = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 5.0]];
        assert!((sagitta(&s, t) - (2.0 - 2f64.sqrt())).abs() < 1e-12);
    }
}
