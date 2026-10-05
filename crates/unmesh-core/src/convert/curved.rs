use super::fit::Region;
use super::linalg::{V3, add, cross, dot, norm, outer_add, scale, sub, sym_eigen, unit};
use super::segment::TriInfo;
use super::surface::{Surface, radial};

const AXIS_STEP: f64 = 1e-7;
pub const LM_ITERS: usize = 40;
pub const QUICK_ITERS: usize = 10;
const ROW_GAP: f64 = 0.05;
pub const IRLS_ROUNDS: usize = 3;
pub const MAX_GROUP: usize = 16;
const LAW_GAP_SPREAD: f64 = 0.02;
const LAW_N_REL: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Cylinder,
    Cone { sign: f64 },
    Sphere,
    Torus,
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

pub fn basis(a: V3) -> (V3, V3) {
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
        Kind::Sphere => {
            let dh = h - shape[m.slot];
            (rho * rho + dh * dh).sqrt() - shape[m.slot + 1]
        }
        Kind::Torus => {
            let (dr, dh) = (rho - shape[m.slot + 1], h - shape[m.slot]);
            (dr * dr + dh * dh).sqrt() - shape[m.slot + 2]
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

pub fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
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
    free: Vec<usize>,
    len_step: f64,
}

impl Problem<'_> {
    fn n_axis(&self) -> usize {
        if self.fix_dir { 2 } else { 4 }
    }

    fn n_params(&self) -> usize {
        self.n_axis() + self.free.len()
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
        let mut s = shape.to_vec();
        for (j, &slot) in self.free.iter().enumerate() {
            s[slot] += x[k + j];
        }
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

fn lm(problem: &Problem<'_>, axis: &mut Axis, shape: &mut Vec<f64>, weights: &[f64], iters: usize) {
    let n = problem.n_params();
    let mut r0 = Vec::new();
    let mut rp = Vec::new();
    let mut rm = Vec::new();
    let mut lambda = 1e-6;
    problem.residuals(axis, shape, weights, &mut r0);
    let mut c0 = cost(&r0);
    for _ in 0..iters {
        if c0 == 0.0 {
            break;
        }
        let m = r0.len();
        let mut jac = vec![vec![0.0; m]; n];
        for j in 0..n {
            let h = if j >= problem.n_axis() {
                let s = shape[problem.free[j - problem.n_axis()]];
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
    axis: Axis,
    shape: Vec<f64>,
    fixed: &[bool],
    fix_dir: bool,
    tol: f64,
) -> Joint {
    fit_joint_with(
        members,
        axis,
        shape,
        fixed,
        fix_dir,
        tol,
        IRLS_ROUNDS,
        LM_ITERS,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fit_joint_with(
    members: &[Member],
    mut axis: Axis,
    mut shape: Vec<f64>,
    fixed: &[bool],
    fix_dir: bool,
    tol: f64,
    rounds: usize,
    iters: usize,
) -> Joint {
    let fix_dir = fix_dir || members.iter().all(|m| m.kind == Kind::Sphere);
    let extent = members
        .iter()
        .flat_map(|m| m.pts.iter())
        .map(|(p, _)| norm(sub(*p, axis.c)))
        .fold(0.0, f64::max)
        .max(1e-9);
    let problem = Problem {
        members,
        fix_dir,
        free: (0..shape.len())
            .filter(|&i| !fixed.get(i).copied().unwrap_or(false))
            .collect(),
        len_step: 1e-7 * extent,
    };
    let base: Vec<f64> = members
        .iter()
        .flat_map(|m| m.pts.iter().map(|&(_, w)| w.sqrt()))
        .collect();
    let mut weights = base.clone();
    for round in 0..rounds {
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
        lm(&problem, &mut axis, &mut shape, &weights, iters);
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

pub fn kasa(pts: &[(V3, f64)], a: V3) -> Option<(V3, f64)> {
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

/// Cone axis from triangle normals that fall into two tilt populations (a
/// staggered band, where triangles with two corners on the upper circle tilt
/// differently from those with two on the lower): each population is
/// centred on its own mean before the scatter, so the tilt gap does not
/// masquerade as spread along the axis.
fn split_axis(tris: &[Tri], a0: V3) -> Option<V3> {
    let mut s: Vec<f64> = tris.iter().map(|t| dot(t.normal, a0)).collect();
    s.sort_by(f64::total_cmp);
    let (lo, hi) = (s[0], s[s.len() - 1]);
    if hi - lo <= 0.0 {
        return None;
    }
    let mut cut = 0.5 * (lo + hi);
    for _ in 0..8 {
        let (mut a, mut na, mut b, mut nb) = (0.0, 0.0, 0.0, 0.0);
        for &x in &s {
            if x < cut {
                a += x;
                na += 1.0;
            } else {
                b += x;
                nb += 1.0;
            }
        }
        if na == 0.0 || nb == 0.0 {
            return None;
        }
        cut = 0.5 * (a / na + b / nb);
    }
    let mut means = [[0.0; 3]; 2];
    let mut w = [0.0; 2];
    for t in tris {
        let k = usize::from(dot(t.normal, a0) >= cut);
        means[k] = add(means[k], scale(t.normal, t.area));
        w[k] += t.area;
    }
    if w[0] <= 0.0 || w[1] <= 0.0 {
        return None;
    }
    let means = [scale(means[0], 1.0 / w[0]), scale(means[1], 1.0 / w[1])];
    let mut m = [[0.0; 3]; 3];
    for t in tris {
        let k = usize::from(dot(t.normal, a0) >= cut);
        let d = sub(t.normal, means[k]);
        outer_add(&mut m, d, d, t.area);
    }
    let (_, vecs) = sym_eigen(m);
    Some(vecs[0])
}

/// Apex and half-angle about a given axis direction from the vertices alone:
/// `|q - c|² = t² (h - h0)²` (q the projection across the axis, h the height
/// along it) is linear in `c`, `t²`, `-2 t² h0` and a constant, so a
/// Kasa-style least squares is exact on exact vertices.
fn cone_about(pts: &[(V3, f64)], a: V3) -> Option<(Axis, f64)> {
    let (u, v) = basis(a);
    let mut sw = 0.0;
    let mut mean = [0.0; 3];
    for &(p, w) in pts {
        sw += w;
        mean = add(mean, scale(p, w));
    }
    if sw <= 0.0 {
        return None;
    }
    let mean = scale(mean, 1.0 / sw);
    let mut m = vec![vec![0.0; 5]; 5];
    let mut b = vec![0.0; 5];
    for &(p, w) in pts {
        let d = sub(p, mean);
        let (x, y, h) = (dot(d, u), dot(d, v), dot(d, a));
        let row = [2.0 * x, 2.0 * y, h * h, h, 1.0];
        let rhs = x * x + y * y;
        for i in 0..5 {
            b[i] += w * row[i] * rhs;
            for j in 0..5 {
                m[i][j] += w * row[i] * row[j];
            }
        }
    }
    let sol = solve_dense(m, b)?;
    let t2 = sol[2];
    if t2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let h0 = -sol[3] / (2.0 * t2);
    let mut axis = a;
    let mut apex = add(
        mean,
        add(add(scale(u, sol[0]), scale(v, sol[1])), scale(a, h0)),
    );
    let mut hm = 0.0;
    for &(p, w) in pts {
        hm += w * dot(sub(p, apex), axis);
    }
    if hm < 0.0 {
        axis = scale(axis, -1.0);
    }
    apex = add(apex, [0.0; 3]);
    let alpha = t2.sqrt().atan();
    (alpha > 0.0 && alpha < std::f64::consts::FRAC_PI_2)
        .then_some((Axis { a: axis, c: apex }, alpha))
}

/// Axis from the rows of a band: vertices of a tessellated cone or cylinder
/// sit on circles across the axis, so heights along a rough axis cluster by
/// row, and each row's plane normal (its smallest scatter direction) is the
/// axis, to the accuracy of the vertices rather than of the facet normals.
fn row_axis(pts: &[(V3, f64)], rough: V3) -> Option<V3> {
    let mut hs: Vec<(f64, usize)> = pts
        .iter()
        .enumerate()
        .map(|(i, &(p, _))| (dot(p, rough), i))
        .collect();
    hs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let range = hs[hs.len() - 1].0 - hs[0].0;
    if range <= 0.0 {
        return None;
    }
    let mut rows: Vec<Vec<usize>> = vec![vec![hs[0].1]];
    for w in hs.windows(2) {
        if w[1].0 - w[0].0 > ROW_GAP * range {
            rows.push(Vec::new());
        }
        rows.last_mut()?.push(w[1].1);
    }
    let mut acc = [0.0; 3];
    for row in rows.iter().filter(|r| r.len() >= 3) {
        let n = row.len() as f64;
        let c = scale(row.iter().fold([0.0; 3], |s, &i| add(s, pts[i].0)), 1.0 / n);
        let mut m = [[0.0; 3]; 3];
        for &i in row {
            let d = sub(pts[i].0, c);
            outer_add(&mut m, d, d, 1.0);
        }
        let (vals, vecs) = sym_eigen(m);
        if vals[1] <= 1e-6 * vals[2] {
            continue;
        }
        let mut a = vecs[0];
        if dot(a, rough) < 0.0 {
            a = scale(a, -1.0);
        }
        acc = add(acc, scale(a, vals[1]));
    }
    let len = norm(acc);
    (len > 0.0).then(|| scale(acc, 1.0 / len))
}

/// Cone starts for `fit_quick`: the plane-intersection estimate of
/// `init_cone`, and the vertex solve about the plain and the two-population
/// normal axes, best first by largest vertex residual.
fn cone_starts(pts: &[(V3, f64)], tris: &[Tri]) -> Vec<(Axis, f64, f64)> {
    let mut out = Vec::new();
    let mut push = |s: Option<(Axis, f64)>| {
        if let Some((axis, alpha)) = s {
            let r = max_residual(Kind::Cone { sign: 1.0 }, &axis, &[0.0, alpha], pts);
            if r.is_finite() {
                out.push((axis, alpha, r));
            }
        }
    };
    push(init_cone(pts, tris));
    if let Some((vals, vecs, _)) = normal_scatter(tris, true)
        && vals[1].is_finite()
    {
        push(cone_about(pts, vecs[0]));
        if let Some(a) = split_axis(tris, vecs[0]) {
            push(cone_about(pts, a));
            if let Some(b) = row_axis(pts, a) {
                push(cone_about(pts, b));
            }
        }
        if let Some(b) = row_axis(pts, vecs[0]) {
            push(cone_about(pts, b));
        }
    }
    out.sort_by(|x, y| x.2.total_cmp(&y.2));
    out
}

/// Radius from the tessellation law: a circle cut into `N` equal chords `c`
/// turns by `2π/N` at every ruling, so `r = c / (2 sin(π/N))`. Chords and
/// turning angles are measured between rulings and do not depend on the
/// fitted centre; only the ruling order does. `None` unless the chords and
/// turning angles are equal within the noise `tol` allows, the mean turning
/// angle is `2π/N` to within its own measurement uncertainty, and the chord
/// count agrees with `N` (all `N` for a closed ring, fewer for an arc).
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
    let len_tol = (4.0 * tol).min(LAW_GAP_SPREAD * c);
    let turn_tol = (8.0 * tol / c).min(LAW_GAP_SPREAD * theta);
    if theta <= 0.0
        || lens.iter().any(|l| (l - c).abs() > len_tol)
        || turns.iter().any(|t| (t - theta).abs() > turn_tol)
    {
        return None;
    }
    let n = (tau / theta).round();
    let quantum = (2.0 * tol / c / (turns.len() as f64).sqrt()).min(LAW_N_REL * theta);
    let steps = chords.len() as f64;
    if n < 3.0
        || (tau / n - theta).abs() > quantum
        || (closed && steps != n)
        || (!closed && steps >= n)
    {
        return None;
    }
    Some(c / (2.0 * (std::f64::consts::PI / n).sin()))
}

pub fn orientation(surface: &Surface, tris: &[Tri]) -> Surface {
    if matches!(surface, Surface::Plane { .. } | Surface::Facets) {
        return *surface;
    }
    let natural = surface.with_reversed(false);
    let mut s = 0.0;
    for t in tris {
        if let Some(n) = natural.normal_at(t.centroid) {
            s += t.area * dot(n, t.normal);
        }
    }
    natural.with_reversed(s < 0.0)
}

fn canonical(mut a: V3) -> V3 {
    let big = (0..3)
        .max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs()))
        .unwrap();
    if a[big] < 0.0 {
        a = scale(a, -1.0);
    }
    a
}

pub fn surface_of(axis: &Axis, shape: &[f64], m: &Member) -> Surface {
    match m.kind {
        Kind::Cylinder => {
            let a = canonical(axis.a);
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
        Kind::Sphere => Surface::Sphere {
            center: add(axis.c, scale(axis.a, shape[m.slot])),
            radius: shape[m.slot + 1],
            reversed: false,
        },
        Kind::Torus => Surface::Torus {
            center: add(axis.c, scale(axis.a, shape[m.slot])),
            axis: canonical(axis.a),
            major: shape[m.slot + 1],
            minor: shape[m.slot + 2],
            reversed: false,
        },
    }
}

/// Upper bound on the distance from any point of the triangle to the surface:
/// the chord sagitta plus how far the corners are off it. Exact for a
/// cylinder (the distance to the axis is convex, so its minimum over the
/// triangle is the 2D point-triangle distance in the plane normal to the
/// axis) and for a cone (see `cone_sagitta`).
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
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } => cone_sagitta(surface, apex, axis, half_angle, t),
        Surface::Sphere { center, radius, .. } => super::doubly::sphere_sagitta(center, radius, t),
        Surface::Torus {
            center,
            axis,
            major,
            minor,
            ..
        } => super::doubly::torus_sagitta(center, axis, major, minor, t),
        _ => 0.0,
    }
}

/// For points on the widening side of the apex (`h >= 0`), the signed
/// distance to the cone is `f = rho cos(a) - h sin(a)`. It is convex (rho is
/// a distance to a line, h is affine), so its maximum over the triangle is at
/// a corner and its minimum is at a corner, at a stationary point of `f`
/// along an edge (a quadratic in the edge parameter), or where the axis
/// pierces the triangle (`f` is not differentiable there). No other interior
/// point can be stationary: on a plane crossing the axis, `c|y| - s l(y)`
/// with `l` affine has no stationary point off `y = 0`, and on a plane
/// parallel to the axis `h` keeps a nonzero slope. A triangle reaching
/// behind the apex falls back to the 1-Lipschitz bound of the distance.
fn cone_sagitta(surface: &Surface, apex: V3, axis: V3, alpha: f64, t: [V3; 3]) -> f64 {
    let (s, c) = alpha.sin_cos();
    let f = |p: V3| {
        let (h, rho, _) = radial(p, apex, axis);
        rho * c - h * s
    };
    let corner = t
        .iter()
        .map(|p| surface.distance(*p).abs())
        .fold(0.0, f64::max);
    if t.iter().any(|p| dot(sub(*p, apex), axis) < 0.0) {
        let edge = (0..3)
            .map(|i| norm(sub(t[(i + 1) % 3], t[i])))
            .fold(0.0, f64::max);
        return corner + edge;
    }
    let mut fmin = t.iter().map(|p| f(*p)).fold(f64::INFINITY, f64::min);
    for i in 0..3 {
        let (p0, p1) = (t[i], t[(i + 1) % 3]);
        let d = sub(p1, p0);
        let dh = dot(d, axis);
        let w = sub(d, scale(axis, dh));
        let q0 = {
            let q = sub(p0, apex);
            sub(q, scale(axis, dot(q, axis)))
        };
        let (qa, qb, qc) = (dot(w, w), dot(q0, w), dot(q0, q0));
        let k = c * c * qa - s * s * dh * dh;
        let (a2, b2, c2) = (qa * k, 2.0 * qb * k, c * c * qb * qb - s * s * dh * dh * qc);
        let mut roots = Vec::new();
        if a2.abs() > 0.0 {
            let disc = b2 * b2 - 4.0 * a2 * c2;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                roots.push((-b2 + sq) / (2.0 * a2));
                roots.push((-b2 - sq) / (2.0 * a2));
            }
        }
        if qa > 0.0 {
            roots.push(-qb / qa);
        }
        for u in roots {
            if (0.0..=1.0).contains(&u) {
                fmin = fmin.min(f(add(p0, scale(d, u))));
            }
        }
    }
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let denom = dot(n, axis);
    if denom != 0.0 {
        let x = add(apex, scale(axis, dot(n, sub(t[0], apex)) / denom));
        let inside = (0..3).all(|i| dot(cross(sub(t[(i + 1) % 3], t[i]), sub(x, t[i])), n) >= 0.0);
        if inside {
            fmin = fmin.min(f(x));
        }
    }
    corner.max(-fmin)
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
    law: bool,
    kind: Kind,
    axis: Axis,
    shape: Vec<f64>,
    rms: f64,
    max: f64,
}

pub fn region_tris(vc: &[V3], faces: &[[u32; 3]], info: &[TriInfo], r: &Region) -> Vec<Tri> {
    face_tris(vc, faces, info, &r.faces)
}

pub fn face_tris(vc: &[V3], faces: &[[u32; 3]], info: &[TriInfo], list: &[u32]) -> Vec<Tri> {
    list.iter()
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

pub type Single = (Kind, Axis, Vec<f64>, f64, f64, bool);

fn law_fits(law_rms: f64, lsq_rms: f64, n: usize, r: f64) -> bool {
    let dof = (n as f64 - 5.0).max(1.0);
    law_rms * law_rms <= lsq_rms * lsq_rms * (1.0 + 4.0 / dof) + (1e-12 * r).powi(2)
}

pub fn fit_single(pts: &[(V3, f64)], tris: &[Tri], tol: f64) -> Option<Single> {
    fit_single_gated(pts, tris, tol, f64::INFINITY)
        .or_else(|| super::doubly::fit_doubly(pts, tris, tol, f64::INFINITY))
}

/// A cheap screen for seeds: the gated initial estimates refined by one
/// short unweighted round, without the tessellation law. On failure, the
/// smallest largest-residual of the initial estimates.
pub fn fit_quick(pts: &[(V3, f64)], tris: &[Tri], tol: f64, gate: f64) -> Result<Single, f64> {
    let member = |kind| {
        [Member {
            pts: pts.to_vec(),
            kind,
            slot: 0,
        }]
    };
    let mut best = f64::INFINITY;
    if let Some((axis, r)) = init_cylinder(pts, tris) {
        let init = max_residual(Kind::Cylinder, &axis, &[r], pts);
        best = init;
        if init <= gate {
            let m = member(Kind::Cylinder);
            let j = fit_joint_with(&m, axis, vec![r], &[false], false, tol, 1, QUICK_ITERS);
            let (rms, max) = j.fit[0];
            if max <= tol && valid_shape(&j.shape, &m[0]) {
                return Ok((Kind::Cylinder, j.axis, j.shape, rms, max, false));
            }
        }
    }
    let Some((axis, alpha, r)) = cone_starts(pts, tris).into_iter().next() else {
        return Err(best);
    };
    let best = best.min(r);
    if r > gate {
        return Err(best);
    }
    let m = member(Kind::Cone { sign: 1.0 });
    let j = fit_joint_with(
        &m,
        axis,
        vec![0.0, alpha],
        &[false, false],
        false,
        tol,
        1,
        QUICK_ITERS,
    );
    let (rms, max) = j.fit[0];
    if max <= tol && valid_shape(&j.shape, &m[0]) {
        Ok((Kind::Cone { sign: 1.0 }, j.axis, j.shape, rms, max, false))
    } else {
        Err(best)
    }
}

/// `fit_single`, skipping the refinement of any initial estimate whose
/// largest vertex residual exceeds `gate` (the estimates are exact on exact
/// chord facets, so a large residual means the region is not that surface).
pub fn fit_single_gated(pts: &[(V3, f64)], tris: &[Tri], tol: f64, gate: f64) -> Option<Single> {
    let cyl_init = init_cylinder(pts, tris).filter(|(axis, r)| {
        gate.is_infinite() || max_residual(Kind::Cylinder, axis, &[*r], pts) <= gate
    });
    if let Some((axis, r)) = cyl_init {
        let m = [Member {
            pts: pts.to_vec(),
            kind: Kind::Cylinder,
            slot: 0,
        }];
        let j = fit_joint(&m, axis, vec![r], &[false], false, tol);
        let (rms, max) = j.fit[0];
        if max <= tol && j.shape[0] > 0.0 {
            if let Some(rl) = tessellation_radius(pts, &j.axis, j.shape[0], tol) {
                let jl = fit_joint(&m, j.axis, vec![rl], &[true], false, tol);
                let (lr, lm) = jl.fit[0];
                if lm <= tol && law_fits(lr, rms, pts.len(), rl) {
                    return Some((Kind::Cylinder, jl.axis, jl.shape, lr, lm, true));
                }
            }
            return Some((Kind::Cylinder, j.axis, j.shape, rms, max, false));
        }
    }
    let (axis, alpha) = init_cone(pts, tris).filter(|(axis, alpha)| {
        gate.is_infinite()
            || max_residual(Kind::Cone { sign: 1.0 }, axis, &[0.0, *alpha], pts) <= gate
    })?;
    let m = [Member {
        pts: pts.to_vec(),
        kind: Kind::Cone { sign: 1.0 },
        slot: 0,
    }];
    let j = fit_joint(&m, axis, vec![0.0, alpha], &[false, false], false, tol);
    let (rms, max) = j.fit[0];
    (max <= tol && valid_shape(&j.shape, &m[0])).then_some((
        Kind::Cone { sign: 1.0 },
        j.axis,
        j.shape,
        rms,
        max,
        false,
    ))
}

pub fn max_residual(kind: Kind, axis: &Axis, shape: &[f64], pts: &[(V3, f64)]) -> f64 {
    let m = Member {
        pts: Vec::new(),
        kind,
        slot: 0,
    };
    pts.iter()
        .map(|&(p, _)| point_residual(axis, shape, &m, p).abs())
        .fold(0.0, f64::max)
}

pub fn refit(pts: &[(V3, f64)], kind: Kind, axis: Axis, shape: &[f64], tol: f64) -> Option<Single> {
    refit_with(pts, kind, axis, shape, tol, IRLS_ROUNDS, LM_ITERS)
}

/// `refit` with one short unweighted round, for growth steps.
pub fn refit_quick(
    pts: &[(V3, f64)],
    kind: Kind,
    axis: Axis,
    shape: &[f64],
    tol: f64,
) -> Option<Single> {
    refit_with(pts, kind, axis, shape, tol, 1, QUICK_ITERS)
}

fn refit_with(
    pts: &[(V3, f64)],
    kind: Kind,
    axis: Axis,
    shape: &[f64],
    tol: f64,
    rounds: usize,
    iters: usize,
) -> Option<Single> {
    let m = [Member {
        pts: pts.to_vec(),
        kind,
        slot: 0,
    }];
    let j = fit_joint_with(
        &m,
        axis,
        shape.to_vec(),
        &vec![false; shape.len()],
        false,
        tol,
        rounds,
        iters,
    );
    let (rms, max) = j.fit[0];
    (max <= tol && valid_shape(&j.shape, &m[0])).then_some((kind, j.axis, j.shape, rms, max, false))
}

pub fn valid_shape(shape: &[f64], m: &Member) -> bool {
    match m.kind {
        Kind::Cylinder => shape[m.slot] > 0.0,
        Kind::Cone { .. } => {
            let a = shape[m.slot + 1];
            shape[m.slot].is_finite() && a > 0.0 && a < std::f64::consts::FRAC_PI_2
        }
        Kind::Sphere => shape[m.slot].is_finite() && shape[m.slot + 1] > 0.0,
        Kind::Torus => {
            let (major, minor) = (shape[m.slot + 1], shape[m.slot + 2]);
            shape[m.slot].is_finite() && minor > 0.0 && major > minor && major.is_finite()
        }
    }
}

fn line_distance(p: V3, axis: &Axis) -> f64 {
    radial(p, axis.c, axis.a).1
}

fn coaxial(a: &Fitted, b: &Fitted, cos_lim: f64, dist: f64) -> bool {
    a.kind != Kind::Sphere
        && b.kind != Kind::Sphere
        && dot(a.axis.a, b.axis.a).abs() >= cos_lim
        && line_distance(b.axis.c, &a.axis) <= dist
        && line_distance(a.axis.c, &b.axis) <= dist
}

struct GroupFit {
    members: Vec<Member>,
    axis: Axis,
    shape: Vec<f64>,
    fixed: Vec<bool>,
}

fn build_group(fits: &[&Fitted], pts: &[Vec<(V3, f64)>], tol: f64) -> GroupFit {
    let lead = fits
        .iter()
        .max_by(|x, y| pts[x.region].len().cmp(&pts[y.region].len()))
        .unwrap();
    let axis = lead.axis;
    let mut shape: Vec<f64> = Vec::new();
    let mut fixed: Vec<bool> = Vec::new();
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
                        fixed.push(false);
                        radii.push((r, shape.len() - 1));
                        shape.len() - 1
                    }
                };
                if f.law && !fixed[slot] {
                    shape[slot] = r;
                    fixed[slot] = true;
                }
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
                fixed.extend([false, false]);
                members.push(Member {
                    pts: ps,
                    kind: Kind::Cone { sign },
                    slot: shape.len() - 2,
                });
            }
            Kind::Sphere | Kind::Torus => {
                let center = add(f.axis.c, scale(f.axis.a, f.shape[0]));
                shape.push(dot(sub(center, axis.c), axis.a));
                shape.extend_from_slice(&f.shape[1..]);
                fixed.extend(std::iter::repeat_n(false, f.shape.len()));
                members.push(Member {
                    pts: ps,
                    kind: f.kind,
                    slot: shape.len() - f.shape.len(),
                });
            }
        }
    }
    GroupFit {
        members,
        axis,
        shape,
        fixed,
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

fn accept(fit: &Joint, members: &[Member], tol: f64) -> bool {
    fit.fit
        .iter()
        .all(|&(rms, max)| max <= tol && rms <= 0.5 * tol)
        && members.iter().all(|m| valid_shape(&fit.shape, m))
}

pub fn refine_regions(
    vc: &[V3],
    faces: &[[u32; 3]],
    info: &[TriInfo],
    regions: &mut [Region],
    tol: f64,
    snap_deg: f64,
) {
    refine_regions_with(vc, faces, info, regions, tol, snap_deg, Vec::new(), true);
}

/// As `refine_regions`, but the regions listed in `prefit` (region index and
/// its fit, every vertex within `tol`) skip their own fit and join the
/// coaxial snapping with it. Other regions that exceed `tol` as planes are
/// fitted on their own only when `fit_loose` is set.
#[allow(clippy::too_many_arguments)]
pub fn refine_regions_with(
    vc: &[V3],
    faces: &[[u32; 3]],
    info: &[TriInfo],
    regions: &mut [Region],
    tol: f64,
    snap_deg: f64,
    prefit: Vec<(usize, Single)>,
    fit_loose: bool,
) {
    let mut fits: Vec<Fitted> = Vec::new();
    let mut pts: Vec<Vec<(V3, f64)>> = vec![Vec::new(); regions.len()];
    let mut tris: Vec<Vec<Tri>> = (0..regions.len()).map(|_| Vec::new()).collect();
    let mut given: Vec<Option<Single>> = vec![None; regions.len()];
    for (ri, s) in prefit {
        given[ri] = Some(s);
    }
    for (ri, r) in regions.iter().enumerate() {
        if r.area <= 0.0
            || (given[ri].is_none() && (!fit_loose || r.max <= tol || !r.surface.is_analytic()))
        {
            continue;
        }
        let p: Vec<(V3, f64)> = r.verts.iter().map(|&(v, w)| (vc[v as usize], w)).collect();
        let t = region_tris(vc, faces, info, r);
        let fit = match given[ri].take() {
            Some(s) => Some(s),
            None => fit_single(&p, &t, tol),
        };
        if let Some((kind, axis, shape, rms, max, law)) = fit {
            fits.push(Fitted {
                region: ri,
                law,
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
            let j = fit_joint(
                &gf.members,
                gf.axis,
                gf.shape.clone(),
                &gf.fixed,
                false,
                tol,
            );
            if accept(&j, &gf.members, tol) {
                gf.axis = j.axis;
                gf.shape = j.shape.clone();
                joint = Some(j);
            }
        }
        if (g.len() == 1 || joint.is_some())
            && fits[g[0]].kind != Kind::Sphere
            && let Some(e) = world_axis(gf.axis.a, cos_lim)
        {
            let axis = Axis { a: e, c: gf.axis.c };
            let j = fit_joint(&gf.members, axis, gf.shape.clone(), &gf.fixed, true, tol);
            if accept(&j, &gf.members, tol) {
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
    fn tessellation_law_rounds_n_to_the_nearest_step_count() {
        let frame = Frame::tilted();
        let step = std::f64::consts::TAU / 36.0 * (1.0 - 1e-7);
        let angles: Vec<f64> = (0..=8).map(|k| 0.2 + step * k as f64).collect();
        let (pts, _) = strip(&frame, |_| 10.0, &angles, &[0.0, 1.0]);
        let axis = Axis {
            a: frame.dir([0.0, 0.0, 1.0]),
            c: frame.off,
        };
        let law = tessellation_radius(&pts, &axis, 10.0, 1e-6).unwrap();
        assert!((law - 10.0).abs() < 1e-5, "{law}");
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
            let (kind, axis, shape, _, max, _) = fit_single(&pts, &tris, 1e-7).unwrap();
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
            let (kind, axis, shape, _, max, _) = fit_single(&pts, &tris, 1e-7).unwrap();
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
        let (kind, axis, shape, _, _, _) = fit_single(&pts, &tris, 1e-4).unwrap();
        assert_eq!(kind, Kind::Cylinder);
        assert_ne!(axis.a.map(f64::abs), [0.0, 0.0, 1.0]);
        let member = Member {
            pts: pts.clone(),
            kind,
            slot: 0,
        };
        let members = [member];
        let snapped = fit_joint(
            &members,
            Axis {
                a: [0.0, 0.0, 1.0],
                c: axis.c,
            },
            shape,
            &[false],
            true,
            1e-4,
        );
        assert!(accept(&snapped, &members, 1e-4));
        assert_eq!(snapped.axis.a, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn cone_sagitta_bounds_a_dense_max_over_random_triangles() {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut lo, mut hi): (f64, f64) = (f64::INFINITY, 0.0);
        for _ in 0..400 {
            let alpha = (3.0 + 82.0 * rnd()).to_radians();
            let axis = unit([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]);
            let apex = [10.0 * rnd(), -5.0 * rnd(), 3.0 * rnd()];
            let s = Surface::Cone {
                apex,
                axis,
                half_angle: alpha,
                reversed: rnd() < 0.5,
            };
            let (u, v) = basis(axis);
            let spread = (5.0 + 175.0 * rnd()).to_radians();
            let phi0 = std::f64::consts::TAU * rnd();
            let t: [V3; 3] = std::array::from_fn(|_| {
                let h = 0.2 + 10.0 * rnd();
                let phi = phi0 + spread * rnd();
                let r = h * alpha.tan();
                add(
                    apex,
                    add(
                        scale(axis, h),
                        add(scale(u, r * phi.cos()), scale(v, r * phi.sin())),
                    ),
                )
            });
            let bound = sagitta(&s, t);
            let k = 120;
            let mut dense: f64 = 0.0;
            for i in 0..=k {
                for j in 0..=k - i {
                    let (a, b) = (i as f64 / k as f64, j as f64 / k as f64);
                    let p = add(
                        add(scale(t[0], a), scale(t[1], b)),
                        scale(t[2], 1.0 - a - b),
                    );
                    dense = dense.max(s.distance(p).abs());
                }
            }
            if dense < 1e-9 {
                continue;
            }
            assert!(
                bound >= dense * (1.0 - 1e-9),
                "bound {bound} < dense {dense}"
            );
            assert!(bound <= 2.0 * dense, "bound {bound} > 2x dense {dense}");
            lo = lo.min(bound / dense);
            hi = hi.max(bound / dense);
        }
        eprintln!("cone sagitta bound / dense max: min {lo:.6} max {hi:.6}");
    }

    #[test]
    fn group_refit_refuses_an_out_of_range_cone() {
        let members = [Member {
            pts: Vec::new(),
            kind: Kind::Cone { sign: 1.0 },
            slot: 0,
        }];
        let axis = Axis {
            a: [0.0, 0.0, 1.0],
            c: [0.0; 3],
        };
        for (alpha, ok) in [(0.5, true), (-0.1, false), (1.6, false)] {
            let joint = Joint {
                axis,
                shape: vec![0.0, alpha],
                fit: vec![(0.0, 0.0)],
            };
            assert_eq!(accept(&joint, &members, 1e-3), ok, "{alpha}");
        }
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
