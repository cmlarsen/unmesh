use super::curved::{
    Axis, IRLS_ROUNDS, Kind, LM_ITERS, Member, QUICK_ITERS, Single, Tri, fit_joint_with, kasa,
    max_residual, solve_dense, valid_shape,
};
use super::linalg::{V3, add, cross, dot, norm, outer_add, scale, sub, sym_eigen};

const SPINE_GRID: usize = 48;
const SPINE_GOLDEN: usize = 48;
const SPINE_RANGE: (f64, f64) = (1e-3, 2.0);
const ENCLOSE_REL: f64 = 1e-3;
const MAX_PIECES: usize = 512;
const MAX_SPAN_COS: f64 = std::f64::consts::FRAC_1_SQRT_2;
const GMAX_SLACK: f64 = 0.02;
const GMAX_DEPTH: usize = 6;
const MIN_RINGS: usize = 4;

/// Sphere through the vertices: `|p|² + D·p + E = 0` is linear in `D` and
/// `E`, exact on exact vertices (solved about the weighted mean).
pub fn init_sphere(pts: &[(V3, f64)]) -> Option<(Axis, Vec<f64>)> {
    let (mean, _) = weighted_mean(pts.iter().copied())?;
    let mut m = vec![vec![0.0; 4]; 4];
    let mut b = vec![0.0; 4];
    for &(p, w) in pts {
        let d = sub(p, mean);
        let row = [2.0 * d[0], 2.0 * d[1], 2.0 * d[2], 1.0];
        let rhs = dot(d, d);
        for i in 0..4 {
            b[i] += w * row[i] * rhs;
            for j in 0..4 {
                m[i][j] += w * row[i] * row[j];
            }
        }
    }
    let sol = solve_dense(m, b)?;
    let c = [sol[0], sol[1], sol[2]];
    let r2 = sol[3] + dot(c, c);
    (r2 > 0.0).then(|| {
        (
            Axis {
                a: [0.0, 0.0, 1.0],
                c: add(mean, c),
            },
            vec![0.0, r2.sqrt()],
        )
    })
}

fn weighted_mean(it: impl Iterator<Item = (V3, f64)>) -> Option<(V3, f64)> {
    let mut sw = 0.0;
    let mut m = [0.0; 3];
    for (p, w) in it {
        sw += w;
        m = add(m, scale(p, w));
    }
    (sw > 0.0).then(|| (scale(m, 1.0 / sw), sw))
}

/// The circle through the centres of curvature `centroid - s * normal`: for
/// the tube radius `s` (signed by orientation) of a torus they all lie on its
/// spine circle. Returns the circle (centre, axis, radius) and its mean
/// squared misfit.
fn spine_circle(tris: &[Tri], s: f64) -> Option<(V3, V3, f64, f64)> {
    let pts: Vec<(V3, f64)> = tris
        .iter()
        .map(|t| (sub(t.centroid, scale(t.normal, s)), t.area))
        .collect();
    let (mean, sw) = weighted_mean(pts.iter().copied())?;
    let mut cov = [[0.0; 3]; 3];
    for &(p, w) in &pts {
        let d = sub(p, mean);
        outer_add(&mut cov, d, d, w / sw);
    }
    let (_, vecs) = sym_eigen(cov);
    let a = vecs[0];
    if !a.iter().all(|x| x.is_finite()) || norm(a) == 0.0 {
        return None;
    }
    let (c, big) = kasa(&pts, a)?;
    let center = add(c, scale(a, dot(mean, a)));
    let mut cost = 0.0;
    for &(p, w) in &pts {
        let q = sub(p, center);
        let h = dot(q, a);
        let rho = norm(sub(q, scale(a, h)));
        cost += w * (h * h + (rho - big) * (rho - big));
    }
    Some((center, a, big, cost / sw))
}

/// Torus start from the spine circle traced through the centres of
/// curvature: the tube radius minimising the circle misfit (log grid on both
/// orientations, then golden section), the circle's centre, axis and radius,
/// and the mean distance of the vertices from the spine as the tube radius.
pub fn init_torus(pts: &[(V3, f64)], tris: &[Tri]) -> Option<(Axis, Vec<f64>)> {
    let (lo, hi) = bounds(pts);
    let extent = norm(sub(hi, lo));
    if extent <= 0.0 || tris.len() < 4 {
        return None;
    }
    let ln = |k: f64| {
        let t = k / (SPINE_GRID - 1) as f64;
        (SPINE_RANGE.0.ln() * (1.0 - t) + SPINE_RANGE.1.ln() * t).exp() * extent
    };
    let cost = |s: f64| spine_circle(tris, s).map_or(f64::INFINITY, |c| c.3);
    let mut best: Option<(f64, f64, usize)> = None;
    for sign in [1.0, -1.0] {
        for k in 0..SPINE_GRID {
            let s = sign * ln(k as f64);
            let c = cost(s);
            if c.is_finite() && best.is_none_or(|b| c < b.0) {
                best = Some((c, sign, k));
            }
        }
    }
    let (_, sign, k) = best?;
    let (mut a, mut b) = (
        ln(k.saturating_sub(1) as f64).ln(),
        ln((k + 1).min(SPINE_GRID - 1) as f64).ln(),
    );
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let f = |x: f64| cost(sign * x.exp());
    let (mut x1, mut x2) = (b - g * (b - a), a + g * (b - a));
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..SPINE_GOLDEN {
        if f1 <= f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - g * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + g * (b - a);
            f2 = f(x2);
        }
    }
    let s = sign * (0.5 * (a + b)).exp();
    let (center, axis, major, _) = spine_circle(tris, s)?;
    let mut sw = 0.0;
    let mut minor = 0.0;
    for &(p, w) in pts {
        let q = sub(p, center);
        let h = dot(q, axis);
        let rho = norm(sub(q, scale(axis, h)));
        sw += w;
        minor += w * ((rho - major).powi(2) + h * h).sqrt();
    }
    let minor = minor / sw.max(f64::MIN_POSITIVE);
    Some((Axis { a: axis, c: center }, vec![0.0, major, minor]))
}

fn bounds(pts: &[(V3, f64)]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &(p, _) in pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn refine(
    pts: &[(V3, f64)],
    kind: Kind,
    axis: Axis,
    shape: Vec<f64>,
    tol: f64,
    quick: bool,
) -> Option<Single> {
    let m = [Member {
        pts: pts.to_vec(),
        kind,
        slot: 0,
    }];
    let (rounds, iters) = if quick {
        (1, QUICK_ITERS)
    } else {
        (IRLS_ROUNDS, LM_ITERS)
    };
    let fixed = vec![false; shape.len()];
    let j = fit_joint_with(&m, axis, shape, &fixed, false, tol, rounds, iters);
    let (rms, max) = j.fit[0];
    (max <= tol && valid_shape(&j.shape, &m[0])).then_some((kind, j.axis, j.shape, rms, max, false))
}

/// A sphere, else a torus, refined from the starts above; a start whose
/// largest vertex residual exceeds `gate` is not refined. `Err` carries the
/// smallest largest-residual of the starts.
pub fn fit_doubly_with(
    pts: &[(V3, f64)],
    tris: &[Tri],
    tol: f64,
    gate: f64,
    quick: bool,
) -> Result<Single, f64> {
    let mut best = f64::INFINITY;
    if pts.len() >= 5
        && let Some((axis, shape)) = init_sphere(pts)
    {
        let r = max_residual(Kind::Sphere, &axis, &shape, pts);
        best = best.min(r);
        if r <= gate
            && let Some(f) = refine(pts, Kind::Sphere, axis, shape, tol, quick)
        {
            return Ok(f);
        }
    }
    if pts.len() >= 8
        && let Some((axis, shape)) = init_torus(pts, tris)
    {
        let r = max_residual(Kind::Torus, &axis, &shape, pts);
        best = best.min(r);
        if r <= gate
            && let Some(f) = refine(pts, Kind::Torus, axis, shape, tol, quick)
            && rings(pts, &f.1, &f.2, tol) >= MIN_RINGS
        {
            return Ok(f);
        }
    }
    Err(best)
}

/// The fewest rings (bands of the tube angle, `2 tol` wide on the tube)
/// that hold every vertex. Any three coaxial circles lie on a torus, so a
/// torus is only evidence of one when its vertices need at least four.
fn rings(pts: &[(V3, f64)], axis: &Axis, shape: &[f64], tol: f64) -> usize {
    let (major, minor) = (shape[1], shape[2]);
    let mut v: Vec<f64> = pts
        .iter()
        .map(|&(p, _)| {
            let q = sub(p, axis.c);
            let h = dot(q, axis.a) - shape[0];
            let rho = norm(sub(q, scale(axis.a, dot(q, axis.a))));
            h.atan2(rho - major)
        })
        .collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return 0;
    }
    let tau = std::f64::consts::TAU;
    let wrap = v[0] + tau - v[n - 1];
    let (cut, _) = (0..n - 1)
        .map(|i| (i + 1, v[i + 1] - v[i]))
        .fold((0, wrap), |b, x| if x.1 > b.1 { x } else { b });
    if cut != 0 {
        v.rotate_left(cut);
        for x in v.iter_mut().skip(n - cut) {
            *x += tau;
        }
    }
    let width = 2.0 * tol / minor;
    let mut count = 0;
    let mut start = f64::NEG_INFINITY;
    for &x in &v {
        if x > start + width {
            count += 1;
            start = x;
        }
    }
    count
}

pub fn fit_doubly(pts: &[(V3, f64)], tris: &[Tri], tol: f64, gate: f64) -> Option<Single> {
    fit_doubly_with(pts, tris, tol, gate, false).ok()
}

/// Exact: the distance to the centre is convex, so its maximum over the
/// triangle is at a corner and its minimum is the point-triangle distance.
pub fn sphere_sagitta(center: V3, radius: f64, t: [V3; 3]) -> f64 {
    let dmin = norm(sub(closest_on_triangle(center, t), center));
    let dmax = t.iter().map(|p| norm(sub(*p, center))).fold(0.0, f64::max);
    (radius - dmin).max(dmax - radius).max(0.0)
}

/// Upper bound on the distance from any point of the triangle to the torus,
/// `|g - minor|` with `g` the distance to the spine circle. With `q` the
/// offset from the centre, `h` its height along the axis and `rho` its
/// distance from it, `g² = (rho - major)² + h²`.
///
/// Largest `g`: for any unit `d` across the axis `rho >= d·q`, so
/// `(rho - major)²` is at most `max((rho - major)₊², (major - d·q)₊²)`, the
/// larger of two squares of non-negative convex functions. With `h²` added the
/// bound is convex and its maximum over the triangle is at a corner.
///
/// Smallest `g`: the nearest spine point of any point of the triangle lies
/// within the triangle's range of azimuths (the triangle's projection across
/// the axis does not contain it), so `g` is at least the distance from the
/// triangle to that arc. The arc is covered by pieces, each inside the
/// triangle of its chord and its end tangents (at most `ENCLOSE_REL` of the
/// chord sagitta scale outside the arc), and the distance between two
/// triangles is exact (zero if they cross, else the least vertex-face or
/// edge-edge distance), so the minimum over the pieces is a lower bound.
///
/// The slack of the first bound grows with the square of the triangle's span
/// about the axis, so a triangle is split into four (exactly, at its edge
/// midpoints) until the bound is within `GMAX_SLACK` of a value the true
/// maximum is known to reach.
///
/// A triangle spanning more than 90° about the axis falls back to the
/// 1-Lipschitz bound of the distance.
pub fn torus_sagitta(center: V3, axis: V3, major: f64, minor: f64, t: [V3; 3]) -> f64 {
    let q: [V3; 3] = t.map(|p| sub(p, center));
    let h: [f64; 3] = q.map(|x| dot(x, axis));
    let perp: [V3; 3] = std::array::from_fn(|i| sub(q[i], scale(axis, h[i])));
    let rho: [f64; 3] = perp.map(norm);
    let g = |i: usize| ((rho[i] - major).powi(2) + h[i] * h[i]).sqrt();
    let corner = (0..3).map(|i| (g(i) - minor).abs()).fold(0.0, f64::max);
    let edge = (0..3)
        .map(|i| norm(sub(t[(i + 1) % 3], t[i])))
        .fold(0.0, f64::max);
    let fallback = corner + edge;
    let mean = add(add(perp[0], perp[1]), perp[2]);
    let len = norm(mean);
    if len <= 0.0 || rho.iter().any(|&r| r <= 0.0) {
        return fallback;
    }
    let d = scale(mean, 1.0 / len);
    if (0..3).any(|i| dot(perp[i], d) < MAX_SPAN_COS * rho[i]) {
        return fallback;
    }
    let e = cross(axis, d);
    let phi: [f64; 3] = perp.map(|p| dot(p, e).atan2(dot(p, d)));
    let lo = phi.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = phi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale_sag = edge * edge / (8.0 * minor);
    let eps = (ENCLOSE_REL * scale_sag).max(1e-12 * (major + minor));
    let step = 2.0 * (major / (major + eps)).min(1.0).acos();
    let pieces = if step > 0.0 {
        (((hi - lo) / step).ceil() as usize).clamp(1, MAX_PIECES)
    } else {
        MAX_PIECES
    };
    let at = |phi: f64, r: f64| {
        add(
            center,
            add(scale(d, r * phi.cos()), scale(e, r * phi.sin())),
        )
    };
    let mut gmin = f64::INFINITY;
    for k in 0..pieces {
        let a = lo + (hi - lo) * k as f64 / pieces as f64;
        let b = lo + (hi - lo) * (k + 1) as f64 / pieces as f64;
        let dist = if b - a <= 1e-15 {
            norm(sub(closest_on_triangle(at(a, major), t), at(a, major)))
        } else {
            let half = 0.5 * (b - a);
            let tip = at(0.5 * (a + b), major / half.cos());
            triangle_distance(t, [at(a, major), at(b, major), tip])
        };
        gmin = gmin.min(dist);
    }
    let reached = (minor - gmin).max(corner);
    let geo = Torus {
        center,
        axis,
        major,
    };
    let gmax = geo.gmax(t, minor + reached * (1.0 + GMAX_SLACK), GMAX_DEPTH);
    let bound = (gmax - minor).max(minor - gmin).max(0.0);
    bound.min(fallback)
}

struct Torus {
    center: V3,
    axis: V3,
    major: f64,
}

impl Torus {
    /// Upper bound on the distance to the spine over the triangle (see
    /// `torus_sagitta`), split until it is at most `enough` or `depth` runs
    /// out.
    fn gmax(&self, t: [V3; 3], enough: f64, depth: usize) -> f64 {
        let q: [V3; 3] = t.map(|p| sub(p, self.center));
        let h: [f64; 3] = q.map(|x| dot(x, self.axis));
        let perp: [V3; 3] = std::array::from_fn(|i| sub(q[i], scale(self.axis, h[i])));
        let mean = add(add(perp[0], perp[1]), perp[2]);
        let len = norm(mean);
        if len <= 0.0 {
            return f64::INFINITY;
        }
        let d = scale(mean, 1.0 / len);
        let g2 = (0..3)
            .map(|i| {
                let out = (norm(perp[i]) - self.major).max(0.0);
                let inn = (self.major - dot(d, perp[i])).max(0.0);
                out.max(inn).powi(2) + h[i] * h[i]
            })
            .fold(0.0, f64::max);
        let g = g2.sqrt();
        if g <= enough || depth == 0 {
            return g;
        }
        let m = [0, 1, 2].map(|i| scale(add(t[i], t[(i + 1) % 3]), 0.5));
        [
            [t[0], m[0], m[2]],
            [m[0], t[1], m[1]],
            [m[2], m[1], t[2]],
            [m[0], m[1], m[2]],
        ]
        .into_iter()
        .map(|s| self.gmax(s, enough, depth - 1))
        .fold(0.0, f64::max)
    }
}

/// Closest point of triangle `t` to `p` (Ericson, Real-Time Collision
/// Detection 5.1.5).
pub fn closest_on_triangle(p: V3, t: [V3; 3]) -> V3 {
    let [a, b, c] = t;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = va + vb + vc;
    if denom == 0.0 {
        return a;
    }
    let v = vb / denom;
    let w = vc / denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

/// Distance between segments `p1q1` and `p2q2` (Ericson 5.1.9).
fn segment_distance(p1: V3, q1: V3, p2: V3, q2: V3) -> f64 {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let (s, t);
    if a <= 0.0 && e <= 0.0 {
        return norm(r);
    }
    if a <= 0.0 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= 0.0 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s0 = if denom > 0.0 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t0 = 0.0;
                s0 = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t0 = 1.0;
                s0 = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s0;
            t = t0;
        }
    }
    norm(sub(add(p1, scale(d1, s)), add(p2, scale(d2, t))))
}

fn segment_crosses(p: V3, q: V3, t: [V3; 3]) -> bool {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let (sp, sq) = (dot(n, sub(p, t[0])), dot(n, sub(q, t[0])));
    if (sp > 0.0 && sq > 0.0) || (sp < 0.0 && sq < 0.0) || sp == sq {
        return false;
    }
    let x = add(p, scale(sub(q, p), sp / (sp - sq)));
    (0..3).all(|i| dot(cross(sub(t[(i + 1) % 3], t[i]), sub(x, t[i])), n) >= 0.0)
}

/// Distance between two triangles: zero when an edge of one crosses the
/// other, else the least vertex-face or edge-edge distance.
pub fn triangle_distance(a: [V3; 3], b: [V3; 3]) -> f64 {
    for i in 0..3 {
        let j = (i + 1) % 3;
        if segment_crosses(a[i], a[j], b) || segment_crosses(b[i], b[j], a) {
            return 0.0;
        }
    }
    let mut best = f64::INFINITY;
    for i in 0..3 {
        best = best.min(norm(sub(closest_on_triangle(a[i], b), a[i])));
        best = best.min(norm(sub(closest_on_triangle(b[i], a), b[i])));
        for j in 0..3 {
            best = best.min(segment_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::super::curved::{fit_single, sagitta, surface_of};
    use super::super::linalg::unit;
    use super::super::surface::Surface;
    use super::super::tests::Frame;
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn dense_max(s: &Surface, t: [V3; 3]) -> f64 {
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
        dense
    }

    fn torus_point(c: V3, axis: V3, big: f64, small: f64, u: f64, v: f64) -> V3 {
        let (e1, e2) = basis_of(axis);
        let radial = add(scale(e1, u.cos()), scale(e2, u.sin()));
        let rho = big + small * v.cos();
        add(c, add(scale(radial, rho), scale(axis, small * v.sin())))
    }

    fn basis_of(a: V3) -> (V3, V3) {
        super::super::curved::basis(a)
    }

    #[test]
    fn sphere_sagitta_bounds_a_dense_max() {
        let mut rnd = rng(0x9E3779B97F4A7C15);
        let (mut lo, mut hi): (f64, f64) = (f64::INFINITY, 0.0);
        for _ in 0..300 {
            let r = 0.5 + 20.0 * rnd();
            let c = [rnd() * 10.0, rnd() * -4.0, rnd()];
            let s = Surface::Sphere {
                center: c,
                radius: r,
                reversed: rnd() < 0.5,
            };
            let spread = (2.0 + 60.0 * rnd()).to_radians();
            let d0 = unit([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]);
            let t: [V3; 3] = std::array::from_fn(|_| {
                let d = unit(add(
                    d0,
                    scale([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5], spread),
                ));
                add(c, scale(d, r * (1.0 + 2e-3 * (rnd() - 0.5))))
            });
            let bound = sagitta(&s, t);
            let dense = dense_max(&s, t);
            assert!(bound >= dense * (1.0 - 1e-9) - 1e-12, "{bound} < {dense}");
            assert!(bound <= dense * (1.0 + 1e-4) + 1e-9, "{bound} > {dense}");
            lo = lo.min(bound / dense);
            hi = hi.max(bound / dense);
        }
        eprintln!("sphere sagitta bound / dense max: min {lo:.6} max {hi:.6}");
    }

    #[test]
    fn torus_sagitta_bounds_a_dense_max() {
        let mut rnd = rng(0x2545F4914F6CDD1D);
        let (mut lo, mut hi): (f64, f64) = (f64::INFINITY, 0.0);
        for i in 0..600 {
            let big = 2.0 + 48.0 * rnd();
            let small = big * (0.02 + 0.9 * rnd());
            let axis = unit([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]);
            let c = [5.0 * rnd(), 3.0 * rnd(), -2.0 * rnd()];
            let s = Surface::Torus {
                center: c,
                axis,
                major: big,
                minor: small,
                reversed: rnd() < 0.5,
            };
            let u0 = std::f64::consts::TAU * rnd();
            let v0 = std::f64::consts::TAU * rnd();
            let du = (0.5 + 25.0 * rnd()).to_radians();
            let dv = (0.5 + 40.0 * rnd()).to_radians();
            let noise = if i % 3 == 0 { 1e-3 * small } else { 0.0 };
            let t: [V3; 3] = std::array::from_fn(|_| {
                let p = torus_point(c, axis, big, small, u0 + du * rnd(), v0 + dv * rnd());
                add(p, scale([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5], noise))
            });
            let bound = sagitta(&s, t);
            let dense = dense_max(&s, t);
            if dense < 1e-9 {
                continue;
            }
            assert!(
                bound >= dense * (1.0 - 1e-9),
                "bound {bound} < dense {dense} (R {big} r {small})"
            );
            assert!(bound <= 1.1 * dense, "bound {bound} > 1.1 x dense {dense}");
            lo = lo.min(bound / dense);
            hi = hi.max(bound / dense);
        }
        eprintln!("torus sagitta bound / dense max: min {lo:.6} max {hi:.6}");
    }

    #[test]
    fn torus_sagitta_is_tight_on_fillet_chords() {
        let (big, small) = (40.0, 2.0);
        let s = Surface::Torus {
            center: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            major: big,
            minor: small,
            reversed: false,
        };
        for (du, dv) in [(0.02, 0.2), (0.05, 0.05), (0.004, 0.4)] {
            for v0 in [0.0, 0.7, 1.4, 2.5, 3.5] {
                let p = |u: f64, v: f64| torus_point([0.0; 3], [0.0, 0.0, 1.0], big, small, u, v);
                let t = [p(0.3, v0), p(0.3 + du, v0), p(0.3 + du, v0 + dv)];
                let bound = sagitta(&s, t);
                let dense = dense_max(&s, t);
                assert!(bound >= dense * (1.0 - 1e-9));
                assert!(
                    bound <= 1.05 * dense + 1e-12,
                    "{du} {dv} {v0}: {bound} vs {dense}"
                );
            }
        }
    }

    fn sample(f: impl Fn(f64, f64) -> V3, us: &[f64], vs: &[f64]) -> (Vec<(V3, f64)>, Vec<Tri>) {
        let mut pts = Vec::new();
        for &u in us {
            for &v in vs {
                pts.push((f(u, v), 1.0));
            }
        }
        let mut tris = Vec::new();
        for a in us.windows(2) {
            for b in vs.windows(2) {
                let (p, q, r, s) = (f(a[0], b[0]), f(a[1], b[0]), f(a[1], b[1]), f(a[0], b[1]));
                for t in [[p, q, r], [p, r, s]] {
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

    fn span(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..=n)
            .map(|k| lo + (hi - lo) * k as f64 / n as f64)
            .collect()
    }

    #[test]
    fn sphere_patches_fit_exactly() {
        let frame = Frame::tilted();
        for (r, v_hi, n) in [(5.0, 1.5, 12), (0.4, 0.3, 4), (80.0, 0.6, 8)] {
            let f = |u: f64, v: f64| {
                frame.map([
                    r * v.cos() * u.cos() + 3.0,
                    r * v.cos() * u.sin(),
                    r * v.sin() - 1.0,
                ])
            };
            let (pts, tris) = sample(f, &span(0.2, 1.4, n), &span(0.1, v_hi, n));
            let (kind, axis, shape, _, max, _) = fit_single(&pts, &tris, 1e-7 * r).unwrap();
            assert_eq!(kind, Kind::Sphere, "r={r}");
            let sf = surface_of(&axis, &shape, &member(kind));
            let Surface::Sphere { center, radius, .. } = sf else {
                panic!()
            };
            assert!((radius - r).abs() < 1e-9 * r, "r={r}: {radius}");
            assert!(norm(sub(center, frame.map([3.0, 0.0, -1.0]))) < 1e-8 * r);
            assert!(max < 1e-9 * r);
        }
    }

    fn member(kind: Kind) -> Member {
        Member {
            pts: Vec::new(),
            kind,
            slot: 0,
        }
    }

    fn torus_case(big: f64, small: f64, us: (f64, f64, usize), vs: (f64, f64, usize), tol: f64) {
        let frame = Frame::tilted();
        let c = [1.0, -2.0, 0.5];
        let f = |u: f64, v: f64| frame.map(torus_point(c, [0.0, 0.0, 1.0], big, small, u, v));
        let (pts, tris) = sample(f, &span(us.0, us.1, us.2), &span(vs.0, vs.1, vs.2));
        let (kind, axis, shape, _, max, _) =
            fit_single(&pts, &tris, tol).unwrap_or_else(|| panic!("R={big} r={small}: no fit"));
        assert_eq!(kind, Kind::Torus, "R={big} r={small}");
        let Surface::Torus {
            center,
            axis: a,
            major,
            minor,
            ..
        } = surface_of(&axis, &shape, &member(kind))
        else {
            panic!()
        };
        assert!((major - big).abs() < 1e-6 * big, "R={big}: {major}");
        assert!((minor - small).abs() < 1e-6 * small, "r={small}: {minor}");
        assert!(dot(a, frame.dir([0.0, 0.0, 1.0])).abs() > 1.0 - 1e-10);
        assert!(norm(sub(center, frame.map(c))) < 1e-6 * big);
        assert!(max <= tol);
    }

    #[test]
    fn torus_patches_fit_exactly() {
        let tau = std::f64::consts::TAU;
        torus_case(20.0, 4.0, (0.0, tau, 48), (0.0, tau, 16), 1e-7);
        torus_case(30.0, 2.0, (0.0, tau, 64), (0.0, 1.57, 6), 1e-7);
        torus_case(30.0, 2.0, (0.0, tau, 64), (3.2, 4.7, 6), 1e-7);
        torus_case(15.0, 5.0, (0.3, 1.2, 10), (-0.5, 0.9, 8), 1e-7);
        torus_case(50.0, 0.5, (0.0, 0.8, 16), (0.0, 1.57, 5), 1e-7);
    }

    #[test]
    fn near_degenerate_tori() {
        let tau = std::f64::consts::TAU;
        torus_case(10.0, 9.0, (0.0, tau, 48), (0.0, tau, 24), 1e-7);
        torus_case(400.0, 1.0, (0.0, 0.3, 24), (0.0, 1.57, 6), 1e-6);
    }

    #[test]
    fn torus_close_to_a_cylinder_prefers_the_cylinder() {
        let tau = std::f64::consts::TAU;
        let (big, small) = (1e6, 3.0);
        let f = |u: f64, v: f64| torus_point([0.0; 3], [0.0, 0.0, 1.0], big, small, u, v);
        let (pts, tris) = sample(f, &span(0.0, 2e-6, 4), &span(0.0, tau, 24));
        let (kind, ..) = fit_single(&pts, &tris, 1e-6).unwrap();
        assert_eq!(kind, Kind::Cylinder);
    }

    #[test]
    fn three_rings_of_a_torus_are_not_evidence_of_one() {
        let (big, small) = (20.0, 3.0);
        let f = |u: f64, v: f64| torus_point([0.0; 3], [0.0, 0.0, 1.0], big, small, u, v);
        let tau = std::f64::consts::TAU;
        let (pts, tris) = sample(f, &span(0.0, tau, 48), &[0.2, 0.9, 1.5]);
        assert!(fit_doubly(&pts, &tris, 1e-7, f64::INFINITY).is_none());
        let (pts, tris) = sample(f, &span(0.0, tau, 48), &[0.2, 0.6, 0.9, 1.5]);
        let (kind, ..) = fit_doubly(&pts, &tris, 1e-7, f64::INFINITY).unwrap();
        assert_eq!(kind, Kind::Torus);
    }

    #[test]
    fn spindle_fit_is_refused() {
        let f = |u: f64, v: f64| torus_point([0.0; 3], [0.0, 0.0, 1.0], 2.0, 3.0, u, v);
        let (pts, tris) = sample(f, &span(0.0, 6.0, 40), &span(-1.0, 1.0, 8));
        assert!(fit_doubly(&pts, &tris, 1e-7, f64::INFINITY).is_none());
    }
}
