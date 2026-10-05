use super::doubly::closest_on_triangle;
use super::linalg::{M3, V3, add, angle_deg, dot, norm, outer_add, scale, sub, sym_eigen};
use super::surface::Surface;

const EIGEN_MIN: f64 = 1e-6;
const ITERATIONS: usize = 60;
const STEP_STOP: f64 = 1e-14;
const CROSSING_STEPS: usize = 40;

/// The least-squares point of `surfaces` reached by Gauss-Newton from
/// `start`. Directions the surfaces do not determine (eigenvalues of the
/// normal matrix below `EIGEN_MIN`) are pulled to `anchor`, so the point on an
/// intersection curve nearest the anchor is chosen and a near-singular corner
/// does not run away.
pub fn lsq_point(anchor: V3, start: V3, surfaces: &[&Surface]) -> V3 {
    let mut x = start;
    for _ in 0..ITERATIONS {
        let mut m: M3 = [[0.0; 3]; 3];
        let mut r = [0.0; 3];
        let mut any = false;
        for s in surfaces {
            if let Some(c) = s.linearize(x, 1.0) {
                outer_add(&mut m, c.normal, c.normal, 1.0);
                r = add(r, scale(c.normal, c.offset - dot(c.normal, x)));
                any = true;
            }
        }
        if !any {
            return x;
        }
        let (vals, vecs) = sym_eigen(m);
        let to_anchor = sub(anchor, x);
        let mut step = [0.0; 3];
        for j in 0..3 {
            let d = if vals[j] >= EIGEN_MIN {
                dot(vecs[j], r) / vals[j]
            } else {
                dot(vecs[j], to_anchor)
            };
            step = add(step, scale(vecs[j], d));
        }
        x = add(x, step);
        if norm(step) <= STEP_STOP * (1.0 + norm(x)) {
            break;
        }
    }
    x
}

pub fn off_surfaces(x: V3, surfaces: &[&Surface]) -> f64 {
    surfaces
        .iter()
        .map(|s| s.distance(x).abs())
        .fold(0.0, f64::max)
}

/// Distance from `x` to the nearest of `tris`; an upper bound on its distance
/// to the whole mesh.
pub fn mesh_distance(x: V3, tris: impl Iterator<Item = [V3; 3]>) -> f64 {
    tris.map(|t| norm(sub(x, closest_on_triangle(x, t))))
        .fold(f64::INFINITY, f64::min)
}

pub fn dihedral(a: &Surface, b: &Surface, p: V3) -> Option<f64> {
    Some(angle_deg(a.normal_at(p)?, b.normal_at(p)?))
}

/// The point between `p` and `q` (both near the intersection of `a` and
/// `b`) where the dihedral of the two surfaces crosses `threshold`, on both
/// surfaces. `None` when the dihedral at the two ends does not bracket it.
pub fn onto_crossing(a: &Surface, b: &Surface, p: V3, q: V3, threshold: f64) -> Option<V3> {
    let on = |t: f64| {
        let y = add(p, scale(sub(q, p), t));
        lsq_point(y, y, &[a, b])
    };
    let f = |t: f64| dihedral(a, b, on(t)).map(|d| d - threshold);
    let (mut lo, mut hi) = (0.0, 1.0);
    let (flo, fhi) = (f(lo)?, f(hi)?);
    if flo == 0.0 {
        return Some(on(lo));
    }
    if fhi == 0.0 {
        return Some(on(hi));
    }
    if (flo > 0.0) == (fhi > 0.0) {
        return None;
    }
    for _ in 0..CROSSING_STEPS {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid)?;
        if (fm > 0.0) == (flo > 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(on(0.5 * (lo + hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::linalg::unit;

    fn plane(n: V3, offset: f64) -> Surface {
        Surface::Plane {
            normal: unit(n),
            offset,
        }
    }

    #[test]
    fn plane_plane_cylinder_corner_is_exact() {
        let cyl = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 5.0,
            reversed: true,
        };
        let top = plane([0.0, 0.0, 1.0], 10.0);
        let side = plane([0.0, 1.0, 0.0], 3.0);
        let truth = [4.0, 3.0, 10.0];
        let x0 = add(truth, [0.004, -0.003, 0.002]);
        let x = lsq_point(x0, x0, &[&cyl, &top, &side]);
        assert!(norm(sub(x, truth)) < 1e-12, "{x:?}");
    }

    #[test]
    fn plane_cone_stays_on_the_circle_nearest_the_mesh_vertex() {
        let a = 30f64.to_radians();
        let cone = Surface::Cone {
            apex: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            half_angle: a,
            reversed: false,
        };
        let cap = plane([0.0, 0.0, 1.0], 6.0);
        let r = 6.0 * a.tan();
        let x0 = [r * 0.6 + 0.01, r * 0.8 - 0.02, 6.003];
        let x = lsq_point(x0, x0, &[&cone, &cap]);
        assert!(cone.distance(x).abs() < 1e-12 && cap.distance(x).abs() < 1e-12);
        let want = [r * 0.6, r * 0.8, 6.0];
        let radial = unit([x0[0], x0[1], 0.0]);
        assert!(norm(sub(unit([x[0], x[1], 0.0]), radial)) < 1e-9);
        assert!(norm(sub(x, want)) < 0.03);
    }

    #[test]
    fn three_cylinders_meet_at_their_common_point() {
        let cyl = |axis: V3| Surface::Cylinder {
            origin: [0.0; 3],
            axis,
            radius: 4.0,
            reversed: true,
        };
        let (x, y, z) = (
            cyl([1.0, 0.0, 0.0]),
            cyl([0.0, 1.0, 0.0]),
            cyl([0.0, 0.0, 1.0]),
        );
        let c = 4.0 / 2f64.sqrt();
        let truth = [c, c, c];
        let x0 = add(truth, [0.01, -0.02, 0.015]);
        let p = lsq_point(x0, x0, &[&x, &y, &z]);
        assert!(norm(sub(p, truth)) < 1e-10, "{p:?}");
    }

    #[test]
    fn near_parallel_planes_reach_their_exact_intersection() {
        let tilt = 1.75f64.to_radians();
        let a = plane([0.0, 0.0, 1.0], 0.0);
        let b = plane([tilt.sin(), 0.0, tilt.cos()], 2e-4);
        let c = plane([0.0, 1.0, 0.0], 0.0);
        let x0 = [0.0, 0.001, 0.0];
        let x = lsq_point(x0, x0, &[&a, &b, &c]);
        for s in [&a, &b, &c] {
            assert!(s.distance(x).abs() < 1e-12);
        }
        assert!((x[0] - 2e-4 / tilt.sin()).abs() < 1e-9);
    }

    #[test]
    fn crossing_lands_where_the_dihedral_meets_the_threshold() {
        let cyl = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 10.0,
            reversed: false,
        };
        let beta = 1f64.to_radians();
        let cut = plane([beta.cos(), 0.0, beta.sin()], 10.0);
        let at = |deg: f64| {
            let x = 10.0 * deg.to_radians().cos() / beta.cos();
            [
                x,
                (100.0 - x * x).sqrt(),
                (10.0 - x * beta.cos()) / beta.sin(),
            ]
        };
        let (p, q) = (at(2.0), at(5.0));
        assert!((dihedral(&cyl, &cut, p).unwrap() - 2.0).abs() < 1e-6);
        let x = onto_crossing(&cyl, &cut, p, q, 3.0).unwrap();
        assert!(cyl.distance(x).abs() < 1e-10 && cut.distance(x).abs() < 1e-10);
        assert!((dihedral(&cyl, &cut, x).unwrap() - 3.0).abs() < 1e-6);
        assert!(onto_crossing(&cyl, &cut, p, at(2.5), 3.0).is_none());
    }
}
