use crate::ir::{Point, Surface};

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: Point, s: f64) -> Point {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

fn unit(a: Point) -> Point {
    let n = norm(a);
    if n == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        scale(a, 1.0 / n)
    }
}

fn any_perpendicular(axis: Point) -> Point {
    let helper = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let c = [
        axis[1] * helper[2] - axis[2] * helper[1],
        axis[2] * helper[0] - axis[0] * helper[2],
        axis[0] * helper[1] - axis[1] * helper[0],
    ];
    unit(c)
}

struct AxisFrame {
    h: f64,
    rho: f64,
    radial: Point,
}

fn axis_frame(p: Point, origin: Point, axis: Point) -> AxisFrame {
    let v = sub(p, origin);
    let h = dot(v, axis);
    let r = sub(v, scale(axis, h));
    let rho = norm(r);
    let radial = if rho > 1e-300 {
        scale(r, 1.0 / rho)
    } else {
        any_perpendicular(axis)
    };
    AxisFrame { h, rho, radial }
}

fn onto_sphere(p: Point, center: Point, radius: f64) -> Point {
    let v = sub(p, center);
    let n = norm(v);
    let dir = if n > 1e-300 {
        scale(v, 1.0 / n)
    } else {
        [0.0, 0.0, 1.0]
    };
    add(center, scale(dir, radius))
}

pub fn project_onto_surface(surface: &Surface, p: Point) -> Point {
    match surface {
        Surface::Plane { origin, normal } => {
            let n = unit(*normal);
            sub(p, scale(n, dot(sub(p, *origin), n)))
        }
        Surface::Sphere { center, radius, .. } => onto_sphere(p, *center, *radius),
        Surface::Cylinder {
            origin,
            axis,
            radius,
            ..
        } => {
            let a = unit(*axis);
            let f = axis_frame(p, *origin, a);
            add(add(*origin, scale(a, f.h)), scale(f.radial, *radius))
        }
        Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } => {
            let a = unit(*axis);
            let f = axis_frame(p, *apex, a);
            let (s, c) = half_angle.sin_cos();
            let t = (f.h * c + f.rho * s).max(0.0);
            add(add(*apex, scale(a, t * c)), scale(f.radial, t * s))
        }
        Surface::Torus {
            center,
            axis,
            major_radius,
            minor_radius,
            ..
        } => {
            let a = unit(*axis);
            let f = axis_frame(p, *center, a);
            let tube = add(*center, scale(f.radial, *major_radius));
            onto_sphere(p, tube, *minor_radius)
        }
        Surface::Facets { .. } => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Orientation;

    fn close(a: Point, b: Point) -> bool {
        norm(sub(a, b)) < 1e-12
    }

    #[test]
    fn plane_projection_drops_the_normal_component() {
        let s = Surface::Plane {
            origin: [0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 2.0],
        };
        assert!(close(
            project_onto_surface(&s, [3.0, 4.0, 9.0]),
            [3.0, 4.0, 1.0]
        ));
    }

    #[test]
    fn cylinder_projection_lands_on_radius() {
        let s = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 2.0,
            orientation: Orientation::Same,
        };
        assert!(close(
            project_onto_surface(&s, [5.0, 0.0, 7.0]),
            [2.0, 0.0, 7.0]
        ));
        let on_axis = project_onto_surface(&s, [0.0, 0.0, 3.0]);
        assert!((norm([on_axis[0], on_axis[1], 0.0]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cone_projection_lands_on_a_generator() {
        let half_angle = 0.4_f64;
        let s = Surface::Cone {
            apex: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            half_angle,
            orientation: Orientation::Same,
        };
        let q = project_onto_surface(&s, [3.0, 0.0, 5.0]);
        let rho = (q[0] * q[0] + q[1] * q[1]).sqrt();
        assert!((rho - q[2] * half_angle.tan()).abs() < 1e-12);
        assert!(close(project_onto_surface(&s, q), q));
    }

    #[test]
    fn sphere_and_torus_projection_are_idempotent() {
        let sphere = Surface::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 2.5,
            orientation: Orientation::Same,
        };
        let q = project_onto_surface(&sphere, [4.0, 4.0, 4.0]);
        assert!((norm(sub(q, [1.0, 2.0, 3.0])) - 2.5).abs() < 1e-12);
        let torus = Surface::Torus {
            center: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            major_radius: 5.0,
            minor_radius: 1.0,
            orientation: Orientation::Same,
        };
        let q = project_onto_surface(&torus, [7.0, 0.0, 0.5]);
        assert!(
            close(q, [6.0 - 0.0, 0.0, 0.0]) || (norm(sub(q, [5.0, 0.0, 0.0])) - 1.0).abs() < 1e-12
        );
        assert!(close(project_onto_surface(&torus, q), q));
    }
}
