use super::linalg::{V3, add, cross, dot, norm, scale, sub, unit};

/// Analytic parameters live in the centred frame: `distance`, `linearize`
/// and `normal_at` take centred positions (`Welded::vc`), and `to_world`
/// converts to world coordinates for IR output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    Plane {
        normal: V3,
        offset: f64,
    },
    Cylinder {
        origin: V3,
        axis: V3,
        radius: f64,
        reversed: bool,
    },
    Cone {
        apex: V3,
        axis: V3,
        half_angle: f64,
        reversed: bool,
    },
    Sphere {
        center: V3,
        radius: f64,
        reversed: bool,
    },
    Torus {
        center: V3,
        axis: V3,
        major: f64,
        minor: f64,
        reversed: bool,
    },
    Facets,
}

pub fn radial(p: V3, origin: V3, axis: V3) -> (f64, f64, Option<V3>) {
    let q = sub(p, origin);
    let h = dot(q, axis);
    let r = sub(q, scale(axis, h));
    let rho = norm(r);
    let dir = if rho > 0.0 {
        Some(scale(r, 1.0 / rho))
    } else {
        None
    };
    (h, rho, dir)
}

pub fn spine_point(p: V3, center: V3, axis: V3, major: f64) -> V3 {
    let (_, _, dir) = radial(p, center, axis);
    let dir = dir.unwrap_or_else(|| any_perp(axis));
    add(center, scale(dir, major))
}

fn any_perp(a: V3) -> V3 {
    let e = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    unit(cross(a, e))
}

impl Surface {
    pub fn is_analytic(&self) -> bool {
        !matches!(self, Surface::Facets)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Surface::Plane { .. } => "plane",
            Surface::Cylinder { .. } => "cylinder",
            Surface::Cone { .. } => "cone",
            Surface::Sphere { .. } => "sphere",
            Surface::Torus { .. } => "torus",
            Surface::Facets => "facets",
        }
    }

    pub fn distance(&self, p: V3) -> f64 {
        match *self {
            Surface::Plane { normal, offset } => dot(normal, p) - offset,
            Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed,
            } => {
                let (_, rho, _) = radial(p, origin, axis);
                let d = rho - radius;
                if reversed { -d } else { d }
            }
            Surface::Cone { reversed, .. } => {
                let q = self.closest_point(p);
                let d = norm(sub(p, q));
                let n = self.natural_normal(q).unwrap_or([0.0; 3]);
                let s = if dot(sub(p, q), n) >= 0.0 { d } else { -d };
                if reversed { -s } else { s }
            }
            Surface::Sphere {
                center,
                radius,
                reversed,
            } => {
                let d = norm(sub(p, center)) - radius;
                if reversed { -d } else { d }
            }
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                reversed,
            } => {
                let d = norm(sub(p, spine_point(p, center, axis, major))) - minor;
                if reversed { -d } else { d }
            }
            Surface::Facets => 0.0,
        }
    }

    pub fn closest_point(&self, p: V3) -> V3 {
        match *self {
            Surface::Plane { normal, offset } => sub(p, scale(normal, dot(normal, p) - offset)),
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => {
                let (h, _, dir) = radial(p, origin, axis);
                let dir = dir.unwrap_or_else(|| any_perp(axis));
                add(add(origin, scale(axis, h)), scale(dir, radius))
            }
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                let (h, rho, dir) = radial(p, apex, axis);
                let dir = dir.unwrap_or_else(|| any_perp(axis));
                let (s, c) = half_angle.sin_cos();
                let g = add(scale(axis, c), scale(dir, s));
                let t = (h * c + rho * s).max(0.0);
                add(apex, scale(g, t))
            }
            Surface::Sphere { center, radius, .. } => add(center, scale(away(p, center), radius)),
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => {
                let s = spine_point(p, center, axis, major);
                let dir = match unit_or_none(sub(p, s)) {
                    Some(d) => d,
                    None => unit(sub(s, center)),
                };
                add(s, scale(dir, minor))
            }
            Surface::Facets => p,
        }
    }

    fn natural_normal(&self, q: V3) -> Option<V3> {
        match *self {
            Surface::Plane { normal, .. } => Some(normal),
            Surface::Cylinder { origin, axis, .. } => radial(q, origin, axis).2,
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                let dir = radial(q, apex, axis).2?;
                let (s, c) = half_angle.sin_cos();
                Some(sub(scale(dir, c), scale(axis, s)))
            }
            Surface::Sphere { center, .. } => unit_or_none(sub(q, center)),
            Surface::Torus {
                center,
                axis,
                major,
                ..
            } => unit_or_none(sub(q, spine_point(q, center, axis, major))),
            Surface::Facets => None,
        }
    }

    pub fn linearize(&self, p: V3, weight: f64) -> Option<Constraint> {
        match *self {
            Surface::Plane { normal, offset } => Some(Constraint {
                normal,
                offset,
                weight,
            }),
            Surface::Cylinder { .. }
            | Surface::Cone { .. }
            | Surface::Sphere { .. }
            | Surface::Torus { .. } => {
                let q = self.closest_point(p);
                let normal = self.natural_normal(q)?;
                Some(Constraint {
                    normal,
                    offset: dot(normal, q),
                    weight,
                })
            }
            Surface::Facets => None,
        }
    }

    /// Outward normal at the surface point closest to `p`.
    pub fn normal_at(&self, p: V3) -> Option<V3> {
        match *self {
            Surface::Plane { normal, .. } => Some(normal),
            Surface::Cylinder { reversed, .. }
            | Surface::Cone { reversed, .. }
            | Surface::Sphere { reversed, .. }
            | Surface::Torus { reversed, .. } => {
                let n = self.natural_normal(self.closest_point(p))?;
                Some(if reversed { scale(n, -1.0) } else { n })
            }
            Surface::Facets => None,
        }
    }

    pub fn as_plane(&self) -> Option<(V3, f64)> {
        match *self {
            Surface::Plane { normal, offset } => Some((normal, offset)),
            _ => None,
        }
    }

    pub fn to_world(self, center: V3) -> Surface {
        match self {
            Surface::Plane { normal, offset } => Surface::Plane {
                normal,
                offset: offset + dot(normal, center),
            },
            Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed,
            } => Surface::Cylinder {
                origin: add(origin, center),
                axis,
                radius,
                reversed,
            },
            Surface::Cone {
                apex,
                axis,
                half_angle,
                reversed,
            } => Surface::Cone {
                apex: add(apex, center),
                axis,
                half_angle,
                reversed,
            },
            Surface::Sphere {
                center: c,
                radius,
                reversed,
            } => Surface::Sphere {
                center: add(c, center),
                radius,
                reversed,
            },
            Surface::Torus {
                center: c,
                axis,
                major,
                minor,
                reversed,
            } => Surface::Torus {
                center: add(c, center),
                axis,
                major,
                minor,
                reversed,
            },
            Surface::Facets => Surface::Facets,
        }
    }

    pub fn with_reversed(self, flag: bool) -> Surface {
        match self {
            Surface::Cylinder {
                origin,
                axis,
                radius,
                ..
            } => Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed: flag,
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
                reversed: flag,
            },
            Surface::Sphere { center, radius, .. } => Surface::Sphere {
                center,
                radius,
                reversed: flag,
            },
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                ..
            } => Surface::Torus {
                center,
                axis,
                major,
                minor,
                reversed: flag,
            },
            s => s,
        }
    }
}

fn unit_or_none(v: V3) -> Option<V3> {
    let n = norm(v);
    (n > 0.0).then(|| scale(v, 1.0 / n))
}

fn away(p: V3, center: V3) -> V3 {
    unit_or_none(sub(p, center)).unwrap_or([0.0, 0.0, 1.0])
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraint {
    pub normal: V3,
    pub offset: f64,
    pub weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylinder_normal_is_evaluated_at_the_closest_point() {
        let s = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 2.0,
            reversed: true,
        };
        let n = s.normal_at([0.5, 0.0, 3.0]).unwrap();
        assert_eq!(n, [-1.0, 0.0, 0.0]);
        assert!((s.distance([3.0, 0.0, 0.0]) + 1.0).abs() < 1e-15);
        let q = s.closest_point([0.0, 5.0, 1.0]);
        assert!(norm(sub(q, [0.0, 2.0, 1.0])) < 1e-15);
    }

    #[test]
    fn cone_distance_and_normal() {
        let a = 30f64.to_radians();
        let s = Surface::Cone {
            apex: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            half_angle: a,
            reversed: false,
        };
        let on = [a.tan() * 4.0, 0.0, 4.0];
        assert!(s.distance(on).abs() < 1e-12);
        let n = s.normal_at(on).unwrap();
        assert!(norm(sub(n, [a.cos(), 0.0, -a.sin()])) < 1e-12);
        let off = add(on, scale(n, 0.3));
        assert!((s.distance(off) - 0.3).abs() < 1e-12);
        assert!(norm(sub(s.normal_at(off).unwrap(), n)) < 1e-12);
        let c = s.linearize(off, 1.0).unwrap();
        assert!((dot(c.normal, on) - c.offset).abs() < 1e-12);
    }
}
