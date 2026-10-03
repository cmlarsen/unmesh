use super::linalg::{V3, dot};

/// Analytic parameters live in the centred frame: `distance`, `linearize`
/// and `normal_at` take centred positions (`Welded::vc`), and `to_world`
/// converts to world coordinates for IR output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    Plane { normal: V3, offset: f64 },
    Facets,
}

impl Surface {
    pub fn is_analytic(&self) -> bool {
        !matches!(self, Surface::Facets)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Surface::Plane { .. } => "plane",
            Surface::Facets => "facets",
        }
    }

    pub fn distance(&self, p: V3) -> f64 {
        match *self {
            Surface::Plane { normal, offset } => dot(normal, p) - offset,
            Surface::Facets => 0.0,
        }
    }

    pub fn linearize(&self, _p: V3, weight: f64) -> Option<Constraint> {
        match *self {
            Surface::Plane { normal, offset } => Some(Constraint {
                normal,
                offset,
                weight,
            }),
            Surface::Facets => None,
        }
    }

    pub fn normal_at(&self, _p: V3) -> Option<V3> {
        match *self {
            Surface::Plane { normal, .. } => Some(normal),
            Surface::Facets => None,
        }
    }

    pub fn to_world(&self, center: V3) -> Surface {
        match *self {
            Surface::Plane { normal, offset } => Surface::Plane {
                normal,
                offset: offset + dot(normal, center),
            },
            Surface::Facets => Surface::Facets,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraint {
    pub normal: V3,
    pub offset: f64,
    pub weight: f64,
}
