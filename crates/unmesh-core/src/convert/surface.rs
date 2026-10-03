use super::linalg::{V3, dot};

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

    pub fn outward_normal(&self, facet_normal: V3) -> V3 {
        match *self {
            Surface::Plane { normal, .. } => normal,
            Surface::Facets => facet_normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constraint {
    pub normal: V3,
    pub offset: f64,
    pub weight: f64,
}
