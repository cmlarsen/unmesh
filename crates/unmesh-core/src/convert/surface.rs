use super::linalg::V3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surface {
    Plane { normal: V3, offset: f64 },
    Facets,
}

impl Surface {
    pub fn is_analytic(&self) -> bool {
        matches!(self, Surface::Plane { .. })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Surface::Plane { .. } => "plane",
            Surface::Facets => "facets",
        }
    }

    pub fn constraint(&self, weight: f64) -> Option<Constraint> {
        match *self {
            Surface::Plane { normal, offset } => Some(Constraint {
                normal,
                offset,
                weight,
            }),
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
