use crate::ir::{Ir, Tolerances};
use crate::mesh::IndexedMesh;

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOptions {
    pub linear_tolerance: Option<f64>,
    pub angular_snap_deg: f64,
    pub tangent_threshold_deg: f64,
    pub vertex_merge: f64,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        let t = Tolerances::default();
        Self {
            linear_tolerance: None,
            angular_snap_deg: t.angular_snap_deg,
            tangent_threshold_deg: t.tangent_threshold_deg,
            vertex_merge: t.vertex_merge,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Warning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Report {
    pub max_deviation: f64,
    pub rms_deviation: f64,
    pub analytic_area_fraction: f64,
    pub region_counts: std::collections::BTreeMap<String, u32>,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOutput {
    pub ir: Ir,
    pub report: Report,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    NotImplemented,
    EmptyMesh,
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::NotImplemented => write!(f, "convert is not implemented yet"),
            ConvertError::EmptyMesh => write!(f, "mesh has no triangles"),
        }
    }
}

impl std::error::Error for ConvertError {}

pub fn convert(
    mesh: &IndexedMesh,
    _options: &ConvertOptions,
) -> Result<ConvertOutput, ConvertError> {
    if mesh.faces.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    Err(ConvertError::NotImplemented)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_errors() {
        assert_eq!(
            convert(&IndexedMesh::default(), &ConvertOptions::default()).unwrap_err(),
            ConvertError::EmptyMesh
        );
    }
}
