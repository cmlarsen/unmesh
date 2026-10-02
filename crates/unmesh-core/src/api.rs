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
pub struct ConvertWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Report {
    pub max_deviation: f64,
    pub rms_deviation: f64,
    pub analytic_area_fraction: f64,
    pub region_counts: std::collections::BTreeMap<String, u32>,
    pub warnings: Vec<ConvertWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvertOutput {
    pub ir: Ir,
    pub report: Report,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    EmptyMesh,
    InvalidInput(String),
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::EmptyMesh => write!(f, "mesh has no triangles"),
            ConvertError::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for ConvertError {}

pub fn convert(
    mesh: &IndexedMesh,
    options: &ConvertOptions,
) -> Result<ConvertOutput, ConvertError> {
    if mesh.faces.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    let n = mesh.vertices.len();
    if mesh.faces.iter().flatten().any(|&i| i as usize >= n) {
        return Err(ConvertError::InvalidInput(
            "face index out of range".to_string(),
        ));
    }
    convert_soup(&mesh.to_soup(), options)
}

pub use crate::convert::convert_soup;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_errors() {
        assert_eq!(
            convert(&IndexedMesh::default(), &ConvertOptions::default()).unwrap_err(),
            ConvertError::EmptyMesh
        );
    }

    #[test]
    fn bad_index_errors() {
        let mesh = IndexedMesh {
            vertices: vec![[0.0; 3]],
            faces: vec![[0, 1, 2]],
        };
        assert!(matches!(
            convert(&mesh, &ConvertOptions::default()),
            Err(ConvertError::InvalidInput(_))
        ));
    }
}
