use std::fmt;

use rustc_hash::FxHashMap;

use crate::mesh::{IndexedMesh, Point, TriangleSoup};

#[derive(Debug, Clone, PartialEq)]
pub struct WeldReport {
    pub input_corners: usize,
    pub unique_vertices: usize,
    pub degenerate_dropped: usize,
    pub max_merge: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WeldError {
    NonFiniteCoordinate,
    InvalidTolerance,
}

impl fmt::Display for WeldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeldError::NonFiniteCoordinate => {
                write!(f, "triangle soup has a non-finite coordinate")
            }
            WeldError::InvalidTolerance => write!(f, "tolerance must be finite and >= 0"),
        }
    }
}

impl std::error::Error for WeldError {}

pub fn weld(
    soup: &TriangleSoup,
    tolerance: f64,
) -> Result<(IndexedMesh, Vec<u32>, WeldReport), WeldError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(WeldError::InvalidTolerance);
    }
    if soup
        .triangles
        .iter()
        .flatten()
        .flatten()
        .any(|c| !c.is_finite())
    {
        return Err(WeldError::NonFiniteCoordinate);
    }
    let mut vertices: Vec<Point> = Vec::new();
    let mut faces = Vec::with_capacity(soup.len());
    let mut source = Vec::with_capacity(soup.len());
    let mut degenerate_dropped = 0;
    let mut index = SpatialIndex::new(tolerance);

    for (i, tri) in soup.triangles.iter().enumerate() {
        let ids = tri.map(|p| index.find_or_insert(p, &mut vertices));
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
            degenerate_dropped += 1;
            continue;
        }
        faces.push(ids);
        source.push(i as u32);
    }

    let report = WeldReport {
        input_corners: soup.len() * 3,
        unique_vertices: vertices.len(),
        degenerate_dropped,
        max_merge: index.max_merge,
    };
    Ok((IndexedMesh { vertices, faces }, source, report))
}

struct SpatialIndex {
    tolerance: f64,
    cells: FxHashMap<[i64; 3], Vec<u32>>,
    exact: FxHashMap<[u64; 3], u32>,
    max_merge: f64,
}

impl SpatialIndex {
    fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            cells: FxHashMap::default(),
            exact: FxHashMap::default(),
            max_merge: 0.0,
        }
    }

    fn find_or_insert(&mut self, p: Point, vertices: &mut Vec<Point>) -> u32 {
        if self.tolerance <= 0.0 {
            let key = p.map(|c| (c + 0.0).to_bits());
            return *self.exact.entry(key).or_insert_with(|| {
                vertices.push(p);
                (vertices.len() - 1) as u32
            });
        }
        let cell_size = 2.0 * self.tolerance;
        let tol2 = self.tolerance * self.tolerance;
        let mut cell = [0i64; 3];
        let mut reach = [[0i64; 2]; 3];
        for axis in 0..3 {
            let scaled = p[axis] / cell_size;
            let base = scaled.floor();
            cell[axis] = base as i64;
            let step = if scaled - base < 0.5 { -1 } else { 1 };
            reach[axis] = [0, step];
        }
        for dx in reach[0] {
            for dy in reach[1] {
                for dz in reach[2] {
                    let key = [
                        cell[0].saturating_add(dx),
                        cell[1].saturating_add(dy),
                        cell[2].saturating_add(dz),
                    ];
                    if let Some(ids) = self.cells.get(&key) {
                        for &id in ids {
                            let d2 = dist2(vertices[id as usize], p);
                            if d2 <= tol2 {
                                let d = d2.sqrt();
                                if d > self.max_merge {
                                    self.max_merge = d;
                                }
                                return id;
                            }
                        }
                    }
                }
            }
        }
        let id = vertices.len() as u32;
        vertices.push(p);
        self.cells.entry(cell).or_default().push(id);
        id
    }
}

fn dist2(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(offset: f64) -> TriangleSoup {
        TriangleSoup {
            triangles: vec![
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0 + offset, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
        }
    }

    #[test]
    fn exact_weld_shares_identical_corners() {
        let (mesh, source, report) = weld(&quad(0.0), 0.0).unwrap();
        assert_eq!(source, [0, 1]);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(report.input_corners, 6);
        assert_eq!(mesh.faces.len(), 2);
    }

    #[test]
    fn exact_weld_keeps_a_gap_open() {
        let (mesh, _, _) = weld(&quad(1e-9), 0.0).unwrap();
        assert_eq!(mesh.vertices.len(), 5);
    }

    #[test]
    fn tolerant_weld_closes_a_gap_across_cell_boundaries() {
        let (mesh, _, _) = weld(&quad(1e-9), 1e-6).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
    }

    #[test]
    fn merge_distance_is_tracked() {
        let (_, _, exact) = weld(&quad(0.0), 1e-6).unwrap();
        assert_eq!(exact.max_merge, 0.0);
        let (_, _, report) = weld(&quad(1e-9), 1e-6).unwrap();
        assert!((report.max_merge - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn collapsed_triangle_is_dropped() {
        let soup = TriangleSoup {
            triangles: vec![[[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        };
        let (mesh, source, report) = weld(&soup, 1e-6).unwrap();
        assert!(mesh.faces.is_empty());
        assert!(source.is_empty());
        assert_eq!(report.degenerate_dropped, 1);
    }

    #[test]
    fn non_finite_and_huge_coordinates() {
        for x in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY] {
            let soup = TriangleSoup {
                triangles: vec![[[x, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            };
            assert_eq!(weld(&soup, 1e-6), Err(WeldError::NonFiniteCoordinate));
        }
        let soup = TriangleSoup {
            triangles: vec![[[1e20, 0.0, 0.0], [-1e20, 0.0, 0.0], [0.0, 1e20, 0.0]]],
        };
        assert!(weld(&soup, 1e-6).is_ok());
    }

    #[test]
    fn source_map_skips_dropped_triangles() {
        let soup = TriangleSoup {
            triangles: vec![
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
        };
        let (mesh, source, _) = weld(&soup, 1e-6).unwrap();
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(source, [0, 2]);
    }
}
