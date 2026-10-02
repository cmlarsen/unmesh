use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::mesh::{IndexedMesh, Point, TriangleSoup};

#[derive(Debug, Clone, PartialEq)]
pub struct WeldReport {
    pub input_corners: usize,
    pub unique_vertices: usize,
    pub degenerate_dropped: usize,
}

pub fn weld(soup: &TriangleSoup, tolerance: f64) -> (IndexedMesh, WeldReport) {
    let mut vertices: Vec<Point> = Vec::new();
    let mut faces = Vec::with_capacity(soup.len());
    let mut degenerate_dropped = 0;
    let mut index = SpatialIndex::new(tolerance);

    for tri in &soup.triangles {
        let ids = tri.map(|p| index.find_or_insert(p, &mut vertices));
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
            degenerate_dropped += 1;
            continue;
        }
        faces.push(ids);
    }

    let report = WeldReport {
        input_corners: soup.len() * 3,
        unique_vertices: vertices.len(),
        degenerate_dropped,
    };
    (IndexedMesh { vertices, faces }, report)
}

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64);
    }

    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

struct SpatialIndex {
    tolerance: f64,
    cells: FastMap<[i64; 3], Vec<u32>>,
    exact: FastMap<[u64; 3], u32>,
}

impl SpatialIndex {
    fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            cells: FastMap::default(),
            exact: FastMap::default(),
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
                    let key = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    if let Some(ids) = self.cells.get(&key) {
                        for &id in ids {
                            if dist2(vertices[id as usize], p) <= tol2 {
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
        let (mesh, report) = weld(&quad(0.0), 0.0);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(report.input_corners, 6);
        assert_eq!(mesh.faces.len(), 2);
    }

    #[test]
    fn exact_weld_keeps_a_gap_open() {
        let (mesh, _) = weld(&quad(1e-9), 0.0);
        assert_eq!(mesh.vertices.len(), 5);
    }

    #[test]
    fn tolerant_weld_closes_a_gap_across_cell_boundaries() {
        let (mesh, _) = weld(&quad(1e-9), 1e-6);
        assert_eq!(mesh.vertices.len(), 4);
    }

    #[test]
    fn collapsed_triangle_is_dropped() {
        let soup = TriangleSoup {
            triangles: vec![[[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        };
        let (mesh, report) = weld(&soup, 1e-6);
        assert!(mesh.faces.is_empty());
        assert_eq!(report.degenerate_dropped, 1);
    }
}
