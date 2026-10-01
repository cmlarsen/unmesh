pub type Point = [f64; 3];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriangleSoup {
    pub triangles: Vec<[Point; 3]>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexedMesh {
    pub vertices: Vec<Point>,
    pub faces: Vec<[u32; 3]>,
}

impl TriangleSoup {
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn bbox(&self) -> Option<(Point, Point)> {
        let mut points = self.triangles.iter().flatten();
        let first = *points.next()?;
        Some(points.fold((first, first), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
            )
        }))
    }
}

impl IndexedMesh {
    pub fn to_soup(&self) -> TriangleSoup {
        TriangleSoup {
            triangles: self
                .faces
                .iter()
                .map(|f| f.map(|i| self.vertices[i as usize]))
                .collect(),
        }
    }
}
