use rustc_hash::FxHashMap;

use super::linalg::{V3, add, norm, scale, sub};
use crate::api::{ConvertError, ConvertWarning};
use crate::mesh::{Point, TriangleSoup};
use crate::weld::weld as weld_soup;

pub struct Welded {
    pub faces: Vec<[u32; 3]>,
    pub fsrc: Vec<u32>,
    pub vc: Vec<V3>,
    pub orig: Vec<Point>,
    pub center: V3,
    pub diag: f64,
    pub unique_vertices: u32,
}

pub fn run(
    soup: &TriangleSoup,
    vertex_merge: f64,
) -> Result<(Welded, Vec<ConvertWarning>), ConvertError> {
    if soup.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    let (mesh, src, wrep) =
        weld_soup(soup, vertex_merge).map_err(|e| ConvertError::InvalidInput(e.to_string()))?;
    if mesh.faces.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    let mut warnings = Vec::new();

    let mut seen: FxHashMap<([u32; 3], bool), ()> = FxHashMap::default();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut fsrc: Vec<u32> = Vec::with_capacity(mesh.faces.len());
    let mut duplicates = 0usize;
    for (f, &s) in mesh.faces.iter().zip(&src) {
        let mut key = *f;
        key.sort_unstable();
        let even = (f[0] == key[0] && f[1] == key[1])
            || (f[0] == key[1] && f[1] == key[2])
            || (f[0] == key[2] && f[1] == key[0]);
        if seen.insert((key, even), ()).is_some() {
            duplicates += 1;
            continue;
        }
        faces.push(*f);
        fsrc.push(s);
    }
    let dropped = wrep.degenerate_dropped + duplicates;
    if dropped > 0 {
        warnings.push(ConvertWarning {
            code: "degenerate_triangles".to_string(),
            message: format!("{dropped} degenerate or duplicate triangles were dropped"),
        });
    }

    let (mut lo, mut hi) = (mesh.vertices[0], mesh.vertices[0]);
    for p in &mesh.vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let center: V3 = scale(add(lo, hi), 0.5);
    let diag = norm(sub(hi, lo));
    let orig: Vec<Point> = mesh.vertices.clone();
    let vc: Vec<V3> = orig.iter().map(|p| sub(*p, center)).collect();

    Ok((
        Welded {
            faces,
            fsrc,
            vc,
            orig,
            center,
            diag,
            unique_vertices: wrep.unique_vertices as u32,
        },
        warnings,
    ))
}
