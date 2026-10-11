use rustc_hash::FxHashMap;

use super::dsu::Dsu;
use super::linalg::{V3, add, norm, scale, sub};
use crate::api::{ConvertError, ConvertWarning};
use crate::mesh::{Point, TriangleSoup};
use crate::weld::weld as weld_soup;

fn parity(f: &[u32; 3], key: &[u32; 3]) -> bool {
    (f[0] == key[0] && f[1] == key[1])
        || (f[0] == key[1] && f[1] == key[2])
        || (f[0] == key[2] && f[1] == key[0])
}

fn component_closed(faces: &[[u32; 3]], members: &[usize]) -> bool {
    let mut edges: Vec<u64> = Vec::with_capacity(members.len() * 3);
    for &f in members {
        let t = faces[f];
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            edges.push(((u.min(v) as u64) << 32) | u.max(v) as u64);
        }
    }
    edges.sort_unstable();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        if j - i != 2 {
            return false;
        }
        i = j;
    }
    true
}

pub struct Welded {
    pub faces: Vec<[u32; 3]>,
    pub fsrc: Vec<u32>,
    pub vc: Vec<V3>,
    pub orig: Vec<Point>,
    pub center: V3,
    pub diag: f64,
    /// The coordinate rounding step the tolerance floor covers, estimated from
    /// the coordinates (one f32 ulp for an all-f32 mesh, the detected decimal
    /// or rescaled grid otherwise, 0 for float64 CAD).
    pub quant_abs: f64,
    pub unique_vertices: u32,
    pub merge_dev: f64,
}

/// One unit in the last place of the float32 value nearest `x`, i.e. the step
/// of the float32 grid at `|x|` (about 6e-5 mm at 1000 mm). Zero for `x == 0`
/// and for a non-finite `x`.
pub fn f32_ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 || !a.is_finite() {
        return 0.0;
    }
    let bits = (a as f32).to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    if exp == 0 {
        // Subnormal: the float32 ulp is the smallest subnormal.
        return f64::from(f32::from_bits(1));
    }
    2f64.powi(exp - 127 - 23)
}

/// The largest candidate divisor of the smallest coordinate gap tried when
/// fitting a rounding step (see `axis_rounding_step`).
const STEP_DIVISORS: u64 = 8192;

/// Safety factor on a rounding step estimated from the data. A binary grid
/// doubles its step at a binade boundary, so coordinates that cross one can
/// under-report the step at the largest coordinate by up to two; the factor
/// covers that. An all-f32 mesh knows its step exactly (`f32_ulp`) and is not
/// scaled.
const ROUNDING_STEP_FACTOR: f64 = 2.0;

/// Whether the coordinates are exactly representable as f32, allowing a stray
/// vertex not to be. A binary STL's coordinates always are, so such a mesh is
/// on the f32 grid and its step is `f32_ulp` at the largest coordinate; one
/// coordinate nudged off the grid must not move it off that determination.
fn all_float32_exact(soup: &TriangleSoup) -> bool {
    let mut total = 0usize;
    let mut off = 0usize;
    for t in &soup.triangles {
        for p in t {
            for &c in p {
                total += 1;
                if (c as f32) as f64 != c {
                    off += 1;
                }
            }
        }
    }
    off <= 1 + total / 1000
}

/// Whether `d` (positive coordinate differences from the first distinct value)
/// are all within a quarter of `step` of an integer multiple of `step`, the
/// test for `step` being the coordinate rounding grid. The quarter tolerates
/// the float64 representation error of a decimal or scaled grid (about
/// `1e-16 * |x|`), far below a real step.
fn multiples_of(d: &[f64], step: f64, abstol: f64) -> bool {
    let tol = (0.25 * step).max(abstol);
    d.iter().all(|&x| {
        let q = (x / step).round();
        (x - q * step).abs() <= tol
    })
}

/// The coordinate rounding step of one axis, from the sorted distinct
/// coordinates `coords` whose magnitude is at least half the mesh's largest
/// coordinate. The smallest positive gap is a multiple of the step, so the
/// step is the coarsest divisor of that gap for which every difference is an
/// integer multiple: the first divisor that fits, scanning from the smallest
/// gap down. Finer divisors can keep fitting when every difference happens to
/// be an even multiple, so the finest fit is not the grid; and below the
/// float64 noise floor (`abstol`) every divisor fits, so the scan stops there.
fn axis_rounding_step(coords: &[f64], max_abs: f64) -> f64 {
    let v0 = coords[0];
    let d: Vec<f64> = coords[1..]
        .iter()
        .map(|&v| v - v0)
        .filter(|&x| x > 0.0)
        .collect();
    if d.len() < 2 {
        return 0.0;
    }
    let gap = d.iter().copied().fold(f64::INFINITY, f64::min);
    if !gap.is_finite() || gap <= 0.0 {
        return 0.0;
    }
    let abstol = (1e-11 * max_abs).max(1e-15);
    let smin = 64.0 * abstol;
    for k in 1..=STEP_DIVISORS {
        let s = gap / k as f64;
        if s < smin {
            break;
        }
        if multiples_of(&d, s, abstol) {
            return s;
        }
    }
    0.0
}

/// The coordinate rounding step the tolerance floor covers, estimated from the
/// coordinates themselves. An all-f32 mesh is on the f32 grid, so the step is
/// one f32 ulp at the largest coordinate, exactly. Otherwise each axis is
/// tested for a rounding grid (a decimal STL's 7 significant digits, a metre-
/// or inch-rescaled f32 mesh); the largest axis step, scaled by
/// `ROUNDING_STEP_FACTOR`, is the coordinate precision. Float64 CAD
/// coordinates are not on any grid, so no step is found and the floor is the
/// diagonal term alone. A step coarser than the term the always-on floor used
/// (`5e-7 * max_abs`) is rejected: such a "grid" is the geometry's own vertex
/// spacing (or the two f64 coordinates of a coarse mesh), not coordinate
/// rounding, and must not make the floor follow translation. The result is
/// capped there as well, so the floor never exceeds that always-on term.
pub fn coordinate_rounding_step(soup: &TriangleSoup) -> f64 {
    let mut max_abs = 0.0f64;
    for t in &soup.triangles {
        for p in t {
            for &c in p {
                max_abs = max_abs.max(c.abs());
            }
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    if all_float32_exact(soup) {
        return f32_ulp(max_abs);
    }
    let cap = 5e-7 * max_abs;
    let mut best = 0.0f64;
    for axis in 0..3 {
        let mut coords: Vec<f64> = soup
            .triangles
            .iter()
            .flatten()
            .map(|p| p[axis])
            .filter(|c| c.abs() >= 0.5 * max_abs)
            .collect();
        coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        coords.dedup();
        if coords.len() < 8 {
            continue;
        }
        let step = axis_rounding_step(&coords, max_abs);
        // A step coarser than the term the always-on floor used is the
        // geometry's own vertex spacing, not coordinate rounding: reject it.
        if step > 0.0 && step <= cap {
            best = best.max(step);
        }
    }
    (ROUNDING_STEP_FACTOR * best).min(cap)
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

    let mut seen: FxHashMap<([u32; 3], bool), usize> = FxHashMap::default();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut fsrc: Vec<u32> = Vec::with_capacity(mesh.faces.len());
    let mut copies: Vec<Vec<u32>> = Vec::with_capacity(mesh.faces.len());
    let mut duplicates = 0usize;
    for (f, &s) in mesh.faces.iter().zip(&src) {
        let mut key = *f;
        key.sort_unstable();
        match seen.get(&(key, parity(f, &key))) {
            Some(&i) => {
                copies[i].push(s);
                duplicates += 1;
            }
            None => {
                seen.insert((key, parity(f, &key)), faces.len());
                faces.push(*f);
                fsrc.push(s);
                copies.push(Vec::new());
            }
        }
    }
    let mut first: FxHashMap<[u32; 3], (usize, bool)> = FxHashMap::default();
    let mut twins: Vec<(usize, usize)> = Vec::new();
    for (i, f) in faces.iter().enumerate() {
        let mut key = *f;
        key.sort_unstable();
        let even = parity(f, &key);
        match first.get(&key) {
            None => {
                first.insert(key, (i, even));
            }
            Some(&(j, prev)) => {
                if prev != even {
                    twins.push((j, i));
                }
            }
        }
    }
    let mut drop_cand = vec![false; faces.len()];
    for &(_, c) in &twins {
        drop_cand[c] = true;
    }
    let mut edges: Vec<(u64, u32)> = Vec::with_capacity(faces.len() * 3);
    for (i, f) in faces.iter().enumerate() {
        if drop_cand[i] {
            continue;
        }
        for k in 0..3 {
            let (u, v) = (f[k], f[(k + 1) % 3]);
            edges.push((((u.min(v) as u64) << 32) | u.max(v) as u64, i as u32));
        }
    }
    edges.sort_unstable();
    let mut dsu = Dsu::new(faces.len());
    let mut bad = vec![false; faces.len()];
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 {
            j += 1;
        }
        match j - i {
            1 => bad[edges[i].1 as usize] = true,
            2 => dsu.union(edges[i].1, edges[i + 1].1),
            _ => {
                for e in &edges[i..j] {
                    bad[e.1 as usize] = true;
                    dsu.union(edges[i].1, e.1);
                }
            }
        }
        i = j;
    }
    let mut comp_bad = vec![false; faces.len()];
    for (fi, b) in bad.iter().enumerate() {
        if *b {
            comp_bad[dsu.find(fi as u32) as usize] = true;
        }
    }
    let mut healed = 0usize;
    for &(k, c) in &twins {
        if comp_bad[dsu.find(k as u32) as usize] {
            drop_cand[c] = false;
        } else {
            healed += 1;
        }
    }
    if healed > 0 {
        let mut kept_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() - healed);
        let mut kept_fsrc: Vec<u32> = Vec::with_capacity(faces.len() - healed);
        let mut kept_copies: Vec<Vec<u32>> = Vec::with_capacity(faces.len() - healed);
        for (i, f) in faces.iter().enumerate() {
            if !drop_cand[i] {
                kept_faces.push(*f);
                kept_fsrc.push(fsrc[i]);
                kept_copies.push(std::mem::take(&mut copies[i]));
            }
        }
        faces = kept_faces;
        fsrc = kept_fsrc;
        copies = kept_copies;
    }

    let mut vertices: Vec<Point> = mesh.vertices.clone();
    let mut stacked = 0usize;
    let mut coincident_comps = 0usize;
    let mut coincident_shells = 0usize;
    let mut coincident_copies = 0usize;
    {
        let mut stacks = Dsu::new(faces.len());
        let mut edges: Vec<(u64, u32)> = Vec::with_capacity(faces.len() * 3);
        for (i, f) in faces.iter().enumerate() {
            for k in 0..3 {
                let (u, v) = (f[k], f[(k + 1) % 3]);
                edges.push((((u.min(v) as u64) << 32) | u.max(v) as u64, i as u32));
            }
        }
        edges.sort_unstable();
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j].0 == edges[i].0 {
                stacks.union(edges[i].1, edges[j].1);
                j += 1;
            }
            i = j;
        }
        let mut comps: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for f in 0..faces.len() {
            comps.entry(stacks.find(f as u32)).or_default().push(f);
        }
        let mut stack_comps: Vec<&Vec<usize>> = comps
            .values()
            .filter(|members| {
                let mult = members
                    .iter()
                    .map(|&f| 1 + copies[f].len())
                    .min()
                    .unwrap_or(0);
                mult >= 2
                    && members.iter().all(|&f| 1 + copies[f].len() == mult)
                    && component_closed(&faces, members)
            })
            .collect();
        stack_comps.sort_unstable();
        for members in stack_comps {
            let mult = members.iter().map(|&f| 1 + copies[f].len()).min().unwrap();
            coincident_comps += 1;
            coincident_shells += mult;
            coincident_copies += mult - 1;
            let mut used: Vec<u32> = members
                .iter()
                .flat_map(|&f| faces[f].iter().copied())
                .collect();
            used.sort_unstable();
            used.dedup();
            for layer in 1..mult {
                let mut remap: FxHashMap<u32, u32> = FxHashMap::default();
                for &v in &used {
                    let p = vertices[v as usize];
                    remap.insert(v, vertices.len() as u32);
                    vertices.push(p);
                }
                for &f in members {
                    let t = faces[f];
                    faces.push([remap[&t[0]], remap[&t[1]], remap[&t[2]]]);
                    fsrc.push(copies[f][layer - 1]);
                }
                stacked += members.len();
            }
        }
    }

    let dropped = wrep.degenerate_dropped + (duplicates - stacked);
    if coincident_comps > 0 {
        warnings.push(ConvertWarning {
            code: "coincident_shells".to_string(),
            message: format!(
                "{coincident_comps} coincident component(s) with {coincident_copies} extra copy(ies) were kept as {coincident_shells} coincident shells"
            ),
        });
    }
    if dropped > 0 {
        warnings.push(ConvertWarning {
            code: "degenerate_triangles".to_string(),
            message: format!("{dropped} degenerate or duplicate triangles were dropped"),
        });
    }
    if healed > 0 {
        warnings.push(ConvertWarning {
            code: "repaired_winding".to_string(),
            message: format!("{healed} reverse-oriented duplicate triangles were dropped"),
        });
    }

    let (mut lo, mut hi) = (vertices[0], vertices[0]);
    for p in &vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let center: V3 = scale(add(lo, hi), 0.5);
    let diag = norm(sub(hi, lo));
    let quant_abs = coordinate_rounding_step(soup);
    let orig: Vec<Point> = vertices;
    let vc: Vec<V3> = orig.iter().map(|p| sub(*p, center)).collect();

    Ok((
        Welded {
            faces,
            fsrc,
            vc,
            orig,
            center,
            diag,
            quant_abs,
            unique_vertices: wrep.unique_vertices as u32,
            merge_dev: wrep.max_merge,
        },
        warnings,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_ulp_matches_the_float32_grid_step() {
        assert_eq!(f32_ulp(0.0), 0.0);
        // 58 is in the [32, 64) binade: 2^(5-23).
        assert_eq!(f32_ulp(58.0), 2f64.powi(5 - 23));
        // 1000 is in the [512, 1024) binade: 2^(9-23) ~ 6.1e-5.
        assert_eq!(f32_ulp(1000.0), 2f64.powi(9 - 23));
        // The step is one f32 value away at a binade boundary.
        assert_eq!(f32_ulp(1024.0), 2f64.powi(10 - 23));
    }

    fn soup(points: &[[f64; 3]]) -> TriangleSoup {
        let triangles = points
            .windows(3)
            .map(|w| [w[0], w[1], w[2]])
            .collect::<Vec<_>>();
        TriangleSoup { triangles }
    }

    #[test]
    fn all_f32_coordinates_give_the_f32_step() {
        let pts: Vec<[f64; 3]> = (0..12)
            .map(|i| [1000.0 + i as f64, 1500.0, 2000.0])
            .collect();
        let step = coordinate_rounding_step(&soup(&pts));
        assert_eq!(step, f32_ulp(2011.0));
    }

    #[test]
    fn decimal_grid_is_found_without_f32_coordinates() {
        let mut pts: Vec<[f64; 3]> = (0..12)
            .map(|i| [5000.0 + i as f64 * 0.001, 5000.0, 5000.0])
            .collect();
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "the fixture must not be all-f32"
        );
        // The per-axis detector finds the raw 1e-3 digit step.
        let coords: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        let raw = axis_rounding_step(&coords, 5000.011);
        assert!((raw - 0.001).abs() < 1e-9, "raw={raw}");

        // The floor's step is doubled to cover a binade crossing.
        let step = coordinate_rounding_step(&soup(&pts));
        assert!((step - 0.002).abs() < 1e-9, "step={step}");

        // A single coordinate nudged off the grid must not change the step.
        pts[5][0] += 1e-9;
        let step2 = coordinate_rounding_step(&soup(&pts));
        assert!((step2 - 0.002).abs() < 1e-9, "step2={step2}");
    }

    #[test]
    fn float64_coordinates_have_no_grid() {
        let pts: Vec<[f64; 3]> = (0..12)
            .map(|i| {
                let t = i as f64 * 0.7;
                [
                    1000.0 + t,
                    1000.0 + t * t * 0.013 + t,
                    1000.0 + (t * 2.3).sin() * 50.0,
                ]
            })
            .collect();
        let step = coordinate_rounding_step(&soup(&pts));
        assert!(step < 1e-6, "spurious step {step}");
    }
}
