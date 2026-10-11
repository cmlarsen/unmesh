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

/// A candidate coordinate rounding grid must reproduce at least this share of
/// the distinct coordinates exactly (to within `ROUNDING_ULPS` float64 ulps)
/// before it is accepted. Exact match is what makes a chance fit impossible: a
/// random float64 lands on a 9-significant-digit grid with probability about
/// 1e-7, not the half a loose acceptance band would allow.
const ROUNDING_MIN_SHARE: f64 = 0.99;

/// Fewer distinct coordinates than this cannot separate a rounding grid from
/// the geometry's own vertex spacing, so no model is accepted and the floor is
/// the diagonal term alone.
const ROUNDING_MIN_DISTINCT: usize = 64;

/// How many float64 ulps a coordinate and a model's rounded value may differ
/// by and still count as reproduced. Covers the representation error of
/// `%.{d}e` and of `x * 10^p` arithmetic, and is far below any real step.
const ROUNDING_ULPS: f64 = 4.0;

/// A decimal model's last digit (the digit at the model's step position) must
/// take at least this many of the ten possible values, each holding at least
/// `ROUNDING_MIN_DIGIT_SHARE` of the distinct coordinates. Rounded data has an
/// essentially uniform last digit; design data sitting on a coarser grid, or
/// shifted by a constant fractional offset, leaves the digit degenerate (an
/// 0.1 mm grid reads 0 in the hundredths place, an integer-mm plate on a .37
/// offset reads only 7). Requiring most of the digit's values to appear
/// rejects those without rejecting genuinely rounded data.
const ROUNDING_MIN_DIGIT_VALUES: usize = 8;

/// The share of distinct coordinates a single last-digit value needs to count
/// towards `ROUNDING_MIN_DIGIT_VALUES`.
const ROUNDING_MIN_DIGIT_SHARE: f64 = 0.02;

/// An f32 or rescaled-f32 model must have the lowest mantissa bit of the value
/// zero for at least this share and one for at least this share. Real
/// float32-quantized data has that bit essentially uniform; design data whose
/// coordinates are exactly representable on a grid coarser than two f32 ulps
/// leaves the bit always zero, so this rejects a coarse grid dressed as f32.
const ROUNDING_MIN_F32_BIT_SHARE: f64 = 0.25;

/// Multiple of a data-fitted rounding step the floor covers. The step is the
/// grid the coordinates sit on exactly, so the coordinate rounding error is at
/// most half a step; the full step covers that with a factor of two in hand, so
/// no further multiplier is needed. The all-f32 path knows its step exactly
/// (`f32_ulp`) and is not scaled.
const ROUNDING_STEP_FACTOR: f64 = 1.0;

/// Upper bound on the quantization term as a fraction of the bounding-box
/// diagonal: the same cap the initial tolerance uses. The floor must never
/// exceed the tolerance it floors.
const ROUNDING_CAP_REL: f64 = 5e-4;

/// Unit conversions a mesh is commonly rescaled by before being written as a
/// binary STL: the coordinates are then `f32` on the *converted* scale.
const SCALED_F32: [f64; 6] = [1000.0, 0.001, 25.4, 1.0 / 25.4, 10.0, 0.1];

/// Significant-digit models, matching `%.{d-1}e` (`%g` is 6).
const SIG_DIGITS: [u32; 4] = [6, 7, 8, 9];

/// Fixed-decimal models, matching `%.{p}f`.
const FIXED_DECIMALS: [u32; 5] = [2, 3, 4, 5, 6];

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

/// The float64 ulp at `x` (the spacing down to the next representable value).
/// Zero for `x == 0` and non-finite `x`.
fn f64_ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 || !a.is_finite() {
        return 0.0;
    }
    a - f64::from_bits(a.to_bits() - 1)
}

/// `floor(log10(|x|))`, corrected against float64 `log10` rounding at a power
/// of ten. `x` must be finite and non-zero.
fn decade(x: f64) -> i32 {
    let a = x.abs();
    let mut e = a.log10().floor() as i32;
    while 10f64.powi(e + 1) <= a {
        e += 1;
    }
    while 10f64.powi(e) > a {
        e -= 1;
    }
    e
}

/// `x` rounded to `d` significant decimal digits.
fn round_sig(x: f64, d: u32) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let scale = 10f64.powi(d as i32 - 1 - decade(x));
    (x * scale).round() / scale
}

/// The step of the `d`-significant-digit grid at `|x|`. Every other coordinate
/// has a smaller-or-equal step, so the largest-coordinate step is the largest.
fn sig_step(x: f64, d: u32) -> f64 {
    10f64.powi(decade(x) - d as i32 + 1)
}

/// `x` rounded to `p` decimal places.
fn round_fixed(x: f64, p: u32) -> f64 {
    let scale = 10f64.powi(p as i32);
    (x * scale).round() / scale
}

/// Whether `x` and a model's rounded `y` agree to within a few float64 ulps.
fn ulps_close(x: f64, y: f64) -> bool {
    if x == y {
        return true;
    }
    let u = f64_ulp(x).max(f64_ulp(y));
    u > 0.0 && (x - y).abs() <= ROUNDING_ULPS * u
}

/// The last digit an `d`-significant-digit model would print for `v`, i.e. the
/// digit at the model's step position (zero for `v == 0`, which has no decade).
fn sig_last_digit(v: f64, d: u32) -> usize {
    if v == 0.0 {
        return 0;
    }
    let scale = 10f64.powi(d as i32 - 1 - decade(v));
    ((v * scale).round() as i64).rem_euclid(10) as usize
}

/// The digit a `p`-decimal-place model would print for `v` in the last place.
fn fixed_last_digit(v: f64, p: u32) -> usize {
    let scale = 10f64.powi(p as i32);
    ((v * scale).round() as i64).rem_euclid(10) as usize
}

/// Whether the last digit of every distinct coordinate under a model is
/// populated: at least `ROUNDING_MIN_DIGIT_VALUES` of the ten digit values
/// each hold at least `ROUNDING_MIN_DIGIT_SHARE` of the coordinates. The
/// per-axis values are pooled, as the match share is: a rounding grid is a
/// property of the whole mesh, and pooling gives every axis the same votes.
fn last_digit_populated(vals: &[f64], digit: impl Fn(f64) -> usize) -> bool {
    let n = vals.len() as f64;
    let mut counts = [0usize; 10];
    for &v in vals {
        counts[digit(v)] += 1;
    }
    counts
        .iter()
        .filter(|&&c| c as f64 / n >= ROUNDING_MIN_DIGIT_SHARE)
        .count()
        >= ROUNDING_MIN_DIGIT_VALUES
}

/// Whether the lowest mantissa bit of `f32(v / scale)` is zero for at least
/// `ROUNDING_MIN_F32_BIT_SHARE` and one for at least that share of the
/// coordinates. `scale` is 1 for the plain all-f32 model and the unit factor
/// for a rescaled one.
fn f32_low_bit_mixed(vals: &[f64], scale: f64) -> bool {
    let n = vals.len() as f64;
    let ones = vals
        .iter()
        .filter(|&&v| ((v / scale) as f32).to_bits() & 1 == 1)
        .count() as f64;
    let share = ones / n;
    (ROUNDING_MIN_F32_BIT_SHARE..=1.0 - ROUNDING_MIN_F32_BIT_SHARE).contains(&share)
}

/// The coordinate rounding step the tolerance floor covers, estimated from the
/// coordinates themselves by testing explicit rounding models.
///
/// An all-f32 mesh is on the f32 grid, so the step is one f32 ulp at the
/// largest coordinate, exactly. Otherwise every distinct coordinate (all axes
/// pooled) is tested against a short list of models: a unit-rescaled f32 mesh,
/// a decimal mesh rounded to 6..9 significant digits, or one rounded to 2..6
/// decimal places. A model is accepted only with at least
/// `ROUNDING_MIN_DISTINCT` distinct values, `ROUNDING_MIN_SHARE` of them
/// reproduced exactly, and a populated last digit: the digit at the model's
/// step position takes `ROUNDING_MIN_DIGIT_VALUES` values at
/// `ROUNDING_MIN_DIGIT_SHARE` each (f32 models, whose "digit" is the lowest
/// mantissa bit, need it mixed as `ROUNDING_MIN_F32_BIT_SHARE`). The last-digit
/// gate is what keeps a design mesh on a round grid from matching a coarser
/// model exactly although nothing was rounded. The coarsest accepted model (the
/// largest step at the largest coordinate) is the coordinate precision. A
/// float64 CAD mesh matches none, so no step is found and the floor is the
/// diagonal term alone. The step is scaled by `ROUNDING_STEP_FACTOR` and capped
/// at `ROUNDING_CAP_REL` of the diagonal, so the floor never exceeds the
/// initial tolerance.
pub fn coordinate_rounding_step(soup: &TriangleSoup) -> f64 {
    let mut max_abs = 0.0f64;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for t in &soup.triangles {
        for p in t {
            for (a, &c) in p.iter().enumerate() {
                max_abs = max_abs.max(c.abs());
                lo[a] = lo[a].min(c);
                hi[a] = hi[a].max(c);
            }
        }
    }
    if max_abs == 0.0 {
        return 0.0;
    }
    if all_float32_exact(soup) {
        return f32_ulp(max_abs);
    }
    let mut vals: Vec<f64> = soup
        .triangles
        .iter()
        .flatten()
        .flat_map(|p| p.iter().copied())
        .collect();
    vals.sort_unstable_by(f64::total_cmp);
    vals.dedup();
    if vals.len() < ROUNDING_MIN_DISTINCT {
        return 0.0;
    }
    let n = vals.len() as f64;
    let share = |f: &dyn Fn(f64) -> bool| vals.iter().filter(|&&v| f(v)).count() as f64 / n;

    let mut best = 0.0f64;
    // Only reached for a mesh that is not all-f32: one stray beyond the early
    // return's allowance leaves a few coordinates off the f32 grid.
    if share(&|v| (v as f32) as f64 == v) >= ROUNDING_MIN_SHARE && f32_low_bit_mixed(&vals, 1.0) {
        best = best.max(f32_ulp(max_abs));
    }
    for &s in &SCALED_F32 {
        if share(&|v| ulps_close(((v / s) as f32) as f64 * s, v)) >= ROUNDING_MIN_SHARE
            && f32_low_bit_mixed(&vals, s)
        {
            best = best.max(s * f32_ulp(max_abs / s));
        }
    }
    for &d in &SIG_DIGITS {
        if share(&|v| ulps_close(v, round_sig(v, d))) >= ROUNDING_MIN_SHARE
            && last_digit_populated(&vals, |v| sig_last_digit(v, d))
        {
            best = best.max(sig_step(max_abs, d));
        }
    }
    for &p in &FIXED_DECIMALS {
        if share(&|v| ulps_close(v, round_fixed(v, p))) >= ROUNDING_MIN_SHARE
            && last_digit_populated(&vals, |v| fixed_last_digit(v, p))
        {
            best = best.max(10f64.powi(-(p as i32)));
        }
    }
    if best == 0.0 {
        return 0.0;
    }
    let dx = hi[0] - lo[0];
    let dy = hi[1] - lo[1];
    let dz = hi[2] - lo[2];
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    (ROUNDING_STEP_FACTOR * best).min(ROUNDING_CAP_REL * diag)
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

    /// The largest `|coordinate|` and the bounding-box diagonal of a point set.
    fn extent(points: &[[f64; 3]]) -> (f64, f64) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        let mut max_abs = 0.0f64;
        for p in points {
            for a in 0..3 {
                max_abs = max_abs.max(p[a].abs());
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let d = (0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f64>().sqrt();
        (max_abs, d)
    }

    #[test]
    fn scaled_float32_metre_is_found() {
        let pts: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                [
                    (i as f32 * 1.7) as f64 * 1000.0,
                    (i as f32 * 2.3 + 0.5) as f64 * 1000.0,
                    (i as f32 * 3.1 + 1.25) as f64 * 1000.0,
                ]
            })
            .collect();
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "the fixture must not be all-f32"
        );
        let (max_abs, diag) = extent(&pts);
        let expected = (ROUNDING_STEP_FACTOR * 1000.0 * f32_ulp(max_abs / 1000.0))
            .min(ROUNDING_CAP_REL * diag);
        let step = coordinate_rounding_step(&soup(&pts));
        assert!(
            (step - expected).abs() < 1e-12,
            "step={step} expected={expected}"
        );
    }

    #[test]
    fn scaled_float32_inch_is_found() {
        let pts: Vec<[f64; 3]> = (0..200)
            .map(|i| {
                let inch = |v: f64| (v as f32) as f64 * 25.4;
                [
                    inch(i as f64 * 1.7),
                    inch(i as f64 * 2.3 + 0.5),
                    inch(i as f64 * 3.1 + 1.25),
                ]
            })
            .collect();
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "the fixture must not be all-f32"
        );
        let (max_abs, diag) = extent(&pts);
        let expected =
            (ROUNDING_STEP_FACTOR * 25.4 * f32_ulp(max_abs / 25.4)).min(ROUNDING_CAP_REL * diag);
        let step = coordinate_rounding_step(&soup(&pts));
        assert!(
            (step - expected).abs() < 1e-12,
            "step={step} expected={expected}"
        );
    }

    #[test]
    fn decimal_significant_digits_are_found() {
        // 1000.000..1005.000 in 1e-3 steps: 5000 distinct 7-significant-digit
        // decimals, spread so the tolerance cap does not bind.
        let pts: Vec<[f64; 3]> = (0..5000)
            .map(|i| {
                let t = i as f64 * 0.001;
                [1000.0 + t, 1000.0 + 2.0 * t, 1000.0 + 3.0 * t]
            })
            .collect();
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "the fixture must not be all-f32"
        );
        let (max_abs, diag) = extent(&pts);
        let expected = (ROUNDING_STEP_FACTOR * sig_step(max_abs, 7)).min(ROUNDING_CAP_REL * diag);
        let step = coordinate_rounding_step(&soup(&pts));
        assert!(
            (step - expected).abs() < 1e-12,
            "step={step} expected={expected}"
        );
        // The raw 1e-3 step is the answer (the cap does not bind).
        assert!((step - 0.001).abs() < 1e-12, "step={step}");
    }

    #[test]
    fn decimal_fixed_places_are_found() {
        // 1000.00..1050.00 in 1e-2 steps: 5000 distinct two-decimal values.
        let pts: Vec<[f64; 3]> = (0..5000)
            .map(|i| {
                let t = i as f64 * 0.01;
                [1000.0 + t, 1000.0 + 2.0 * t, 1000.0 + 3.0 * t]
            })
            .collect();
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "the fixture must not be all-f32"
        );
        let (max_abs, diag) = extent(&pts);
        let expected = (ROUNDING_STEP_FACTOR * 0.01).min(ROUNDING_CAP_REL * diag);
        let _ = max_abs;
        let step = coordinate_rounding_step(&soup(&pts));
        assert!(
            (step - expected).abs() < 1e-12,
            "step={step} expected={expected}"
        );
        assert!((step - 0.01).abs() < 1e-12, "step={step}");
    }

    #[test]
    fn sixty_four_random_float64_values_match_nothing() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 3]> = (0..100)
            .map(|_| {
                [
                    1000.0 + next() * 1000.0,
                    1000.0 + next() * 1000.0,
                    1000.0 + next() * 1000.0,
                ]
            })
            .collect();
        let step = coordinate_rounding_step(&soup(&pts));
        assert_eq!(step, 0.0, "spurious step {step}");
    }

    #[test]
    fn fewer_than_sixty_four_distinct_values_have_no_grid() {
        // A perfect 1e-3 decimal grid, but only 20 distinct values per axis.
        let pts: Vec<[f64; 3]> = (0..20)
            .map(|i| {
                let t = i as f64 * 0.001;
                [5000.0 + t, 5000.0, 5000.0]
            })
            .collect();
        let step = coordinate_rounding_step(&soup(&pts));
        assert_eq!(step, 0.0, "spurious step {step}");
    }

    #[test]
    fn float64_coordinates_have_no_grid() {
        let pts: Vec<[f64; 3]> = (0..100)
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
        assert_eq!(step, 0.0, "spurious step {step}");
    }

    /// Points on a `step` grid clustered every millimetre, so the tolerance cap
    /// cannot bind, walking every one of the ten last digits within a cluster.
    fn clustered_grid(step: f64) -> Vec<[f64; 3]> {
        let mut pts = Vec::new();
        for k in 0..40 {
            for d in 0..10 {
                let base = 1000.0 + k as f64;
                let off = d as f64 * step;
                pts.push([
                    base + off,
                    1000.0 + 0.5 * k as f64 + off,
                    1000.0 + 0.25 * k as f64 + off,
                ]);
            }
        }
        pts
    }

    #[test]
    fn six_and_seven_significant_digits_are_found() {
        // 6 significant digits (`%g`) at max_abs in [1000, 10000) has step
        // 1e-2, 7 (`%.6e`) has 1e-3; both are also fixed-decimal grids of the
        // same step, and their last digit is uniform.
        for (step, d) in [(1e-2, 6u32), (1e-3, 7u32)] {
            let pts = clustered_grid(step);
            let (max_abs, _) = extent(&pts);
            assert_eq!(sig_step(max_abs, d), step, "fixture at {d} digits");
            let got = coordinate_rounding_step(&soup(&pts));
            assert!(
                (got - step).abs() < 1e-12,
                "{d} significant digits: step={got} expected={step}"
            );
        }
    }

    #[test]
    fn two_to_six_decimal_places_are_found() {
        for p in 2..=6u32 {
            let step = 10f64.powi(-(p as i32));
            let pts = clustered_grid(step);
            let got = coordinate_rounding_step(&soup(&pts));
            assert!(
                (got - step).abs() < 1e-12,
                "{p} decimals: step={got} expected={step}"
            );
        }
    }

    /// A design mesh on a round grid carries no rounding; no model may match
    /// its last digit even though a coarse model reproduces it exactly.
    fn assert_no_decimal_model(pts: &[[f64; 3]], label: &str) {
        assert!(
            pts.iter().flatten().any(|&c| (c as f32) as f64 != c),
            "{label}: fixture must not be all-f32"
        );
        let step = coordinate_rounding_step(&soup(pts));
        assert_eq!(step, 0.0, "{label}: spurious step {step}");
    }

    #[test]
    fn a_tenth_millimetre_design_grid_has_no_model() {
        let pts: Vec<[f64; 3]> = (0..400)
            .map(|i| {
                let k = (i / 10) as f64;
                let d = (i % 10) as f64;
                [k + 0.1 * d, 0.5 * k + 0.1 * d, 0.25 * k + 0.1 * d]
            })
            .collect();
        assert_no_decimal_model(&pts, "0.1 mm grid");
    }

    #[test]
    fn an_integer_millimetre_grid_at_a_fractional_offset_has_no_model() {
        // 1000.37, -2000.11, 500.29: every coordinate is two-decimal, so the
        // hundredths model reproduces it exactly while its last digit is the
        // constant 7, 1 or 9.
        let pts: Vec<[f64; 3]> = (0..400)
            .map(|k| {
                let k = k as f64;
                [1000.37 + k, -2000.11 + k, 500.29 + k]
            })
            .collect();
        assert_no_decimal_model(&pts, "integer mm at a fractional offset");
    }

    #[test]
    fn a_half_millimetre_design_grid_has_no_model() {
        // 0.5 mm spacing at a fractional offset: every coordinate is two
        // decimals, but the hundredths digit is the constant 7, 1 or 9.
        let pts: Vec<[f64; 3]> = (0..400)
            .map(|k| {
                let k = k as f64 * 0.5;
                [1000.37 + k, -2000.11 + k, 500.29 + k]
            })
            .collect();
        assert_no_decimal_model(&pts, "0.5 mm grid");
    }

    #[test]
    fn an_eighth_inch_design_grid_has_no_model() {
        // Every coordinate a whole multiple of 1/8 inch (3.175 mm): the
        // thousandths model reproduces it exactly, but its last digit is only
        // ever 0 or 5.
        let i = 25.4 / 8.0;
        let pts: Vec<[f64; 3]> = (0..400)
            .map(|k| {
                let k = k as f64 * i;
                [1050.0 + k, 500.0 + k, k]
            })
            .collect();
        assert_no_decimal_model(&pts, "1/8 inch grid");
    }

    #[test]
    fn one_stray_coordinate_does_not_change_a_decimal_grid() {
        let mut pts: Vec<[f64; 3]> = (0..5000)
            .map(|i| {
                let t = i as f64 * 0.001;
                [1000.0 + t, 1000.0 + 2.0 * t, 1000.0 + 3.0 * t]
            })
            .collect();
        pts[5][0] += 1e-9;
        let step = coordinate_rounding_step(&soup(&pts));
        assert!((step - 0.001).abs() < 1e-12, "step={step}");
    }
}
