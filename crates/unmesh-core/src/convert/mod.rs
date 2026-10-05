#![allow(clippy::needless_range_loop)]

mod adjacency;
mod curved;
mod dsu;
mod emit;
mod fit;
mod grow;
mod linalg;
mod noise;
mod project;
mod segment;
mod snap;
mod surface;
mod topology;
mod weld;

use std::collections::BTreeMap;

use crate::api::{ConvertError, ConvertOptions, ConvertOutput, ConvertWarning, Report};
use crate::ir::Tolerances;
use crate::mesh::{Point, TriangleSoup};

use emit::Asm;
use fit::{label_faces, region_pairs};
use linalg::{V3, add};
use project::Projected;
use snap::NOISE_FACTOR;
use topology::NONE;

const INITIAL_TOL_REL: f64 = 5e-4;
const MIN_TOL_REL: f64 = 1e-6;
const FLOAT_TOL_REL: f64 = 5e-7;

fn floor(diag: f64, max_abs: f64) -> f64 {
    (MIN_TOL_REL * diag).max(FLOAT_TOL_REL * max_abs)
}

pub fn convert_soup(
    soup: &TriangleSoup,
    options: &ConvertOptions,
) -> Result<ConvertOutput, ConvertError> {
    let (w, shells, info, tol, auto_tol, warnings) = prepare(soup, options)?;
    let mut scratch = fit::Scratch::new(w.vc.len());
    let mut fit_once = |tol: f64| {
        fit::fit_regions(fit::FitArgs {
            vc: &w.vc,
            faces: &w.faces,
            nbr: &shells.topo.nbr,
            info: &info,
            eligible: &shells.eligible,
            tol,
            snap_deg: options.angular_snap_deg,
            scratch: &mut scratch,
        })
    };
    finish(
        &w,
        &shells,
        &info,
        soup.len() as u32,
        options,
        tol,
        auto_tol,
        warnings,
        &mut fit_once,
    )
}

pub fn validate_labels(labels: &[u32], triangles: usize) -> Result<(), ConvertError> {
    if labels.len() != triangles {
        return Err(ConvertError::InvalidInput(format!(
            "label count {} does not match triangle count {}",
            labels.len(),
            triangles
        )));
    }
    if labels.contains(&NONE) {
        return Err(ConvertError::InvalidInput(format!(
            "label {NONE} (u32::MAX) is reserved"
        )));
    }
    let n = labels.iter().max().map_or(0, |&m| m as usize + 1);
    if n > labels.len() {
        return Err(ConvertError::InvalidInput(format!(
            "labels are sparse: largest label {} exceeds the triangle count",
            n - 1
        )));
    }
    let mut used = vec![false; n];
    for &l in labels {
        used[l as usize] = true;
    }
    if let Some(gap) = used.iter().position(|u| !u) {
        return Err(ConvertError::InvalidInput(format!(
            "labels must be contiguous from 0: label {gap} is unused"
        )));
    }
    Ok(())
}

fn split_components(nbr: &[[u32; 3]], label: &[u32]) -> (Vec<u32>, usize) {
    let mut out = vec![NONE; label.len()];
    let mut next = 0u32;
    let mut stack = Vec::new();
    for f in 0..label.len() {
        if label[f] == NONE || out[f] != NONE {
            continue;
        }
        out[f] = next;
        stack.push(f as u32);
        while let Some(g) = stack.pop() {
            for &h in &nbr[g as usize] {
                if h != NONE && out[h as usize] == NONE && label[h as usize] == label[f] {
                    out[h as usize] = next;
                    stack.push(h);
                }
            }
        }
        next += 1;
    }
    (out, next as usize)
}

/// Fits one surface per labelled region (the oracle segmentation) instead of
/// segmenting. Regions are never merged with each other; a label whose
/// triangles fall apart into several edge-connected pieces becomes one region
/// per piece.
pub fn convert_soup_from_labels(
    soup: &TriangleSoup,
    source_labels: &[u32],
    options: &ConvertOptions,
) -> Result<ConvertOutput, ConvertError> {
    validate_labels(source_labels, soup.len())?;
    let (w, shells, info, tol, auto_tol, warnings) = prepare(soup, options)?;
    let raw: Vec<u32> = w
        .fsrc
        .iter()
        .enumerate()
        .map(|(f, &s)| {
            if shells.eligible[f] {
                source_labels[s as usize]
            } else {
                NONE
            }
        })
        .collect();
    let (face_labels, n_seg) = split_components(&shells.topo.nbr, &raw);
    let mut scratch = fit::Scratch::new(w.vc.len());
    let mut fit_once = |tol: f64| {
        fit::run(fit::RunArgs {
            vc: &w.vc,
            faces: &w.faces,
            info: &info,
            label: &face_labels,
            n_seg,
            tol,
            snap_deg: options.angular_snap_deg,
            scratch: &mut scratch,
        })
    };
    finish(
        &w,
        &shells,
        &info,
        soup.len() as u32,
        options,
        tol,
        auto_tol,
        warnings,
        &mut fit_once,
    )
}

type Prepared = (
    weld::Welded,
    topology::Shells,
    Vec<segment::TriInfo>,
    f64,
    bool,
    Vec<ConvertWarning>,
);

fn prepare(soup: &TriangleSoup, options: &ConvertOptions) -> Result<Prepared, ConvertError> {
    let (mut w, mut warnings) = weld::run(soup, options.vertex_merge)?;

    let auto_tol = options.linear_tolerance.is_none();
    let tol = match options.linear_tolerance {
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(_) => {
            return Err(ConvertError::InvalidInput(
                "linear_tolerance must be finite and > 0".to_string(),
            ));
        }
        None => (INITIAL_TOL_REL * w.diag).max(floor(w.diag, w.max_abs)),
    };

    let (shells, shell_warnings) = topology::prepare(&mut w.faces, &mut w.vc, &mut w.orig);
    warnings.extend(shell_warnings);

    let info = segment::tri_info(&w.vc, &w.faces);
    Ok((w, shells, info, tol, auto_tol, warnings))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    w: &weld::Welded,
    shells: &topology::Shells,
    info: &[segment::TriInfo],
    source_triangles: u32,
    options: &ConvertOptions,
    mut tol: f64,
    auto_tol: bool,
    mut warnings: Vec<ConvertWarning>,
    fit_once: &mut dyn FnMut(f64) -> (Vec<u32>, Vec<fit::Region>),
) -> Result<ConvertOutput, ConvertError> {
    let mut sigma = tol / NOISE_FACTOR;
    if auto_tol {
        let fl = floor(w.diag, w.max_abs);
        if let Some(est) =
            noise::estimate_sigma(&w.vc, &w.faces, &shells.topo.nbr, info, &shells.eligible)
        {
            tol = tol.min((NOISE_FACTOR * est).max(fl));
            sigma = est.min(tol / NOISE_FACTOR);
        }
    }
    let (label2, mut regions) = fit_once(tol);
    snap::snap_normals(
        &w.vc,
        &mut regions,
        tol,
        sigma,
        options.angular_snap_deg,
        w.diag,
        w.max_abs,
    );

    let pairs = region_pairs(&shells.topo.nbr, &label2);
    let (finals, flabel) = fit::finalize(&regions, &pairs, &shells.comp_of, &shells.topo, tol);

    let Projected {
        positions: pv,
        report_dev,
        report_rms,
        areas,
    } = project::run(project::ProjectArgs {
        vc: &w.vc,
        finals: &finals,
        flabel: &flabel,
        faces: &w.faces,
        info,
        tol,
        diag: w.diag,
        center: w.center,
    });

    let total_area: f64 = info.iter().map(|t| t.area).sum();
    let plane_area: f64 = finals
        .iter()
        .zip(&areas)
        .filter(|(f, _)| f.surface.is_analytic())
        .map(|(_, a)| a)
        .sum();
    let mut region_counts = BTreeMap::new();
    for fr in &finals {
        *region_counts
            .entry(fr.surface.name().to_string())
            .or_insert(0u32) += 1;
    }

    let tolerances = Tolerances {
        linear: tol,
        angular_snap_deg: options.angular_snap_deg,
        tangent_threshold_deg: options.tangent_threshold_deg,
        vertex_merge: options.vertex_merge,
    };
    let positions = |pv: &[V3]| -> Vec<Point> {
        pv.iter()
            .enumerate()
            .map(|(i, p)| {
                if *p == w.vc[i] {
                    w.orig[i]
                } else {
                    add(*p, w.center)
                }
            })
            .collect()
    };
    let asm = Asm {
        tolerances,
        source_triangles,
        center: w.center,
        source_vertices: w.unique_vertices,
        faces: &w.faces,
        fsrc: &w.fsrc,
        nbr: &shells.topo.nbr,
        metas: &shells.metas,
        orig: &w.orig,
    };
    let out_pos = positions(&pv);
    let result = emit::assemble(&asm, &finals, &flabel, &out_pos);
    let merge = w.merge_dev;
    let (ir, report_dev, report_rms, region_counts, area_fraction) = match result {
        Ok(ir) => (
            ir,
            report_dev + merge,
            report_rms,
            region_counts,
            if total_area > 0.0 {
                plane_area / total_area
            } else {
                0.0
            },
        ),
        Err(errors) => {
            warnings.push(ConvertWarning {
                code: "fallback_facets".to_string(),
                message: format!(
                    "analytic assembly failed validation ({}); every shell is kept as facets",
                    errors.join("; ")
                ),
            });
            let fb: Vec<fit::Final> = shells
                .topo
                .comps
                .iter()
                .enumerate()
                .map(|(ci, c)| fit::Final {
                    faces: c.faces.clone(),
                    surface: surface::Surface::Facets,
                    rms: 0.0,
                    max: 0.0,
                    sag: 0.0,
                    comp: ci,
                })
                .collect();
            let fl = label_faces(&fb, w.faces.len());
            let orig_pos = positions(&w.vc);
            let ir = emit::assemble(&asm, &fb, &fl, &orig_pos)
                .map_err(|e| ConvertError::InvalidInput(e.join("; ")))?;
            let mut counts = BTreeMap::new();
            counts.insert("facets".to_string(), fb.len() as u32);
            (ir, merge, 0.0, counts, 0.0)
        }
    };

    Ok(ConvertOutput {
        ir,
        report: Report {
            max_deviation: report_dev,
            rms_deviation: report_rms,
            analytic_area_fraction: area_fraction,
            region_counts,
            warnings,
        },
    })
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::linalg::{dot, norm, scale, sub};
    use super::*;
    use crate::ir::{Kind, ShellRole, Surface, VertexRole};

    pub(super) fn grid_box(lo: V3, hi: V3, div: usize, inward: bool) -> Vec<[V3; 3]> {
        let mut tris = Vec::new();
        for axis in 0..3 {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            for side in 0..2 {
                let mut o = lo;
                o[axis] = if side == 0 { lo[axis] } else { hi[axis] };
                let du = (hi[u] - lo[u]) / div as f64;
                let dv = (hi[v] - lo[v]) / div as f64;
                for i in 0..div {
                    for j in 0..div {
                        let p = |a: usize, b: usize| {
                            let mut q = o;
                            q[u] = lo[u] + du * (i + a) as f64;
                            q[v] = lo[v] + dv * (j + b) as f64;
                            q
                        };
                        let (p00, p10, p11, p01) = (p(0, 0), p(1, 0), p(1, 1), p(0, 1));
                        let (a, b) = if (side == 1) != inward {
                            ([p00, p10, p11], [p00, p11, p01])
                        } else {
                            ([p00, p11, p10], [p00, p01, p11])
                        };
                        tris.push(a);
                        tris.push(b);
                    }
                }
            }
        }
        tris
    }

    fn convert_tris(tris: Vec<[V3; 3]>) -> ConvertOutput {
        convert_soup(
            &TriangleSoup { triangles: tris },
            &ConvertOptions::default(),
        )
        .unwrap()
    }

    #[test]
    fn box_is_six_planes() {
        let out = convert_tris(grid_box([0.0; 3], [10.0, 20.0, 30.0], 1, false));
        let ir = &out.ir;
        assert_eq!(ir.regions.len(), 6, "{:?}", out.report.warnings);
        assert_eq!(ir.vertices.len(), 8);
        assert_eq!(ir.adjacencies.len(), 12);
        assert!(ir.adjacencies.iter().all(|a| a.boundaries.len() == 1));
        assert!(out.report.max_deviation < 1e-9);
        assert_eq!(ir.shells.len(), 1);
        assert!(ir.vertices.iter().all(|v| v.role == VertexRole::Junction));
        ir.validate().unwrap();
        for a in &ir.adjacencies {
            assert_eq!(a.boundaries[0].kind, Kind::Transversal);
            assert!((a.boundaries[0].dihedral_deg - 90.0).abs() < 1e-9);
        }
    }

    #[test]
    fn clean_300mm_plate_gets_micron_floor() {
        let out = convert_tris(grid_box([0.0, 0.0, 0.0], [300.0, 300.0, 10.0], 1, false));
        assert_eq!(out.ir.regions.len(), 6);
        let diag = (300.0_f64.powi(2) * 2.0 + 10.0_f64.powi(2)).sqrt();
        assert!((out.ir.tolerances.linear - 1e-6 * diag).abs() < 1e-12);
        assert!(out.ir.tolerances.linear < 1e-3);
    }

    #[test]
    fn floor_scales_with_part_size() {
        for s in [1000.0, 0.001] {
            let tris: Vec<[V3; 3]> = grid_box([0.0; 3], [10.0, 20.0, 30.0], 1, false)
                .into_iter()
                .map(|t| t.map(|p| scale(p, s)))
                .collect();
            let out = convert_tris(tris);
            assert_eq!(out.ir.regions.len(), 6, "scale {s}");
            let diag = super::linalg::norm(sub([10.0 * s, 20.0 * s, 30.0 * s], [0.0; 3]));
            assert!(
                (out.ir.tolerances.linear / diag - 1e-6).abs() < 1e-9,
                "scale {s}: linear={}",
                out.ir.tolerances.linear
            );
        }
    }

    #[test]
    fn offset_part_gets_float_precision_floor() {
        let out = convert_tris(grid_box(
            [1e6, 0.0, 0.0],
            [1e6 + 10.0, 10.0, 10.0],
            1,
            false,
        ));
        assert_eq!(out.ir.regions.len(), 6);
        let expect = 5e-7 * 1_000_010.0;
        assert!(
            (out.ir.tolerances.linear - expect).abs() < 1e-9,
            "linear={}",
            out.ir.tolerances.linear
        );
    }

    #[test]
    fn subdivided_box_still_six_planes() {
        let out = convert_tris(grid_box([0.0; 3], [10.0, 20.0, 30.0], 6, false));
        assert_eq!(out.ir.regions.len(), 6);
        assert_eq!(out.ir.vertices.len(), 8);
    }

    #[test]
    fn inside_out_box_is_flipped() {
        let out = convert_tris(grid_box([0.0; 3], [10.0, 20.0, 30.0], 2, true));
        assert_eq!(out.ir.regions.len(), 6);
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "flipped_winding")
        );
        for r in &out.ir.regions {
            if let Surface::Plane { origin, normal } = &r.surface {
                let c = [5.0, 10.0, 15.0];
                assert!(dot(sub(*origin, c), *normal) > 0.0);
            }
        }
    }

    #[test]
    fn cavity_and_two_bodies() {
        let mut tris = grid_box([0.0; 3], [30.0; 3], 2, false);
        tris.extend(grid_box([10.0; 3], [20.0; 3], 2, true));
        tris.extend(grid_box([100.0; 3], [110.0; 3], 1, false));
        let out = convert_tris(tris);
        let ir = &out.ir;
        assert_eq!(ir.shells.len(), 3);
        assert_eq!(ir.shells[1].role, ShellRole::Cavity);
        assert_eq!(ir.shells[1].parent, Some(0));
        assert_eq!(ir.shells[2].role, ShellRole::Outer);
        assert_eq!(ir.regions.len(), 18);
    }

    #[test]
    fn open_box_is_one_facets_region() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        tris.pop();
        let out = convert_tris(tris);
        assert_eq!(out.ir.regions.len(), 1);
        assert!(out.ir.regions[0].surface.is_facets());
        assert!(!out.ir.shells[0].closed);
        assert!(out.report.warnings.iter().any(|w| w.code == "open_edges"));
    }

    #[test]
    fn noisy_box_stays_six_planes_and_reports_honestly() {
        let mut tris = grid_box([0.0; 3], [40.0, 30.0, 20.0], 8, false);
        let mut state = 12345u64;
        let mut rnd = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
        };
        let mut moved: FxHashMap<[u64; 3], V3> = FxHashMap::default();
        for t in tris.iter_mut() {
            for p in t.iter_mut() {
                let key = p.map(f64::to_bits);
                let d = *moved.entry(key).or_insert_with(|| {
                    scale(linalg::unit([rnd(), rnd(), rnd()]), 0.01 * rnd().abs())
                });
                *p = add(*p, d);
            }
        }
        let out = convert_tris(tris);
        assert_eq!(out.ir.regions.len(), 6, "{:?}", out.report.warnings);
        assert!(out.report.max_deviation >= 0.0 && out.report.max_deviation < 0.03);
    }

    #[test]
    fn sub_weld_split_is_repaired_and_reported() {
        let mut tris = grid_box([0.0; 3], [10.0, 20.0, 30.0], 1, false);
        tris[0][0][0] += 5e-7;
        let out = convert_tris(tris);
        assert_eq!(out.ir.regions.len(), 6, "{:?}", out.report.warnings);
        assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
        assert!(out.report.max_deviation >= 5e-7 - 1e-12);
        assert!(out.report.max_deviation < 2e-6);
    }

    #[test]
    fn bodies_touching_at_a_corner_stay_separate() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        tris.extend(grid_box([10.0; 3], [20.0; 3], 1, false));
        let out = convert_tris(tris);
        assert_eq!(out.ir.shells.len(), 2);
        assert_eq!(out.ir.regions.len(), 12);
        assert_eq!(out.ir.vertices.len(), 16);
        assert!(out.report.warnings.is_empty(), "{:?}", out.report.warnings);
    }

    #[test]
    fn stray_reverse_duplicate_heals_to_six_planes() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        let t = tris[0];
        tris.push([t[0], t[2], t[1]]);
        let out = convert_tris(tris);
        let ir = &out.ir;
        assert_eq!(ir.regions.len(), 6, "{:?}", out.report.warnings);
        assert_eq!(ir.shells.len(), 1);
        assert!(ir.shells[0].closed);
        assert!(ir.regions.iter().all(|r| !r.surface.is_facets()));
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "repaired_winding")
        );
        assert!(
            out.report
                .warnings
                .iter()
                .all(|w| w.code != "degenerate_triangles")
        );
        ir.validate().unwrap();
    }

    #[test]
    fn two_stray_reverse_duplicates_heal_to_six_planes() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        let (a, b) = (tris[0], tris[5]);
        tris.push([a[0], a[2], a[1]]);
        tris.push([b[0], b[2], b[1]]);
        let out = convert_tris(tris);
        let ir = &out.ir;
        assert_eq!(ir.regions.len(), 6, "{:?}", out.report.warnings);
        assert_eq!(ir.shells.len(), 1);
        assert!(ir.shells[0].closed);
        assert!(ir.regions.iter().all(|r| !r.surface.is_facets()));
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "repaired_winding")
        );
        ir.validate().unwrap();
    }

    #[test]
    fn face_touching_boxes_keep_opposite_winding_duplicates() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        tris.extend(grid_box([10.0, 0.0, 0.0], [20.0, 10.0, 10.0], 1, false));
        let out = convert_tris(tris);
        let ir = &out.ir;
        let mut covered: Vec<u32> = ir
            .regions
            .iter()
            .flat_map(|r| r.triangles.iter().copied())
            .collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..24).collect::<Vec<_>>());
        assert!(
            out.report
                .warnings
                .iter()
                .all(|w| w.code != "degenerate_triangles")
        );
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "non_manifold_edges")
        );
        assert_eq!(ir.shells.len(), 1);
        assert!(!ir.shells[0].closed);
        assert_eq!(ir.regions.len(), 1);
        assert!(ir.regions[0].surface.is_facets());
        ir.validate().unwrap();
    }

    #[test]
    fn edge_touching_boxes_are_one_non_manifold_shell() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 1, false);
        tris.extend(grid_box([10.0, 10.0, 0.0], [20.0, 20.0, 10.0], 1, false));
        let out = convert_tris(tris);
        let ir = &out.ir;
        assert_eq!(ir.shells.len(), 1);
        assert!(!ir.shells[0].closed);
        assert_eq!(ir.regions.len(), 1);
        assert!(ir.regions[0].surface.is_facets());
        let mut covered: Vec<u32> = ir
            .regions
            .iter()
            .flat_map(|r| r.triangles.iter().copied())
            .collect();
        covered.sort_unstable();
        assert_eq!(covered, (0..24).collect::<Vec<_>>());
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "non_manifold_edges")
        );
        ir.validate().unwrap();
    }

    #[test]
    fn oracle_labels_reproduce_segmented_box() {
        let soup = TriangleSoup {
            triangles: grid_box([0.0; 3], [10.0, 20.0, 30.0], 1, false),
        };
        let labels: Vec<u32> = (0..12).map(|i| i / 2).collect();
        let a = convert_soup(&soup, &ConvertOptions::default()).unwrap();
        let b = convert_soup_from_labels(&soup, &labels, &ConvertOptions::default()).unwrap();
        assert_eq!(a.ir, b.ir);
        assert_eq!(a.report, b.report);
    }

    #[test]
    fn mixed_winding_is_repaired() {
        let mut tris = grid_box([0.0; 3], [10.0; 3], 2, false);
        for t in tris.iter_mut().step_by(5) {
            t.swap(1, 2);
        }
        let out = convert_tris(tris);
        assert_eq!(out.ir.regions.len(), 6);
        assert!(
            out.report
                .warnings
                .iter()
                .any(|w| w.code == "repaired_winding")
        );
    }

    pub(super) struct Frame {
        pub rot: [V3; 3],
        pub off: V3,
    }

    impl Frame {
        pub fn identity() -> Self {
            Frame {
                rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                off: [0.0; 3],
            }
        }

        pub fn tilted() -> Self {
            let (a, b) = (0.37_f64, 0.91_f64);
            let (sa, ca) = a.sin_cos();
            let (sb, cb) = b.sin_cos();
            Frame {
                rot: [
                    [cb, -sb * ca, sb * sa],
                    [sb, cb * ca, -cb * sa],
                    [0.0, sa, ca],
                ],
                off: [12.5, -7.25, 3.0],
            }
        }

        pub fn map(&self, p: V3) -> V3 {
            add(
                [
                    dot(self.rot[0], p),
                    dot(self.rot[1], p),
                    dot(self.rot[2], p),
                ],
                self.off,
            )
        }

        pub fn dir(&self, p: V3) -> V3 {
            [
                dot(self.rot[0], p),
                dot(self.rot[1], p),
                dot(self.rot[2], p),
            ]
        }
    }

    pub(super) fn ring(r: f64, z: f64, i: usize, n: usize) -> V3 {
        let t = std::f64::consts::TAU * (i % n) as f64 / n as f64;
        [r * t.cos(), r * t.sin(), z]
    }

    /// Washer: outer cylinder (label 0), bore (labels 1 and, when
    /// `split_bore`, 4 for the second half), top (2) and bottom (3) annuli.
    pub(super) fn washer(
        ro: f64,
        ri: f64,
        h: f64,
        n: usize,
        frame: &Frame,
        split_bore: bool,
    ) -> (TriangleSoup, Vec<u32>) {
        let mut tris = Vec::new();
        let mut labels = Vec::new();
        let m = |p: V3| frame.map(p);
        for i in 0..n {
            let (o0, o1) = (ring(ro, 0.0, i, n), ring(ro, 0.0, i + 1, n));
            let (o2, o3) = (ring(ro, h, i + 1, n), ring(ro, h, i, n));
            tris.push([m(o0), m(o1), m(o2)]);
            tris.push([m(o0), m(o2), m(o3)]);
            labels.extend([0, 0]);
            let (i0, i1) = (ring(ri, 0.0, i, n), ring(ri, 0.0, i + 1, n));
            let (i2, i3) = (ring(ri, h, i + 1, n), ring(ri, h, i, n));
            let bl = if split_bore && i >= n / 2 { 4 } else { 1 };
            tris.push([m(i0), m(i2), m(i1)]);
            tris.push([m(i0), m(i3), m(i2)]);
            labels.extend([bl, bl]);
            tris.push([m(i3), m(o3), m(o2)]);
            tris.push([m(i3), m(o2), m(i2)]);
            labels.extend([2, 2]);
            tris.push([m(i0), m(o1), m(o0)]);
            tris.push([m(i0), m(i1), m(o1)]);
            labels.extend([3, 3]);
        }
        (TriangleSoup { triangles: tris }, labels)
    }

    /// Frustum: cone side (label 0) from radius `r0` at z=0 to `r1` at z=h,
    /// bottom disc (1), top disc (2).
    pub(super) fn frustum(
        r0: f64,
        r1: f64,
        h: f64,
        n: usize,
        frame: &Frame,
    ) -> (TriangleSoup, Vec<u32>) {
        twisted_frustum(r0, r1, h, n, 0.0, frame)
    }

    pub(super) fn twisted_frustum(
        r0: f64,
        r1: f64,
        h: f64,
        n: usize,
        twist: f64,
        frame: &Frame,
    ) -> (TriangleSoup, Vec<u32>) {
        let top = |i: usize| {
            let t = std::f64::consts::TAU * (i % n) as f64 / n as f64 + twist;
            [r1 * t.cos(), r1 * t.sin(), h]
        };
        let mut tris = Vec::new();
        let mut labels = Vec::new();
        let m = |p: V3| frame.map(p);
        for i in 0..n {
            let (b0, b1) = (ring(r0, 0.0, i, n), ring(r0, 0.0, i + 1, n));
            let (t1, t0) = (top(i + 1), top(i));
            tris.push([m(b0), m(b1), m(t1)]);
            tris.push([m(b0), m(t1), m(t0)]);
            labels.extend([0, 0]);
            tris.push([m([0.0, 0.0, 0.0]), m(b1), m(b0)]);
            labels.push(1);
            tris.push([m([0.0, 0.0, h]), m(t0), m(t1)]);
            labels.push(2);
        }
        (TriangleSoup { triangles: tris }, labels)
    }

    fn labelled(soup: &TriangleSoup, labels: &[u32]) -> ConvertOutput {
        convert_soup_from_labels(soup, labels, &ConvertOptions::default()).unwrap()
    }

    fn cylinder_of(ir: &crate::ir::Ir, tri: u32) -> (V3, V3, f64, crate::ir::Orientation) {
        let r = ir
            .regions
            .iter()
            .find(|r| r.triangles.contains(&tri))
            .unwrap();
        match &r.surface {
            Surface::Cylinder {
                origin,
                axis,
                radius,
                orientation,
            } => (*origin, *axis, *radius, *orientation),
            s => panic!("triangle {tri}: expected a cylinder, got {s:?}"),
        }
    }

    #[test]
    fn labelled_washer_gives_two_cylinders_and_two_planes() {
        use crate::ir::Orientation;
        for (frame, n) in [
            (Frame::identity(), 48),
            (Frame::tilted(), 6),
            (Frame::tilted(), 37),
        ] {
            let (soup, labels) = washer(10.0, 4.0, 5.0, n, &frame, false);
            let out = labelled(&soup, &labels);
            let ir = &out.ir;
            ir.validate().unwrap();
            assert_eq!(ir.regions.len(), 4, "n={n}");
            assert_eq!(out.report.region_counts.get("cylinder"), Some(&2));
            assert_eq!(out.report.region_counts.get("plane"), Some(&2));
            let axis = frame.dir([0.0, 0.0, 1.0]);
            for (tri, r, orient) in [
                (0, 10.0, Orientation::Same),
                (2, 4.0, Orientation::Reversed),
            ] {
                let (o, a, radius, or) = cylinder_of(ir, tri);
                assert!((radius - r).abs() < 1e-9, "n={n}: radius {radius} vs {r}");
                assert!(dot(a, axis).abs() > 1.0 - 1e-12, "n={n}: axis {a:?}");
                let d = sub(o, frame.off);
                assert!(norm(sub(d, scale(axis, dot(d, axis)))) < 1e-9);
                assert_eq!(or, orient);
            }
            let sag = 10.0 * (1.0 - (std::f64::consts::PI / n as f64).cos());
            assert!(out.report.max_deviation >= sag - 1e-9, "n={n}");
            assert!(out.report.max_deviation < sag + 1e-6, "n={n}");
            for a in &ir.adjacencies {
                for b in &a.boundaries {
                    assert!((b.dihedral_deg - 90.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn labelled_frustum_gives_a_cone() {
        use crate::ir::Orientation;
        for (frame, r0, r1, n) in [
            (Frame::identity(), 8.0, 3.0, 64),
            (Frame::tilted(), 2.0, 9.0, 7),
        ] {
            let h = 5.0;
            let (soup, labels) = frustum(r0, r1, h, n, &frame);
            let out = labelled(&soup, &labels);
            let ir = &out.ir;
            ir.validate().unwrap();
            assert_eq!(ir.regions.len(), 3);
            let r = ir
                .regions
                .iter()
                .find(|r| r.triangles.contains(&0))
                .unwrap();
            let Surface::Cone {
                apex,
                axis,
                half_angle,
                orientation,
            } = &r.surface
            else {
                panic!("expected a cone, got {:?}", r.surface);
            };
            let alpha = (r0 - r1).abs().atan2(h);
            let z_apex = r0 * h / (r0 - r1);
            let widen = if r0 > r1 { -1.0 } else { 1.0 };
            assert!((half_angle - alpha).abs() < 1e-9);
            assert!(norm(sub(*apex, frame.map([0.0, 0.0, z_apex]))) < 1e-8);
            assert!(dot(*axis, frame.dir([0.0, 0.0, widen])) > 1.0 - 1e-12);
            assert_eq!(*orientation, Orientation::Same);
        }
    }

    /// Prism over a circular sector: `segments` equal chords turning by
    /// `step`, the arc wall labelled 0, the two radial walls 1 and 2, the
    /// caps 3 and 4. Every distinct vertex is moved by up to `noise`.
    pub(super) fn sector_prism(
        r: f64,
        segments: usize,
        step: f64,
        noise: f64,
        frame: &Frame,
    ) -> (TriangleSoup, Vec<u32>) {
        let h = 5.0;
        let start = -std::f64::consts::FRAC_PI_2;
        let mut profile = vec![[0.0, 0.0]];
        for k in 0..=segments {
            let t = start + step * k as f64;
            profile.push([r * t.cos(), r * t.sin()]);
        }
        let m = profile.len();
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let pts: Vec<[V3; 2]> = profile
            .iter()
            .map(|q| {
                let mut at = |z: f64| {
                    let p = frame.map([q[0], q[1], z]);
                    add(p, scale([jitter(), jitter(), jitter()], noise))
                };
                [at(0.0), at(h)]
            })
            .collect();
        let mut tris = Vec::new();
        let mut labels = Vec::new();
        for i in 0..m {
            let j = (i + 1) % m;
            let label = if i == 0 {
                1
            } else if j == 0 {
                2
            } else {
                0
            };
            tris.push([pts[i][0], pts[j][0], pts[j][1]]);
            tris.push([pts[i][0], pts[j][1], pts[i][1]]);
            labels.extend([label, label]);
        }
        for i in 1..m - 1 {
            tris.push([pts[0][1], pts[i][1], pts[i + 1][1]]);
            labels.push(4);
            tris.push([pts[0][0], pts[i + 1][0], pts[i][0]]);
            labels.push(3);
        }
        (TriangleSoup { triangles: tris }, labels)
    }

    fn sector_radius(seg: usize, step: f64, noise: f64, tol: f64) -> f64 {
        let (soup, labels) = sector_prism(10.0, seg, step, noise, &Frame::tilted());
        let out = convert_soup_from_labels(
            &soup,
            &labels,
            &ConvertOptions {
                linear_tolerance: Some(tol),
                ..ConvertOptions::default()
            },
        )
        .unwrap();
        out.ir.validate().unwrap();
        cylinder_of(&out.ir, 2).2
    }

    #[test]
    fn tessellation_law_recovers_short_noisy_off_axis_arcs() {
        let tau = std::f64::consts::TAU;
        for (seg, n) in [(3, 36), (2, 24), (3, 12), (9, 36)] {
            let r = sector_radius(seg, tau / n as f64, 1e-4, 1e-3);
            assert!((r - 10.0).abs() < 1e-4, "{seg} of {n}: radius {r}");
        }
        for (seg, n) in [(2, 6), (3, 8), (5, 40)] {
            let r = sector_radius(seg, tau / n as f64, 0.0, 1e-6);
            assert!((r - 10.0).abs() < 1e-9, "clean {seg} of {n}: radius {r}");
        }
    }

    #[test]
    fn tessellation_law_never_replaces_an_exact_radius() {
        let step = 90.05f64.to_radians() / 9.0;
        for tol in [1e-2, 1e-3] {
            let r = sector_radius(9, step, 0.0, tol);
            assert!((r - 10.0).abs() < 1e-9, "tol {tol}: radius {r}");
        }
    }

    fn dense_cone_deviation(soup: &TriangleSoup, labels: &[u32], ir: &crate::ir::Ir) -> f64 {
        let r = ir
            .regions
            .iter()
            .find(|r| r.triangles.contains(&0))
            .unwrap();
        let Surface::Cone {
            apex,
            axis,
            half_angle,
            ..
        } = r.surface
        else {
            panic!("expected a cone");
        };
        let cone = surface::Surface::Cone {
            apex,
            axis,
            half_angle,
            reversed: false,
        };
        let k = 200;
        let mut worst: f64 = 0.0;
        for (t, &l) in soup.triangles.iter().zip(labels) {
            if l != 0 {
                continue;
            }
            for i in 0..=k {
                for j in 0..=k - i {
                    let (a, b) = (i as f64 / k as f64, j as f64 / k as f64);
                    let p = add(
                        add(scale(t[0], a), scale(t[1], b)),
                        scale(t[2], 1.0 - a - b),
                    );
                    worst = worst.max(cone.distance(p).abs());
                }
            }
        }
        worst
    }

    #[test]
    fn twisted_cone_never_under_reports() {
        for frame in [Frame::identity(), Frame::tilted()] {
            let (soup, labels) = twisted_frustum(10.0, 2.0, 4.0, 48, 60f64.to_radians(), &frame);
            let out = labelled(&soup, &labels);
            out.ir.validate().unwrap();
            assert_eq!(out.report.region_counts.get("cone"), Some(&1));
            let measured = dense_cone_deviation(&soup, &labels, &out.ir);
            assert!(measured > 0.284, "{measured}");
            assert!(
                out.report.max_deviation >= measured,
                "reported {} < measured {measured}",
                out.report.max_deviation
            );
            assert!(out.report.max_deviation < measured * 1.001);
        }
    }

    #[test]
    fn split_bore_shares_one_cylinder() {
        let frame = Frame::tilted();
        let (soup, labels) = washer(10.0, 4.0, 5.0, 40, &frame, true);
        let out = labelled(&soup, &labels);
        let ir = &out.ir;
        ir.validate().unwrap();
        assert_eq!(ir.regions.len(), 5);
        let first = cylinder_of(ir, 2);
        let last = cylinder_of(ir, 8 * 39 + 2);
        assert_eq!(first, last);
        assert!((first.2 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn coaxial_snap_fixes_noisy_halves_to_one_cylinder() {
        let frame = Frame::tilted();
        let (mut soup, labels) = washer(10.0, 4.0, 5.0, 40, &frame, true);
        let mut state = 7u64;
        let mut moved: FxHashMap<[u64; 3], V3> = FxHashMap::default();
        for t in soup.triangles.iter_mut() {
            for p in t.iter_mut() {
                let key = p.map(f64::to_bits);
                let d = *moved.entry(key).or_insert_with(|| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                    scale(linalg::unit([u, 1.0 - u, 0.5]), 2e-5 * u)
                });
                *p = add(*p, d);
            }
        }
        let out = convert_soup_from_labels(
            &soup,
            &labels,
            &ConvertOptions {
                linear_tolerance: Some(1e-4),
                ..ConvertOptions::default()
            },
        )
        .unwrap();
        let first = cylinder_of(&out.ir, 2);
        let last = cylinder_of(&out.ir, 8 * 39 + 2);
        assert_eq!(first, last);
        assert!((first.2 - 4.0).abs() < 1e-4);
    }

    #[test]
    fn labels_are_validated() {
        let soup = TriangleSoup {
            triangles: grid_box([0.0; 3], [1.0; 3], 1, false),
        };
        let opts = ConvertOptions::default();
        let good: Vec<u32> = (0..12).map(|i| i / 2).collect();
        assert!(convert_soup_from_labels(&soup, &good, &opts).is_ok());
        let mut reserved = good.clone();
        reserved[3] = u32::MAX;
        let mut sparse = good.clone();
        sparse[0] = 7;
        sparse[1] = 7;
        let mut huge = good.clone();
        huge[0] = 1000;
        for bad in [reserved, sparse, huge, good[..11].to_vec()] {
            assert!(
                matches!(
                    convert_soup_from_labels(&soup, &bad, &opts),
                    Err(ConvertError::InvalidInput(_))
                ),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn labels_are_never_merged() {
        let soup = TriangleSoup {
            triangles: grid_box([0.0; 3], [10.0, 20.0, 30.0], 2, false),
        };
        let labels: Vec<u32> = (0..soup.len() as u32).map(|i| i / 2).collect();
        let out = convert_soup_from_labels(&soup, &labels, &ConvertOptions::default()).unwrap();
        assert_eq!(out.ir.regions.len(), soup.len() / 2);
        assert!(
            out.ir
                .regions
                .iter()
                .all(|r| matches!(r.surface, Surface::Plane { .. }))
        );
        out.ir.validate().unwrap();
    }

    #[test]
    fn unfittable_label_becomes_facets() {
        let (soup, mut labels) = washer(10.0, 4.0, 5.0, 24, &Frame::identity(), false);
        for (l, t) in labels.iter_mut().zip(&soup.triangles) {
            if *l == 2 && t.iter().all(|p| p[0] > 0.0) {
                *l = 0;
            }
        }
        let out = labelled(&soup, &labels);
        out.ir.validate().unwrap();
        assert!(out.ir.regions.iter().any(|r| r.surface.is_facets()));
        assert_eq!(out.report.region_counts.get("cylinder"), Some(&1));
    }
}
