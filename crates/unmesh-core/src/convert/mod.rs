#![allow(clippy::needless_range_loop)]

mod adjacency;
mod dsu;
mod emit;
mod fit;
mod linalg;
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
use topology::NONE;

const MIN_TOLERANCE: f64 = 1e-3;
const NOISE_FACTOR: f64 = 5.0;

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

#[allow(dead_code)]
pub(crate) fn convert_from_labels(
    soup: &TriangleSoup,
    options: &ConvertOptions,
    source_labels: &[u32],
) -> Result<ConvertOutput, ConvertError> {
    if source_labels.len() != soup.len() {
        return Err(ConvertError::InvalidInput(format!(
            "label count {} does not match triangle count {}",
            source_labels.len(),
            soup.len()
        )));
    }
    let (w, shells, info, tol, auto_tol, warnings) = prepare(soup, options)?;
    let face_labels: Vec<u32> = w
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
    let n_seg = face_labels
        .iter()
        .filter(|&&l| l != NONE)
        .max()
        .map(|m| *m as usize + 1)
        .unwrap_or(0);
    let mut scratch = fit::Scratch::new(w.vc.len());
    let mut fit_once = |tol: f64| {
        fit::run(fit::RunArgs {
            vc: &w.vc,
            faces: &w.faces,
            nbr: &shells.topo.nbr,
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
        None => (5e-4 * w.diag).max(MIN_TOLERANCE),
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
    let (mut label2, mut regions) = fit_once(tol);
    let mut sigma = tol / NOISE_FACTOR;
    if auto_tol {
        sigma = snap::estimate_noise(&w.vc, &regions);
        let derived = (NOISE_FACTOR * sigma).max(MIN_TOLERANCE);
        if derived < tol {
            tol = derived;
            (label2, regions) = fit_once(tol);
        }
    }
    snap::snap_normals(&w.vc, &mut regions, tol, sigma, options.angular_snap_deg);

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
    let (ir, report_dev, report_rms, region_counts, area_fraction) = match result {
        Ok(ir) => (
            ir,
            report_dev,
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
                    comp: ci,
                })
                .collect();
            let fl = label_faces(&fb, w.faces.len());
            let orig_pos = positions(&w.vc);
            let ir = emit::assemble(&asm, &fb, &fl, &orig_pos)
                .map_err(|e| ConvertError::InvalidInput(e.join("; ")))?;
            let mut counts = BTreeMap::new();
            counts.insert("facets".to_string(), fb.len() as u32);
            (ir, 0.0, 0.0, counts, 0.0)
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

    use super::linalg::{dot, scale, sub};
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
        let b = convert_from_labels(&soup, &ConvertOptions::default(), &labels).unwrap();
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
}
