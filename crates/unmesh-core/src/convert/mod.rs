#![allow(clippy::needless_range_loop)]

mod emit;
mod linalg;
mod planes;
mod project;
mod segment;
mod topology;

use rustc_hash::FxHashMap;

use crate::api::{ConvertError, ConvertOptions, ConvertOutput, ConvertWarning, Report};
use crate::ir::{ShellRole, Tolerances};
use crate::mesh::{Point, TriangleSoup};
use crate::weld::weld;

use emit::{Asm, Final};
use linalg::{V3, add, dot, norm, scale, sub};
use planes::{Scratch, build_regions, estimate_noise, merge_coplanar, region_pairs, snap_normals};
use project::{PlaneRef, project_vertex};
use segment::{segment, tri_info};
use topology::{CompKind, NONE, build, point_in_shell, signed_volume};

const MIN_TOLERANCE: f64 = 1e-3;
const NOISE_FACTOR: f64 = 5.0;

pub(crate) struct CompMeta {
    pub closed: bool,
    pub role: ShellRole,
    pub parent: Option<usize>,
}

fn warn(warnings: &mut Vec<ConvertWarning>, code: &str, message: String) {
    warnings.push(ConvertWarning {
        code: code.to_string(),
        message,
    });
}

pub fn convert_soup(
    soup: &TriangleSoup,
    options: &ConvertOptions,
) -> Result<ConvertOutput, ConvertError> {
    if soup.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    let (mesh, src, wrep) =
        weld(soup, options.vertex_merge).map_err(|e| ConvertError::InvalidInput(e.to_string()))?;
    if mesh.faces.is_empty() {
        return Err(ConvertError::EmptyMesh);
    }
    let mut warnings = Vec::new();

    let mut seen: FxHashMap<[u32; 3], ()> = FxHashMap::default();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut fsrc: Vec<u32> = Vec::with_capacity(mesh.faces.len());
    let mut duplicates = 0usize;
    for (f, &s) in mesh.faces.iter().zip(&src) {
        let mut key = *f;
        key.sort_unstable();
        if seen.insert(key, ()).is_some() {
            duplicates += 1;
            continue;
        }
        faces.push(*f);
        fsrc.push(s);
    }
    let dropped = wrep.degenerate_dropped + duplicates;
    if dropped > 0 {
        warn(
            &mut warnings,
            "degenerate_triangles",
            format!("{dropped} degenerate or duplicate triangles were dropped"),
        );
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
    let mut orig: Vec<Point> = mesh.vertices.clone();
    let mut vc: Vec<V3> = orig.iter().map(|p| sub(*p, center)).collect();

    let auto_tol = options.linear_tolerance.is_none();
    let tol = match options.linear_tolerance {
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(_) => {
            return Err(ConvertError::InvalidInput(
                "linear_tolerance must be finite and > 0".to_string(),
            ));
        }
        None => (5e-4 * diag).max(MIN_TOLERANCE),
    };

    let mut topo = build(&mut faces);
    if topo.repaired {
        warn(
            &mut warnings,
            "repaired_winding",
            "inconsistent triangle winding was repaired".to_string(),
        );
    }

    let n_comps = topo.comps.len();
    let volumes: Vec<f64> = topo
        .comps
        .iter()
        .map(|c| {
            if c.kind == CompKind::Closed {
                signed_volume(&vc, &faces, &c.faces)
            } else {
                0.0
            }
        })
        .collect();
    let mut metas: Vec<CompMeta> = Vec::with_capacity(n_comps);
    let mut flip_comps = Vec::new();
    let mut open_edges = 0usize;
    let mut non_manifold = 0usize;
    for (ci, comp) in topo.comps.iter().enumerate() {
        match comp.kind {
            CompKind::OpenEdges => open_edges += 1,
            CompKind::NonManifold => non_manifold += 1,
            CompKind::Closed => {}
        }
        if comp.kind != CompKind::Closed {
            metas.push(CompMeta {
                closed: false,
                role: ShellRole::Outer,
                parent: None,
            });
            continue;
        }
        if volumes[ci] >= 0.0 {
            metas.push(CompMeta {
                closed: true,
                role: ShellRole::Outer,
                parent: None,
            });
            continue;
        }
        let probe = {
            let t = faces[comp.faces[0] as usize];
            scale(
                add(add(vc[t[0] as usize], vc[t[1] as usize]), vc[t[2] as usize]),
                1.0 / 3.0,
            )
        };
        let parent = (0..n_comps)
            .filter(|&o| {
                o != ci
                    && topo.comps[o].kind == CompKind::Closed
                    && volumes[o] > 0.0
                    && point_in_shell(&vc, &faces, &topo.comps[o].faces, probe)
            })
            .min_by(|&a, &b| volumes[a].total_cmp(&volumes[b]));
        match parent {
            Some(p) => metas.push(CompMeta {
                closed: true,
                role: ShellRole::Cavity,
                parent: Some(p),
            }),
            None => {
                flip_comps.push(ci);
                metas.push(CompMeta {
                    closed: true,
                    role: ShellRole::Outer,
                    parent: None,
                });
            }
        }
    }
    if open_edges > 0 {
        warn(
            &mut warnings,
            "open_edges",
            format!("{open_edges} shell(s) have open edges and are kept as facets"),
        );
    }
    if non_manifold > 0 {
        warn(
            &mut warnings,
            "non_manifold_edges",
            format!("{non_manifold} shell(s) have non-manifold edges and are kept as facets"),
        );
    }
    if !flip_comps.is_empty() {
        for &ci in &flip_comps {
            for &f in &topo.comps[ci].faces {
                faces[f as usize].swap(1, 2);
            }
        }
        warn(
            &mut warnings,
            "flipped_winding",
            format!(
                "{} shell(s) had inward-facing winding and were flipped",
                flip_comps.len()
            ),
        );
        topo = build(&mut faces);
    }

    let mut claimed: Vec<Option<usize>> = vec![None; vc.len()];
    let mut dup: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    for (ci, comp) in topo.comps.iter().enumerate() {
        for &f in &comp.faces {
            for slot in faces[f as usize].iter_mut() {
                let v = *slot;
                match claimed[v as usize] {
                    None => claimed[v as usize] = Some(ci),
                    Some(c) if c == ci => {}
                    Some(_) => {
                        let id = *dup.entry((v, ci)).or_insert_with(|| {
                            vc.push(vc[v as usize]);
                            orig.push(orig[v as usize]);
                            (vc.len() - 1) as u32
                        });
                        *slot = id;
                    }
                }
            }
        }
    }
    if !dup.is_empty() {
        topo = build(&mut faces);
    }

    let nf = faces.len();
    let comp_of: Vec<u32> = {
        let mut v = vec![0u32; nf];
        for (ci, c) in topo.comps.iter().enumerate() {
            for &f in &c.faces {
                v[f as usize] = ci as u32;
            }
        }
        v
    };
    let eligible: Vec<bool> = comp_of
        .iter()
        .map(|&c| topo.comps[c as usize].kind == CompKind::Closed)
        .collect();

    let info = tri_info(&vc, &faces);
    let mut scratch = Scratch::new(vc.len());
    let mut stage = |tol: f64| -> (Vec<u32>, Vec<planes::Region>) {
        let (label, n_seg) = segment(&vc, &faces, &topo.nbr, &info, &eligible, tol);
        let regions0 = build_regions(&vc, &faces, &info, &label, n_seg, tol, &mut scratch);
        let pairs0 = region_pairs(&topo.nbr, &label);
        let root = merge_coplanar(&vc, &regions0, &pairs0, tol, options.angular_snap_deg);
        let mut compact = vec![NONE; n_seg];
        let mut n_merged = 0u32;
        for r in 0..n_seg {
            if root[r] as usize == r {
                compact[r] = n_merged;
                n_merged += 1;
            }
        }
        let label2: Vec<u32> = label
            .iter()
            .map(|&l| {
                if l == NONE {
                    NONE
                } else {
                    compact[root[l as usize] as usize]
                }
            })
            .collect();
        let regions = build_regions(
            &vc,
            &faces,
            &info,
            &label2,
            n_merged as usize,
            tol,
            &mut scratch,
        );
        (label2, regions)
    };
    let (mut label2, mut regions) = stage(tol);
    let mut tol = tol;
    let mut sigma = tol / NOISE_FACTOR;
    if auto_tol {
        sigma = estimate_noise(&vc, &regions);
        let derived = (NOISE_FACTOR * sigma).max(MIN_TOLERANCE);
        if derived < tol {
            tol = derived;
            (label2, regions) = stage(tol);
        }
    }
    snap_normals(&vc, &mut regions, tol, sigma, options.angular_snap_deg);

    let pairs = region_pairs(&topo.nbr, &label2);
    let is_plane: Vec<bool> = regions
        .iter()
        .map(|r| r.area > 0.0 && r.max <= tol)
        .collect();
    let mut fdsu: Vec<u32> = (0..regions.len() as u32).collect();
    fn find(d: &mut [u32], mut x: u32) -> u32 {
        while d[x as usize] != x {
            d[x as usize] = d[d[x as usize] as usize];
            x = d[x as usize];
        }
        x
    }
    for &(a, b) in &pairs {
        if !is_plane[a as usize] && !is_plane[b as usize] {
            let (ra, rb) = (find(&mut fdsu, a), find(&mut fdsu, b));
            if ra != rb {
                fdsu[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    let mut finals: Vec<Final> = Vec::new();
    let mut group_of: FxHashMap<u32, usize> = FxHashMap::default();
    for (ri, r) in regions.iter().enumerate() {
        if r.faces.is_empty() {
            continue;
        }
        if is_plane[ri] {
            finals.push(Final {
                faces: r.faces.clone(),
                plane: Some((r.n, r.d)),
                rms: r.rms,
                max: r.max,
                comp: comp_of[r.faces[0] as usize] as usize,
            });
        } else {
            let g = find(&mut fdsu, ri as u32);
            let idx = *group_of.entry(g).or_insert_with(|| {
                finals.push(Final {
                    faces: Vec::new(),
                    plane: None,
                    rms: 0.0,
                    max: 0.0,
                    comp: comp_of[r.faces[0] as usize] as usize,
                });
                finals.len() - 1
            });
            finals[idx].faces.extend_from_slice(&r.faces);
        }
    }
    for (ci, c) in topo.comps.iter().enumerate() {
        if c.kind != CompKind::Closed {
            finals.push(Final {
                faces: c.faces.clone(),
                plane: None,
                rms: 0.0,
                max: 0.0,
                comp: ci,
            });
        }
    }
    for fr in finals.iter_mut() {
        fr.faces.sort_unstable();
    }
    finals.sort_by_key(|f| f.faces[0]);

    let mut flabel = vec![NONE; nf];
    for (i, fr) in finals.iter().enumerate() {
        for &f in &fr.faces {
            flabel[f as usize] = i as u32;
        }
    }

    let mut vr: Vec<u64> = Vec::with_capacity(nf * 3);
    for (f, t) in faces.iter().enumerate() {
        for &v in t {
            vr.push(((v as u64) << 32) | flabel[f] as u64);
        }
    }
    vr.sort_unstable();
    vr.dedup();

    let areas: Vec<f64> = finals
        .iter()
        .map(|fr| fr.faces.iter().map(|&f| info[f as usize].area).sum())
        .collect();
    let mut pv = vc.clone();
    let mut dev_sum2 = 0.0;
    let mut dev_max: f64 = 0.0;
    let mut dev_count = 0usize;
    let mut i = 0;
    while i < vr.len() {
        let v = (vr[i] >> 32) as usize;
        let mut j = i;
        let mut planes: Vec<PlaneRef> = Vec::new();
        let mut plane_idx: Vec<usize> = Vec::new();
        while j < vr.len() && (vr[j] >> 32) as usize == v {
            let r = (vr[j] & 0xffff_ffff) as usize;
            if let Some((n, d)) = finals[r].plane {
                planes.push(PlaneRef {
                    n,
                    d,
                    weight: areas[r],
                });
                plane_idx.push(r);
            }
            j += 1;
        }
        if !planes.is_empty() {
            let x = project_vertex(vc[v], &planes, tol);
            pv[v] = x;
            let disp = norm(sub(x, vc[v]));
            let off = planes
                .iter()
                .map(|p| (dot(p.n, x) - p.d).abs())
                .fold(0.0, f64::max);
            let dev = disp + off;
            dev_max = dev_max.max(dev);
            dev_sum2 += dev * dev;
            dev_count += 1;
        }
        i = j;
    }
    for fr in &finals {
        dev_max = dev_max.max(if fr.plane.is_some() { fr.max } else { 0.0 });
    }
    let slack = 8.0 * f64::EPSILON * (diag + norm(center));
    let report_dev = if dev_count == 0 { 0.0 } else { dev_max + slack };
    let report_rms = if dev_count == 0 {
        0.0
    } else {
        (dev_sum2 / dev_count as f64).sqrt()
    };

    let total_area: f64 = info.iter().map(|t| t.area).sum();
    let plane_area: f64 = finals
        .iter()
        .zip(&areas)
        .filter(|(f, _)| f.plane.is_some())
        .map(|(_, a)| a)
        .sum();
    let mut region_counts = std::collections::BTreeMap::new();
    for fr in &finals {
        let key = if fr.plane.is_some() {
            "plane"
        } else {
            "facets"
        };
        *region_counts.entry(key.to_string()).or_insert(0u32) += 1;
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
                if *p == vc[i] {
                    orig[i]
                } else {
                    add(*p, center)
                }
            })
            .collect()
    };
    let asm = Asm {
        tolerances,
        source_triangles: soup.len() as u32,
        center,
        source_vertices: wrep.unique_vertices as u32,
        faces: &faces,
        fsrc: &fsrc,
        nbr: &topo.nbr,
        metas: &metas,
        orig: &orig,
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
            warn(
                &mut warnings,
                "fallback_facets",
                format!(
                    "analytic assembly failed validation ({}); every shell is kept as facets",
                    errors.join("; ")
                ),
            );
            let fb: Vec<Final> = topo
                .comps
                .iter()
                .enumerate()
                .map(|(ci, c)| Final {
                    faces: c.faces.clone(),
                    plane: None,
                    rms: 0.0,
                    max: 0.0,
                    comp: ci,
                })
                .collect();
            let mut fl = vec![NONE; nf];
            for (i, fr) in fb.iter().enumerate() {
                for &f in &fr.faces {
                    fl[f as usize] = i as u32;
                }
            }
            let orig_pos = positions(&vc);
            let ir = emit::assemble(&asm, &fb, &fl, &orig_pos)
                .map_err(|e| ConvertError::InvalidInput(e.join("; ")))?;
            let mut counts = std::collections::BTreeMap::new();
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
    use super::*;
    use crate::ir::{Kind, Surface, VertexRole};

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
