use rustc_hash::FxHashMap;

use super::dsu::Dsu;
use super::linalg::{V3, dot, scale, sub, sym_eigen, unit};
use super::segment::{self, TriInfo};
use super::surface::Surface;
use super::topology::{CompKind, NONE, Topology};

const RESPLIT_MIN_DIHEDRAL_DEG: f64 = 15.0;

fn has_crease(
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    label: &[u32],
    region: u32,
    faces: &[u32],
) -> bool {
    let limit = RESPLIT_MIN_DIHEDRAL_DEG.to_radians().cos();
    for &f in faces {
        let nb = &nbr[f as usize];
        if info[f as usize].area <= 0.0 {
            continue;
        }
        for &g in nb {
            if g == NONE || label[g as usize] != region || info[g as usize].area <= 0.0 {
                continue;
            }
            if dot(info[f as usize].normal, info[g as usize].normal) < limit {
                return true;
            }
        }
    }
    false
}

#[derive(Clone)]
pub struct Region {
    pub faces: Vec<u32>,
    pub surface: Surface,
    pub area: f64,
    pub width: f64,
    pub verts: Vec<(u32, f64)>,
    pub rms: f64,
    pub max: f64,
    pub sag: f64,
}

pub struct Scratch {
    mark: Vec<u32>,
    pos: Vec<u32>,
    token: u32,
}

impl Scratch {
    pub fn new(n_verts: usize) -> Self {
        Self {
            mark: vec![0; n_verts],
            pos: vec![0; n_verts],
            token: 0,
        }
    }
}

pub fn collect_verts(
    f: &[[u32; 3]],
    info: &[TriInfo],
    faces: &[u32],
    scratch: &mut Scratch,
) -> Vec<(u32, f64)> {
    scratch.token += 1;
    let token = scratch.token;
    let mut list: Vec<(u32, f64)> = Vec::new();
    for &fi in faces {
        let w = info[fi as usize].area / 3.0;
        for &vi in &f[fi as usize] {
            let vi = vi as usize;
            if scratch.mark[vi] == token {
                list[scratch.pos[vi] as usize].1 += w;
            } else {
                scratch.mark[vi] = token;
                scratch.pos[vi] = list.len() as u32;
                list.push((vi as u32, w));
            }
        }
    }
    list
}

fn huber(r: f64, k: f64) -> f64 {
    let r = r.abs();
    if r <= k { 1.0 } else { k / r }
}

pub fn irls_plane(v: &[V3], list: &[(u32, f64)], init: V3, tol: f64) -> (V3, f64, f64) {
    let k = 0.25 * tol;
    let mut n = init;
    let mut weights: Vec<f64> = list.iter().map(|x| x.1).collect();
    let mut c = [0.0; 3];
    let mut width = 0.0;
    for _ in 0..6 {
        let sw: f64 = weights.iter().sum();
        if sw <= 0.0 {
            break;
        }
        c = [0.0; 3];
        for (i, &(vi, _)) in list.iter().enumerate() {
            let p = v[vi as usize];
            for a in 0..3 {
                c[a] += weights[i] * p[a];
            }
        }
        c = scale(c, 1.0 / sw);
        let mut cov = [[0.0; 3]; 3];
        for (i, &(vi, _)) in list.iter().enumerate() {
            let q = sub(v[vi as usize], c);
            super::linalg::outer_add(&mut cov, q, q, weights[i] / sw);
        }
        let (vals, vecs) = sym_eigen(cov);
        if vals[1] > 1e-24 {
            let mut nn = vecs[0];
            if dot(nn, n) < 0.0 {
                nn = scale(nn, -1.0);
            }
            n = nn;
        }
        width = (12.0 * vals[1].max(0.0)).sqrt();
        let mut changed = false;
        for (i, &(vi, w)) in list.iter().enumerate() {
            let nw = w * huber(dot(n, sub(v[vi as usize], c)), k);
            changed |= nw != weights[i];
            weights[i] = nw;
        }
        if !changed {
            break;
        }
    }
    (n, dot(n, c), width)
}

pub fn offset_fit(v: &[V3], list: &[(u32, f64)], n: V3, tol: f64) -> f64 {
    let k = 0.25 * tol;
    let mut weights: Vec<f64> = list.iter().map(|x| x.1).collect();
    let mut d = 0.0;
    for _ in 0..5 {
        let sw: f64 = weights.iter().sum();
        if sw <= 0.0 {
            break;
        }
        let mut c = [0.0; 3];
        for (i, &(vi, _)) in list.iter().enumerate() {
            let p = v[vi as usize];
            for a in 0..3 {
                c[a] += weights[i] * p[a];
            }
        }
        d = dot(n, scale(c, 1.0 / sw));
        for (i, &(vi, w)) in list.iter().enumerate() {
            weights[i] = w * huber(dot(n, v[vi as usize]) - d, k);
        }
    }
    d
}

pub fn residual(v: &[V3], list: &[(u32, f64)], n: V3, d: f64) -> (f64, f64) {
    let mut sw = 0.0;
    let mut s2 = 0.0;
    let mut mx: f64 = 0.0;
    for &(vi, w) in list {
        let r = dot(n, v[vi as usize]) - d;
        sw += w;
        s2 += w * r * r;
        mx = mx.max(r.abs());
    }
    ((s2 / sw.max(f64::MIN_POSITIVE)).sqrt(), mx)
}

pub fn build_regions(
    v: &[V3],
    f: &[[u32; 3]],
    info: &[TriInfo],
    label: &[u32],
    n_regions: usize,
    tol: f64,
    scratch: &mut Scratch,
) -> Vec<Region> {
    let mut faces: Vec<Vec<u32>> = vec![Vec::new(); n_regions];
    for (fi, &l) in label.iter().enumerate() {
        if l != NONE {
            faces[l as usize].push(fi as u32);
        }
    }
    faces
        .into_iter()
        .map(|fl| build_region(v, f, info, fl, tol, scratch))
        .collect()
}

pub fn build_region(
    v: &[V3],
    f: &[[u32; 3]],
    info: &[TriInfo],
    fl: Vec<u32>,
    tol: f64,
    scratch: &mut Scratch,
) -> Region {
    let verts = collect_verts(f, info, &fl, scratch);
    let mut nsum = [0.0; 3];
    let mut area = 0.0;
    for &fi in &fl {
        let t = &info[fi as usize];
        for a in 0..3 {
            nsum[a] += t.normal[a] * t.area;
        }
        area += t.area;
    }
    let init = unit(nsum);
    let (n, d, width) = if area > 0.0 {
        irls_plane(v, &verts, init, tol)
    } else {
        ([0.0; 3], 0.0, 0.0)
    };
    let (rms, max) = if area > 0.0 {
        residual(v, &verts, n, d)
    } else {
        (0.0, f64::INFINITY)
    };
    Region {
        faces: fl,
        surface: Surface::Plane {
            normal: n,
            offset: d,
        },
        area,
        width,
        verts,
        rms,
        max,
        sag: 0.0,
    }
}

pub fn region_pairs(f_nbr: &[[u32; 3]], label: &[u32]) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for (fi, nb) in f_nbr.iter().enumerate() {
        let a = label[fi];
        if a == NONE {
            continue;
        }
        for &g in nb {
            if g == NONE {
                continue;
            }
            let b = label[g as usize];
            if b != NONE && b != a && a < b {
                pairs.push((a, b));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

struct Comp {
    verts: Vec<(u32, f64)>,
    n: V3,
    width: f64,
    area: f64,
}

fn union_verts(a: &[(u32, f64)], b: &[(u32, f64)]) -> Vec<(u32, f64)> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push((a[i].0, a[i].1 + b[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

pub fn merge_coplanar(
    v: &[V3],
    regions: &[Region],
    pairs: &[(u32, u32)],
    tol: f64,
    snap_deg: f64,
) -> Vec<u32> {
    let n = regions.len();
    let mut dsu = Dsu::new(n);
    let mut comps: Vec<Comp> = regions
        .iter()
        .map(|r| {
            let mut verts = r.verts.clone();
            verts.sort_unstable_by_key(|x| x.0);
            Comp {
                verts,
                n: r.surface.as_plane().map(|(n, _)| n).unwrap_or([0.0; 3]),
                width: r.width,
                area: r.area,
            }
        })
        .collect();
    let snap = snap_deg.to_radians();
    for _ in 0..4 {
        let mut changed = false;
        for &(a, b) in pairs {
            let (ra, rb) = (dsu.find(a), dsu.find(b));
            if ra == rb {
                continue;
            }
            let (ca, cb) = (&comps[ra as usize], &comps[rb as usize]);
            if ca.area <= 0.0 || cb.area <= 0.0 {
                continue;
            }
            let allow = snap
                + (2.0 * tol).atan2(ca.width.max(1e-300))
                + (2.0 * tol).atan2(cb.width.max(1e-300));
            let cosang = dot(ca.n, cb.n);
            if cosang <= 0.0 || (allow < std::f64::consts::FRAC_PI_2 && cosang < allow.cos()) {
                continue;
            }
            let (big, small) = if ca.area >= cb.area {
                (ra, rb)
            } else {
                (rb, ra)
            };
            let (cbig, csmall) = (&comps[big as usize], &comps[small as usize]);
            let verts = union_verts(&cbig.verts, &csmall.verts);
            let (nn, d, width) = irls_plane(v, &verts, cbig.n, tol);
            let (_, max) = residual(v, &verts, nn, d);
            if max > tol {
                continue;
            }
            let area = cbig.area + csmall.area;
            dsu.union_into(small, big);
            comps[big as usize] = Comp {
                verts,
                n: nn,
                width,
                area,
            };
            comps[small as usize].verts = Vec::new();
            changed = true;
        }
        if !changed {
            break;
        }
    }
    (0..n as u32).map(|i| dsu.find(i)).collect()
}

pub struct Final {
    pub faces: Vec<u32>,
    pub surface: Surface,
    pub rms: f64,
    pub max: f64,
    pub sag: f64,
    pub comp: usize,
}

pub struct FitArgs<'a> {
    pub vc: &'a [V3],
    pub faces: &'a [[u32; 3]],
    pub nbr: &'a [[u32; 3]],
    pub info: &'a [TriInfo],
    pub eligible: &'a [bool],
    pub tol: f64,
    pub sigma: f64,
    pub noisy: bool,
    pub snap_deg: f64,
    pub scratch: &'a mut Scratch,
}

pub fn fit_regions(args: FitArgs<'_>) -> (Vec<u32>, Vec<Region>) {
    let FitArgs {
        vc,
        faces,
        nbr,
        info,
        eligible,
        tol,
        sigma,
        noisy,
        snap_deg,
        scratch,
    } = args;
    let (label, n_seg) = segment::run(vc, faces, nbr, info, eligible, tol, false);
    super::timing::lap("segment");
    let (label2, regions) =
        merge_and_rebuild(vc, faces, info, nbr, &label, n_seg, tol, snap_deg, scratch);
    super::timing::lap("merge");
    let (label3, regions) = match resplit_loose(vc, faces, nbr, info, &label2, &regions, tol) {
        Some((combined, n_all)) => merge_and_rebuild(
            vc, faces, info, nbr, &combined, n_all, tol, snap_deg, scratch,
        ),
        None => (label2, regions),
    };
    super::timing::lap("resplit");
    let grown = super::grow::run(vc, faces, nbr, info, &label3, regions, tol, sigma, noisy);
    super::timing::lap("grow");
    let mut regions = grown.regions;
    super::curved::refine_regions_with(
        vc,
        faces,
        info,
        &mut regions,
        tol,
        snap_deg,
        grown.prefit,
        false,
    );
    super::timing::lap("curved");
    let mut label = grown.label;
    absorb_remnants(vc, faces, nbr, info, &mut label, &mut regions, tol);
    super::timing::lap("absorb");
    (label, regions)
}

const REMNANT_FACES: usize = 2;

fn curved(s: &Surface) -> bool {
    s.is_analytic() && s.as_plane().is_none()
}

/// Gives a remnant of at most `REMNANT_FACES` triangles that is not itself
/// curved to an adjacent curved region whose surface passes within `tol` of
/// every one of its vertices and whose largest chord sagitta, plus the
/// remnant's largest vertex distance (noise lifts a chord's sagitta by up to
/// that much), already bounds the remnant's: the long sliver triangles a tessellator leaves where a
/// curved face meets another, which growth refused. A flat cut into the
/// surface (a D-flat) has its corners on it too, but its sagitta is the flat's
/// depth, far beyond the tessellation's, so it stays a plane. The curved
/// surface is kept; its residual takes the remnant's vertex distances.
pub fn absorb_remnants(
    vc: &[V3],
    faces: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    label: &mut [u32],
    regions: &mut [Region],
    tol: f64,
) {
    loop {
        let mut changed = false;
        for r in 0..regions.len() {
            let n = regions[r].faces.len();
            if n == 0 || n > REMNANT_FACES || curved(&regions[r].surface) {
                continue;
            }
            let mut cands: Vec<u32> = regions[r]
                .faces
                .iter()
                .flat_map(|&f| nbr[f as usize])
                .filter(|&g| g != NONE)
                .map(|g| label[g as usize])
                .filter(|&l| l != NONE && l as usize != r)
                .filter(|&l| {
                    let c = &regions[l as usize];
                    curved(&c.surface) && c.max <= tol
                })
                .collect();
            cands.sort_unstable();
            cands.dedup();
            let best = cands
                .into_iter()
                .filter_map(|c| {
                    let target = &regions[c as usize];
                    let s = &target.surface;
                    let worst = regions[r]
                        .faces
                        .iter()
                        .flat_map(|&f| faces[f as usize])
                        .map(|v| s.distance(vc[v as usize]).abs())
                        .fold(0.0, f64::max);
                    let sag = regions[r]
                        .faces
                        .iter()
                        .map(|&f| {
                            super::curved::sagitta(s, faces[f as usize].map(|v| vc[v as usize]))
                        })
                        .fold(0.0, f64::max);
                    (worst <= tol && sag <= target.sag.max(tol) + worst).then_some((worst, sag, c))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
            let Some((worst, sag, c)) = best else {
                continue;
            };
            let moved = std::mem::take(&mut regions[r].faces);
            let area: f64 = moved.iter().map(|&f| info[f as usize].area).sum();
            let target = &mut regions[c as usize];
            for &f in &moved {
                label[f as usize] = c;
            }
            target.faces.extend_from_slice(&moved);
            target.max = target.max.max(worst);
            target.sag = target.sag.max(sag);
            target.area += area;
            regions[r].area = 0.0;
            changed = true;
        }
        if !changed {
            break;
        }
    }
}

pub struct RunArgs<'a> {
    pub vc: &'a [V3],
    pub faces: &'a [[u32; 3]],
    pub info: &'a [TriInfo],
    pub label: &'a [u32],
    pub n_seg: usize,
    pub tol: f64,
    pub snap_deg: f64,
    pub scratch: &'a mut Scratch,
}

pub fn run(args: RunArgs<'_>) -> (Vec<u32>, Vec<Region>) {
    let RunArgs {
        vc,
        faces,
        info,
        label,
        n_seg,
        tol,
        snap_deg,
        scratch,
    } = args;
    let mut regions = build_regions(vc, faces, info, label, n_seg, tol, scratch);
    super::curved::refine_regions(vc, faces, info, &mut regions, tol, snap_deg);
    (label.to_vec(), regions)
}

pub fn resplit_loose(
    vc: &[V3],
    faces: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    label: &[u32],
    regions: &[Region],
    tol: f64,
) -> Option<(Vec<u32>, usize)> {
    let loose: Vec<u32> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.area > 0.0 && r.max > tol)
        .map(|(i, _)| i as u32)
        .collect();
    if loose.is_empty() {
        return None;
    }
    let mut combined = vec![NONE; faces.len()];
    let mut next_id = 0u32;
    for r in regions {
        if r.area > 0.0 && r.max > tol {
            continue;
        }
        for &f in &r.faces {
            combined[f as usize] = next_id;
        }
        next_id += 1;
    }
    let mut creased = vec![false; regions.len()];
    for &r in &loose {
        let rf = &regions[r as usize].faces;
        creased[r as usize] = has_crease(nbr, info, label, r, rf);
        if !creased[r as usize] {
            for &f in rf {
                combined[f as usize] = next_id;
            }
            next_id += 1;
        }
    }
    if !creased.iter().any(|&c| c) {
        return None;
    }
    let eligible: Vec<bool> = label
        .iter()
        .map(|&l| l != NONE && creased[l as usize])
        .collect();
    let (sub, n_sub) = segment::run_fenced(vc, faces, nbr, info, &eligible, Some(label), tol, true);
    for (f, &l) in sub.iter().enumerate() {
        if l != NONE {
            combined[f] = next_id + l;
        }
    }
    next_id += n_sub as u32;
    Some((combined, next_id as usize))
}

#[allow(clippy::too_many_arguments)]
fn merge_and_rebuild(
    vc: &[V3],
    faces: &[[u32; 3]],
    info: &[TriInfo],
    nbr: &[[u32; 3]],
    label: &[u32],
    n_seg: usize,
    tol: f64,
    snap_deg: f64,
    scratch: &mut Scratch,
) -> (Vec<u32>, Vec<Region>) {
    let regions0 = build_regions(vc, faces, info, label, n_seg, tol, scratch);
    let pairs0 = region_pairs(nbr, label);
    let root = merge_coplanar(vc, &regions0, &pairs0, tol, snap_deg);
    let mut compact = vec![NONE; n_seg];
    let mut members = Vec::new();
    for r in 0..n_seg {
        if root[r] as usize == r {
            compact[r] = members.len() as u32;
            members.push(0u32);
        }
    }
    for r in 0..n_seg {
        members[compact[root[r] as usize] as usize] += 1;
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
    let mut merged_faces: Vec<Vec<u32>> = vec![Vec::new(); members.len()];
    for (fi, &l) in label2.iter().enumerate() {
        if l != NONE && members[l as usize] > 1 {
            merged_faces[l as usize].push(fi as u32);
        }
    }
    let mut regions: Vec<Option<Region>> = (0..members.len()).map(|_| None).collect();
    for (r, reg) in regions0.into_iter().enumerate() {
        let id = compact[root[r] as usize] as usize;
        if members[id] == 1 {
            regions[id] = Some(reg);
        }
    }
    let regions = regions
        .into_iter()
        .zip(merged_faces)
        .map(|(reg, fl)| reg.unwrap_or_else(|| build_region(vc, faces, info, fl, tol, scratch)))
        .collect();
    (label2, regions)
}

pub fn finalize(
    regions: &[Region],
    pairs: &[(u32, u32)],
    comp_of: &[u32],
    topo: &Topology,
    tol: f64,
) -> (Vec<Final>, Vec<u32>) {
    let is_analytic: Vec<bool> = regions
        .iter()
        .map(|r| r.area > 0.0 && r.max <= tol)
        .collect();
    let mut fdsu = Dsu::new(regions.len());
    for &(a, b) in pairs {
        if !is_analytic[a as usize] && !is_analytic[b as usize] {
            fdsu.union(a, b);
        }
    }
    let mut finals: Vec<Final> = Vec::new();
    let mut group_of: FxHashMap<u32, usize> = FxHashMap::default();
    for (ri, r) in regions.iter().enumerate() {
        if r.faces.is_empty() {
            continue;
        }
        if is_analytic[ri] {
            finals.push(Final {
                faces: r.faces.clone(),
                surface: r.surface,
                rms: r.rms,
                max: r.max,
                sag: r.sag,
                comp: comp_of[r.faces[0] as usize] as usize,
            });
        } else {
            let g = fdsu.find(ri as u32);
            let idx = *group_of.entry(g).or_insert_with(|| {
                finals.push(Final {
                    faces: Vec::new(),
                    surface: Surface::Facets,
                    rms: 0.0,
                    max: 0.0,
                    sag: 0.0,
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
                surface: Surface::Facets,
                rms: 0.0,
                max: 0.0,
                sag: 0.0,
                comp: ci,
            });
        }
    }
    for fr in finals.iter_mut() {
        fr.faces.sort_unstable();
    }
    finals.sort_by_key(|f| f.faces[0]);

    let flabel = label_faces(&finals, comp_of.len());
    (finals, flabel)
}

pub fn label_faces(finals: &[Final], n_faces: usize) -> Vec<u32> {
    let mut flabel = vec![NONE; n_faces];
    for (i, fr) in finals.iter().enumerate() {
        for &f in &fr.faces {
            flabel[f as usize] = i as u32;
        }
    }
    flabel
}

#[cfg(test)]
mod tests {
    use super::super::segment::tri_info;
    use super::*;

    /// A strip of a radius-10 cylinder between z = 0 and z = 1, in steps of
    /// `step` radians, then one more triangle (the remnant) spanning `span`
    /// radians from the strip's last ring, every corner on the cylinder.
    fn strip_with_remnant(step: f64, span: f64) -> (bool, f64) {
        let ring = |a: f64, z: f64| [10.0 * a.cos(), 10.0 * a.sin(), z];
        let n = 8;
        let mut v = Vec::new();
        for i in 0..=n {
            let a = step * i as f64;
            v.push(ring(a, 0.0));
            v.push(ring(a, 1.0));
        }
        let mut f: Vec<[u32; 3]> = Vec::new();
        for i in 0..n as u32 {
            let (b0, t0, b1, t1) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
            f.push([b0, b1, t1]);
            f.push([b0, t1, t0]);
        }
        let last = 2 * n as u32;
        v.push(ring(step * n as f64 + span, 0.5));
        f.push([last, v.len() as u32 - 1, last + 1]);
        let info = tri_info(&v, &f);
        let mut nbr = vec![[NONE; 3]; f.len()];
        let rem = f.len() - 1;
        nbr[rem][2] = rem as u32 - 1;
        nbr[rem - 1][1] = rem as u32;
        let surface = Surface::Cylinder {
            origin: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            radius: 10.0,
            reversed: false,
        };
        let sag = 10.0 * (1.0 - (step / 2.0).cos());
        let tol = 1e-4;
        let mut regions = vec![
            Region {
                faces: (0..rem as u32).collect(),
                surface,
                area: 1.0,
                width: 1.0,
                verts: Vec::new(),
                rms: 0.0,
                max: 0.0,
                sag,
            },
            Region {
                faces: vec![rem as u32],
                surface: Surface::Plane {
                    normal: info[rem].normal,
                    offset: dot(info[rem].normal, v[last as usize]),
                },
                area: info[rem].area,
                width: 0.1,
                verts: Vec::new(),
                rms: 0.0,
                max: 0.0,
                sag: 0.0,
            },
        ];
        let mut label: Vec<u32> = (0..f.len()).map(|i| (i == rem) as u32).collect();
        absorb_remnants(&v, &f, &nbr, &info, &mut label, &mut regions, tol);
        (label[rem] == 0, regions[0].sag)
    }

    #[test]
    fn a_sliver_within_the_tessellation_sagitta_is_absorbed() {
        let (absorbed, sag) = strip_with_remnant(0.1, 0.05);
        assert!(absorbed);
        assert!(sag >= 10.0 * (1.0 - 0.025f64.cos()) - 1e-12);
    }

    #[test]
    fn a_d_flat_deeper_than_the_tessellation_sagitta_stays_a_plane() {
        assert!(!strip_with_remnant(0.1, 0.6).0);
        assert!(strip_with_remnant(0.1, 0.1).0);
    }

    fn hinge(tilt_deg: f64, base: u32) -> (Vec<V3>, Vec<[u32; 3]>, Vec<[u32; 3]>) {
        let t = tilt_deg.to_radians().tan();
        let v = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0 * t],
            [3.0, 0.0, 3.0 * t],
        ];
        let b = base as usize;
        let f = vec![
            [b as u32, b as u32 + 1, b as u32 + 2],
            [b as u32 + 3, b as u32 + 5, b as u32 + 4],
        ];
        let nbr = vec![[1, NONE, NONE], [0, NONE, NONE]];
        (v, f, nbr)
    }

    #[test]
    fn union_resplit_splits_every_creased_loose_region() {
        let (v0, f0, n0) = hinge(20.0, 0);
        let (v1, f1, n1) = hinge(20.0, 6);
        let mut v = v0;
        v.extend(v1);
        let mut f = f0;
        f.extend(f1);
        let mut nbr = n0;
        let off2 = nbr.len() as u32;
        nbr.extend(
            n1.into_iter()
                .map(|t| t.map(|g| if g == NONE { NONE } else { g + off2 })),
        );
        let info = tri_info(&v, &f);
        let label = vec![0, 0, 1, 1];
        let mut scratch = Scratch::new(v.len());
        let regions = build_regions(&v, &f, &info, &label, 2, 0.03, &mut scratch);
        assert!(regions.iter().all(|r| r.max > 0.03));
        let (combined, n_all) = resplit_loose(&v, &f, &nbr, &info, &label, &regions, 0.03).unwrap();
        assert_eq!(n_all, 4);
        let (label2, regions) = merge_and_rebuild(
            &v,
            &f,
            &info,
            &nbr,
            &combined,
            n_all,
            0.03,
            0.5,
            &mut scratch,
        );
        let mut distinct = label2.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 4);
        assert_eq!(regions.len(), 4);
        assert!(regions.iter().all(|r| r.max <= 0.03));
        for (fi, &l) in label2.iter().enumerate() {
            assert!(regions[l as usize].faces.contains(&(fi as u32)));
        }
    }
}
