use rustc_hash::FxHashMap;

use super::linalg::{V3, dot, scale, sub, sym_eigen, unit};
use super::segment::{TriInfo, segment};
use super::surface::Surface;
use super::topology::{CompKind, NONE, Topology};

pub struct Region {
    pub faces: Vec<u32>,
    pub n: V3,
    pub d: f64,
    pub area: f64,
    pub width: f64,
    pub verts: Vec<(u32, f64)>,
    pub rms: f64,
    pub max: f64,
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
        for (i, &(vi, w)) in list.iter().enumerate() {
            weights[i] = w * huber(dot(n, sub(v[vi as usize], c)), k);
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
        .map(|fl| {
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
                n,
                d,
                area,
                width,
                verts,
                rms,
                max,
            }
        })
        .collect()
}

struct Dsu(Vec<u32>);

impl Dsu {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[x as usize];
            self.0[x as usize] = self.0[p as usize];
            x = self.0[x as usize];
        }
        x
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
    let mut dsu = Dsu((0..n as u32).collect());
    let mut comps: Vec<Comp> = regions
        .iter()
        .map(|r| {
            let mut verts = r.verts.clone();
            verts.sort_unstable_by_key(|x| x.0);
            Comp {
                verts,
                n: r.n,
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
            dsu.0[small as usize] = big;
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
    pub comp: usize,
}

pub struct FitArgs<'a> {
    pub vc: &'a [V3],
    pub faces: &'a [[u32; 3]],
    pub nbr: &'a [[u32; 3]],
    pub info: &'a [TriInfo],
    pub eligible: &'a [bool],
    pub tol: f64,
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
        snap_deg,
        scratch,
    } = args;
    let (label, n_seg) = segment(vc, faces, nbr, info, eligible, tol);
    let regions0 = build_regions(vc, faces, info, &label, n_seg, tol, scratch);
    let pairs0 = region_pairs(nbr, &label);
    let root = merge_coplanar(vc, &regions0, &pairs0, tol, snap_deg);
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
    let regions = build_regions(vc, faces, info, &label2, n_merged as usize, tol, scratch);
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
    let mut fdsu: Vec<u32> = (0..regions.len() as u32).collect();
    fn find(d: &mut [u32], mut x: u32) -> u32 {
        while d[x as usize] != x {
            d[x as usize] = d[d[x as usize] as usize];
            x = d[x as usize];
        }
        x
    }
    for &(a, b) in pairs {
        if !is_analytic[a as usize] && !is_analytic[b as usize] {
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
        if is_analytic[ri] {
            finals.push(Final {
                faces: r.faces.clone(),
                surface: Surface::Plane {
                    normal: r.n,
                    offset: r.d,
                },
                rms: r.rms,
                max: r.max,
                comp: comp_of[r.faces[0] as usize] as usize,
            });
        } else {
            let g = find(&mut fdsu, ri as u32);
            let idx = *group_of.entry(g).or_insert_with(|| {
                finals.push(Final {
                    faces: Vec::new(),
                    surface: Surface::Facets,
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
                surface: Surface::Facets,
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

    let mut flabel = vec![NONE; comp_of.len()];
    for (i, fr) in finals.iter().enumerate() {
        for &f in &fr.faces {
            flabel[f as usize] = i as u32;
        }
    }
    (finals, flabel)
}
