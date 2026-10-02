use super::linalg::{V3, add, cross, dot, scale, sub};
use crate::api::ConvertWarning;
use crate::ir::ShellRole;
use crate::mesh::Point;

pub const NONE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompKind {
    Closed,
    OpenEdges,
    NonManifold,
}

#[derive(Debug, Clone)]
pub struct Comp {
    pub faces: Vec<u32>,
    pub kind: CompKind,
}

pub struct Topology {
    pub nbr: Vec<[u32; 3]>,
    pub comps: Vec<Comp>,
    pub repaired: bool,
}

pub struct CompMeta {
    pub closed: bool,
    pub role: ShellRole,
    pub parent: Option<usize>,
}

pub struct Shells {
    pub topo: Topology,
    pub metas: Vec<CompMeta>,
    pub comp_of: Vec<u32>,
    pub eligible: Vec<bool>,
}

pub fn prepare(
    faces: &mut [[u32; 3]],
    vc: &mut Vec<V3>,
    orig: &mut Vec<Point>,
) -> (Shells, Vec<ConvertWarning>) {
    use rustc_hash::FxHashMap;

    let mut warnings = Vec::new();
    let mut topo = build(faces);
    if topo.repaired {
        warnings.push(ConvertWarning {
            code: "repaired_winding".to_string(),
            message: "inconsistent triangle winding was repaired".to_string(),
        });
    }

    let n_comps = topo.comps.len();
    let volumes: Vec<f64> = topo
        .comps
        .iter()
        .map(|c| {
            if c.kind == CompKind::Closed {
                signed_volume(vc, faces, &c.faces)
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
                    && point_in_shell(vc, faces, &topo.comps[o].faces, probe)
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
        warnings.push(ConvertWarning {
            code: "open_edges".to_string(),
            message: format!("{open_edges} shell(s) have open edges and are kept as facets"),
        });
    }
    if non_manifold > 0 {
        warnings.push(ConvertWarning {
            code: "non_manifold_edges".to_string(),
            message: format!(
                "{non_manifold} shell(s) have non-manifold edges and are kept as facets"
            ),
        });
    }
    if !flip_comps.is_empty() {
        for &ci in &flip_comps {
            for &f in &topo.comps[ci].faces {
                faces[f as usize].swap(1, 2);
            }
        }
        warnings.push(ConvertWarning {
            code: "flipped_winding".to_string(),
            message: format!(
                "{} shell(s) had inward-facing winding and were flipped",
                flip_comps.len()
            ),
        });
        topo = build(faces);
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
        topo = build(faces);
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

    (
        Shells {
            topo,
            metas,
            comp_of,
            eligible,
        },
        warnings,
    )
}

struct Dsu(Vec<u32>);

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu((0..n as u32).collect())
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            let p = self.0[x as usize];
            self.0[x as usize] = self.0[p as usize];
            x = self.0[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b) as usize] = a.min(b);
        }
    }
}

struct Raw {
    nbr: Vec<[u32; 3]>,
    same_dir: Vec<[bool; 3]>,
    comp_of: Vec<u32>,
    n_comps: usize,
    open: Vec<bool>,
    nonmanifold: Vec<bool>,
}

fn analyse(faces: &[[u32; 3]]) -> Raw {
    let nf = faces.len();
    let mut edges: Vec<(u64, u32, u8)> = Vec::with_capacity(nf * 3);
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            let (u, v) = (f[k], f[(k + 1) % 3]);
            let key = ((u.min(v) as u64) << 32) | u.max(v) as u64;
            edges.push((key, fi as u32, k as u8));
        }
    }
    edges.sort_unstable();
    let mut nbr = vec![[NONE; 3]; nf];
    let mut same_dir = vec![[false; 3]; nf];
    let mut dsu = Dsu::new(nf);
    let mut open_face = vec![false; nf];
    let mut nm_face = vec![false; nf];
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 {
            j += 1;
        }
        match j - i {
            1 => open_face[edges[i].1 as usize] = true,
            2 => {
                let (a, b) = (edges[i], edges[i + 1]);
                let (fa, fb) = (a.1 as usize, b.1 as usize);
                let fwd_a = faces[fa][a.2 as usize] == faces[fb][b.2 as usize];
                nbr[fa][a.2 as usize] = b.1;
                nbr[fb][b.2 as usize] = a.1;
                same_dir[fa][a.2 as usize] = fwd_a;
                same_dir[fb][b.2 as usize] = fwd_a;
                dsu.union(a.1, b.1);
            }
            _ => {
                for e in &edges[i..j] {
                    nm_face[e.1 as usize] = true;
                    dsu.union(edges[i].1, e.1);
                }
            }
        }
        i = j;
    }
    let mut id_of_root = vec![NONE; nf];
    let mut comp_of = vec![0u32; nf];
    let mut n_comps = 0usize;
    for f in 0..nf {
        let r = dsu.find(f as u32) as usize;
        if id_of_root[r] == NONE {
            id_of_root[r] = n_comps as u32;
            n_comps += 1;
        }
        comp_of[f] = id_of_root[r];
    }
    let mut open = vec![false; n_comps];
    let mut nonmanifold = vec![false; n_comps];
    for f in 0..nf {
        open[comp_of[f] as usize] |= open_face[f];
        nonmanifold[comp_of[f] as usize] |= nm_face[f];
    }
    Raw {
        nbr,
        same_dir,
        comp_of,
        n_comps,
        open,
        nonmanifold,
    }
}

pub fn build(faces: &mut [[u32; 3]]) -> Topology {
    let mut raw = analyse(faces);
    let mut repaired = false;
    let inconsistent = (0..faces.len()).any(|f| (0..3).any(|k| raw.same_dir[f][k]));
    if inconsistent {
        let mut flip = vec![None::<bool>; faces.len()];
        let mut bad_comp = vec![false; raw.n_comps];
        let mut stack = Vec::new();
        for start in 0..faces.len() {
            let c = raw.comp_of[start] as usize;
            if flip[start].is_some() || raw.open[c] || raw.nonmanifold[c] {
                continue;
            }
            flip[start] = Some(false);
            stack.push(start);
            while let Some(f) = stack.pop() {
                let ff = flip[f].unwrap();
                for k in 0..3 {
                    let g = raw.nbr[f][k];
                    if g == NONE {
                        continue;
                    }
                    let want = ff ^ raw.same_dir[f][k];
                    match flip[g as usize] {
                        None => {
                            flip[g as usize] = Some(want);
                            stack.push(g as usize);
                        }
                        Some(have) => {
                            if have != want {
                                bad_comp[c] = true;
                            }
                        }
                    }
                }
            }
        }
        let mut any = false;
        for f in 0..faces.len() {
            let c = raw.comp_of[f] as usize;
            if flip[f] == Some(true) && !bad_comp[c] {
                faces[f].swap(1, 2);
                any = true;
            }
        }
        for (c, bad) in bad_comp.iter().enumerate() {
            if *bad {
                raw.nonmanifold[c] = true;
            }
        }
        if any {
            repaired = true;
            let nm = raw.nonmanifold.clone();
            let comp_old = raw.comp_of.clone();
            raw = analyse(faces);
            for f in 0..faces.len() {
                if nm[comp_old[f] as usize] {
                    raw.nonmanifold[raw.comp_of[f] as usize] = true;
                }
            }
        }
    }
    let mut comps: Vec<Comp> = (0..raw.n_comps)
        .map(|c| Comp {
            faces: Vec::new(),
            kind: if raw.nonmanifold[c] {
                CompKind::NonManifold
            } else if raw.open[c] {
                CompKind::OpenEdges
            } else {
                CompKind::Closed
            },
        })
        .collect();
    for (f, &c) in raw.comp_of.iter().enumerate() {
        comps[c as usize].faces.push(f as u32);
    }
    Topology {
        nbr: raw.nbr,
        comps,
        repaired,
    }
}

pub fn signed_volume(verts: &[V3], faces: &[[u32; 3]], members: &[u32]) -> f64 {
    members
        .iter()
        .map(|&f| {
            let t = faces[f as usize];
            dot(
                verts[t[0] as usize],
                cross(verts[t[1] as usize], verts[t[2] as usize]),
            ) / 6.0
        })
        .sum()
}

pub fn point_in_shell(verts: &[V3], faces: &[[u32; 3]], members: &[u32], p: V3) -> bool {
    let dir: V3 = [0.5773502691896258, 0.3203174832, 0.7507943105];
    let mut crossings = 0;
    for &f in members {
        let t = faces[f as usize];
        let (a, b, c) = (
            verts[t[0] as usize],
            verts[t[1] as usize],
            verts[t[2] as usize],
        );
        let e1 = sub(b, a);
        let e2 = sub(c, a);
        let h = cross(dir, e2);
        let det = dot(e1, h);
        if det.abs() < 1e-14 {
            continue;
        }
        let inv = 1.0 / det;
        let s = sub(p, a);
        let u = inv * dot(s, h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = cross(s, e1);
        let v = inv * dot(dir, q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if inv * dot(e2, q) > 0.0 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}
