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
    pub max_abs: f64,
    pub unique_vertices: u32,
    pub merge_dev: f64,
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
    let max_abs = vertices
        .iter()
        .flatten()
        .map(|x| x.abs())
        .fold(0.0, f64::max);
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
            max_abs,
            unique_vertices: wrep.unique_vertices as u32,
            merge_dev: wrep.max_merge,
        },
        warnings,
    ))
}
