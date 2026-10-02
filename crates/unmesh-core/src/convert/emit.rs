use rustc_hash::FxHashMap;

use super::CompMeta;
use super::linalg::{V3, angle_deg, cross, dot, scale, sub, unit};
use super::topology::NONE;
use crate::ir::{
    Adjacency, Boundary, Ir, Kind, Region, Residual, Shell, Source, Surface, Tolerances, Vertex,
    VertexRole, validate,
};
use crate::mesh::Point;

pub struct Final {
    pub faces: Vec<u32>,
    pub plane: Option<(V3, f64)>,
    pub rms: f64,
    pub max: f64,
    pub comp: usize,
}

pub struct Asm<'a> {
    pub tolerances: Tolerances,
    pub source_triangles: u32,
    pub source_vertices: u32,
    pub center: V3,
    pub faces: &'a [[u32; 3]],
    pub fsrc: &'a [u32],
    pub nbr: &'a [[u32; 3]],
    pub metas: &'a [CompMeta],
    pub orig: &'a [Point],
}

struct EdgeRec {
    ra: u32,
    rb: u32,
    u: u32,
    v: u32,
    fa: u32,
    fb: u32,
}

struct VertexReg {
    ids: FxHashMap<u32, usize>,
    list: Vec<(u32, VertexRole, Vec<u32>)>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn lex(a: &Point, b: &Point) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

pub fn assemble(
    asm: &Asm,
    finals: &[Final],
    flabel: &[u32],
    pos: &[Point],
) -> Result<Ir, Vec<String>> {
    let nv = pos.len();
    let mut regions = Vec::with_capacity(finals.len());
    for (i, fr) in finals.iter().enumerate() {
        let triangles: Vec<u32> = fr.faces.iter().map(|&f| asm.fsrc[f as usize]).collect();
        let (surface, residual) = match fr.plane {
            Some((n, d)) => {
                let dw = d + dot(n, asm.center);
                let p = pos[asm.faces[fr.faces[0] as usize][0] as usize];
                let off = dot(n, p) - dw;
                (
                    Surface::Plane {
                        origin: sub(p, scale(n, off)),
                        normal: n,
                    },
                    Some(Residual {
                        rms: fr.rms,
                        max: fr.max,
                    }),
                )
            }
            None => {
                let mut local: FxHashMap<u32, u32> = FxHashMap::default();
                let mut vertices = Vec::new();
                let mut faces = Vec::with_capacity(fr.faces.len());
                for &f in &fr.faces {
                    let t = asm.faces[f as usize].map(|v| {
                        *local.entry(v).or_insert_with(|| {
                            vertices.push(pos[v as usize]);
                            (vertices.len() - 1) as u32
                        })
                    });
                    faces.push(t);
                }
                (Surface::Facets { vertices, faces }, None)
            }
        };
        regions.push(Region {
            id: i as u32,
            surface,
            triangles,
            residual,
        });
    }

    let shells: Vec<Shell> = asm
        .metas
        .iter()
        .enumerate()
        .map(|(ci, m)| Shell {
            closed: m.closed,
            role: m.role,
            parent: m.parent.map(|p| p as u32),
            regions: finals
                .iter()
                .enumerate()
                .filter(|(_, f)| f.comp == ci)
                .map(|(i, _)| i as u32)
                .collect(),
        })
        .collect();

    let mut vr: Vec<u64> = Vec::new();
    for (f, t) in asm.faces.iter().enumerate() {
        if flabel[f] == NONE {
            continue;
        }
        for &v in t {
            vr.push(((v as u64) << 32) | flabel[f] as u64);
        }
    }
    vr.sort_unstable();
    vr.dedup();
    let mut vstart = vec![0usize; nv + 1];
    for k in &vr {
        vstart[(k >> 32) as usize + 1] += 1;
    }
    for i in 0..nv {
        vstart[i + 1] += vstart[i];
    }
    let vregs = |v: u32| -> Vec<u32> {
        vr[vstart[v as usize]..vstart[v as usize + 1]]
            .iter()
            .map(|k| (k & 0xffff_ffff) as u32)
            .collect()
    };
    let is_junction = |v: u32| vstart[v as usize + 1] - vstart[v as usize] >= 3;

    let tri_normal = |f: u32| -> V3 {
        let t = asm.faces[f as usize];
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        unit(cross(sub(b, a), sub(c, a)))
    };
    let normal_of = |r: u32, f: u32| -> V3 {
        match finals[r as usize].plane {
            Some((n, _)) => n,
            None => tri_normal(f),
        }
    };

    let mut edges: Vec<EdgeRec> = Vec::new();
    for (f, nb) in asm.nbr.iter().enumerate() {
        let ra = flabel[f];
        if ra == NONE {
            continue;
        }
        for (k, &g) in nb.iter().enumerate() {
            if g == NONE {
                continue;
            }
            let rb = flabel[g as usize];
            if rb != NONE && ra < rb {
                edges.push(EdgeRec {
                    ra,
                    rb,
                    u: asm.faces[f][k],
                    v: asm.faces[f][(k + 1) % 3],
                    fa: f as u32,
                    fb: g,
                });
            }
        }
    }
    edges.sort_by_key(|e| (e.ra, e.rb, e.u, e.v));

    let thr = asm.tolerances.tangent_threshold_deg;
    let mut reg = VertexReg {
        ids: FxHashMap::default(),
        list: Vec::new(),
    };
    let mut adjacencies: Vec<Adjacency> = Vec::new();
    let mut s = 0;
    while s < edges.len() {
        let mut e = s + 1;
        while e < edges.len() && edges[e].ra == edges[s].ra && edges[e].rb == edges[s].rb {
            e += 1;
        }
        let group = &edges[s..e];
        let (ra, rb) = (group[0].ra, group[0].rb);
        let dih: Vec<f64> = group
            .iter()
            .map(|g| angle_deg(normal_of(ra, g.fa), normal_of(rb, g.fb)))
            .collect();
        let tangent: Vec<bool> = dih.iter().map(|d| *d < thr).collect();
        let mut out: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        let mut in_edge: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (i, g) in group.iter().enumerate() {
            out.entry(g.u).or_default().push(i);
            in_edge.entry(g.v).or_default().push(i);
        }
        let is_break = |v: u32| -> bool {
            if is_junction(v) {
                return true;
            }
            match (out.get(&v), in_edge.get(&v)) {
                (Some(o), Some(i)) if o.len() == 1 && i.len() == 1 => {
                    tangent[o[0]] != tangent[i[0]]
                }
                _ => true,
            }
        };
        let mut visited = vec![false; group.len()];
        let mut boundaries: Vec<Boundary> = Vec::new();
        let mut make = |chain: &[usize], closed: bool, reg: &mut VertexReg| {
            let mut points: Vec<Point> = Vec::new();
            if closed {
                points.extend(chain.iter().map(|&i| pos[group[i].u as usize]));
                let first = (0..points.len())
                    .min_by(|&a, &b| lex(&points[a], &points[b]))
                    .unwrap();
                points.rotate_left(first);
            } else {
                points.push(pos[group[chain[0]].u as usize]);
                points.extend(chain.iter().map(|&i| pos[group[i].v as usize]));
            }
            let d = median(chain.iter().map(|&i| dih[i]).collect());
            let mut vid = |v: u32| -> u32 {
                let len = reg.list.len();
                let id = *reg.ids.entry(v).or_insert(len);
                if id == len {
                    let (role, regions) = if is_junction(v) {
                        (VertexRole::Junction, vregs(v))
                    } else {
                        (VertexRole::KindChange, vec![ra, rb])
                    };
                    reg.list.push((v, role, regions));
                }
                id as u32
            };
            let (sv, ev) = if closed {
                (None, None)
            } else {
                (
                    Some(vid(group[chain[0]].u)),
                    Some(vid(group[*chain.last().unwrap()].v)),
                )
            };
            boundaries.push(Boundary {
                kind: if d < thr {
                    Kind::Tangent
                } else {
                    Kind::Transversal
                },
                dihedral_deg: d,
                closed,
                start_vertex: sv,
                end_vertex: ev,
                points,
            });
        };
        for i in 0..group.len() {
            if visited[i] || !is_break(group[i].u) {
                continue;
            }
            let mut chain = vec![i];
            visited[i] = true;
            let mut head = group[i].v;
            while !is_break(head) {
                let next = out[&head][0];
                if visited[next] {
                    break;
                }
                visited[next] = true;
                chain.push(next);
                head = group[next].v;
            }
            make(&chain, false, &mut reg);
        }
        for i in 0..group.len() {
            if visited[i] {
                continue;
            }
            let mut chain = vec![i];
            visited[i] = true;
            let mut head = group[i].v;
            while head != group[i].u {
                let next = out[&head][0];
                if visited[next] {
                    break;
                }
                visited[next] = true;
                chain.push(next);
                head = group[next].v;
            }
            make(&chain, true, &mut reg);
        }
        boundaries.sort_by(|a, b| lex(&a.points[0], &b.points[0]));
        adjacencies.push(Adjacency {
            regions: [ra, rb],
            boundaries,
        });
        s = e;
    }

    let mut order: Vec<usize> = (0..reg.list.len()).collect();
    order.sort_by(|&a, &b| {
        lex(&pos[reg.list[a].0 as usize], &pos[reg.list[b].0 as usize])
            .then(reg.list[a].2.cmp(&reg.list[b].2))
    });
    let mut new_id = vec![0u32; order.len()];
    for (n, &o) in order.iter().enumerate() {
        new_id[o] = n as u32;
    }
    for a in adjacencies.iter_mut() {
        for b in a.boundaries.iter_mut() {
            b.start_vertex = b.start_vertex.map(|v| new_id[v as usize]);
            b.end_vertex = b.end_vertex.map(|v| new_id[v as usize]);
        }
    }
    let vertices: Vec<Vertex> = order
        .iter()
        .enumerate()
        .map(|(n, &o)| {
            let (v, role, regions) = &reg.list[o];
            Vertex {
                id: n as u32,
                role: *role,
                position: pos[*v as usize],
                regions: regions.clone(),
                source_positions: vec![asm.orig[*v as usize]],
            }
        })
        .collect();

    let ir = Ir {
        ir_version: crate::ir::IR_VERSION,
        tolerances: asm.tolerances.clone(),
        source: Source {
            triangle_count: asm.source_triangles,
            vertex_count: asm.source_vertices,
        },
        shells,
        regions,
        adjacencies,
        vertices,
    };
    let errors = validate(&ir);
    if errors.is_empty() {
        Ok(ir)
    } else {
        Err(errors)
    }
}
