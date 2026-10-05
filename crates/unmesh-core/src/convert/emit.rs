use rustc_hash::FxHashMap;

use super::adjacency::{self, lex};
use super::fit::Final;
use super::linalg::{V3, dot, scale, sub};
use super::surface::Surface;
use super::topology::CompMeta;
use crate::ir::{
    Adjacency, Boundary, Ir, Orientation, Region, Residual, Shell, Source, Tolerances, Vertex,
    validate,
};
use crate::mesh::Point;

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

fn orientation(reversed: bool) -> Orientation {
    if reversed {
        Orientation::Reversed
    } else {
        Orientation::Same
    }
}

pub fn assemble(
    asm: &Asm,
    finals: &[Final],
    flabel: &[u32],
    pos: &[Point],
) -> Result<Ir, Vec<String>> {
    let mut regions = Vec::with_capacity(finals.len());
    for (i, fr) in finals.iter().enumerate() {
        let triangles: Vec<u32> = fr.faces.iter().map(|&f| asm.fsrc[f as usize]).collect();
        let (surface, residual) = match fr.surface.to_world(asm.center) {
            Surface::Plane {
                normal: n,
                offset: dw,
            } => {
                let p = pos[asm.faces[fr.faces[0] as usize][0] as usize];
                let off = dot(n, p) - dw;
                (
                    crate::ir::Surface::Plane {
                        origin: sub(p, scale(n, off)),
                        normal: n,
                    },
                    Some(Residual {
                        rms: fr.rms,
                        max: fr.max,
                    }),
                )
            }
            Surface::Cylinder {
                origin,
                axis,
                radius,
                reversed,
            } => (
                crate::ir::Surface::Cylinder {
                    origin,
                    axis,
                    radius,
                    orientation: orientation(reversed),
                },
                Some(Residual {
                    rms: fr.rms,
                    max: fr.max.max(fr.sag),
                }),
            ),
            Surface::Cone {
                apex,
                axis,
                half_angle,
                reversed,
            } => (
                crate::ir::Surface::Cone {
                    apex,
                    axis,
                    half_angle,
                    orientation: orientation(reversed),
                },
                Some(Residual {
                    rms: fr.rms,
                    max: fr.max.max(fr.sag),
                }),
            ),
            Surface::Sphere {
                center,
                radius,
                reversed,
            } => (
                crate::ir::Surface::Sphere {
                    center,
                    radius,
                    orientation: orientation(reversed),
                },
                Some(Residual {
                    rms: fr.rms,
                    max: fr.max.max(fr.sag),
                }),
            ),
            Surface::Torus {
                center,
                axis,
                major,
                minor,
                reversed,
            } => (
                crate::ir::Surface::Torus {
                    center,
                    axis,
                    major_radius: major,
                    minor_radius: minor,
                    orientation: orientation(reversed),
                },
                Some(Residual {
                    rms: fr.rms,
                    max: fr.max.max(fr.sag),
                }),
            ),
            Surface::Facets => {
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
                (crate::ir::Surface::Facets { vertices, faces }, None)
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

    let (raw, reg) = adjacency::build(
        asm.nbr,
        asm.faces,
        flabel,
        finals,
        pos,
        asm.center,
        asm.tolerances.tangent_threshold_deg,
    );

    let mut order: Vec<usize> = (0..reg.len()).collect();
    order.sort_by(|&a, &b| {
        lex(&pos[reg[a].mesh as usize], &pos[reg[b].mesh as usize])
            .then(reg[a].regions.cmp(&reg[b].regions))
    });
    let mut new_id = vec![0u32; order.len()];
    for (n, &o) in order.iter().enumerate() {
        new_id[o] = n as u32;
    }
    let adjacencies: Vec<Adjacency> = raw
        .into_iter()
        .map(|a| Adjacency {
            regions: a.regions,
            boundaries: a
                .boundaries
                .into_iter()
                .map(|b| Boundary {
                    kind: b.kind,
                    dihedral_deg: b.dihedral_deg,
                    closed: b.closed,
                    start_vertex: b.start.map(|v| new_id[v as usize]),
                    end_vertex: b.end.map(|v| new_id[v as usize]),
                    points: b.points,
                })
                .collect(),
        })
        .collect();
    let vertices: Vec<Vertex> = order
        .iter()
        .enumerate()
        .map(|(n, &o)| {
            let e = &reg[o];
            Vertex {
                id: n as u32,
                role: e.role,
                position: pos[e.mesh as usize],
                regions: e.regions.clone(),
                source_positions: vec![asm.orig[e.mesh as usize]],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Kind, ShellRole, VertexRole};

    fn asm<'a>(
        faces: &'a [[u32; 3]],
        fsrc: &'a [u32],
        nbr: &'a [[u32; 3]],
        metas: &'a [CompMeta],
        pos: &'a [Point],
    ) -> Asm<'a> {
        Asm {
            tolerances: Tolerances::default(),
            source_triangles: fsrc.len() as u32,
            source_vertices: pos.len() as u32,
            center: [0.0; 3],
            faces,
            fsrc,
            nbr,
            metas,
            orig: pos,
        }
    }

    fn outer_metas() -> Vec<CompMeta> {
        vec![CompMeta {
            closed: true,
            role: ShellRole::Outer,
            parent: None,
        }]
    }

    fn plane_final(faces: Vec<u32>, normal: V3) -> Final {
        Final {
            faces,
            surface: Surface::Plane {
                normal,
                offset: 0.0,
            },
            rms: 0.0,
            max: 0.0,
            sag: 0.0,
            comp: 0,
        }
    }

    fn facets_final(faces: Vec<u32>) -> Final {
        Final {
            faces,
            surface: Surface::Facets,
            rms: 0.0,
            max: 0.0,
            sag: 0.0,
            comp: 0,
        }
    }

    #[test]
    fn pinch_vertex_routes_through_without_kind_change() {
        let pos: Vec<Point> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 2, 4],
            [1, 4, 2],
            [1, 3, 4],
            [0, 4, 3],
            [0, 5, 2],
            [1, 2, 5],
            [1, 5, 3],
            [0, 3, 5],
        ];
        let fsrc: Vec<u32> = (0..8).collect();
        let topo = super::super::topology::build(&mut faces);
        let finals = vec![
            plane_final(vec![0, 2], [0.0, 0.0, 1.0]),
            facets_final(vec![1, 3, 4, 5, 6, 7]),
        ];
        let flabel = vec![0, 1, 0, 1, 1, 1, 1, 1];
        let metas = outer_metas();
        let asm = asm(&faces, &fsrc, &topo.nbr, &metas, &pos);
        let ir = assemble(&asm, &finals, &flabel, &pos).unwrap();
        ir.validate().unwrap();
        assert!(ir.vertices.is_empty());
        assert_eq!(ir.adjacencies.len(), 1);
        assert!(ir.adjacencies[0].boundaries.iter().all(|b| b.closed));
    }

    #[test]
    fn pinch_fixture_stable_under_vertex_relabeling() {
        let base_pos: Vec<Point> = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let base_faces: Vec<[u32; 3]> = vec![
            [0, 2, 4],
            [1, 4, 2],
            [1, 3, 4],
            [0, 4, 3],
            [0, 5, 2],
            [1, 2, 5],
            [1, 5, 3],
            [0, 3, 5],
        ];
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
        for _ in 0..50 {
            let mut perm: [usize; 6] = [0, 1, 2, 3, 4, 5];
            for i in (1..6).rev() {
                let j = rng() % (i + 1);
                perm.swap(i, j);
            }
            let mut inv = [0u32; 6];
            for (n, &o) in perm.iter().enumerate() {
                inv[o] = n as u32;
            }
            let pos: Vec<Point> = perm.iter().map(|&o| base_pos[o]).collect();
            let mut faces: Vec<[u32; 3]> = base_faces
                .iter()
                .map(|t| t.map(|v| inv[v as usize]))
                .collect();
            let fsrc: Vec<u32> = (0..8).collect();
            let topo = super::super::topology::build(&mut faces);
            let finals = vec![
                plane_final(vec![0, 2], [0.0, 0.0, 1.0]),
                facets_final(vec![1, 3, 4, 5, 6, 7]),
            ];
            let flabel = vec![0, 1, 0, 1, 1, 1, 1, 1];
            let metas = outer_metas();
            let asm = asm(&faces, &fsrc, &topo.nbr, &metas, &pos);
            let ir = assemble(&asm, &finals, &flabel, &pos).unwrap();
            ir.validate().unwrap();
            assert!(ir.vertices.is_empty());
            assert_eq!(ir.adjacencies.len(), 1);
            let boundaries = &ir.adjacencies[0].boundaries;
            assert_eq!(boundaries.len(), 2);
            for b in boundaries {
                assert!(b.closed);
                assert_eq!(b.points.len(), 3);
                let mut bits: Vec<[u64; 3]> =
                    b.points.iter().map(|p| p.map(f64::to_bits)).collect();
                bits.sort_unstable();
                bits.dedup();
                assert_eq!(bits.len(), 3);
            }
        }
    }

    #[test]
    fn genuine_kind_change_still_emits_a_vertex() {
        let pos: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [5.0, 1.0, 0.1],
        ];
        let mut faces: Vec<[u32; 3]> = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        let fsrc: Vec<u32> = (0..4).collect();
        let topo = super::super::topology::build(&mut faces);
        let finals = vec![
            plane_final(vec![0], [0.0, 0.0, 1.0]),
            facets_final(vec![1, 2, 3]),
        ];
        let flabel = vec![0, 1, 1, 1];
        let metas = outer_metas();
        let asm = asm(&faces, &fsrc, &topo.nbr, &metas, &pos);
        let ir = assemble(&asm, &finals, &flabel, &pos).unwrap();
        ir.validate().unwrap();
        let kc: Vec<_> = ir
            .vertices
            .iter()
            .filter(|v| v.role == VertexRole::KindChange)
            .collect();
        assert_eq!(kc.len(), 2);
        assert_eq!(ir.adjacencies.len(), 1);
        let kinds: Vec<Kind> = ir.adjacencies[0]
            .boundaries
            .iter()
            .map(|b| b.kind)
            .collect();
        assert!(kinds.contains(&Kind::Tangent) && kinds.contains(&Kind::Transversal));
    }
}
