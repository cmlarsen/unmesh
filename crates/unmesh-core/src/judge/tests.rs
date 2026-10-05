use super::*;
use crate::ir::{Ir, Orientation, Region, Shell, ShellRole, Source, Tolerances};

fn ir_of(regions: Vec<Region>, triangles: usize) -> Ir {
    let n = regions.len() as u32;
    Ir {
        ir_version: 0,
        tolerances: Tolerances::default(),
        source: Source {
            triangle_count: triangles as u32,
            vertex_count: 0,
        },
        shells: vec![Shell {
            closed: true,
            role: ShellRole::Outer,
            parent: None,
            regions: (0..n).collect(),
        }],
        regions,
        adjacencies: vec![],
        vertices: vec![],
    }
}

fn region(id: u32, surface: Surface, triangles: Vec<u32>) -> Region {
    Region {
        id,
        surface,
        triangles,
        residual: None,
    }
}

type Quad = ([f64; 3], [f64; 3], [f64; 3], [f64; 3]);

fn cube(side: f64) -> (Vec<Triangle>, Vec<Region>) {
    let faces: [Quad; 6] = [
        (
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ),
        (
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ),
        (
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ),
        (
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ),
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ),
    ];
    let normals = [
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let mut tris = Vec::new();
    let mut regions = Vec::new();
    for (i, ((a, b, c, d), n)) in faces.iter().zip(normals).enumerate() {
        let s = |p: &[f64; 3]| [p[0] * side, p[1] * side, p[2] * side];
        tris.push([s(a), s(b), s(c)]);
        tris.push([s(a), s(c), s(d)]);
        let origin = s(a);
        regions.push(region(
            i as u32,
            Surface::Plane { origin, normal: n },
            vec![2 * i as u32, 2 * i as u32 + 1],
        ));
    }
    (tris, regions)
}

fn opts() -> JudgeOptions {
    JudgeOptions {
        samples_per_mm2: 5.0,
        seed: 11,
        include_vertices: true,
    }
}

#[test]
fn exact_planar_ir_scores_zero_both_ways() {
    let (tris, regions) = cube(10.0);
    let ir = ir_of(regions, tris.len());
    let r = judge(&ir, &tris, Some(&tris), &opts()).unwrap();
    for c in [r.input, r.truth.unwrap()] {
        assert!(c.ir_to_mesh.max < 1e-9, "{c:?}");
        assert!(c.mesh_to_ir.max < 1e-9, "{c:?}");
        assert!(c.ir_to_mesh.count > 1000);
    }
}

#[test]
fn shifted_plane_reports_the_offset_in_that_region_only() {
    let (tris, mut regions) = cube(10.0);
    let d = 0.25;
    if let Surface::Plane { origin, .. } = &mut regions[1].surface {
        origin[2] += d;
    }
    let ir = ir_of(regions, tris.len());
    let r = judge(&ir, &tris, None, &opts()).unwrap();
    assert!((r.input.ir_to_mesh.max - d).abs() < 1e-9);
    assert!((r.input.mesh_to_ir.max - d).abs() < 1e-9);
    assert!((r.region_max[1] - d).abs() < 1e-9);
    assert!(
        r.region_max
            .iter()
            .enumerate()
            .all(|(i, m)| i == 1 || *m < 1e-9)
    );
}

#[test]
fn calibration_is_negative_when_the_converter_under_reports() {
    let (tris, mut regions) = cube(10.0);
    if let Surface::Plane { origin, .. } = &mut regions[0].surface {
        origin[2] -= 0.1;
    }
    let ir = ir_of(regions, tris.len());
    let r = judge(&ir, &tris, None, &opts()).unwrap();
    assert!((calibration(0.05, &r.input) + 0.05).abs() < 1e-9);
    assert!((calibration(0.1, &r.input)).abs() < 1e-9);
}

fn prism(n: usize, radius: f64, height: f64) -> Vec<Triangle> {
    let ring = |k: usize, z: f64| {
        let a = std::f64::consts::TAU * (k % n) as f64 / n as f64;
        [radius * a.cos(), radius * a.sin(), z]
    };
    let mut tris = Vec::new();
    for k in 0..n {
        tris.push([ring(k, 0.0), ring(k + 1, 0.0), ring(k + 1, height)]);
        tris.push([ring(k, 0.0), ring(k + 1, height), ring(k, height)]);
    }
    tris
}

#[test]
fn cylinder_against_its_tessellation_reports_the_chord_error() {
    let (n, radius) = (24, 5.0);
    let tris = prism(n, radius, 8.0);
    let surface = Surface::Cylinder {
        origin: [0.0; 3],
        axis: [0.0, 0.0, 1.0],
        radius,
        orientation: Orientation::Same,
    };
    let ir = ir_of(
        vec![region(0, surface, (0..tris.len() as u32).collect())],
        tris.len(),
    );
    let sagitta = radius * (1.0 - (std::f64::consts::PI / n as f64).cos());
    let r = judge(
        &ir,
        &tris,
        None,
        &JudgeOptions {
            samples_per_mm2: 20.0,
            ..opts()
        },
    )
    .unwrap();
    assert!((r.input.ir_to_mesh.max - sagitta).abs() < 0.02 * sagitta);
    assert!(r.input.mesh_to_ir.max <= sagitta * 1.001);
    assert!(r.input.ir_to_mesh.mean < sagitta);
}

fn quad(x0: f64, x1: f64, z: f64) -> [Triangle; 2] {
    let (a, b, c, d) = ([x0, 0.0, z], [x1, 0.0, z], [x1, 1.0, z], [x0, 1.0, z]);
    [[a, b, c], [a, c, d]]
}

#[test]
fn reverse_direction_does_not_charge_a_neighbouring_chord_gap() {
    let (w, radius, chord_z) = (0.8_f64, 1.0_f64, 0.15);
    let sagitta = radius - (radius * radius - w * w).sqrt();
    let plane_offset = 0.1;
    let mut tris = quad(0.0, 1.0, plane_offset).to_vec();
    tris.extend(quad(0.5 - w, 0.5 + w, chord_z));
    let plane = Surface::Plane {
        origin: [0.0; 3],
        normal: [0.0, 0.0, 1.0],
    };
    let cylinder = Surface::Cylinder {
        origin: [0.5, 0.0, chord_z - (radius - sagitta)],
        axis: [0.0, 1.0, 0.0],
        radius,
        orientation: Orientation::Same,
    };
    let ir = ir_of(
        vec![
            region(0, plane, vec![0, 1]),
            region(1, cylinder, vec![2, 3]),
        ],
        tris.len(),
    );
    let r = judge(
        &ir,
        &tris,
        None,
        &JudgeOptions {
            samples_per_mm2: 50.0,
            ..opts()
        },
    )
    .unwrap();
    assert!((r.input.ir_to_mesh.max - sagitta).abs() < 1e-6, "{r:?}");
    assert!((r.input.mesh_to_ir.max - sagitta).abs() < 1e-6, "{r:?}");
    assert!(!under_reports(sagitta, &r.input));
    assert!(under_reports(0.5 * sagitta, &r.input));
}

#[test]
fn facets_regions_are_sampled_on_their_own_triangles() {
    let (tris, _) = cube(10.0);
    let mut verts = Vec::new();
    let mut faces = Vec::new();
    for t in &tris {
        let base = verts.len() as u32;
        verts.extend_from_slice(t);
        faces.push([base, base + 1, base + 2]);
    }
    let ir = ir_of(
        vec![region(
            0,
            Surface::Facets {
                vertices: verts,
                faces,
            },
            (0..tris.len() as u32).collect(),
        )],
        tris.len(),
    );
    let r = judge(&ir, &tris, None, &opts()).unwrap();
    assert!(r.input.max() < 1e-9);
}

#[test]
fn missing_coverage_shows_in_the_reverse_direction() {
    let (tris, mut regions) = cube(10.0);
    regions.pop();
    let ir = ir_of(regions, tris.len());
    let r = judge(&ir, &tris, None, &opts()).unwrap();
    assert!(r.input.ir_to_mesh.max < 1e-9);
    assert!(r.input.mesh_to_ir.max > 4.0);
}

#[test]
fn rejects_out_of_range_triangle_ids_and_source_mismatch() {
    let (tris, mut regions) = cube(10.0);
    regions[0].triangles.push(99);
    let ir = ir_of(regions, tris.len());
    assert!(matches!(
        judge(&ir, &tris, None, &opts()),
        Err(JudgeError::TriangleIdOutOfRange { .. })
    ));
    let (tris, regions) = cube(10.0);
    let ir = ir_of(regions, 3);
    assert!(matches!(
        judge(&ir, &tris, None, &opts()),
        Err(JudgeError::SourceMismatch { .. })
    ));
}

#[test]
fn stats_percentiles() {
    let s = Stats::of((0..=100).map(f64::from).collect());
    assert_eq!(s.max, 100.0);
    assert_eq!(s.p99, 99.0);
    assert_eq!(s.p95, 95.0);
    assert_eq!(s.mean, 50.0);
}
