use rustc_hash::FxHashSet;

use crate::ir::Point;

pub type Triangle = [Point; 3];

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_f64(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

pub fn triangle_area(t: &Triangle) -> f64 {
    let u = [t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]];
    let v = [t[2][0] - t[0][0], t[2][1] - t[0][1], t[2][2] - t[0][2]];
    let c = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

fn point_in(t: &Triangle, r1: f64, r2: f64) -> Point {
    let s = r1.sqrt();
    let (a, b, c) = (1.0 - s, s * (1.0 - r2), s * r2);
    [
        a * t[0][0] + b * t[1][0] + c * t[2][0],
        a * t[0][1] + b * t[1][1] + c * t[2][1],
        a * t[0][2] + b * t[1][2] + c * t[2][2],
    ]
}

fn key(p: &Point) -> [u64; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

/// Draws points uniformly by area on `tris`. A triangle of area `A` receives `floor(A*density)`
/// points plus one more with probability `frac(A*density)`, so the count is exact in expectation
/// and the draw for a triangle depends only on `(seed, salt, index)`. With `include_vertices`,
/// every distinct triangle corner is also emitted once. Each point carries its triangle index.
pub fn sample_triangles(
    tris: &[Triangle],
    density: f64,
    seed: u64,
    salt: u64,
    include_vertices: bool,
) -> Vec<(Point, u32)> {
    let mut out = Vec::new();
    let mut seen: FxHashSet<[u64; 3]> = FxHashSet::default();
    for (i, t) in tris.iter().enumerate() {
        let area = triangle_area(t);
        if area == 0.0 || !area.is_finite() {
            continue;
        }
        let mut state = seed
            ^ salt.wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let expected = area * density;
        let mut n = expected.floor() as usize;
        if unit_f64(&mut state) < expected - expected.floor() {
            n += 1;
        }
        for _ in 0..n {
            let (r1, r2) = (unit_f64(&mut state), unit_f64(&mut state));
            out.push((point_in(t, r1, r2), i as u32));
        }
        if include_vertices {
            for p in t {
                if seen.insert(key(p)) {
                    out.push((*p, i as u32));
                }
            }
        }
    }
    out
}

pub fn total_area(tris: &[Triangle]) -> f64 {
    tris.iter().map(triangle_area).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_match_area_times_density() {
        let tris = vec![[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]];
        let pts = sample_triangles(&tris, 20.0, 7, 1, false);
        assert!((pts.len() as f64 - 1000.0).abs() <= 1.0);
        assert!(
            pts.iter()
                .all(|(p, _)| p[0] + p[1] <= 10.0 + 1e-9 && p[2] == 0.0)
        );
    }

    #[test]
    fn deterministic_and_seed_sensitive() {
        let tris = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
        let a = sample_triangles(&tris, 100.0, 3, 1, false);
        assert_eq!(a, sample_triangles(&tris, 100.0, 3, 1, false));
        assert_ne!(a, sample_triangles(&tris, 100.0, 4, 1, false));
    }

    #[test]
    fn vertices_are_emitted_once() {
        let tris = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ];
        assert_eq!(sample_triangles(&tris, 0.0, 0, 0, true).len(), 4);
    }

    #[test]
    fn coverage_is_uniform_by_area() {
        let tris = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [4.0, 0.0, 1.0], [0.0, 4.0, 1.0]],
        ];
        let pts = sample_triangles(&tris, 500.0, 1, 1, false);
        let big = pts.iter().filter(|(_, i)| *i == 1).count() as f64;
        let small = pts.iter().filter(|(_, i)| *i == 0).count() as f64;
        assert!((big / small - 16.0).abs() < 0.5);
    }
}
