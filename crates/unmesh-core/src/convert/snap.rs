use super::fit::{Region, offset_fit, residual};
use super::linalg::{V3, dot, scale, sub, unit};
use super::surface::Surface;

pub const NOISE_FACTOR: f64 = 5.0;

const POOL_GAP_RATIO: f64 = 10.0;
const NEAR_EXACT_FLOOR_RATIO: f64 = 3.0;
const NEAR_EXACT_MIN_FACES: usize = 8;

pub struct NoiseEstimate {
    pub sigma: f64,
    pub snap_sigma: f64,
}

pub fn snap_normals(
    v: &[V3],
    regions: &mut [Region],
    tol: f64,
    sigma: f64,
    snap_deg: f64,
    diag: f64,
    max_abs: f64,
) {
    let rms_limit = (2.5 * sigma).max(super::floor(diag, max_abs));
    let n = regions.len();
    let mut done = vec![false; n];
    let mut classes: Vec<V3> = Vec::new();

    let allow = |r: &Region| -> f64 {
        snap_deg.to_radians() + (3.0 * sigma / r.width.max(f64::MIN_POSITIVE)).atan()
    };
    let allow_cos = |r: &Region| -> f64 { allow(r).cos() };
    let try_normal = |r: &Region, cand: V3| -> Option<(f64, f64, f64)> {
        let d = offset_fit(v, &r.verts, cand, tol);
        let (rms, max) = residual(v, &r.verts, cand, d);
        (max <= tol && rms <= rms_limit).then_some((d, rms, max))
    };
    let adopt = |r: &mut Region, cand: V3, res: (f64, f64, f64)| {
        r.surface = Surface::Plane {
            normal: cand,
            offset: res.0,
        };
        r.rms = res.1;
        r.max = res.2;
    };

    let mut used_axis = [false; 3];
    for (i, r) in regions.iter_mut().enumerate() {
        if r.area <= 0.0 {
            continue;
        }
        let Some((n, _)) = r.surface.as_plane() else {
            continue;
        };
        for a in 0..3 {
            let mut e = [0.0; 3];
            e[a] = if n[a] >= 0.0 { 1.0 } else { -1.0 };
            if dot(n, e) >= allow_cos(r) {
                if let Some(res) = try_normal(r, e) {
                    adopt(r, e, res);
                    done[i] = true;
                    used_axis[a] = true;
                }
                break;
            }
        }
    }
    for (a, used) in used_axis.iter().enumerate() {
        if *used {
            let mut e = [0.0; 3];
            e[a] = 1.0;
            classes.push(e);
        }
    }

    let mut order: Vec<usize> = (0..n)
        .filter(|&i| !done[i] && regions[i].area > 0.0)
        .collect();
    order.sort_by(|&a, &b| regions[b].area.total_cmp(&regions[a].area).then(a.cmp(&b)));
    for i in order {
        let Some((ni, _)) = regions[i].surface.as_plane() else {
            continue;
        };
        let cos_r = allow_cos(&regions[i]);
        if let Some(u) = classes.iter().find(|u| dot(ni, **u).abs() >= cos_r) {
            let s = if dot(ni, *u) >= 0.0 { 1.0 } else { -1.0 };
            let cand = scale(*u, s);
            if let Some(res) = try_normal(&regions[i], cand) {
                adopt(&mut regions[i], cand, res);
            }
            continue;
        }
        let sin_snap = allow(&regions[i]).sin();
        let mut u = ni;
        for e in &classes {
            let c = dot(u, *e);
            if c.abs() <= sin_snap {
                u = unit(sub(u, scale(*e, c)));
            }
        }
        if u != ni
            && let Some(res) = try_normal(&regions[i], u)
        {
            adopt(&mut regions[i], u, res);
            classes.push(u);
            continue;
        }
        classes.push(ni);
    }
}

pub fn estimate_noise(v: &[V3], regions: &[Region], tol: f64, floor: f64) -> NoiseEstimate {
    match pool_noise(v, regions, floor) {
        Some((sigma, snap_sigma)) => NoiseEstimate { sigma, snap_sigma },
        None => NoiseEstimate {
            sigma: tol / NOISE_FACTOR,
            snap_sigma: 0.0,
        },
    }
}

struct Stats {
    rms: f64,
    ss: f64,
    dof: f64,
    faces: usize,
    planar: bool,
}

fn region_stats(v: &[V3], regions: &[Region]) -> Vec<Stats> {
    let mut stats: Vec<Stats> = Vec::new();
    for r in regions {
        let nv = r.verts.len();
        if r.area <= 0.0 || nv < 4 || !r.surface.is_analytic() {
            continue;
        }
        let dof = nv.saturating_sub(r.surface.param_count());
        if dof == 0 {
            continue;
        }
        let ss: f64 = r
            .verts
            .iter()
            .map(|&(vi, _)| r.surface.distance(v[vi as usize]).powi(2))
            .sum();
        let dof = dof as f64;
        stats.push(Stats {
            rms: (ss / dof).sqrt(),
            ss,
            dof,
            faces: r.faces.len(),
            planar: r.surface.as_plane().is_some(),
        });
    }
    stats.sort_by(|a, b| a.rms.total_cmp(&b.rms));
    stats
}

fn pooled(stats: &[Stats]) -> f64 {
    let ss: f64 = stats.iter().map(|x| x.ss).sum();
    let dof: f64 = stats.iter().map(|x| x.dof).sum();
    (ss / dof).sqrt()
}

fn pool_noise(v: &[V3], regions: &[Region], floor: f64) -> Option<(f64, f64)> {
    let stats = region_stats(v, regions);
    if stats.is_empty() {
        return None;
    }
    if stats.len() == 1 {
        let sigma = pooled(&stats);
        return Some((sigma, sigma));
    }
    let mut cut = stats.len();
    let mut best = POOL_GAP_RATIO;
    for i in 0..stats.len().saturating_sub(1) {
        let gap = stats[i + 1].rms / stats[i].rms.max(floor * 1e-9);
        if gap > best {
            best = gap;
            cut = i + 1;
        }
    }
    let quiet = &stats[..cut];
    let near_exact_faces: usize = quiet
        .iter()
        .filter(|s| s.planar && s.rms <= NEAR_EXACT_FLOOR_RATIO * floor)
        .map(|s| s.faces)
        .sum();
    if near_exact_faces >= NEAR_EXACT_MIN_FACES {
        let sigma = pooled(quiet);
        return Some((sigma, sigma));
    }
    Some((pooled(&stats), 0.0))
}

#[cfg(test)]
mod tests {
    use super::super::linalg::dot;
    use super::*;

    fn tilted_strip(tilt_deg: f64) -> (Vec<V3>, Region) {
        let t = tilt_deg.to_radians().tan();
        let mut v = Vec::new();
        for i in 0..6 {
            let x = -0.025 + 0.01 * i as f64;
            for &y in &[0.0, 0.5, 1.0] {
                v.push([x, y, x * t]);
            }
        }
        let n = [-t, 0.0, 1.0];
        let l = (t * t + 1.0).sqrt();
        let n = [n[0] / l, 0.0, n[2] / l];
        let verts: Vec<(u32, f64)> = (0..v.len() as u32).map(|i| (i, 1.0)).collect();
        let r = Region {
            faces: vec![0],
            surface: Surface::Plane {
                normal: n,
                offset: 0.0,
            },
            area: 0.05,
            width: 0.05,
            verts,
            rms: 0.0,
            max: 0.0,
        };
        (v, r)
    }

    fn snap_tilt(tilt_deg: f64) -> V3 {
        let (v, r) = tilted_strip(tilt_deg);
        let mut regions = vec![r];
        snap_normals(&v, &mut regions, 1e-3, 1e-4, 0.5, 50.0, 50.0);
        regions[0].surface.as_plane().unwrap().0
    }

    #[test]
    fn one_degree_tilt_on_50um_face_stays_unsnapped() {
        let n = snap_tilt(1.0);
        let cos = dot(n, [0.0, 0.0, 1.0]);
        assert!(
            cos < 0.5_f64.to_radians().cos(),
            "1.0 deg tilt was snapped to the axis"
        );
        assert!((cos - 1.0_f64.to_radians().cos()).abs() < 1e-12);
    }

    #[test]
    fn sub_snap_tilt_still_snaps() {
        let n = snap_tilt(0.2);
        assert_eq!(n, [0.0, 0.0, 1.0]);
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as f64 / (1u64 << 31) as f64
        }
    }

    fn noisy_wall_top(seed: u64, noise: f64) -> (Vec<V3>, Region) {
        let mut rng = Lcg(seed);
        let mut v = Vec::new();
        for ix in 0..3 {
            for iy in 0..11 {
                v.push([
                    -0.025 + 0.025 * ix as f64,
                    iy as f64 * 0.1,
                    (rng.next() * 2.0 - 1.0) * noise,
                ]);
            }
        }
        let list: Vec<(u32, f64)> = (0..v.len() as u32).map(|i| (i, 1.0)).collect();
        let (n, d, width) = super::super::fit::irls_plane(&v, &list, [0.0, 0.0, 1.0], 0.05);
        let r = Region {
            faces: vec![0],
            surface: Surface::Plane {
                normal: n,
                offset: d,
            },
            area: 0.1,
            width,
            verts: list,
            rms: 0.0,
            max: 0.0,
        };
        (v, r)
    }

    #[test]
    fn noisy_50um_wall_top_snaps() {
        let mut snapped = 0;
        for seed in 0..12 {
            let noise = 0.005 + (seed as f64) * 0.005 / 11.0;
            let (v, r) = noisy_wall_top(seed, noise);
            let regions = vec![r];
            let est = estimate_noise(&v, &regions, 0.05, 1e-9);
            let mut regions = regions;
            let tol = 5.0 * est.sigma;
            snap_normals(&v, &mut regions, tol, est.snap_sigma, 0.5, 50.0, 50.0);
            if regions[0].surface.as_plane().unwrap().0 == [0.0, 0.0, 1.0] {
                snapped += 1;
            }
        }
        assert!(snapped >= 11, "snapped {snapped}/12");
    }

    fn tilted_patch(tilt_deg: f64, width: f64) -> (Vec<V3>, Region) {
        let t = tilt_deg.to_radians().tan();
        let mut v = Vec::new();
        for i in 0..6 {
            let x = -width / 2.0 + width * i as f64 / 5.0;
            for &y in &[0.0, 0.5, 1.0] {
                v.push([x, y, (x + width / 2.0) * t]);
            }
        }
        let n = [-t, 0.0, 1.0];
        let l = (t * t + 1.0).sqrt();
        let n = [n[0] / l, 0.0, n[2] / l];
        let verts: Vec<(u32, f64)> = (0..v.len() as u32).map(|i| (i, 1.0)).collect();
        let r = Region {
            faces: vec![0],
            surface: Surface::Plane {
                normal: n,
                offset: 0.0,
            },
            area: width,
            width,
            verts,
            rms: 0.0,
            max: 0.0,
        };
        (v, r)
    }

    #[test]
    fn deliberate_06deg_tilt_stays_unsnapped_at_all_scales() {
        for width in [20.0, 0.2, 0.05] {
            let (v, r) = tilted_patch(0.6, width);
            let mut regions = vec![r];
            snap_normals(&v, &mut regions, 1e-3, 0.0, 0.5, 50.0, 50.0);
            let n = regions[0].surface.as_plane().unwrap().0;
            let cos = dot(n, [0.0, 0.0, 1.0]);
            assert!(
                cos < 0.5_f64.to_radians().cos(),
                "width {width}: 0.6 deg tilt was snapped"
            );
        }
    }

    fn pool_case(bases: &[f64], areas: &[f64]) -> NoiseEstimate {
        pool_case_nv(bases, areas, &vec![60; bases.len()], 1.0)
    }

    fn pool_case_nv(bases: &[f64], areas: &[f64], nvs: &[usize], floor: f64) -> NoiseEstimate {
        pool_case_nfaces(bases, areas, nvs, nvs, floor)
    }

    fn pool_case_nfaces(
        bases: &[f64],
        areas: &[f64],
        nvs: &[usize],
        nfaces: &[usize],
        floor: f64,
    ) -> NoiseEstimate {
        let mut v = Vec::new();
        let mut regions = Vec::new();
        for (k, (((&b, &a), &nv), &nf)) in bases.iter().zip(areas).zip(nvs).zip(nfaces).enumerate()
        {
            let mut verts = Vec::new();
            for i in 0..nv {
                let z = b * (1.0 + 0.01 * ((i % 7) as f64 - 3.0));
                v.push([i as f64 * 0.01, k as f64, z]);
                verts.push((v.len() as u32 - 1, 1.0));
            }
            regions.push(Region {
                faces: (0..nf as u32).collect(),
                surface: Surface::Plane {
                    normal: [0.0, 0.0, 1.0],
                    offset: 0.0,
                },
                area: a,
                width: 1.0,
                verts,
                rms: 0.0,
                max: 0.0,
            });
        }
        estimate_noise(&v, &regions, 1.0, floor)
    }

    #[test]
    fn pool_uses_quiet_population_below_a_decade_gap() {
        let est = pool_case(&[1e-9, 2e-9, 1.5e-9, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        assert!(est.sigma < 1e-7, "sigma={}", est.sigma);
    }

    #[test]
    fn pool_keeps_a_gapless_continuum_whole() {
        let est = pool_case(&[0.001, 0.003, 0.01, 0.03], &[1.0, 1.0, 1.0, 1.0]);
        assert!(est.sigma > 0.012, "sigma={}", est.sigma);
    }

    #[test]
    fn pool_ignores_a_chance_tiny_sliver() {
        let est = pool_case_nfaces(
            &[1e-12, 0.005, 0.006, 0.007],
            &[0.01, 1.0, 1.0, 1.0],
            &[60, 60, 60, 60],
            &[2, 60, 60, 60],
            1.0,
        );
        assert!((0.004..0.008).contains(&est.sigma), "sigma={}", est.sigma);
        assert_eq!(est.snap_sigma, 0.0);
    }

    #[test]
    fn pool_trusts_quiet_set_despite_small_area_share() {
        let est = pool_case_nv(&[1e-9, 1.0], &[1.0, 3.0], &[60, 60], 1e-3);
        assert!(est.sigma < 1e-6, "sigma={}", est.sigma);
        assert_eq!(est.sigma, est.snap_sigma);
    }

    #[test]
    fn pool_falls_back_only_when_no_near_exact_region() {
        let est = pool_case_nv(&[0.5, 1.0], &[1.0, 3.0], &[60, 60], 1e-3);
        assert!(est.sigma > 0.1, "sigma={}", est.sigma);
        assert_eq!(est.snap_sigma, 0.0);
    }

    #[test]
    fn pool_ignores_lone_quad_strip_without_widening() {
        let est = pool_case_nfaces(&[1e-12, 1.0], &[0.01, 1.0], &[4, 60], &[2, 60], 1e-3);
        assert!(est.sigma > 0.1, "sigma={}", est.sigma);
        assert_eq!(est.snap_sigma, 0.0);
    }

    #[test]
    fn pool_uses_near_exact_pool_despite_large_curved_area() {
        let est = pool_case_nv(&[1e-9, 1.0, 1.2], &[1.0, 10.0, 10.0], &[60, 60, 60], 1e-3);
        assert!(est.sigma < 1e-6, "sigma={}", est.sigma);
        assert_eq!(est.sigma, est.snap_sigma);
    }

    #[test]
    fn pool_keeps_small_quiet_set_when_near_exact_exists() {
        let est = pool_case_nv(&[1e-9, 1.0, 1.2], &[1.0, 10.0, 10.0], &[60, 30, 30], 1e-3);
        assert!(est.sigma < 1e-7, "sigma={}", est.sigma);
        assert_eq!(est.sigma, est.snap_sigma);
    }

    #[test]
    fn pool_falls_back_without_widening_when_nothing_near_exact() {
        let est = pool_case_nv(&[0.5, 1.0, 1.2], &[1.0, 10.0, 10.0], &[60, 30, 30], 1e-3);
        assert!(est.sigma > 0.1, "sigma={}", est.sigma);
        assert_eq!(est.snap_sigma, 0.0);
    }
}
