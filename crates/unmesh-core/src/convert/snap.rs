use super::fit::{Region, offset_fit, residual};
use super::linalg::{V3, dot, scale, sub, unit};
use super::surface::Surface;

pub const NOISE_FACTOR: f64 = 5.0;

const ANCHOR_MIN_VERTS: usize = 50;
const POOL_WINDOW_MIN_AREA_FRAC: f64 = 0.5;
const POOL_GAP_RATIO: f64 = 10.0;

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

pub fn estimate_noise(v: &[V3], regions: &[Region], tol: f64) -> f64 {
    pool_noise(v, regions, ANCHOR_MIN_VERTS)
        .or_else(|| pool_noise(v, regions, 4))
        .unwrap_or(tol / NOISE_FACTOR)
}

fn region_stats(v: &[V3], regions: &[Region], min_verts: usize) -> Vec<(f64, f64, f64, f64)> {
    let mut stats: Vec<(f64, f64, f64, f64)> = Vec::new();
    for r in regions {
        let nv = r.verts.len();
        if r.area <= 0.0 || nv < min_verts || !r.surface.is_analytic() {
            continue;
        }
        let ss: f64 = r
            .verts
            .iter()
            .map(|&(vi, _)| r.surface.distance(v[vi as usize]).powi(2))
            .sum();
        let dof = (nv - r.surface.param_count()) as f64;
        stats.push(((ss / dof).sqrt(), ss, dof, r.area));
    }
    stats.sort_by(|a, b| a.0.total_cmp(&b.0));
    stats
}

fn pooled(stats: &[(f64, f64, f64, f64)]) -> f64 {
    let ss: f64 = stats.iter().map(|x| x.1).sum();
    let dof: f64 = stats.iter().map(|x| x.2).sum();
    (ss / dof).sqrt()
}

fn pool_noise(v: &[V3], regions: &[Region], min_verts: usize) -> Option<f64> {
    let all = region_stats(v, regions, 4);
    if all.is_empty() {
        return None;
    }
    let sup = region_stats(v, regions, min_verts);
    if sup.is_empty() {
        return None;
    }
    let total_area: f64 = sup.iter().map(|x| x.3).sum();
    let mut cut = sup.len();
    let mut best = POOL_GAP_RATIO;
    let mut below_area = 0.0;
    for i in 0..sup.len() - 1 {
        below_area += sup[i].3;
        if below_area < POOL_WINDOW_MIN_AREA_FRAC * total_area {
            continue;
        }
        let gap = sup[i + 1].0 / sup[i].0.max(f64::MIN_POSITIVE);
        if gap > best {
            best = gap;
            cut = i + 1;
        }
    }
    let kept_area: f64 = sup[..cut].iter().map(|x| x.3).sum();
    let supported: f64 = all.iter().map(|x| x.3).sum();
    if kept_area >= POOL_WINDOW_MIN_AREA_FRAC * supported {
        Some(pooled(&sup[..cut]))
    } else {
        Some(pooled(&all))
    }
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
            let sigma = estimate_noise(&v, &regions, 0.05);
            let mut regions = regions;
            let tol = 5.0 * sigma;
            snap_normals(&v, &mut regions, tol, sigma, 0.5, 50.0, 50.0);
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

    fn pool_case(bases: &[f64], areas: &[f64]) -> f64 {
        let mut v = Vec::new();
        let mut regions = Vec::new();
        for (k, (&b, &a)) in bases.iter().zip(areas).enumerate() {
            let mut verts = Vec::new();
            for i in 0..60 {
                let z = b * (1.0 + 0.01 * ((i % 7) as f64 - 3.0));
                v.push([i as f64 * 0.01, k as f64, z]);
                verts.push((v.len() as u32 - 1, 1.0));
            }
            regions.push(Region {
                faces: vec![0],
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
        estimate_noise(&v, &regions, 1.0)
    }

    #[test]
    fn pool_uses_quiet_population_below_a_decade_gap() {
        let sigma = pool_case(&[1e-9, 2e-9, 1.5e-9, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        assert!(sigma < 1e-7, "sigma={sigma}");
    }

    #[test]
    fn pool_keeps_a_gapless_continuum_whole() {
        let sigma = pool_case(&[0.001, 0.003, 0.01, 0.03], &[1.0, 1.0, 1.0, 1.0]);
        assert!(sigma > 0.012, "sigma={sigma}");
    }

    #[test]
    fn pool_ignores_a_chance_tiny_sliver() {
        let sigma = pool_case(&[1e-12, 0.005, 0.006, 0.007], &[0.01, 1.0, 1.0, 1.0]);
        assert!((0.004..0.008).contains(&sigma), "sigma={sigma}");
    }

    #[test]
    fn pool_falls_back_to_all_when_quiet_set_is_small() {
        let sigma = pool_case(&[1e-9, 1.0], &[1.0, 3.0]);
        assert!(sigma > 0.1, "sigma={sigma}");
    }
}
