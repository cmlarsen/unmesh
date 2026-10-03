use super::fit::{Region, offset_fit, residual};
use super::linalg::{V3, dot, scale, sub, unit};
use super::surface::Surface;

pub const NOISE_FACTOR: f64 = 5.0;

const ANCHOR_MIN_VERTS: usize = 50;
const ANCHOR_MIN_DOF_FRAC: f64 = 0.25;

pub fn snap_normals(
    v: &[V3],
    regions: &mut [Region],
    tol: f64,
    sigma: f64,
    snap_deg: f64,
    diag: f64,
) {
    let rms_limit = (2.5 * sigma).max(super::floor(diag));
    let n = regions.len();
    let mut done = vec![false; n];
    let mut classes: Vec<V3> = Vec::new();

    let allow = |_r: &Region| -> f64 { snap_deg.to_radians() };
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
    let mut stats: Vec<(f64, f64, f64)> = Vec::new();
    for r in regions {
        let nv = r.verts.len();
        if r.area <= 0.0 || nv < ANCHOR_MIN_VERTS {
            continue;
        }
        let Some((n, d)) = r.surface.as_plane() else {
            continue;
        };
        let ss: f64 = r
            .verts
            .iter()
            .map(|&(vi, _)| (dot(n, v[vi as usize]) - d).powi(2))
            .sum();
        let dof = (nv - 3) as f64;
        stats.push(((ss / dof).sqrt(), ss, dof));
    }
    if stats.is_empty() {
        return tol / NOISE_FACTOR;
    }
    stats.sort_by(|a, b| a.0.total_cmp(&b.0));
    let anchor = stats[0].0;
    let total_dof: f64 = stats.iter().map(|x| x.2).sum();
    let mut keep_ss = 0.0;
    let mut keep_dof = 0.0;
    for x in &stats {
        if x.0 > 10.0 * anchor + 1e-9 {
            break;
        }
        keep_ss += x.1;
        keep_dof += x.2;
    }
    if keep_dof < ANCHOR_MIN_DOF_FRAC * total_dof {
        return tol / NOISE_FACTOR;
    }
    (keep_ss / keep_dof).sqrt()
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
            surface: Surface::Plane { normal: n, offset: 0.0 },
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
        snap_normals(&v, &mut regions, 1e-3, 1e-4, 0.5, 50.0);
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
}
