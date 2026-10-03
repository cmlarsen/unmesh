use super::linalg::{Moments, V3, add, dot, scale, sub, unit};
use super::topology::NONE;

pub struct TriInfo {
    pub normal: V3,
    pub area: f64,
    pub min_alt: f64,
}

pub fn tri_info(v: &[V3], f: &[[u32; 3]]) -> Vec<TriInfo> {
    f.iter()
        .map(|t| {
            let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            let n = super::linalg::cross(sub(b, a), sub(c, a));
            let len = super::linalg::norm(n);
            let longest = [sub(b, a), sub(c, b), sub(a, c)]
                .iter()
                .map(|e| super::linalg::norm(*e))
                .fold(0.0, f64::max);
            if len > 0.0 && longest > 0.0 {
                TriInfo {
                    normal: scale(n, 1.0 / len),
                    area: len / 2.0,
                    min_alt: len / longest,
                }
            } else {
                TriInfo {
                    normal: [0.0; 3],
                    area: 0.0,
                    min_alt: 0.0,
                }
            }
        })
        .collect()
}

const BASE_ANGLE: f64 = 5.0 * std::f64::consts::PI / 180.0;

struct Grower<'a> {
    v: &'a [V3],
    f: &'a [[u32; 3]],
    info: &'a [TriInfo],
    tol: f64,
    strict: bool,
}

impl Grower<'_> {
    fn accept(&self, g: usize, n: V3, c: V3, count: usize) -> bool {
        let limit = if self.strict {
            self.tol
        } else {
            self.tol * (1.0 + 2.0 / (count as f64).sqrt())
        };
        let t = self.f[g];
        for &i in &t {
            if dot(n, sub(self.v[i as usize], c)).abs() > limit {
                return false;
            }
        }
        let info = &self.info[g];
        if info.area > 0.0 {
            let cosang = dot(n, info.normal);
            if cosang <= 0.0 {
                return false;
            }
            let allow = BASE_ANGLE + (2.0 * self.tol).atan2(info.min_alt);
            if allow < std::f64::consts::FRAC_PI_2 && cosang < allow.cos() {
                return false;
            }
        }
        true
    }

    fn tri_points(&self, g: usize) -> [V3; 3] {
        let t = self.f[g];
        [
            self.v[t[0] as usize],
            self.v[t[1] as usize],
            self.v[t[2] as usize],
        ]
    }
}

pub fn run(
    v: &[V3],
    f: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    eligible: &[bool],
    tol: f64,
    strict: bool,
) -> (Vec<u32>, usize) {
    run_fenced(v, f, nbr, info, eligible, None, tol, strict)
}

pub fn run_fenced(
    v: &[V3],
    f: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    eligible: &[bool],
    fence: Option<&[u32]>,
    tol: f64,
    strict: bool,
) -> (Vec<u32>, usize) {
    let grower = Grower {
        v,
        f,
        info,
        tol,
        strict,
    };
    let mut label = vec![NONE; f.len()];
    let mut seeds: Vec<u32> = (0..f.len() as u32)
        .filter(|&i| eligible[i as usize])
        .collect();
    seeds.sort_unstable_by(|&a, &b| {
        info[b as usize]
            .min_alt
            .total_cmp(&info[a as usize].min_alt)
            .then(a.cmp(&b))
    });
    let mut n_regions = 0usize;
    for &seed in &seeds {
        if label[seed as usize] != NONE {
            continue;
        }
        let rid = n_regions as u32;
        n_regions += 1;
        label[seed as usize] = rid;
        let mut mom = Moments::default();
        let mut nsum = scale(info[seed as usize].normal, info[seed as usize].area);
        mom.add_tri(grower.tri_points(seed as usize), info[seed as usize].area);
        let mut count = 1usize;
        let mut next_refit = 2usize;
        if info[seed as usize].area == 0.0 {
            continue;
        }
        let (mut n, mut c) = seed_plane(&grower, seed);
        let mut stack = vec![seed];
        let mut deferred: Vec<u32> = Vec::new();
        loop {
            while let Some(cur) = stack.pop() {
                for k in 0..3 {
                    let g = nbr[cur as usize][k];
                    if g == NONE || label[g as usize] != NONE || !eligible[g as usize] {
                        continue;
                    }
                    if let Some(fence) = fence
                        && fence[g as usize] != fence[cur as usize]
                    {
                        continue;
                    }
                    if grower.accept(g as usize, n, c, count) {
                        label[g as usize] = rid;
                        let ti = &info[g as usize];
                        mom.add_tri(grower.tri_points(g as usize), ti.area);
                        nsum = [
                            nsum[0] + ti.normal[0] * ti.area,
                            nsum[1] + ti.normal[1] * ti.area,
                            nsum[2] + ti.normal[2] * ti.area,
                        ];
                        count += 1;
                        stack.push(g);
                        if count >= next_refit {
                            (n, c) = refit(&mom, nsum, n, c);
                            next_refit = count + count / 4 + 1;
                        }
                    } else {
                        deferred.push(g);
                    }
                }
            }
            (n, c) = refit(&mom, nsum, n, c);
            deferred.sort_unstable();
            deferred.dedup();
            let mut still = Vec::new();
            let mut progressed = false;
            for g in deferred.drain(..) {
                if label[g as usize] != NONE {
                    continue;
                }
                if grower.accept(g as usize, n, c, count) {
                    label[g as usize] = rid;
                    let ti = &info[g as usize];
                    mom.add_tri(grower.tri_points(g as usize), ti.area);
                    nsum = [
                        nsum[0] + ti.normal[0] * ti.area,
                        nsum[1] + ti.normal[1] * ti.area,
                        nsum[2] + ti.normal[2] * ti.area,
                    ];
                    stack.push(g);
                    progressed = true;
                } else {
                    still.push(g);
                }
            }
            deferred = still;
            if !progressed {
                break;
            }
        }
    }
    (label, n_regions)
}

fn seed_plane(grower: &Grower, seed: u32) -> (V3, V3) {
    let t = grower.tri_points(seed as usize);
    let c = scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
    (grower.info[seed as usize].normal, c)
}

fn refit(mom: &Moments, nsum: V3, n_prev: V3, c_prev: V3) -> (V3, V3) {
    match mom.plane() {
        Some((mut n, c, vals)) if vals[1] > 1e-24 => {
            if dot(n, nsum) < 0.0 {
                n = scale(n, -1.0);
            }
            (unit(n), c)
        }
        _ => (n_prev, c_prev),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hinge_pair() -> (Vec<V3>, Vec<[u32; 3]>, Vec<[u32; 3]>) {
        hinge_pair_deg(1.0)
    }

    fn hinge_pair_deg(deg: f64) -> (Vec<V3>, Vec<[u32; 3]>, Vec<[u32; 3]>) {
        let t = deg.to_radians().tan();
        let v = vec![
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 3.0 * t],
            [3.0, 0.0, 3.0 * t],
        ];
        let f = vec![[0, 1, 2], [3, 5, 4]];
        let nbr = vec![[1, NONE, NONE], [0, NONE, NONE]];
        (v, f, nbr)
    }

    fn labels(strict: bool) -> Vec<u32> {
        let (v, f, nbr) = hinge_pair();
        let info = tri_info(&v, &f);
        let eligible = vec![true; 2];
        run(&v, &f, &nbr, &info, &eligible, 0.03, strict).0
    }

    #[test]
    fn lenient_growth_merges_early_but_strict_does_not() {
        let lax = labels(false);
        assert_eq!(lax[0], lax[1]);
        let strict = labels(true);
        assert_ne!(strict[0], strict[1]);
    }

    #[test]
    fn fenced_run_keeps_growth_inside_fence() {
        let (v, f, nbr) = hinge_pair_deg(0.0);
        let info = tri_info(&v, &f);
        let eligible = vec![true; 2];
        let (open, n_open) = run(&v, &f, &nbr, &info, &eligible, 0.03, false);
        assert_eq!((open, n_open), (vec![0, 0], 1));
        let fence = vec![0u32, 1];
        let (fenced, n_fenced) =
            run_fenced(&v, &f, &nbr, &info, &eligible, Some(&fence), 0.03, false);
        assert_eq!((fenced, n_fenced), (vec![0, 1], 2));
    }
}
