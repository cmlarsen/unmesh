use super::linalg::{V3, cross, dot, norm, scale, sub, unit};
use super::segment::TriInfo;
use super::topology::NONE;

const SMOOTH_DIHEDRAL_DEG: f64 = 15.0;
const MAX_TILT_DEG: f64 = 60.0;
const MAX_PATCH_FACES: usize = 256;
const KEEP_FRACTION: f64 = 0.75;
const LTS_MIN_POINTS: usize = 12;
const MIN_DOF: usize = 3;
const C_STEPS: usize = 2;
const NP: usize = 6;
const LTS_SELECTION_SHRINK: f64 = 0.77;
const MEDIAN_CAP: f64 = 4.0;
const POOL_INFO: f64 = 200.0;
const CHI2_LOW_Z: f64 = -2.33;
const CHI2_HIGH_Z: f64 = 4.26;
const PLANE_F_CRIT: f64 = 3.0;
const REWEIGHT_STEPS: usize = 10;
const CHOLESKY_FLOOR: f64 = 1e-6;
const NEAR_RANK: usize = 6;
const NEAR_FACTOR: f64 = 3.0;
const BLIND_RATIO: f64 = 10.0;
const EVIDENCE_RATIO: f64 = 1.1;
const OUTLIER_RATIO: f64 = 15.0;
const CURVE_SPREAD: f64 = 0.02;
const FLAT_F_CRIT: f64 = 10.0;

#[derive(Clone, Copy)]
struct Sample {
    var: f64,
    dof: usize,
    points: usize,
    pool: (f64, usize),
}

impl Sample {
    fn median_unbiased(&self) -> f64 {
        self.var.sqrt() / chi2_quantile(self.dof, 0.0).sqrt()
    }
}

struct Entry {
    plain: Sample,
    r2: Option<f64>,
    weight: f64,
}

fn aggregate(samples: &mut [(f64, f64, Sample)]) -> Option<f64> {
    let median = weighted_median(samples)?;
    let info: f64 = samples
        .iter()
        .map(|(_, _, fit)| fit.pool.1 as f64 / fit.points as f64)
        .sum();
    let pooled = pooled_sigma(samples, median);
    let shrink = POOL_INFO * POOL_INFO / (POOL_INFO * POOL_INFO + info * info);
    Some((shrink * pooled * pooled + (1.0 - shrink) * median * median).sqrt())
}

fn chi2_quantile(dof: usize, z: f64) -> f64 {
    let h = 2.0 / (9.0 * dof as f64);
    (1.0 - h + z * h.sqrt()).max(0.0).powi(3)
}

struct Marks {
    mark: Vec<u32>,
    token: u32,
}

impl Marks {
    fn new(n: usize) -> Self {
        Self {
            mark: vec![0; n],
            token: 0,
        }
    }

    fn next(&mut self) -> u32 {
        self.token += 1;
        self.token
    }

    fn set(&mut self, i: usize, t: u32) -> bool {
        let was = self.mark[i] == t;
        self.mark[i] = t;
        !was
    }

    fn has(&self, i: usize, t: u32) -> bool {
        self.mark[i] == t
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

pub fn estimate_sigma(
    v: &[V3],
    faces: &[[u32; 3]],
    nbr: &[[u32; 3]],
    info: &[TriInfo],
    eligible: &[bool],
) -> Option<f64> {
    let usable = |f: usize| eligible[f] && info[f].area > 0.0;
    let mut start = vec![0u32; v.len() + 1];
    for (f, t) in faces.iter().enumerate() {
        if usable(f) {
            for &vi in t {
                start[vi as usize + 1] += 1;
            }
        }
    }
    for i in 0..v.len() {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut inc = vec![0u32; start[v.len()] as usize];
    for (f, t) in faces.iter().enumerate() {
        if usable(f) {
            for &vi in t {
                inc[fill[vi as usize] as usize] = f as u32;
                fill[vi as usize] += 1;
            }
        }
    }

    let smooth = SMOOTH_DIHEDRAL_DEG.to_radians().cos();
    let tilt = MAX_TILT_DEG.to_radians().cos();
    let mut in_fan = Marks::new(faces.len());
    let mut seen_face = Marks::new(faces.len());
    let mut ring = Marks::new(v.len());
    let mut seen_vert = Marks::new(v.len());
    let mut entries: Vec<Entry> = Vec::new();
    let mut pts: Vec<V3> = Vec::new();
    let mut dist: Vec<f64> = Vec::new();
    let mut queue: Vec<u32> = Vec::new();
    let mut parent: Vec<usize> = Vec::new();

    for vi in 0..v.len() {
        let fan = &inc[start[vi] as usize..start[vi + 1] as usize];
        if fan.is_empty() {
            continue;
        }
        let ft = in_fan.next();
        for &f in fan {
            in_fan.set(f as usize, ft);
        }
        parent.clear();
        parent.extend(0..fan.len());
        for (i, &f) in fan.iter().enumerate() {
            for &g in &nbr[f as usize] {
                if g == NONE || !in_fan.has(g as usize, ft) {
                    continue;
                }
                if dot(info[f as usize].normal, info[g as usize].normal) < smooth {
                    continue;
                }
                let j = fan.iter().position(|&x| x == g).unwrap();
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
        let nsectors = (0..fan.len())
            .filter(|&r| find(&mut parent, r) == r)
            .count();
        let multi = nsectors > 1;
        for root in 0..fan.len() {
            if find(&mut parent, root) != root {
                continue;
            }
            let sector: Vec<u32> = (0..fan.len())
                .filter(|&i| find(&mut parent, i) == root)
                .map(|i| fan[i])
                .collect();
            let mut acc = [0.0; 3];
            let mut area = 0.0;
            let rt = ring.next();
            for &f in &sector {
                let fi = &info[f as usize];
                acc = [
                    acc[0] + fi.area * fi.normal[0],
                    acc[1] + fi.area * fi.normal[1],
                    acc[2] + fi.area * fi.normal[2],
                ];
                area += fi.area;
                for &u in &faces[f as usize] {
                    ring.set(u as usize, rt);
                }
            }
            if norm(acc) <= 0.0 {
                continue;
            }
            let n = unit(acc);
            let st = seen_face.next();
            queue.clear();
            for &f in &sector {
                seen_face.set(f as usize, st);
                queue.push(f);
            }
            let mut head = 0;
            while head < queue.len() && queue.len() < MAX_PATCH_FACES {
                let f = queue[head] as usize;
                head += 1;
                for &g in &nbr[f] {
                    if g == NONE {
                        continue;
                    }
                    let gi = g as usize;
                    if seen_face.has(gi, st) || !usable(gi) {
                        continue;
                    }
                    let ng = info[gi].normal;
                    if dot(info[f].normal, ng) < smooth || dot(n, ng) < tilt {
                        continue;
                    }
                    if !faces[gi].iter().any(|&u| ring.has(u as usize, rt)) {
                        continue;
                    }
                    seen_face.set(gi, st);
                    queue.push(g);
                }
            }
            let pt = seen_vert.next();
            pts.clear();
            for &f in &queue {
                for &u in &faces[f as usize] {
                    if seen_vert.set(u as usize, pt) {
                        pts.push(v[u as usize]);
                    }
                }
            }
            if !near_points(v[vi], &mut pts, &mut dist) {
                continue;
            }
            if let Some((plain, r2)) = quadric_fit(v[vi], n, &pts, multi) {
                entries.push(Entry {
                    plain,
                    r2,
                    weight: area / 3.0,
                });
            }
        }
    }
    let mut base_in: Vec<(f64, f64, Sample)> = entries
        .iter()
        .map(|e| (e.plain.median_unbiased(), e.weight, e.plain))
        .collect();
    let base = aggregate(&mut base_in)?;
    let mut large_plain: Vec<f64> = entries
        .iter()
        .filter(|e| e.plain.points >= NP)
        .map(|e| e.plain.median_unbiased())
        .collect();
    large_plain.sort_by(f64::total_cmp);
    let bulk = if large_plain.is_empty() {
        base
    } else {
        large_plain[large_plain.len() / 2]
    };
    let r2_sample = |e: &Entry| match e.r2 {
        Some(r2) => Sample {
            var: r2 * chi2_quantile(1, 0.0),
            dof: 1,
            points: e.plain.points,
            pool: (r2, 1),
        },
        None => e.plain,
    };
    let mut crease_in: Vec<(f64, f64, Sample)> = entries
        .iter()
        .filter(|e| e.plain.points >= NP)
        .filter(|e| e.plain.median_unbiased() <= OUTLIER_RATIO * bulk)
        .map(|e| {
            let sample = r2_sample(e);
            (sample.median_unbiased(), e.weight, sample)
        })
        .collect();
    let crease = aggregate(&mut crease_in).unwrap_or(0.0);
    let mut all_in: Vec<(f64, f64, Sample)> = entries
        .iter()
        .filter(|e| e.plain.median_unbiased() <= OUTLIER_RATIO * bulk)
        .map(|e| {
            let sample = r2_sample(e);
            (sample.median_unbiased(), e.weight, sample)
        })
        .collect();
    let all = aggregate(&mut all_in).unwrap_or(base);
    if base > 0.0 && all > EVIDENCE_RATIO * base {
        Some(base.max(all).max(crease))
    } else {
        Some(base)
    }
}

fn near_points(center: V3, pts: &mut Vec<V3>, dist: &mut Vec<f64>) -> bool {
    dist.clear();
    dist.extend(pts.iter().map(|&p| norm(sub(p, center))));
    let mut sorted = dist.clone();
    sorted.sort_by(f64::total_cmp);
    let Some(&dk) = sorted.iter().filter(|&&d| d > 0.0).nth(NEAR_RANK - 1) else {
        return true;
    };
    let cut = NEAR_FACTOR * dk;
    let before = pts.len();
    let mut i = 0;
    pts.retain(|_| {
        i += 1;
        dist[i - 1] <= cut
    });
    pts.len() == before || pts.len() >= NP + MIN_DOF
}

fn pooled_sigma(samples: &[(f64, f64, Sample)], median: f64) -> f64 {
    let mut sigma = median;
    for _ in 0..REWEIGHT_STEPS {
        let (mut num, mut den) = (0.0, 0.0);
        let var = sigma * sigma;
        for (_, _, fit) in samples {
            let (v, dof) = fit.pool;
            if v > var * chi2_quantile(dof, CHI2_LOW_Z).max(1e-6)
                && v <= var * chi2_quantile(dof, CHI2_HIGH_Z)
            {
                let w = dof as f64 / fit.points as f64;
                num += w * v;
                den += w;
            }
        }
        if den <= 0.0 {
            break;
        }
        sigma = (num / den).sqrt().min(MEDIAN_CAP * median);
    }
    sigma
}

fn weighted_median(samples: &mut [(f64, f64, Sample)]) -> Option<f64> {
    let total: f64 = samples.iter().map(|s| s.1).sum();
    if samples.is_empty() || total <= 0.0 {
        return None;
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut acc = 0.0;
    for (r, w, _) in samples.iter() {
        acc += w;
        if acc >= 0.5 * total {
            return Some(*r);
        }
    }
    samples.last().map(|s| s.0)
}

fn frame(n: V3) -> (V3, V3) {
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = unit(sub(a, scale(n, dot(a, n))));
    (t1, cross(n, t1))
}

fn quadric_fit(center: V3, n: V3, pts: &[V3], multi: bool) -> Option<(Sample, Option<f64>)> {
    let (t1, t2) = frame(n);
    let mut local: Vec<(f64, f64, f64)> = pts
        .iter()
        .map(|&p| {
            let d = sub(p, center);
            (dot(d, t1), dot(d, t2), dot(d, n))
        })
        .collect();
    let s = local
        .iter()
        .map(|&(x, y, _)| x.abs().max(y.abs()))
        .fold(0.0, f64::max);
    if s <= 0.0 {
        return None;
    }
    for p in local.iter_mut() {
        p.0 /= s;
        p.1 /= s;
    }
    let pair = |sample: Sample| (sample, unseen_center(sample, &local, multi));
    let empty = Nulls::empty();
    let all: Vec<usize> = (0..local.len()).collect();
    let (mut coef, mut rank) = solve(&local, &all, NP, &empty)?;
    let (plane, plane_rank) = solve(&local, &all, 3, &empty)?;
    let plane_ss: f64 = local.iter().map(|p| resid(&plane, *p).powi(2)).sum();
    let plane_fit = || {
        let dof = local.len().checked_sub(plane_rank).filter(|&d| d > 0)?;
        Some(Sample {
            var: plane_ss / dof as f64,
            dof,
            points: local.len(),
            pool: (plane_ss / dof as f64, dof),
        })
    };
    if local.len() < rank + MIN_DOF {
        return plane_fit().map(pair);
    }
    let quad_ss: f64 = local.iter().map(|p| resid(&coef, *p).powi(2)).sum();
    let extra = rank.saturating_sub(plane_rank);
    let plane_suffices = extra == 0
        || (plane_ss - quad_ss) * ((local.len() - rank) as f64)
            <= PLANE_F_CRIT * extra as f64 * quad_ss;
    if plane_suffices && local.len() < LTS_MIN_POINTS {
        return plane_fit().map(pair);
    }
    let plane_pool = plane_suffices
        .then(|| plane_fit().map(|p| p.pool))
        .flatten();
    let keep = if local.len() < LTS_MIN_POINTS {
        local.len()
    } else {
        ((KEEP_FRACTION * local.len() as f64).ceil() as usize)
            .max(rank + MIN_DOF)
            .min(local.len())
    };
    let mut order: Vec<(f64, usize)> = Vec::with_capacity(local.len());
    let mut subset: Vec<usize> = Vec::with_capacity(keep);
    for _ in 0..C_STEPS {
        order.clear();
        order.extend(
            local
                .iter()
                .enumerate()
                .map(|(i, p)| (resid(&coef, *p).abs(), i)),
        );
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        subset.clear();
        subset.extend(order[..keep].iter().map(|x| x.1));
        (coef, rank) = solve(&local, &subset, NP, &empty)?;
    }
    let mut res: Vec<f64> = local.iter().map(|p| resid(&coef, *p).powi(2)).collect();
    res.sort_by(f64::total_cmp);
    let ss: f64 = res[..keep].iter().sum();
    let dof = keep.checked_sub(rank).filter(|&d| d >= MIN_DOF)?;
    let var = ss / dof as f64 / trimmed_variance(keep, local.len());
    let sample = Sample {
        var,
        dof,
        points: local.len(),
        pool: plane_pool.unwrap_or((var, dof)),
    };
    Some(pair(sample))
}

fn unseen_center(sample: Sample, local: &[(f64, f64, f64)], multi: bool) -> Option<f64> {
    if !multi {
        return None;
    }
    let others: Vec<usize> = (0..local.len())
        .filter(|&i| local[i] != (0.0, 0.0, 0.0))
        .collect();
    let (plane, plane_rank) = solve(local, &others, 3, &Nulls::empty())?;
    if others.len() < plane_rank + MIN_DOF {
        return None;
    }
    let r2 = plane[0] * plane[0];
    if r2 <= BLIND_RATIO * sample.var {
        return None;
    }
    let qnulls = Nulls::of(local, NP);
    let (quad, rank) = solve(local, &others, NP, &qnulls)?;
    if others.len() < rank + MIN_DOF {
        return None;
    }
    let plane_ss: f64 = others
        .iter()
        .map(|&i| resid(&plane, local[i]).powi(2))
        .sum();
    let quad_ss: f64 = others.iter().map(|&i| resid(&quad, local[i]).powi(2)).sum();
    let extra = rank.saturating_sub(plane_rank);
    let flat = extra == 0
        || (plane_ss - quad_ss) * ((others.len() - rank) as f64)
            <= FLAT_F_CRIT * extra as f64 * quad_ss;
    if !flat {
        return None;
    }
    Some(r2)
}

fn basis(x: f64, y: f64) -> [f64; NP] {
    [1.0, x, y, x * x, x * y, y * y]
}

fn resid(c: &[f64; NP], p: (f64, f64, f64)) -> f64 {
    let b = basis(p.0, p.1);
    p.2 - (0..NP).map(|k| c[k] * b[k]).sum::<f64>()
}

struct Nulls {
    quad: Vec<[f64; NP]>,
    plane: Vec<[f64; NP]>,
}

impl Nulls {
    fn of(local: &[(f64, f64, f64)], np: usize) -> Self {
        let spacing = local
            .iter()
            .map(|p| p.0.hypot(p.1))
            .filter(|&d| d > 0.0)
            .fold(f64::INFINITY, f64::min);
        let reach = (CURVE_SPREAD * spacing).powi(2);
        let dirs = unsupported(local, np, reach);
        if np == NP {
            Self {
                quad: dirs,
                plane: Vec::new(),
            }
        } else {
            Self {
                quad: Vec::new(),
                plane: dirs,
            }
        }
    }

    fn empty() -> Self {
        Self {
            quad: Vec::new(),
            plane: Vec::new(),
        }
    }

    fn get(&self, np: usize) -> &[[f64; NP]] {
        if np == NP { &self.quad } else { &self.plane }
    }
}

fn gradients(x: f64, y: f64) -> ([f64; NP], [f64; NP]) {
    (
        [0.0, 1.0, 0.0, 2.0 * x, y, 0.0],
        [0.0, 0.0, 1.0, 0.0, x, 2.0 * y],
    )
}

fn unsupported(local: &[(f64, f64, f64)], np: usize, reach: f64) -> Vec<[f64; NP]> {
    let mut a = [[0.0; NP]; NP];
    let mut g = [[0.0; NP]; NP];
    for &(x, y, _) in local {
        let b = basis(x, y);
        let (gx, gy) = gradients(x, y);
        for r in 0..np {
            for c in 0..np {
                a[r][c] += b[r] * b[c];
                g[r][c] += gx[r] * gx[c] + gy[r] * gy[c];
            }
        }
    }
    let (vals, vecs) = jacobi(a);
    let mut out = Vec::new();
    for k in 0..NP {
        let mut vk = [0.0; NP];
        for r in 0..NP {
            vk[r] = vecs[r][k];
        }
        if vk[np..].iter().any(|&c| c.abs() > 0.5) {
            continue;
        }
        let den: f64 = (0..np)
            .map(|r| (0..np).map(|c| vk[r] * g[r][c] * vk[c]).sum::<f64>())
            .sum();
        if den > 0.0 && vals[k].max(0.0) < reach * den {
            out.push(vk);
        }
    }
    out
}

fn solve(
    local: &[(f64, f64, f64)],
    idx: &[usize],
    np: usize,
    nulls: &Nulls,
) -> Option<([f64; NP], usize)> {
    let (mut a, rhs) = normal_equations(local, idx, np);
    let nulls = nulls.get(np);
    if !nulls.is_empty() {
        let top = (0..np).map(|i| a[i][i]).fold(0.0, f64::max);
        for n in nulls {
            for r in 0..np {
                for c in 0..np {
                    a[r][c] += top * n[r] * n[c];
                }
            }
        }
    }
    let (coef, rank) = solve_system(&a, &rhs, np)?;
    Some((coef, rank.saturating_sub(nulls.len())))
}

fn normal_equations(local: &[(f64, f64, f64)], idx: &[usize], np: usize) -> (M6, [f64; NP]) {
    let mut a = [[0.0; NP]; NP];
    let mut rhs = [0.0; NP];
    for &i in idx {
        let (x, y, z) = local[i];
        let b = basis(x, y);
        for r in 0..np {
            rhs[r] += b[r] * z;
            for c in 0..np {
                a[r][c] += b[r] * b[c];
            }
        }
    }
    (a, rhs)
}

fn solve_system(a: &M6, rhs: &[f64; NP], np: usize) -> Option<([f64; NP], usize)> {
    if let Some(coef) = cholesky(a, rhs, np) {
        return Some((coef, np));
    }
    let (vals, vecs) = jacobi(*a);
    let top = vals.iter().cloned().fold(0.0, f64::max);
    if top <= 0.0 {
        return None;
    }
    let mut coef = [0.0; NP];
    let mut rank = 0;
    for k in 0..NP {
        if vals[k] <= 1e-10 * top {
            continue;
        }
        rank += 1;
        let proj: f64 = (0..NP).map(|r| vecs[r][k] * rhs[r]).sum::<f64>() / vals[k];
        for r in 0..NP {
            coef[r] += proj * vecs[r][k];
        }
    }
    Some((coef, rank))
}

/// The full-rank solve, or `None` when a pivot falls under `CHOLESKY_FLOOR`
/// of the largest diagonal entry (the eigen-decomposition then decides the
/// rank).
fn cholesky(a: &M6, rhs: &[f64; NP], np: usize) -> Option<[f64; NP]> {
    let top = (0..np).map(|i| a[i][i]).fold(0.0, f64::max);
    if top <= 0.0 {
        return None;
    }
    let mut l = [[0.0; NP]; NP];
    for i in 0..np {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= CHOLESKY_FLOOR * top {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0.0; NP];
    for i in 0..np {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0; NP];
    for i in (0..np).rev() {
        let mut s = y[i];
        for k in i + 1..np {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

pub type M6 = [[f64; NP]; NP];

pub fn jacobi(mut a: M6) -> ([f64; NP], M6) {
    let mut v = [[0.0; NP]; NP];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for p in 0..NP {
            diag += a[p][p].abs();
            for q in p + 1..NP {
                off += a[p][q].abs();
            }
        }
        if off <= 1e-300 || off <= 1e-16 * diag {
            break;
        }
        for p in 0..NP {
            for q in p + 1..NP {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..NP {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..NP {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut vals = [0.0; NP];
    for (k, val) in vals.iter_mut().enumerate() {
        *val = a[k][k];
    }
    (vals, v)
}

fn trimmed_variance(keep: usize, points: usize) -> f64 {
    thread_local! {
        static CACHE: std::cell::RefCell<rustc_hash::FxHashMap<(usize, usize), f64>> =
            Default::default();
    }
    CACHE.with(|c| {
        *c.borrow_mut()
            .entry((keep, points))
            .or_insert_with(|| trimmed_variance_of(keep as f64 / points as f64))
    })
}

fn trimmed_variance_of(kept: f64) -> f64 {
    if kept >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0, 10.0);
    for _ in 0..60 {
        let q = 0.5 * (lo + hi);
        if erf(q / std::f64::consts::SQRT_2) < kept {
            lo = q;
        } else {
            hi = q;
        }
    }
    let q = 0.5 * (lo + hi);
    let pdf = (-0.5 * q * q).exp() / (2.0 * std::f64::consts::PI).sqrt();
    LTS_SELECTION_SHRINK * (1.0 - 2.0 * q * pdf / kept)
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::segment::tri_info;
    use super::super::topology;
    use super::*;

    pub fn grid_faces(nx: usize, ny: usize) -> Vec<[u32; 3]> {
        let id = |i: usize, j: usize| (i * ny + j) as u32;
        let mut f = Vec::new();
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                f.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                f.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        f
    }

    pub fn sigma_of(v: &[V3], mut faces: Vec<[u32; 3]>) -> Option<f64> {
        let topo = topology::build(&mut faces);
        let info = tri_info(v, &faces);
        estimate_sigma(v, &faces, &topo.nbr, &info, &vec![true; faces.len()])
    }

    pub fn grid_sigma(v: &[V3], nx: usize, ny: usize) -> f64 {
        sigma_of(v, grid_faces(nx, ny)).unwrap()
    }

    struct Lcg(u64);

    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn cylinder(radius: f64, nu: usize, nv: usize, noise: f64, seed: u64) -> Vec<V3> {
        noisy_cylinder(radius, nu, nv, seed, |rng| noise * rng.uniform())
    }

    fn noisy_cylinder(
        radius: f64,
        nu: usize,
        nv: usize,
        seed: u64,
        mut draw: impl FnMut(&mut Lcg) -> f64,
    ) -> Vec<V3> {
        let mut rng = Lcg(seed);
        let mut v = Vec::new();
        for i in 0..nu {
            let a = (3.0 / radius) * i as f64 / (nu - 1) as f64;
            for j in 0..nv {
                let r = radius + draw(&mut rng);
                v.push([r * a.cos(), r * a.sin(), 10.0 * j as f64 / (nv - 1) as f64]);
            }
        }
        v
    }

    #[test]
    fn clean_curved_tessellation_gives_zero_sigma() {
        for (radius, nv) in [(2.0, 2), (30.0, 2), (2.0, 12)] {
            let v = cylinder(radius, 24, nv, 0.0, 1);
            let sigma = grid_sigma(&v, 24, nv);
            assert!(sigma < 2e-5, "r={radius} nv={nv}: sigma={sigma}");
        }
    }

    #[test]
    fn sigma_tracks_noise_independent_of_curvature() {
        let std = 0.01 / 3f64.sqrt();
        for radius in [2.0, 8.0, 1e6] {
            let v = cylinder(radius, 40, 40, 0.01, 7);
            let sigma = grid_sigma(&v, 40, 40);
            assert!(
                (0.75 * std..1.25 * std).contains(&sigma),
                "r={radius}: sigma={sigma} vs std={std}"
            );
        }
    }

    fn gaussian(rng: &mut Lcg) -> f64 {
        let u = 0.5 * (rng.uniform() + 1.0);
        let w = 0.5 * (rng.uniform() + 1.0);
        (-2.0 * u.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * w).cos()
    }

    #[test]
    fn sigma_matches_gaussian_and_uniform_sd() {
        for radius in [2.0, 8.0, 1e6] {
            for seed in 0..4 {
                let g = noisy_cylinder(radius, 40, 40, seed, |r| 0.004 * gaussian(r));
                let u = noisy_cylinder(radius, 40, 40, seed, |r| 0.01 * r.uniform());
                let rg = grid_sigma(&g, 40, 40) / 0.004;
                let ru = grid_sigma(&u, 40, 40) / (0.01 / 3f64.sqrt());
                assert!(
                    (0.93..1.07).contains(&rg),
                    "r={radius} seed={seed}: gauss {rg}"
                );
                assert!(
                    (0.95..1.12).contains(&ru),
                    "r={radius} seed={seed}: uniform {ru}"
                );
            }
        }
    }

    #[test]
    fn cylinder_to_cone_crease_does_not_read_as_noise() {
        let rings = [(10.0, -8.5), (10.0, 0.0), (14.0, 13.7)];
        let mut v = Vec::new();
        for i in 0..120 {
            let a = 1.5 * i as f64 / 119.0;
            for &(r, z) in &rings {
                v.push([r * a.cos(), r * a.sin(), z]);
            }
        }
        let sigma = grid_sigma(&v, 120, 3);
        assert!(sigma < 1e-5, "sigma={sigma}");
    }

    fn indexed(tris: Vec<[V3; 3]>) -> (Vec<V3>, Vec<[u32; 3]>) {
        let mut v: Vec<V3> = Vec::new();
        let mut faces = Vec::new();
        for t in tris {
            let mut ids = [0u32; 3];
            for (k, p) in t.iter().enumerate() {
                ids[k] = match v.iter().position(|q| q == p) {
                    Some(i) => i as u32,
                    None => {
                        v.push(*p);
                        v.len() as u32 - 1
                    }
                };
            }
            faces.push(ids);
        }
        (v, faces)
    }

    #[test]
    fn sigma_is_stable_on_a_coarse_noisy_box() {
        let sd = 0.002;
        let mut ratios = Vec::new();
        for seed in 0..10 {
            let tris = super::super::tests::grid_box([0.0; 3], [40.0, 30.0, 20.0], 2, false);
            let (mut v, faces) = indexed(tris);
            assert_eq!(faces.len(), 48);
            let mut rng = Lcg(seed);
            for p in v.iter_mut() {
                for c in p.iter_mut() {
                    *c += sd * gaussian(&mut rng);
                }
            }
            let ratio = sigma_of(&v, faces).unwrap() / sd;
            assert!(
                (0.65..1.35).contains(&ratio),
                "seed {seed}: sigma/sd = {ratio}"
            );
            ratios.push(ratio);
        }
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!((0.9..1.1).contains(&mean), "mean sigma/sd = {mean}");
    }

    #[test]
    fn sharp_edges_do_not_read_as_noise() {
        let (v, faces) = indexed(super::super::tests::grid_box(
            [0.0; 3],
            [10.0, 20.0, 30.0],
            4,
            false,
        ));
        assert_eq!(sigma_of(&v, faces), Some(0.0));
    }

    fn vertex_normals(v: &[V3], faces: &[[u32; 3]]) -> Vec<V3> {
        let mut acc = vec![[0.0; 3]; v.len()];
        for t in faces {
            let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            let n = cross(sub(b, a), sub(c, a));
            for &u in t {
                let r = &mut acc[u as usize];
                for k in 0..3 {
                    r[k] += n[k];
                }
            }
        }
        acc.iter()
            .map(|&a| if norm(a) > 0.0 { unit(a) } else { a })
            .collect()
    }

    fn normal_noise(v: &[V3], faces: &[[u32; 3]], amp: f64, seed: u64) -> Vec<V3> {
        let n = vertex_normals(v, faces);
        let mut rng = Lcg(seed);
        v.iter()
            .zip(n)
            .map(|(p, m)| {
                let d = amp * rng.uniform();
                [p[0] + d * m[0], p[1] + d * m[1], p[2] + d * m[2]]
            })
            .collect()
    }

    fn isotropic_noise(v: &[V3], amp: f64, seed: u64) -> Vec<V3> {
        let mut rng = Lcg(seed);
        v.iter()
            .map(|p| {
                let d = [rng.uniform(), rng.uniform(), rng.uniform()];
                let len = norm(d);
                let r = amp * (0.5 * (rng.uniform() + 1.0)).cbrt();
                [
                    p[0] + r * d[0] / len,
                    p[1] + r * d[1] / len,
                    p[2] + r * d[2] / len,
                ]
            })
            .collect()
    }

    fn grid_plane(nx: usize, ny: usize, step: f64) -> (Vec<V3>, Vec<[u32; 3]>) {
        let mut v = Vec::with_capacity(nx * ny);
        for i in 0..nx {
            for j in 0..ny {
                v.push([i as f64 * step, j as f64 * step, 0.0]);
            }
        }
        (v, grid_faces(nx, ny))
    }

    fn fillet_strip() -> (Vec<V3>, Vec<[u32; 3]>) {
        use std::f64::consts::FRAC_PI_2;
        let mut profile = vec![(-3.0, 0.0)];
        for k in 0..=4 {
            let a = -FRAC_PI_2 + FRAC_PI_2 * k as f64 / 4.0;
            profile.push((a.cos(), 1.0 + a.sin()));
        }
        profile.push((1.0, 3.0));
        let m = profile.len();
        let nz = 4;
        let mut v = Vec::with_capacity(nz * m);
        for iz in 0..nz {
            for &(x, y) in &profile {
                v.push([x, y, iz as f64]);
            }
        }
        let id = |iz: usize, j: usize| (iz * m + j) as u32;
        let mut faces = Vec::new();
        for iz in 0..nz - 1 {
            for j in 0..m - 1 {
                let (a, b, c, d) = (id(iz, j), id(iz + 1, j), id(iz + 1, j + 1), id(iz, j + 1));
                faces.push([a, b, c]);
                faces.push([a, c, d]);
            }
        }
        (v, faces)
    }

    fn assert_within_two(_v: &[V3], faces: Vec<[u32; 3]>, noisy: Vec<V3>, sd: f64, what: &str) {
        let sigma = sigma_of(&noisy, faces).unwrap();
        let ratio = sigma / sd;
        assert!((0.5..2.0).contains(&ratio), "{what}: sigma/sd = {ratio}");
    }

    #[test]
    fn plane_normal_and_isotropic_noise_are_within_two() {
        let sd_n = 0.01 / 3f64.sqrt();
        let sd_i = 0.01 / 5f64.sqrt();
        let (v, faces) = grid_plane(24, 24, 1.0);
        for seed in 0..4 {
            let n = normal_noise(&v, &faces, 0.01, seed);
            assert_within_two(&v, faces.clone(), n, sd_n, &format!("normal seed {seed}"));
            let i = isotropic_noise(&v, 0.01, seed);
            assert_within_two(
                &v,
                faces.clone(),
                i,
                sd_i,
                &format!("isotropic seed {seed}"),
            );
        }
    }

    #[test]
    fn cylinder_normal_and_isotropic_noise_are_within_two() {
        let sd_n = 0.01 / 3f64.sqrt();
        let sd_i = 0.01 / 5f64.sqrt();
        for radius in [2.0, 8.0] {
            let v = cylinder(radius, 32, 12, 0.0, 1);
            let faces = grid_faces(32, 12);
            for seed in 0..3 {
                let n = normal_noise(&v, &faces, 0.01, seed);
                assert_within_two(
                    &v,
                    faces.clone(),
                    n,
                    sd_n,
                    &format!("radius {radius} normal seed {seed}"),
                );
                let i = isotropic_noise(&v, 0.01, seed);
                assert_within_two(
                    &v,
                    faces.clone(),
                    i,
                    sd_i,
                    &format!("radius {radius} isotropic seed {seed}"),
                );
            }
        }
    }

    #[test]
    fn fillet_strip_normal_and_isotropic_noise_are_within_two() {
        let sd_n = 0.01 / 3f64.sqrt();
        let sd_i = 0.01 / 5f64.sqrt();
        let (v, faces) = fillet_strip();
        for seed in 0..4 {
            let n = normal_noise(&v, &faces, 0.01, seed);
            assert_within_two(&v, faces.clone(), n, sd_n, &format!("normal seed {seed}"));
            let i = isotropic_noise(&v, 0.01, seed);
            assert_within_two(
                &v,
                faces.clone(),
                i,
                sd_i,
                &format!("isotropic seed {seed}"),
            );
        }
    }

    #[test]
    fn coarse_box_normal_noise_reads_within_two() {
        let sd = 0.002;
        let (v, faces) = indexed(super::super::tests::grid_box(
            [0.0; 3],
            [40.0, 30.0, 20.0],
            1,
            false,
        ));
        for seed in 0..8 {
            let noisy = normal_noise(&v, &faces, 3.0 * sd, seed);
            assert_within_two(&v, faces.clone(), noisy, sd, &format!("seed {seed}"));
        }
    }

    #[test]
    fn clean_meshes_sit_below_the_floor() {
        let (plane_v, plane_f) = grid_plane(20, 20, 1.0);
        let (strip_v, strip_f) = fillet_strip();
        let (box_v, box_f) = indexed(super::super::tests::grid_box(
            [0.0; 3],
            [20.0, 30.0, 10.0],
            3,
            false,
        ));
        for (name, v, faces) in [
            ("plane", plane_v, plane_f),
            ("fillet", strip_v, strip_f),
            ("box", box_v, box_f),
        ] {
            let sigma = sigma_of(&v, faces).unwrap();
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            let mut max_abs = 0.0f64;
            for p in &v {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                    max_abs = max_abs.max(p[k].abs());
                }
            }
            let diag = norm(sub(hi, lo));
            let floor = (1e-6 * diag).max(5e-7 * max_abs);
            assert!(
                5.0 * sigma < floor,
                "{name}: 5 sigma {:.3e} not under floor {floor:.3e}",
                5.0 * sigma
            );
        }
    }
}
