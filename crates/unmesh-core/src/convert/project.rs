use super::fit::Final;
use super::linalg::{M3, V3, add, dot, norm, outer_add, scale, sub, sym_eigen};
use super::segment::TriInfo;
use super::surface::{Constraint, Surface};

const EIGEN_FLOOR: f64 = 5e-4;
const GAUSS_NEWTON_STEPS: usize = 3;

fn solve(x0: V3, constraints: &[&Constraint]) -> V3 {
    let mut m: M3 = [[0.0; 3]; 3];
    let mut r = [0.0; 3];
    for p in constraints {
        outer_add(&mut m, p.normal, p.normal, 1.0);
        let e = p.offset - dot(p.normal, x0);
        r = add(r, scale(p.normal, e));
    }
    let (vals, vecs) = sym_eigen(m);
    let mut x = x0;
    for j in 0..3 {
        if vals[j] > EIGEN_FLOOR {
            x = add(x, scale(vecs[j], dot(vecs[j], r) / vals[j]));
        }
    }
    x
}

pub fn project_vertex(x0: V3, surfaces: &[(&Surface, f64)], tol: f64) -> V3 {
    let mut sel: Vec<(&Surface, f64)> = surfaces
        .iter()
        .copied()
        .filter(|(s, _)| s.is_analytic())
        .collect();
    if sel.is_empty() {
        return x0;
    }
    sel.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut x = x0;
    let mut kept = sel.len();
    for step in 0..GAUSS_NEWTON_STEPS {
        let mut cs: Vec<Constraint> = Vec::with_capacity(kept);
        for (s, w) in &sel[..kept] {
            if let Some(c) = s.linearize(x, *w) {
                cs.push(c);
            }
        }
        if cs.is_empty() {
            return x0;
        }
        if step == 0 {
            let (x_new, k) = solve_dropping(x, &cs, tol);
            kept = k;
            x = x_new;
        } else {
            let refs: Vec<&Constraint> = cs.iter().collect();
            let x_new = solve(x, &refs);
            if norm(sub(x_new, x)) <= 1e-6 * tol {
                break;
            }
            x = x_new;
        }
    }
    x
}

fn solve_dropping(x0: V3, constraints: &[Constraint], tol: f64) -> (V3, usize) {
    let mut set: Vec<&Constraint> = constraints.iter().collect();
    set.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    loop {
        let x = solve(x0, &set);
        if set.len() == 1 || norm(sub(x, x0)) <= 5.0 * tol {
            return (x, set.len());
        }
        set.pop();
    }
}

pub struct Projected {
    pub positions: Vec<V3>,
    pub report_dev: f64,
    pub report_rms: f64,
    pub areas: Vec<f64>,
}

pub struct ProjectArgs<'a> {
    pub vc: &'a [V3],
    pub finals: &'a [Final],
    pub flabel: &'a [u32],
    pub faces: &'a [[u32; 3]],
    pub info: &'a [TriInfo],
    pub tol: f64,
    pub diag: f64,
    pub center: V3,
}

pub fn run(args: ProjectArgs<'_>) -> Projected {
    let ProjectArgs {
        vc,
        finals,
        flabel,
        faces,
        info,
        tol,
        diag,
        center,
    } = args;
    let nf = faces.len();
    let mut vr: Vec<u64> = Vec::with_capacity(nf * 3);
    for (f, t) in faces.iter().enumerate() {
        for &v in t {
            vr.push(((v as u64) << 32) | flabel[f] as u64);
        }
    }
    vr.sort_unstable();
    vr.dedup();

    let areas: Vec<f64> = finals
        .iter()
        .map(|fr| fr.faces.iter().map(|&f| info[f as usize].area).sum())
        .collect();
    let mut pv = vc.to_vec();
    let mut dev_sum2 = 0.0;
    let mut dev_max: f64 = 0.0;
    let mut dev_count = 0usize;
    let mut i = 0;
    while i < vr.len() {
        let v = (vr[i] >> 32) as usize;
        let mut j = i;
        let mut surfaces: Vec<(&Surface, f64)> = Vec::new();
        while j < vr.len() && (vr[j] >> 32) as usize == v {
            let r = (vr[j] & 0xffff_ffff) as usize;
            surfaces.push((&finals[r].surface, areas[r]));
            j += 1;
        }
        if surfaces.iter().any(|(s, _)| s.is_analytic()) {
            let x = project_vertex(vc[v], &surfaces, tol);
            pv[v] = x;
            let disp = norm(sub(x, vc[v]));
            let off = surfaces
                .iter()
                .filter(|(s, _)| s.is_analytic())
                .map(|(s, _)| s.distance(x).abs())
                .fold(0.0, f64::max);
            let dev = disp + off;
            dev_max = dev_max.max(dev);
            dev_sum2 += dev * dev;
            dev_count += 1;
        }
        i = j;
    }
    for fr in finals {
        dev_max = dev_max.max(if fr.surface.is_analytic() {
            fr.max
        } else {
            0.0
        });
    }
    let slack = 8.0 * f64::EPSILON * (diag + norm(center));
    let report_dev = if dev_count == 0 { 0.0 } else { dev_max + slack };
    let report_rms = if dev_count == 0 {
        0.0
    } else {
        (dev_sum2 / dev_count as f64).sqrt()
    };

    Projected {
        positions: pv,
        report_dev,
        report_rms,
        areas,
    }
}
