use super::linalg::{M3, V3, add, dot, norm, outer_add, scale, sub, sym_eigen};

pub struct PlaneRef {
    pub n: V3,
    pub d: f64,
    pub weight: f64,
}

const EIGEN_FLOOR: f64 = 5e-4;

fn solve(x0: V3, planes: &[&PlaneRef]) -> V3 {
    let mut m: M3 = [[0.0; 3]; 3];
    let mut r = [0.0; 3];
    for p in planes {
        outer_add(&mut m, p.n, p.n, 1.0);
        let e = p.d - dot(p.n, x0);
        r = add(r, scale(p.n, e));
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

pub fn project_vertex(x0: V3, planes: &[PlaneRef], tol: f64) -> V3 {
    if planes.is_empty() {
        return x0;
    }
    let mut set: Vec<&PlaneRef> = planes.iter().collect();
    set.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    loop {
        let x = solve(x0, &set);
        if set.len() == 1 || norm(sub(x, x0)) <= 5.0 * tol {
            return x;
        }
        set.pop();
    }
}
