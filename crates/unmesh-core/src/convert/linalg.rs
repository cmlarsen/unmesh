pub type V3 = [f64; 3];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}

pub fn angle_deg(a: V3, b: V3) -> f64 {
    let c = norm(cross(a, b));
    c.atan2(dot(a, b)).to_degrees()
}

pub type M3 = [[f64; 3]; 3];

pub fn outer_add(m: &mut M3, a: V3, b: V3, w: f64) {
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell += w * a[i] * b[j];
        }
    }
}

pub fn sym_eigen(a: M3) -> ([f64; 3], [V3; 3]) {
    let mut a = a;
    let mut v: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let diag = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-300 || off <= 1e-17 * diag {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let vkp = row[p];
                let vkq = row[q];
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let vals = idx.map(|i| a[i][i]);
    let vecs = idx.map(|i| unit([v[0][i], v[1][i], v[2][i]]));
    (vals, vecs)
}

#[derive(Clone, Copy, Default)]
pub struct Moments {
    pub area: f64,
    pub s: V3,
    pub xx: M3,
}

impl Moments {
    pub fn add_tri(&mut self, p: [V3; 3], area: f64) {
        let sum = add(add(p[0], p[1]), p[2]);
        self.area += area;
        for k in 0..3 {
            self.s[k] += area * sum[k] / 3.0;
        }
        let w = area / 12.0;
        for pt in p {
            outer_add(&mut self.xx, pt, pt, w);
        }
        outer_add(&mut self.xx, sum, sum, w);
    }

    pub fn plane(&self) -> Option<(V3, V3, [f64; 3])> {
        if self.area <= 0.0 {
            return None;
        }
        let c = scale(self.s, 1.0 / self.area);
        let mut cov = self.xx;
        for row in cov.iter_mut() {
            for cell in row.iter_mut() {
                *cell /= self.area;
            }
        }
        outer_add(&mut cov, c, c, -1.0);
        let (vals, vecs) = sym_eigen(cov);
        Some((vecs[0], c, vals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_recovers_axes() {
        let (vals, vecs) = sym_eigen([[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 9.0]]);
        assert_eq!(vals, [1.0, 4.0, 9.0]);
        assert!((vecs[0][1].abs() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eigen_general() {
        let m = [[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]];
        let (vals, vecs) = sym_eigen(m);
        for i in 0..3 {
            let mv = [dot(m[0], vecs[i]), dot(m[1], vecs[i]), dot(m[2], vecs[i])];
            let lv = scale(vecs[i], vals[i]);
            assert!(norm(sub(mv, lv)) < 1e-10);
        }
    }

    #[test]
    fn moments_plane() {
        let mut m = Moments::default();
        m.add_tri([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]], 2.0);
        m.add_tri([[2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 2.0, 1.0]], 2.0);
        let (n, c, _) = m.plane().unwrap();
        assert!((n[2].abs() - 1.0).abs() < 1e-12);
        assert!((c[2] - 1.0).abs() < 1e-12);
    }
}
