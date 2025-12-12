
// ----------------- helpers (dependency-free) -----------------

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// A = J J^T + lambda I, J: (m x n) row-major
pub fn jj_t_plus_lambda(j: &[f64], m: usize, n: usize, lambda: f64, a: &mut [f64]) {
    a.fill(0.0);
    for i in 0..m {
        for k in 0..n {
            let jik = j[i * n + k];
            for jrow in 0..m {
                a[i * m + jrow] += jik * j[jrow * n + k];
            }
        }
    }
    for i in 0..m {
        a[i * m + i] += lambda;
    }
}

/// out = J^T v
pub fn jt_mul_vec(j: &[f64], m: usize, n: usize, v: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..m {
        let vi = v[i];
        for k in 0..n {
            out[k] += j[i * n + k] * vi;
        }
    }
}

/// Gaussian elimination with partial pivoting (in-place).
/// Solves A x = b, overwriting A and b (b becomes x).
pub fn solve_linear_inplace(a: &mut [f64], b: &mut [f64], dim: usize) -> bool {
    const EPS: f64 = 1e-12;

    for col in 0..dim {
        // pivot
        let mut piv = col;
        let mut max_abs = a[col * dim + col].abs();
        for row in (col + 1)..dim {
            let v = a[row * dim + col].abs();
            if v > max_abs {
                max_abs = v;
                piv = row;
            }
        }
        if max_abs < EPS || !max_abs.is_finite() {
            return false;
        }

        if piv != col {
            for k in col..dim {
                a.swap(col * dim + k, piv * dim + k);
            }
            b.swap(col, piv);
        }

        let diag = a[col * dim + col];
        for row in (col + 1)..dim {
            let f = a[row * dim + col] / diag;
            a[row * dim + col] = 0.0;
            for k in (col + 1)..dim {
                a[row * dim + k] -= f * a[col * dim + k];
            }
            b[row] -= f * b[col];
        }
    }

    for i_rev in 0..dim {
        let i = dim - 1 - i_rev;
        let mut s = b[i];
        for k in (i + 1)..dim {
            s -= a[i * dim + k] * b[k];
        }
        let diag = a[i * dim + i];
        if diag.abs() < EPS || !diag.is_finite() {
            return false;
        }
        b[i] = s / diag;
    }
    true
}
