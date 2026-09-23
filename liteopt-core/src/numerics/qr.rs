//! Dense, column-equilibrated Householder QR with column pivoting.
//! Full-column-rank solves only; no pseudoinverse or hidden regularization.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrError {
    InvalidShape,
    NonFinite,
    RankDeficient,
}
impl QrError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidShape => "qr_invalid_shape",
            Self::NonFinite => "qr_non_finite",
            Self::RankDeficient => "qr_rank_deficient",
        }
    }
}

/// Reusable workspace for min ||J d + r||² + lambda ||d||².
/// Storage is O((m+n)n) with damping, O(mn) without it.
pub struct QrWorkspace {
    a: Vec<f64>,
    b: Vec<f64>,
    scales: Vec<f64>,
    perm: Vec<usize>,
    z: Vec<f64>,
    pub last_error: Option<QrError>,
}
impl QrWorkspace {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            a: vec![0.; rows * cols],
            b: vec![0.; rows],
            scales: vec![0.; cols],
            perm: vec![0; cols],
            z: vec![0.; cols],
            last_error: None,
        }
    }

    /// Fills `out` only on success. Rank threshold is eps * max(rows, cols)
    /// after equilibration to unit column norms; this is a numerical rank test.
    pub fn solve(
        &mut self,
        j: &[f64],
        r: &[f64],
        m: usize,
        n: usize,
        lambda: f64,
        out: &mut [f64],
    ) -> Result<(), QrError> {
        let result = self.run(j, r, m, n, lambda, out);
        self.last_error = result.err();
        result
    }
    fn run(
        &mut self,
        j: &[f64],
        r: &[f64],
        m: usize,
        n: usize,
        lambda: f64,
        out: &mut [f64],
    ) -> Result<(), QrError> {
        let rows = m
            .checked_add(if lambda > 0. { n } else { 0 })
            .ok_or(QrError::InvalidShape)?;
        if n == 0
            || m == 0
            || rows < n
            || m.checked_mul(n) != Some(j.len())
            || r.len() != m
            || out.len() != n
            || self.b.len() != rows
            || self.scales.len() != n
        {
            return Err(QrError::InvalidShape);
        }
        if !lambda.is_finite() || lambda < 0. || j.iter().chain(r).any(|v| !v.is_finite()) {
            return Err(QrError::NonFinite);
        }
        self.a.fill(0.);
        self.a[..j.len()].copy_from_slice(j);
        self.b.fill(0.);
        for i in 0..m {
            self.b[i] = -r[i];
        }
        if lambda > 0. {
            for k in 0..n {
                self.a[(m + k) * n + k] = lambda.sqrt();
            }
        }
        for col in 0..n {
            let norm = (0..rows).fold(0f64, |acc, row| acc.hypot(self.a[row * n + col]));
            if !norm.is_finite() {
                return Err(QrError::NonFinite);
            }
            if norm == 0. {
                return Err(QrError::RankDeficient);
            }
            self.scales[col] = norm;
            self.perm[col] = col;
            for row in 0..rows {
                self.a[row * n + col] /= norm;
            }
        }
        let threshold = f64::EPSILON * rows.max(n) as f64;
        for k in 0..n {
            // Recompute trailing norms instead of unstable norm downdates.
            let mut pivot = k;
            let mut best = 0f64;
            for col in k..n {
                let norm = (k..rows).fold(0f64, |acc, row| acc.hypot(self.a[row * n + col]));
                if !norm.is_finite() {
                    return Err(QrError::NonFinite);
                }
                if norm > best {
                    best = norm;
                    pivot = col;
                }
            }
            if best <= threshold {
                return Err(QrError::RankDeficient);
            }
            if pivot != k {
                for row in 0..rows {
                    self.a.swap(row * n + k, row * n + pivot);
                }
                self.perm.swap(k, pivot);
            }
            let first = self.a[k * n + k];
            let alpha = -best.copysign(first);
            let denom = first - alpha;
            let tau = (alpha - first) / alpha;
            self.a[k * n + k] = alpha;
            for row in k + 1..rows {
                self.a[row * n + k] /= denom;
            }
            for col in k + 1..n {
                let mut dot = self.a[k * n + col];
                for row in k + 1..rows {
                    dot += self.a[row * n + k] * self.a[row * n + col];
                }
                dot *= tau;
                self.a[k * n + col] -= dot;
                for row in k + 1..rows {
                    self.a[row * n + col] -= self.a[row * n + k] * dot;
                }
            }
            let mut dot = self.b[k];
            for row in k + 1..rows {
                dot += self.a[row * n + k] * self.b[row];
            }
            dot *= tau;
            self.b[k] -= dot;
            for row in k + 1..rows {
                self.b[row] -= self.a[row * n + k] * dot;
            }
        }
        for k in (0..n).rev() {
            let mut value = self.b[k];
            for col in k + 1..n {
                value -= self.a[k * n + col] * self.z[col];
            }
            self.z[k] = value / self.a[k * n + k];
            if !(self.z[k] / self.scales[self.perm[k]]).is_finite() {
                return Err(QrError::NonFinite);
            }
        }
        for k in 0..n {
            out[self.perm[k]] = self.z[k] / self.scales[self.perm[k]];
        }
        Ok(())
    }
}
