//! Unpreconditioned conjugate gradients with a checked true residual.
//! Algorithm reference: https://www.netlib.org/templates/templates.html
use super::linalg::dot;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinearSolver {
    #[default]
    Direct,
    Cg,
}

#[derive(Clone, Copy, Debug)]
pub struct CgOptions {
    pub max_iters: usize,
    pub rtol: f64,
    pub atol: f64,
}
impl Default for CgOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            rtol: 1e-6,
            atol: 0.0,
        }
    }
}
impl CgOptions {
    pub fn is_valid(&self) -> bool {
        self.max_iters > 0
            && self.rtol.is_finite()
            && self.rtol > 0.0
            && self.rtol < 1.0
            && self.atol.is_finite()
            && self.atol >= 0.0
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CgReport {
    pub status: &'static str,
    pub iters: usize,
    pub residual_norm: f64,
}
impl CgReport {
    pub fn converged(&self) -> bool {
        self.status == "linear_converged"
    }
}

pub struct CgWorkspace {
    r: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}
impl CgWorkspace {
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![0.; n],
            p: vec![0.; n],
            ap: vec![0.; n],
        }
    }

    /// Solve A*d = -gradient from zero. A must be symmetric positive definite.
    /// Failure leaves an approximate d which callers must not commit.
    pub fn solve(
        &mut self,
        gradient: &[f64],
        d: &mut [f64],
        opts: CgOptions,
        mut apply: impl FnMut(&[f64], &mut [f64]),
    ) -> CgReport {
        let mut iters = 0;
        let mut norm = f64::NAN;
        macro_rules! done {
            ($status:expr) => {
                return CgReport {
                    status: $status,
                    iters,
                    residual_norm: norm,
                }
            };
        }
        if !opts.is_valid() || gradient.len() != self.r.len() || d.len() != self.r.len() {
            done!("linear_invalid_options");
        }
        d.fill(0.);
        for (r, g) in self.r.iter_mut().zip(gradient) {
            *r = -g;
        }
        self.p.copy_from_slice(&self.r);
        let mut rr = dot(&self.r, &self.r);
        norm = rr.sqrt();
        if !norm.is_finite() {
            done!("linear_non_finite");
        }
        let threshold = opts.atol.max(opts.rtol * norm);
        if norm <= threshold {
            done!("linear_converged");
        }
        for k in 0..opts.max_iters {
            apply(&self.p, &mut self.ap);
            if self.ap.iter().any(|v| !v.is_finite()) {
                done!("linear_non_finite");
            }
            let pap = dot(&self.p, &self.ap);
            if !pap.is_finite() {
                done!("linear_non_finite");
            }
            if pap <= 0. {
                done!("linear_breakdown");
            }
            let alpha = rr / pap;
            for i in 0..d.len() {
                d[i] += alpha * self.p[i];
                self.r[i] -= alpha * self.ap[i];
            }
            iters = k + 1;
            let mut next_rr = dot(&self.r, &self.r);
            norm = next_rr.sqrt();
            if !norm.is_finite() || d.iter().any(|v| !v.is_finite()) {
                done!("linear_non_finite");
            }
            // Check the actual residual before success and at exhaustion.
            if norm <= threshold || iters == opts.max_iters {
                apply(d, &mut self.ap);
                for (r, (g, ad)) in self.r.iter_mut().zip(gradient.iter().zip(&self.ap)) {
                    *r = -g - ad;
                }
                next_rr = dot(&self.r, &self.r);
                norm = next_rr.sqrt();
                if !norm.is_finite() {
                    done!("linear_non_finite");
                }
                if norm <= threshold {
                    done!("linear_converged");
                }
                // Restart if recursive residual underestimated the true residual.
                self.p.copy_from_slice(&self.r);
            } else {
                let beta = next_rr / rr;
                for i in 0..d.len() {
                    self.p[i] = self.r[i] + beta * self.p[i];
                }
            }
            rr = next_rr;
        }
        done!("linear_max_iters");
    }
}
