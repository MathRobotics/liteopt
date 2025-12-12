use crate::space::Space;
use crate::linalg::{dot, norm2, jj_t_plus_lambda, jt_mul_vec, solve_linear_inplace};

// ----------------- NonlinearLeastSquares ---------------------
// --------------------------------------
#[derive(Clone, Debug)]
pub struct LeastSquaresResult<P> {
    pub x: P,
    pub cost: f64,        // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    pub dx_norm: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct NonlinearLeastSquares<S: Space<Point = Vec<f64>>> {
    pub space: S,
    pub lambda: f64,          // damping
    pub step_scale: f64,      // alpha in (0,1]
    pub max_iters: usize,
    pub tol_r: f64,           // stop if ||r|| < tol_r
    pub tol_dq: f64,          // stop if ||dq|| < tol_dq
    pub line_search: bool,    // simple backtracking on cost
    pub ls_beta: f64,         // e.g. 0.5
    pub ls_max_steps: usize,  // e.g. 20
}

impl<S: Space<Point = Vec<f64>>> NonlinearLeastSquares<S> {
    /// Solve the nonlinear least squares problem.
    ///
    /// - m: residual dimension (e.g. 2 for 2D position, 3 for 3D position, 6 for SE(3) pose error)
    /// - q0: initial guess (len = n)
    /// - residual_fn(q, r): fill r (len = m)
    /// - jacobian_fn(q, J): fill J (len = m*n), row-major (i-th row, k-th col => J[i*n+k])
    /// - project(q): optional projection (joint limits etc). If not needed, pass |q| {}
    pub fn solve_with_fn<R, JF, P>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
    ) -> LeastSquaresResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        let n = x.len();
        assert!(m > 0 && n > 0);

        let mut r = vec![0.0f64; m];
        let mut j = vec![0.0f64; m * n];  // row-major
        let mut a = vec![0.0f64; m * m];  // A = J J^T + lambda I
        let mut y = vec![0.0f64; m];
        let mut dx = vec![0.0f64; n];

        // for line search
        let mut x_trial = vec![0.0f64; n];
        let mut r_trial = vec![0.0f64; m];

        // initial
        residual_fn(&x, &mut r);
        let mut cost = 0.5 * dot(&r, &r);
        let mut r_norm = norm2(&r);

        let mut x_next = vec![0.0f64; n];
        let mut tmp = vec![0.0f64; n]; // for retract_into

        for it in 0..self.max_iters {
            if r_norm <= self.tol_r {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: true,
                };
            }

            // J(q)
            jacobian_fn(&x, &mut j);

            // A = J J^T + lambda I
            jj_t_plus_lambda(&j, m, n, self.lambda, &mut a);

            // y = A^{-1} r
            y.copy_from_slice(&r);
            let ok = solve_linear_inplace(&mut a, &mut y, m);
            if !ok {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: f64::NAN,
                    converged: false,
                };
            }

            // dx = - J^T y
            jt_mul_vec(&j, m, n, &y, &mut dx);
            for v in dx.iter_mut() {
                *v = -*v;
            }

            let dx_norm = norm2(&dx);
            if dx_norm <= self.tol_dq {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: true,
                };
            }

            let mut alpha = self.step_scale.clamp(0.0, 1.0);
            if alpha == 0.0 {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: false,
                };
            }

            if self.line_search {
                let mut accepted = false;
                for _ in 0..self.ls_max_steps {
                    // q_trial = Retr_q(alpha * dq)
                    self.space.retract_into(&mut x_trial, &x, &dx, alpha, &mut tmp);
                    project(&mut x_trial);

                    residual_fn(&x_trial, &mut r_trial);
                    let cost_trial = 0.5 * dot(&r_trial, &r_trial);

                    if cost_trial.is_finite() && cost_trial <= cost {
                        x.copy_from_slice(&x_trial);
                        r.copy_from_slice(&r_trial);
                        cost = cost_trial;
                        r_norm = norm2(&r);
                        accepted = true;
                        break;
                    }
                    alpha *= self.ls_beta;
                }
                if !accepted {
                    return LeastSquaresResult {
                        x,
                        cost,
                        iters: it + 1,
                        r_norm,
                        dx_norm,
                        converged: false,
                    };
                }
            } else {
                // x = Retr_x(alpha * dx)
                self.space.retract_into(&mut x_next, &x, &dx, alpha, &mut tmp);
                project(&mut x_next);
                std::mem::swap(&mut x, &mut x_next);

                residual_fn(&x, &mut r);
                cost = 0.5 * dot(&r, &r);
                r_norm = norm2(&r);
            }
        }

        LeastSquaresResult {
            x,
            cost,
            iters: self.max_iters,
            r_norm,
            dx_norm: f64::NAN,
            converged: false,
        }
    }
}
