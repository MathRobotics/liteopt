use crate::manifolds::space::Space;
use crate::numerics::linalg::{dot, jj_t_plus_lambda, jt_mul_vec, norm2, solve_linear_inplace};
use crate::problems::least_squares::LeastSquaresProblem;

use super::types::{LevenbergMarquardt, LevenbergMarquardtResult};
use super::workspace::LmWorkspace;

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> LevenbergMarquardt<S> {
    fn compute_direction<JF>(
        &self,
        x: &[f64],
        m: usize,
        n: usize,
        lambda: f64,
        jacobian_fn: &mut JF,
        ws: &mut LmWorkspace,
    ) -> Option<f64>
    where
        JF: FnMut(&[f64], &mut [f64]),
    {
        jacobian_fn(x, &mut ws.j);

        // A = J J^T + lambda I
        jj_t_plus_lambda(&ws.j, m, n, lambda, &mut ws.a);

        // y = A^{-1} r
        ws.y.copy_from_slice(&ws.r);
        if !solve_linear_inplace(&mut ws.a, &mut ws.y, m) {
            return None;
        }

        // dx = -J^T y
        jt_mul_vec(&ws.j, m, n, &ws.y, &mut ws.dx);
        for v in &mut ws.dx {
            *v = -*v;
        }
        Some(self.space.tangent_norm(&ws.dx))
    }

    /// Solve using a problem object.
    pub fn solve<P>(&self, x: Vec<f64>, problem: &P) -> LevenbergMarquardtResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
    {
        let m = problem.residual_dim();
        self.solve_with_fn(
            m,
            x,
            |x, r| problem.residual(x, r),
            |x, j| problem.jacobian(x, j),
            |x| problem.project(x),
        )
    }

    /// Solve nonlinear least squares with LM-style damping updates.
    pub fn solve_with_fn<R, JF, P>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        let n = x.len();
        assert!(m > 0 && n > 0);

        let lambda_up = if self.lambda_up > 1.0 {
            self.lambda_up
        } else {
            10.0
        };
        let lambda_down = if (0.0..1.0).contains(&self.lambda_down) {
            self.lambda_down
        } else {
            0.5
        };

        let mut lambda = self.lambda.max(1e-12);
        let mut ws = LmWorkspace::new(m, n);
        let mut last_dx_norm = f64::NAN;

        residual_fn(&x, &mut ws.r);
        let mut cost = 0.5 * dot(&ws.r, &ws.r);
        let mut r_norm = norm2(&ws.r);

        if self.verbose {
            println!(
                "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | note initial",
                0, cost, r_norm
            );
        }

        for it in 0..self.max_iters {
            if r_norm <= self.tol_r {
                return LevenbergMarquardtResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: true,
                };
            }

            let dx_norm = self.compute_direction(&x, m, n, lambda, &mut jacobian_fn, &mut ws);
            let Some(dx_norm) = dx_norm else {
                lambda *= lambda_up;
                if self.verbose {
                    println!(
                        "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | lambda {:>9.3e} | note linear_solve_failed",
                        it, cost, r_norm, lambda
                    );
                }
                continue;
            };
            last_dx_norm = dx_norm;

            if dx_norm <= self.tol_dq {
                return LevenbergMarquardtResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: true,
                };
            }

            let alpha = self.step_scale.clamp(0.0, 1.0);
            if alpha == 0.0 {
                return LevenbergMarquardtResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: false,
                };
            }

            self.space
                .retract_into(&mut ws.x_trial, &x, &ws.dx, alpha, &mut ws.tmp);
            project(&mut ws.x_trial);

            residual_fn(&ws.x_trial, &mut ws.r_trial);
            let cost_trial = 0.5 * dot(&ws.r_trial, &ws.r_trial);

            if cost_trial.is_finite() && cost_trial < cost {
                x.copy_from_slice(&ws.x_trial);
                ws.r.copy_from_slice(&ws.r_trial);
                cost = cost_trial;
                r_norm = norm2(&ws.r);
                lambda = (lambda * lambda_down).max(1e-12);
                if self.verbose {
                    println!(
                        "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | lambda {:>9.3e} | note accepted",
                        it, cost, r_norm, dx_norm, lambda
                    );
                }
            } else {
                lambda *= lambda_up;
                if self.verbose {
                    println!(
                        "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | lambda {:>9.3e} | note rejected",
                        it, cost, r_norm, dx_norm, lambda
                    );
                }
            }
        }

        LevenbergMarquardtResult {
            x,
            cost,
            iters: self.max_iters,
            r_norm,
            dx_norm: last_dx_norm,
            converged: false,
        }
    }
}
