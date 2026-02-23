use crate::manifolds::space::Space;
use crate::numerics::linalg::{dot, jj_t_plus_lambda, jt_mul_vec, norm2, solve_linear_inplace};
use crate::problems::least_squares::LeastSquaresProblem;
use crate::solvers::common::step_policy::{CostDecrease, LineSearchContext, LineSearchPolicy};

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
        need_dphi0: bool,
        ws: &mut LmWorkspace,
    ) -> Option<(f64, Option<f64>, bool)>
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
        let mut dx_norm = self.space.tangent_norm(&ws.dx);

        if !need_dphi0 {
            return Some((dx_norm, None, false));
        }

        // g = J^T r  (gradient of 0.5||r||^2)
        jt_mul_vec(&ws.j, m, n, &ws.r, &mut ws.g);
        let mut dphi0 = dot(&ws.g, &ws.dx);
        let mut used_steepest_descent = false;

        // Fallback if not descent (or NaN/Inf): dx = -g
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            for i in 0..n {
                ws.dx[i] = -ws.g[i];
            }
            dx_norm = self.space.tangent_norm(&ws.dx);
            dphi0 = dot(&ws.g, &ws.dx);
            used_steepest_descent = true;
        }

        Some((dx_norm, Some(dphi0), used_steepest_descent))
    }

    fn evaluate_trial<R, P>(
        &self,
        x: &Vec<f64>,
        alpha: f64,
        residual_fn: &mut R,
        project: &mut P,
        ws: &mut LmWorkspace,
    ) -> Option<f64>
    where
        R: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        self.space
            .retract_into(&mut ws.x_trial, x, &ws.dx, alpha, &mut ws.tmp);
        project(&mut ws.x_trial);

        residual_fn(&ws.x_trial, &mut ws.r_trial);
        let cost_trial = 0.5 * dot(&ws.r_trial, &ws.r_trial);
        cost_trial.is_finite().then_some(cost_trial)
    }

    fn commit_trial_step(
        &self,
        x: &mut Vec<f64>,
        cost: &mut f64,
        r_norm: &mut f64,
        cost_trial: f64,
        ws: &mut LmWorkspace,
    ) {
        x.copy_from_slice(&ws.x_trial);
        ws.r.copy_from_slice(&ws.r_trial);
        *cost = cost_trial;
        *r_norm = norm2(&ws.r);
    }

    /// Solve using a problem object.
    pub fn solve<P, LS>(
        &self,
        x: Vec<f64>,
        problem: &P,
        line_search: &mut LS,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
        LS: LineSearchPolicy,
    {
        let m = problem.residual_dim();
        self.solve_with_fn(
            m,
            x,
            |x, r| problem.residual(x, r),
            |x, j| problem.jacobian(x, j),
            |x| problem.project(x),
            line_search,
        )
    }

    /// Solve nonlinear least squares with LM-style damping updates.
    pub fn solve_with_fn<R, JF, P, LS>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
        line_search: &mut LS,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
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

            let direction = self.compute_direction(
                &x,
                m,
                n,
                lambda,
                &mut jacobian_fn,
                line_search.requires_directional_derivative(),
                &mut ws,
            );
            let Some((dx_norm, dphi0, used_steepest_descent)) = direction else {
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
            if used_steepest_descent && self.verbose {
                println!(
                    "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | note fallback_to_steepest_descent dphi0={:>13.6e}",
                    it,
                    cost,
                    r_norm,
                    dphi0.unwrap_or(f64::NAN)
                );
            }

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

            let ls_ctx = LineSearchContext {
                iter: it,
                alpha0: alpha,
                cost0: cost,
                dphi0,
                dx_norm,
                lambda,
            };
            let mut eval_cost = |alpha_trial| {
                self.evaluate_trial(&x, alpha_trial, &mut residual_fn, &mut project, &mut ws)
            };
            let ls = line_search.search(&ls_ctx, &mut eval_cost);

            if ls.accepted {
                let cost_trial =
                    self.evaluate_trial(&x, ls.alpha, &mut residual_fn, &mut project, &mut ws);
                let Some(cost_trial) = cost_trial else {
                    lambda *= lambda_up;
                    if self.verbose {
                        println!(
                            "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | alpha {:>8.3e} | lambda {:>9.3e} | note accepted_step_invalid",
                            it, cost, r_norm, dx_norm, ls.alpha, lambda
                        );
                    }
                    continue;
                };

                self.commit_trial_step(&mut x, &mut cost, &mut r_norm, cost_trial, &mut ws);
                lambda = (lambda * lambda_down).max(1e-12);
                if self.verbose {
                    println!(
                        "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | alpha {:>8.3e} | lambda {:>9.3e} | note accepted",
                        it, cost, r_norm, dx_norm, ls.alpha, lambda
                    );
                }
            } else {
                lambda *= lambda_up;
                if self.verbose {
                    println!(
                        "[lm] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | alpha {:>8.3e} | lambda {:>9.3e} | note rejected",
                        it, cost, r_norm, dx_norm, ls.alpha, lambda
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

    /// Solve using default cost-decrease policy (same acceptance behavior as classic LM loop).
    pub fn solve_with_default_line_search<P>(
        &self,
        x: Vec<f64>,
        problem: &P,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
    {
        let mut line_search = CostDecrease;
        self.solve(x, problem, &mut line_search)
    }

    /// Solve callback form using default cost-decrease policy.
    pub fn solve_with_fn_default_line_search<R, JF, P>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        let mut line_search = CostDecrease;
        self.solve_with_fn(m, x, residual_fn, jacobian_fn, project, &mut line_search)
    }
}
