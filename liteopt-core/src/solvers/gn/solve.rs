use crate::manifolds::space::Space;
use crate::numerics::linalg::{dot, jj_t_plus_lambda, jt_mul_vec, norm2, solve_linear_inplace};
use crate::problems::least_squares::LeastSquaresProblem;

use super::line_search::{ArmijoBacktracking, LineSearchContext, LineSearchPolicy};
use super::types::{DirectionResult, GaussNewton, GaussNewtonResult};
use super::workspace::GaussNewtonWorkspace;

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> GaussNewton<S> {
    fn compute_direction<JF>(
        &self,
        x: &[f64],
        m: usize,
        n: usize,
        lambda: f64,
        jacobian_fn: &mut JF,
        need_dphi0: bool,
        ws: &mut GaussNewtonWorkspace,
    ) -> Option<DirectionResult>
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
            return Some(DirectionResult {
                dx_norm,
                dphi0: None,
                used_steepest_descent: false,
            });
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

        Some(DirectionResult {
            dx_norm,
            dphi0: Some(dphi0),
            used_steepest_descent,
        })
    }

    fn evaluate_trial<R, P>(
        &self,
        x: &Vec<f64>,
        alpha: f64,
        residual_fn: &mut R,
        project: &mut P,
        ws: &mut GaussNewtonWorkspace,
    ) -> Option<f64>
    where
        R: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        // x_trial = Retr_x(alpha * dx)
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
        ws: &mut GaussNewtonWorkspace,
    ) {
        x.copy_from_slice(&ws.x_trial);
        ws.r.copy_from_slice(&ws.r_trial);
        *cost = cost_trial;
        *r_norm = norm2(&ws.r);
    }

    fn run_with_line_search<R, JF, P, LS>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
        line_search: &mut LS,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
    {
        let n = x.len();
        assert!(m > 0 && n > 0);

        // LM damping is mutable locally.
        let mut lambda = self.lambda;
        let mut ws = GaussNewtonWorkspace::new(m, n);

        // initial
        residual_fn(&x, &mut ws.r);
        let mut cost = 0.5 * dot(&ws.r, &ws.r);
        let mut r_norm = norm2(&ws.r);

        if self.verbose {
            println!("[gn] debug columns:");
            println!("  iter   : iteration index (0-based)");
            println!("  cost   : 0.5 * ||r(x)||^2");
            println!("  r      : ||r(x)|| (residual norm)");
            println!("  dx     : ||dx|| (step direction norm; after fallback if applied)");
            println!("  alpha  : step size used in retract (after backtracking)");
            println!("  lambda : LM damping (bigger => more conservative step)");
            println!("  note   : state tag (initial/accepted/rejected/...)");
            println!(
                "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | note initial",
                0, cost, r_norm
            );
        }

        for it in 0..self.max_iters {
            // Stop by residual norm
            if r_norm <= self.tol_r {
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | stop r_norm",
                        it, cost, r_norm, 0.0
                    );
                }
                return GaussNewtonResult {
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

            let Some(direction) = direction else {
                // LM-style recovery: increase lambda and retry SAME iteration
                lambda *= 10.0;
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | note linear_solve_failed -> increase lambda {:>9.3e}",
                        it, cost, r_norm, lambda
                    );
                }
                continue;
            };

            let dx_norm = direction.dx_norm;
            let dphi0 = direction.dphi0;
            if direction.used_steepest_descent && self.verbose {
                println!(
                    "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | note fallback_to_steepest_descent dphi0={:>13.6e}",
                    it, cost, r_norm, dphi0.unwrap_or(f64::NAN)
                );
            }

            // Stop by step norm (after possible fallback)
            if dx_norm <= self.tol_dq {
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | stop dx_norm",
                        it, cost, r_norm, dx_norm
                    );
                }
                return GaussNewtonResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: true,
                };
            }

            // step scale alpha
            let alpha = self.step_scale.clamp(0.0, 1.0);
            if alpha == 0.0 {
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | note step_scale_zero",
                        it, cost, r_norm, dx_norm
                    );
                }
                return GaussNewtonResult {
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

            if self.verbose {
                if ls.accepted {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | alpha {:>8.3e} | lambda {:>9.3e} | note accepted",
                        it, cost, r_norm, dx_norm, ls.alpha, lambda
                    );
                } else {
                    println!(
                        "[gn] iter {:>6} | cost {:>13.6e} | r {:>13.6e} | dx {:>13.6e} | alpha {:>8.3e} | lambda {:>9.3e} | note rejected",
                        it, cost, r_norm, dx_norm, ls.alpha, lambda
                    );
                }
            }

            if !ls.accepted {
                // LM-style recovery: increase lambda and retry SAME iteration
                lambda *= 10.0;
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | note rejected -> increase lambda {:>9.3e} and retry",
                        it, lambda
                    );
                }
                continue;
            }

            let cost_trial =
                self.evaluate_trial(&x, ls.alpha, &mut residual_fn, &mut project, &mut ws);

            let Some(cost_trial) = cost_trial else {
                lambda *= 10.0;
                if self.verbose {
                    println!(
                        "[gn] iter {:>6} | note accepted_step_invalid -> increase lambda {:>9.3e} and retry",
                        it, lambda
                    );
                }
                continue;
            };

            self.commit_trial_step(&mut x, &mut cost, &mut r_norm, cost_trial, &mut ws);

            // Optional: relax damping a bit after a successful iteration.
            // This prevents lambda from staying huge forever.
            lambda = (0.5 * lambda).max(self.lambda);
        }

        GaussNewtonResult {
            x,
            cost,
            iters: self.max_iters,
            r_norm,
            dx_norm: f64::NAN,
            converged: false,
        }
    }

    /// Solve using a problem object.
    pub fn solve<P, LS>(
        &self,
        x: Vec<f64>,
        problem: &P,
        line_search: &mut LS,
    ) -> GaussNewtonResult<Vec<f64>>
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

    /// Solve with an explicit line search policy.
    ///
    /// - m: residual dimension (e.g. 2 for 2D position, 3 for 3D position, 6 for SE(3) pose error)
    /// - x: initial guess (len = n)
    /// - residual_fn(x, r): fill r (len = m)
    /// - jacobian_fn(x, J): fill local Jacobian wrt the update vector at x
    ///   (len = m*n, row-major, i-th row and k-th col => J[i*n+k])
    /// - project(x): optional projection (joint limits etc). If not needed, pass |x| {}
    /// - line_search: step size policy
    pub fn solve_with_fn<R, JF, P, LS>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
        line_search: &mut LS,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
    {
        self.run_with_line_search(m, x, residual_fn, jacobian_fn, project, line_search)
    }

    /// Solve using default Armijo backtracking line search settings.
    pub fn solve_with_default_line_search<P>(
        &self,
        x: Vec<f64>,
        problem: &P,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
    {
        let mut line_search = ArmijoBacktracking::default();
        self.solve(x, problem, &mut line_search)
    }

    /// Solve callbacks using default Armijo backtracking line search settings.
    pub fn solve_with_fn_default_line_search<R, JF, P>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        let mut line_search = ArmijoBacktracking::default();
        self.solve_with_fn(m, x, residual_fn, jacobian_fn, project, &mut line_search)
    }
}
