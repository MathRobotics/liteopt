use crate::manifolds::space::Space;
use crate::numerics::linalg::{
    dot, jj_t_plus_lambda, jt_j_plus_lambda, jt_mul_vec, norm2, solve_linear_inplace,
};
use crate::problems::least_squares::LeastSquaresProblem;

use super::line_search::{
    ArmijoBacktracking, LineSearchContext, LineSearchPolicy, NoLineSearch,
    StrictDecreaseBacktracking,
};
use super::types::{
    DirectionResult, GaussNewton, GaussNewtonDampingUpdate, GaussNewtonLineSearchMethod,
    GaussNewtonLinearSystem, GaussNewtonResult,
};
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

        match self.linear_system {
            GaussNewtonLinearSystem::LeftJjT => {
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
            }
            GaussNewtonLinearSystem::NormalJtJ => {
                // A = J^T J + lambda I
                jt_j_plus_lambda(&ws.j, m, n, lambda, &mut ws.an);

                // rhs = -J^T r
                jt_mul_vec(&ws.j, m, n, &ws.r, &mut ws.dx);
                for v in &mut ws.dx {
                    *v = -*v;
                }
                if !solve_linear_inplace(&mut ws.an, &mut ws.dx, n) {
                    return None;
                }
            }
        }

        let mut dx_norm = self.space.tangent_norm(&ws.dx);

        if !need_dphi0 {
            return Some(DirectionResult {
                dx_norm,
                dphi0: None,
            });
        }

        // g = J^T r (gradient of 0.5||r||^2)
        jt_mul_vec(&ws.j, m, n, &ws.r, &mut ws.g);
        let mut dphi0 = dot(&ws.g, &ws.dx);
        // Fallback if not descent (or NaN/Inf): dx = -g
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            for i in 0..n {
                ws.dx[i] = -ws.g[i];
            }
            dx_norm = self.space.tangent_norm(&ws.dx);
            dphi0 = dot(&ws.g, &ws.dx);
        }

        Some(DirectionResult {
            dx_norm,
            dphi0: Some(dphi0),
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

    fn configured_armijo(&self) -> ArmijoBacktracking {
        let beta = if self.ls_beta.is_finite() && (0.0..1.0).contains(&self.ls_beta) {
            self.ls_beta
        } else {
            0.5
        };
        let max_steps = self.ls_max_steps.max(1);
        let c_armijo = if self.c_armijo.is_finite() {
            self.c_armijo
        } else {
            1e-4
        };
        ArmijoBacktracking::new(beta, max_steps, c_armijo)
    }

    fn configured_strict_decrease(&self) -> StrictDecreaseBacktracking {
        let beta = if self.ls_beta.is_finite() && (0.0..1.0).contains(&self.ls_beta) {
            self.ls_beta
        } else {
            0.5
        };
        let min_step = if self.ls_min_step.is_finite() && self.ls_min_step > 0.0 {
            self.ls_min_step
        } else {
            1e-8
        };
        let max_steps = self.ls_max_steps.max(1);
        StrictDecreaseBacktracking::new(beta, min_step, max_steps)
    }

    fn run_with_configured_line_search<R, JF, P>(
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
        match self.line_search_method {
            GaussNewtonLineSearchMethod::Armijo => {
                let mut line_search = self.configured_armijo();
                self.run_with_line_search(m, x, residual_fn, jacobian_fn, project, &mut line_search)
            }
            GaussNewtonLineSearchMethod::StrictDecrease => {
                let mut line_search = self.configured_strict_decrease();
                self.run_with_line_search(m, x, residual_fn, jacobian_fn, project, &mut line_search)
            }
            GaussNewtonLineSearchMethod::None => {
                let mut line_search = NoLineSearch;
                self.run_with_line_search(m, x, residual_fn, jacobian_fn, project, &mut line_search)
            }
        }
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

        let mut lambda = self.lambda;
        let mut ws = GaussNewtonWorkspace::new(m, n);

        residual_fn(&x, &mut ws.r);
        let mut cost = 0.5 * dot(&ws.r, &ws.r);
        let mut r_norm = norm2(&ws.r);

        for it in 0..self.max_iters {
            if r_norm <= self.tol_r {
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
                if self.damping_update == GaussNewtonDampingUpdate::Adaptive {
                    lambda *= 10.0;
                    continue;
                }
                return GaussNewtonResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: false,
                };
            };

            let dx_norm = direction.dx_norm;
            let skip_pre_step_dx_stop = self.line_search_method
                == GaussNewtonLineSearchMethod::StrictDecrease
                && !line_search.requires_directional_derivative();
            if dx_norm <= self.tol_dq && !skip_pre_step_dx_stop {
                return GaussNewtonResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: true,
                };
            }

            let alpha0 = self.step_scale.clamp(0.0, 1.0);
            if alpha0 == 0.0 {
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
                alpha0,
                cost0: cost,
                dphi0: direction.dphi0,
                dx_norm,
                lambda,
            };
            let mut eval_cost = |alpha_trial| {
                self.evaluate_trial(&x, alpha_trial, &mut residual_fn, &mut project, &mut ws)
            };
            let ls = line_search.search(&ls_ctx, &mut eval_cost);

            if !ls.accepted {
                if self.damping_update == GaussNewtonDampingUpdate::Adaptive {
                    lambda *= 10.0;
                    continue;
                }
                return GaussNewtonResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: false,
                };
            }

            let cost_trial =
                self.evaluate_trial(&x, ls.alpha, &mut residual_fn, &mut project, &mut ws);
            let Some(cost_trial) = cost_trial else {
                if self.damping_update == GaussNewtonDampingUpdate::Adaptive {
                    lambda *= 10.0;
                    continue;
                }
                return GaussNewtonResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: false,
                };
            };

            self.commit_trial_step(&mut x, &mut cost, &mut r_norm, cost_trial, &mut ws);

            if self.damping_update == GaussNewtonDampingUpdate::Adaptive {
                lambda = (0.5 * lambda).max(self.lambda);
            }
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
    /// - m: residual dimension
    /// - x: initial guess (len = n)
    /// - residual_fn(x, r): fill r (len = m)
    /// - jacobian_fn(x, J): fill Jacobian (len = m*n, row-major)
    /// - project(x): optional projection
    /// - line_search: external step-size policy
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

    /// Solve using the configured line-search method on the solver.
    pub fn solve_with_default_line_search<P>(
        &self,
        x: Vec<f64>,
        problem: &P,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
    {
        let m = problem.residual_dim();
        self.run_with_configured_line_search(
            m,
            x,
            |x, r| problem.residual(x, r),
            |x, j| problem.jacobian(x, j),
            |x| problem.project(x),
        )
    }

    /// Callback variant of [`solve_with_default_line_search`].
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
        self.run_with_configured_line_search(m, x, residual_fn, jacobian_fn, project)
    }
}
