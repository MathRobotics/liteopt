use crate::manifolds::space::Space;
use crate::numerics::linalg::dot;
use crate::problems::least_squares::LeastSquaresProblem;
use crate::solvers::common::least_squares::{
    commit_trial_state, residual_cost, residual_norm, solve_left_jjt_direction,
};
use crate::solvers::common::step_policy::{CostDecrease, LineSearchContext, LineSearchPolicy};
use crate::solvers::common::trace::{SolverTracer, TraceRow};
use crate::solvers::{Jacobian, LinearSolver};

use super::types::{
    LevenbergMarquardt, LevenbergMarquardtDampingUpdate, LevenbergMarquardtLineSearchMethod,
    LevenbergMarquardtResult,
};
use super::workspace::LmWorkspace;
use crate::solvers::gn::{ArmijoBacktracking, NoLineSearch, StrictDecreaseBacktracking};

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> LevenbergMarquardt<S> {
    fn make_tracer(&self) -> SolverTracer {
        if self.collect_trace {
            SolverTracer::lm_with_history(self.verbose)
        } else {
            SolverTracer::lm(self.verbose)
        }
    }

    fn attach_trace(
        &self,
        mut result: LevenbergMarquardtResult<Vec<f64>>,
        trace: SolverTracer,
    ) -> LevenbergMarquardtResult<Vec<f64>> {
        result.trace = if self.collect_trace {
            Some(trace.into_history())
        } else {
            None
        };
        result
    }

    fn compute_direction<JF: Jacobian>(
        &self,
        x: &[f64],
        jacobian: &mut JF,
        m: usize,
        n: usize,
        lambda: f64,
        need_dphi0: bool,
        ws: &mut LmWorkspace,
    ) -> Option<(f64, Option<f64>, bool)> {
        let solved = if let Some(cg) = &mut ws.cg {
            let report = cg.solve(&ws.g, &mut ws.dx, self.cg, |v, out| {
                jacobian.mul(x, &ws.j, v, &mut ws.jv);
                if ws.jv.iter().any(|v| !v.is_finite()) {
                    out.fill(f64::NAN);
                    return;
                }
                jacobian.transpose_mul(x, &ws.j, &ws.jv, out);
                for (value, vi) in out.iter_mut().zip(v) {
                    *value += lambda * vi;
                }
            });
            ws.linear_report = Some(report);
            report.converged()
        } else if let Some(qr) = &mut ws.qr {
            qr.solve(&ws.j, &ws.r, m, n, lambda, &mut ws.dx).is_ok()
        } else {
            solve_left_jjt_direction(&ws.j, &ws.r, m, n, lambda, &mut ws.a, &mut ws.y, &mut ws.dx)
        };
        if !solved {
            return None;
        }

        let mut dphi0 = need_dphi0.then(|| dot(&ws.g, &ws.dx));
        let fallback = dphi0.is_some_and(|v| !v.is_finite() || v >= 0.0);
        if fallback {
            for (d, g) in ws.dx.iter_mut().zip(&ws.g) {
                *d = -g;
            }
            dphi0 = Some(dot(&ws.g, &ws.dx));
        }
        let dx_norm = self.space.tangent_norm(&ws.dx);

        if !dx_norm.is_finite() || dx_norm < 0.0 || !ws.dx.iter().all(|v| v.is_finite()) {
            return None;
        }
        Some((dx_norm, dphi0, fallback))
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
        if !alpha.is_finite() || alpha <= 0.0 {
            return None;
        }
        self.space
            .retract_into(&mut ws.x_trial, x, &ws.dx, alpha, &mut ws.tmp);
        if !ws.x_trial.iter().all(|v| v.is_finite()) {
            return None;
        }
        project(&mut ws.x_trial);
        if !ws.x_trial.iter().all(|v| v.is_finite()) {
            return None;
        }

        residual_fn(&ws.x_trial, &mut ws.r_trial);
        let cost_trial = residual_cost(&ws.r_trial);
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
        commit_trial_state(
            x,
            &mut ws.r,
            cost,
            r_norm,
            &ws.x_trial,
            &ws.r_trial,
            cost_trial,
        );
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
        LS: LineSearchPolicy + ?Sized,
    {
        let m = problem.residual_dim();
        let trace = self.make_tracer();
        let result = self.run_with_fn(
            m,
            x,
            |x, r| problem.residual(x, r),
            |x: &[f64], j: &mut [f64]| problem.jacobian(x, j),
            |x| problem.project(x),
            line_search,
            &trace,
        );
        self.attach_trace(result, trace)
    }

    fn run_with_fn<R, JF, P, LS>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
        line_search: &mut LS,
        trace: &SolverTracer,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy + ?Sized,
    {
        let nfev = std::cell::Cell::new(0usize);
        let njev = std::cell::Cell::new(0usize);
        let mut residual_fn = |x: &[f64], out: &mut [f64]| {
            nfev.set(nfev.get() + 1);
            residual_fn(x, out);
        };
        let mut n_linear_iters = 0;
        let mut linear_status = None;
        let mut linear_residual_norm = None;
        let matrix_free = jacobian_fn.is_matrix_free();
        let mut n_attempts = 0;
        let mut n_accepted = 0;
        let mut n_retries = 0;
        let mut n_ls_trials = 0;
        let n = x.len();
        let mut it = 0;
        let mut cost = f64::NAN;
        let mut r_norm = f64::NAN;
        let mut last_dx_norm = 0.0;
        let mut grad_norm = f64::NAN;
        // All exits record the state being returned. No small-step exit is success.
        macro_rules! finish {
            ($status:expr, $ok:expr) => {{
                let row = TraceRow::iter(it)
                    .cost(cost)
                    .r_norm(r_norm)
                    .dx_norm(last_dx_norm)
                    .note($status)
                    .phase("final");
                trace.emit(if grad_norm.is_finite() {
                    row.grad_norm(grad_norm)
                } else {
                    row
                });
                return LevenbergMarquardtResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: last_dx_norm,
                    converged: $ok,
                    status: $status,
                    grad_norm: grad_norm.is_finite().then_some(grad_norm),
                    n_linear_iters,
                    linear_status,
                    linear_residual_norm,
                    nfev: nfev.get(),
                    njev: njev.get(),
                    n_attempts,
                    n_accepted,
                    n_retries,
                    n_ls_trials,
                    trace: None,
                };
            }};
        }
        if (matrix_free && self.linear_solver != LinearSolver::Cg) || !self.cg.is_valid() {
            finish!("invalid_options", false);
        }
        if !(self.step_size.is_finite()
            && self.tol_r.is_finite()
            && self.tol_r >= 0.0
            && self.tol_dq.is_finite()
            && self.tol_dq >= 0.0
            && self.tol_grad.is_finite()
            && self.tol_grad >= 0.0
            && self.ls_beta.is_finite()
            && self.ls_beta > 0.0
            && self.ls_beta < 1.0
            && self.c_armijo.is_finite()
            && self.c_armijo > 0.0
            && self.c_armijo < 1.0
            && self.ls_min_step.is_finite()
            && self.ls_min_step > 0.0
            && self.ls_max_steps > 0
            && self.step_size > 0.0
            && self.step_size <= 1.0
            && self.lambda_min.is_finite()
            && self.lambda_min > 0.0
            && self.lambda_max.is_finite()
            && self.lambda_max >= self.lambda_min
            && self.lambda <= self.lambda_max
            && self.lambda.is_finite()
            && self.lambda >= 0.0
            && self.lambda_up.is_finite()
            && self.lambda_up > 1.0
            && self.lambda_down.is_finite()
            && self.lambda_down > 0.0
            && self.lambda_down < 1.0)
        {
            finish!("invalid_options", false);
        }
        if m == 0 || n == 0 {
            finish!("invalid_dimensions", false);
        }
        if !x.iter().all(|v| v.is_finite()) {
            finish!("non_finite_initial", false);
        }
        let mut ws = LmWorkspace::new(m, n, self.linear_system, self.linear_solver, matrix_free);
        let mut lambda = self.lambda.max(self.lambda_min);
        residual_fn(&x, &mut ws.r);
        cost = residual_cost(&ws.r);
        r_norm = residual_norm(&ws.r);
        trace.emit(TraceRow::iter(0).cost(cost).r_norm(r_norm).note("initial"));
        // Preserve the failed attempt before terminating at a damping bound.
        macro_rules! retry_with_damping {
            ($row:expr) => {{
                let row = $row;
                let next = lambda * self.lambda_up;
                if !next.is_finite() || next <= lambda {
                    trace.emit(row);
                    finish!("damping_overflow", false);
                }
                if next > self.lambda_max {
                    trace.emit(row);
                    finish!("damping_limit", false);
                }
                lambda = next;
                n_retries += 1;
                trace.emit(row.lambda_next(lambda));
            }};
        }
        loop {
            grad_norm = f64::NAN;
            if !cost.is_finite() || !r_norm.is_finite() {
                finish!("non_finite_residual", false);
            }
            if !matrix_free {
                njev.set(njev.get() + 1);
            }
            jacobian_fn.prepare(&x, &mut ws.j);
            if !ws.j.iter().all(|v| v.is_finite()) {
                finish!("non_finite_jacobian", false);
            }
            jacobian_fn.transpose_mul(&x, &ws.j, &ws.r, &mut ws.g);
            grad_norm = self.space.tangent_norm(&ws.g);
            if !ws.g.iter().all(|v| v.is_finite()) || !grad_norm.is_finite() || grad_norm < 0.0 {
                finish!("non_finite_gradient", false);
            }
            if r_norm <= self.tol_r {
                finish!("converged_r", true);
            }
            if grad_norm <= self.tol_grad {
                finish!("converged_grad", true);
            }
            if it == self.max_iters {
                finish!("max_iters", false);
            }
            n_attempts += 1;
            let direction = self.compute_direction(
                &x,
                &mut jacobian_fn,
                m,
                n,
                lambda,
                line_search.requires_directional_derivative(),
                &mut ws,
            );
            if let Some(report) = ws.linear_report {
                n_linear_iters += report.iters;
                linear_status = Some(report.status);
                linear_residual_norm = report
                    .residual_norm
                    .is_finite()
                    .then_some(report.residual_norm);
                trace.emit(
                    TraceRow::iter(it)
                        .linear(report)
                        .note(report.status)
                        .phase("linear"),
                );
                if report.status == "linear_non_finite" {
                    finish!("linear_non_finite", false);
                }
            }
            let Some((dx_norm, dphi0, fallback)) = direction else {
                retry_with_damping!(TraceRow::iter(it)
                    .cost(cost)
                    .r_norm(r_norm)
                    .dx_norm(last_dx_norm)
                    .lambda(lambda)
                    .note(
                        ws.linear_report
                            .filter(|r| !r.converged())
                            .map(|r| r.status)
                            .or_else(|| ws
                                .qr
                                .as_ref()
                                .and_then(|qr| qr.last_error)
                                .map(|e| e.as_str()))
                            .unwrap_or("linear_solve_failed"),
                    ));
                it += 1;
                continue;
            };
            if fallback {
                trace.emit(
                    TraceRow::iter(it)
                        .dphi0(dphi0.unwrap_or(f64::NAN))
                        .note("fallback_to_steepest_descent"),
                );
            }
            last_dx_norm = dx_norm;
            if dx_norm <= self.tol_dq {
                finish!("stalled", false);
            }
            let alpha0 = self.step_size.clamp(0.0, 1.0);
            if alpha0 == 0.0 {
                finish!("zero_step_size", false);
            }
            let ctx = LineSearchContext {
                iter: it,
                alpha0,
                cost0: cost,
                dphi0,
                dx_norm,
                lambda,
            };
            let mut ls_trials = 0;
            let ls = line_search.search(&ctx, &mut |alpha| {
                ls_trials += 1;
                self.evaluate_trial(&x, alpha, &mut residual_fn, &mut project, &mut ws)
            });
            n_ls_trials += ls_trials;
            if !ls.accepted {
                retry_with_damping!(TraceRow::iter(it)
                    .cost(cost)
                    .r_norm(r_norm)
                    .dx_norm(last_dx_norm)
                    .grad_norm(grad_norm)
                    .step_size(alpha0)
                    .ls_trials(ls_trials)
                    .alpha(ls.alpha)
                    .lambda(lambda)
                    .note("rejected"));
                it += 1;
                continue;
            }
            let Some(trial) =
                self.evaluate_trial(&x, ls.alpha, &mut residual_fn, &mut project, &mut ws)
            else {
                retry_with_damping!(TraceRow::iter(it)
                    .cost(cost)
                    .r_norm(r_norm)
                    .dx_norm(last_dx_norm)
                    .grad_norm(grad_norm)
                    .step_size(alpha0)
                    .ls_trials(ls_trials)
                    .alpha(ls.alpha)
                    .lambda(lambda)
                    .note("accepted_step_invalid"));
                it += 1;
                continue;
            };
            let mut predicted = f64::NAN;
            let actual = cost - trial;
            let mut ratio = f64::NAN;
            if self.damping_update == LevenbergMarquardtDampingUpdate::GainRatio {
                // Model the actual local displacement, including alpha, projection
                // and retraction. Space::difference(x,y) must return y relative to x.
                self.space.difference_into(&mut ws.tmp, &x, &ws.x_trial);
                if ws.tmp.len() == n && ws.tmp.iter().all(|v| v.is_finite()) {
                    jacobian_fn.mul(&x, &ws.j, &ws.tmp, &mut ws.jv);
                    if ws.jv.iter().any(|v| !v.is_finite()) {
                        finish!("linear_non_finite", false);
                    }
                    predicted = -dot(&ws.g, &ws.tmp) - 0.5 * dot(&ws.jv, &ws.jv);
                    if predicted.is_finite() && predicted > 0.0 {
                        ratio = actual / predicted;
                    }
                }
                if !ratio.is_finite() || ratio <= 1e-4 {
                    let row = TraceRow::iter(it)
                        .cost(cost)
                        .r_norm(r_norm)
                        .grad_norm(grad_norm)
                        .dx_norm(dx_norm)
                        .step_size(alpha0)
                        .alpha(ls.alpha)
                        .ls_trials(ls_trials)
                        .lambda(lambda)
                        .reductions(predicted, actual, ratio)
                        .note(if !ratio.is_finite() {
                            "invalid_prediction"
                        } else {
                            "gain_ratio_rejected"
                        });
                    retry_with_damping!(row);
                    it += 1;
                    continue;
                }
            }
            let next_accepted = if self.damping_update == LevenbergMarquardtDampingUpdate::CostBased
                || ratio > 0.75
            {
                (lambda * self.lambda_down).max(self.lambda_min)
            } else if ratio < 0.25 {
                // Accepted but poorly predicted: be more conservative next time.
                (lambda * self.lambda_up).min(self.lambda_max)
            } else {
                lambda
            };
            let cost_before = cost;
            let r_before = r_norm;
            n_accepted += 1;
            self.commit_trial_step(&mut x, &mut cost, &mut r_norm, trial, &mut ws);
            lambda = next_accepted;
            trace.emit(
                TraceRow::iter(it)
                    .cost(cost_before)
                    .r_norm(r_before)
                    .dx_norm(dx_norm)
                    .grad_norm(grad_norm)
                    .step_size(alpha0)
                    .ls_trials(ls_trials)
                    .alpha(ls.alpha)
                    .lambda(ctx.lambda)
                    .lambda_next(lambda)
                    .reductions(predicted, actual, ratio)
                    .note("accepted"),
            );
            it += 1;
        }
    }

    /// Solve nonlinear least squares with LM-style damping updates.
    pub fn solve_with_fn<R, JF, P, LS>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
        line_search: &mut LS,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy + ?Sized,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn(m, x, residual_fn, jacobian_fn, project, line_search, &trace);
        self.attach_trace(result, trace)
    }

    /// Variant accepting dense callbacks or matrix-free JacobianProducts.
    pub fn solve_with_derivatives<R, JF, P, LS>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
        line_search: &mut LS,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy + ?Sized,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn(m, x, residual_fn, jacobian_fn, project, line_search, &trace);
        self.attach_trace(result, trace)
    }

    /// Solve using the configured built-in step policy (CostDecrease by default).
    pub fn solve_with_default_line_search<P>(
        &self,
        x: Vec<f64>,
        problem: &P,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        P: LeastSquaresProblem<S>,
    {
        self.with_configured_line_search(|line_search| self.solve(x, problem, line_search))
    }

    /// Solve callback form using the configured built-in step policy.
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
        self.with_configured_line_search(|line_search| {
            self.solve_with_fn(m, x, residual_fn, jacobian_fn, project, line_search)
        })
    }

    /// Variant accepting dense callbacks or matrix-free JacobianProducts.
    pub fn solve_with_derivatives_default_line_search<R, JF, P>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
    ) -> LevenbergMarquardtResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
    {
        self.with_configured_line_search(|line_search| {
            self.solve_with_derivatives(m, x, residual_fn, jacobian_fn, project, line_search)
        })
    }
    fn with_configured_line_search<T>(
        &self,
        run: impl FnOnce(&mut dyn LineSearchPolicy) -> T,
    ) -> T {
        let beta = if self.ls_beta.is_finite() && self.ls_beta > 0.0 && self.ls_beta < 1.0 {
            self.ls_beta
        } else {
            0.5
        };
        let max_steps = self.ls_max_steps.max(1);
        match self.line_search_method {
            LevenbergMarquardtLineSearchMethod::CostDecrease => run(&mut CostDecrease),
            LevenbergMarquardtLineSearchMethod::None => run(&mut NoLineSearch),
            LevenbergMarquardtLineSearchMethod::Armijo => {
                let c = if self.c_armijo.is_finite() && self.c_armijo > 0.0 && self.c_armijo < 1.0 {
                    self.c_armijo
                } else {
                    1e-4
                };
                run(&mut ArmijoBacktracking::new(beta, max_steps, c)
                    .with_min_step(self.ls_min_step))
            }
            LevenbergMarquardtLineSearchMethod::StrictDecrease => {
                let min_step = if self.ls_min_step.is_finite() && self.ls_min_step > 0.0 {
                    self.ls_min_step
                } else {
                    1e-8
                };
                run(&mut StrictDecreaseBacktracking::new(
                    beta, min_step, max_steps,
                ))
            }
        }
    }
}
