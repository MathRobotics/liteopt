use crate::manifolds::space::Space;
use crate::numerics::linalg::dot;
use crate::problems::least_squares::LeastSquaresProblem;
use crate::solvers::common::least_squares::{
    commit_trial_state, residual_cost, residual_norm, solve_left_jjt_direction,
    solve_normal_jtj_direction,
};
use crate::solvers::common::trace::{SolverTracer, TraceRow};
use crate::solvers::{Jacobian, LinearSolver};

use super::line_search::{
    ArmijoBacktracking, LineSearchContext, LineSearchPolicy, NoLineSearch,
    StrictDecreaseBacktracking,
};
use super::types::{
    DirectionResult, GaussNewton, GaussNewtonLineSearchMethod, GaussNewtonLinearSystem,
    GaussNewtonResult,
};
use super::workspace::GaussNewtonWorkspace;

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> GaussNewton<S> {
    fn make_tracer(&self) -> SolverTracer {
        if self.collect_trace {
            SolverTracer::gn_with_history(self.verbose)
        } else {
            SolverTracer::gn(self.verbose)
        }
    }

    fn attach_trace(
        &self,
        mut result: GaussNewtonResult<Vec<f64>>,
        trace: SolverTracer,
    ) -> GaussNewtonResult<Vec<f64>> {
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
        need_dphi0: bool,
        ws: &mut GaussNewtonWorkspace,
    ) -> Option<DirectionResult> {
        let solved = if let Some(cg) = &mut ws.cg {
            let report = cg.solve(&ws.g, &mut ws.dx, self.cg, |v, out| {
                jacobian.mul(x, &ws.j, v, &mut ws.jv);
                if ws.jv.iter().any(|v| !v.is_finite()) {
                    out.fill(f64::NAN);
                    return;
                }
                jacobian.transpose_mul(x, &ws.j, &ws.jv, out);
            });
            ws.linear_report = Some(report);
            report.converged()
        } else {
            match self.linear_system {
                GaussNewtonLinearSystem::Qr => ws
                    .qr
                    .as_mut()
                    .unwrap()
                    .solve(&ws.j, &ws.r, m, n, 0., &mut ws.dx)
                    .is_ok(),
                GaussNewtonLinearSystem::LeftJjT => solve_left_jjt_direction(
                    &ws.j, &ws.r, m, n, 0.0, &mut ws.a, &mut ws.y, &mut ws.dx,
                ),
                GaussNewtonLinearSystem::NormalJtJ => {
                    solve_normal_jtj_direction(&ws.j, &ws.r, m, n, 0.0, &mut ws.an, &mut ws.dx)
                }
            }
        };
        if !solved {
            return None;
        }

        let dphi0 = if need_dphi0 {
            Some(dot(&ws.g, &ws.dx))
        } else {
            None
        };
        let dx_norm = self.space.tangent_norm(&ws.dx);
        (dx_norm.is_finite() && dx_norm >= 0.0 && ws.dx.iter().all(|v| v.is_finite()))
            .then_some(DirectionResult { dx_norm, dphi0 })
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
        ws: &mut GaussNewtonWorkspace,
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
        ArmijoBacktracking::new(beta, max_steps, c_armijo).with_min_step(self.ls_min_step)
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
        trace: &SolverTracer,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
    {
        match self.line_search_method {
            GaussNewtonLineSearchMethod::Armijo => {
                let mut line_search = self.configured_armijo();
                self.run_with_line_search(
                    m,
                    x,
                    residual_fn,
                    jacobian_fn,
                    project,
                    &mut line_search,
                    trace,
                )
            }
            GaussNewtonLineSearchMethod::StrictDecrease => {
                let mut line_search = self.configured_strict_decrease();
                self.run_with_line_search(
                    m,
                    x,
                    residual_fn,
                    jacobian_fn,
                    project,
                    &mut line_search,
                    trace,
                )
            }
            GaussNewtonLineSearchMethod::None => {
                let mut line_search = NoLineSearch;
                self.run_with_line_search(
                    m,
                    x,
                    residual_fn,
                    jacobian_fn,
                    project,
                    &mut line_search,
                    trace,
                )
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
        trace: &SolverTracer,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
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
        let n_retries = 0;
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
                return GaussNewtonResult {
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
            && self.ls_max_steps > 0)
        {
            finish!("invalid_options", false);
        }
        if m == 0 || n == 0 {
            finish!("invalid_dimensions", false);
        }
        if !x.iter().all(|v| v.is_finite()) {
            finish!("non_finite_initial", false);
        }
        let mut ws =
            GaussNewtonWorkspace::new(m, n, self.linear_system, self.linear_solver, matrix_free);

        residual_fn(&x, &mut ws.r);
        cost = residual_cost(&ws.r);
        r_norm = residual_norm(&ws.r);
        trace.emit(TraceRow::iter(0).cost(cost).r_norm(r_norm).note("initial"));
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
            let Some(direction) = direction else {
                let status = ws
                    .linear_report
                    .filter(|r| !r.converged())
                    .map(|r| r.status)
                    .or_else(|| {
                        ws.qr
                            .as_ref()
                            .and_then(|qr| qr.last_error)
                            .map(|e| e.as_str())
                    })
                    .unwrap_or("linear_solve_failed");
                finish!(status, false);
            };
            let dx_norm = direction.dx_norm;
            let dphi0 = direction.dphi0;
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
                lambda: 0.0,
            };
            let mut ls_trials = 0;
            let ls = line_search.search(&ctx, &mut |alpha| {
                ls_trials += 1;
                self.evaluate_trial(&x, alpha, &mut residual_fn, &mut project, &mut ws)
            });
            n_ls_trials += ls_trials;
            if !ls.accepted {
                trace.emit(
                    TraceRow::iter(it)
                        .cost(cost)
                        .grad_norm(grad_norm)
                        .step_size(alpha0)
                        .ls_trials(ls_trials)
                        .alpha(ls.alpha)
                        .note("rejected"),
                );
                finish!("rejected", false);
            }
            let Some(trial) =
                self.evaluate_trial(&x, ls.alpha, &mut residual_fn, &mut project, &mut ws)
            else {
                trace.emit(
                    TraceRow::iter(it)
                        .cost(cost)
                        .grad_norm(grad_norm)
                        .step_size(alpha0)
                        .ls_trials(ls_trials)
                        .alpha(ls.alpha)
                        .note("accepted_step_invalid"),
                );
                finish!("accepted_step_invalid", false);
            };
            let cost_before = cost;
            let r_before = r_norm;
            n_accepted += 1;
            self.commit_trial_step(&mut x, &mut cost, &mut r_norm, trial, &mut ws);
            trace.emit(
                TraceRow::iter(it)
                    .cost(cost_before)
                    .r_norm(r_before)
                    .dx_norm(dx_norm)
                    .grad_norm(grad_norm)
                    .step_size(alpha0)
                    .ls_trials(ls_trials)
                    .alpha(ls.alpha)
                    .note("accepted"),
            );

            it += 1;
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

    /// Solve with an explicit line search policy.
    ///
    /// - m: residual dimension
    /// - x: initial guess (len = n)
    /// - residual_fn(x, r): fill r (len = m)
    /// - jacobian_fn(x, J): fill Jacobian (len = m*n, row-major)
    /// - project(x): optional projection
    /// - line_search: external step-size policy
    fn run_with_fn<R, JF, P, LS>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
        line_search: &mut LS,
        trace: &SolverTracer,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
    {
        self.run_with_line_search(m, x, residual_fn, jacobian_fn, project, line_search, trace)
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
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
        LS: LineSearchPolicy,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn(m, x, residual_fn, jacobian_fn, project, line_search, &trace);
        self.attach_trace(result, trace)
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
        let trace = self.make_tracer();
        let result = self.run_with_configured_line_search(
            m,
            x,
            |x, r| problem.residual(x, r),
            |x: &[f64], j: &mut [f64]| problem.jacobian(x, j),
            |x| problem.project(x),
            &trace,
        );
        self.attach_trace(result, trace)
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
        let trace = self.make_tracer();
        let result =
            self.run_with_configured_line_search(m, x, residual_fn, jacobian_fn, project, &trace);
        self.attach_trace(result, trace)
    }

    /// Variant accepting dense callbacks or matrix-free JacobianProducts.
    pub fn solve_with_derivatives_default_line_search<R, JF, P>(
        &self,
        m: usize,
        x: Vec<f64>,
        residual_fn: R,
        jacobian_fn: JF,
        project: P,
    ) -> GaussNewtonResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: Jacobian,
        P: FnMut(&mut [f64]),
    {
        let trace = self.make_tracer();
        let result =
            self.run_with_configured_line_search(m, x, residual_fn, jacobian_fn, project, &trace);
        self.attach_trace(result, trace)
    }
}
