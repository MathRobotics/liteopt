use crate::manifolds::space::Space;
use crate::problems::objective::Objective;
use crate::solvers::common::step_policy::{LineSearchContext, LineSearchPolicy, NoLineSearch};
use crate::solvers::common::trace::{SolverTracer, TraceRow};

use super::types::{GdTermination, GradientDescent, OptimizeResult};

impl<S: Space> GradientDescent<S> {
    fn make_tracer(&self) -> SolverTracer {
        if self.collect_trace {
            SolverTracer::gd_with_history(self.verbose)
        } else {
            SolverTracer::gd(self.verbose)
        }
    }

    fn attach_trace(
        &self,
        mut result: OptimizeResult<S::Point>,
        trace: SolverTracer,
    ) -> OptimizeResult<S::Point> {
        result.trace = if self.collect_trace {
            Some(trace.into_history())
        } else {
            None
        };
        result
    }

    fn run_with_fn<F, G>(
        &self,
        x: S::Point,
        value_fn: F,
        grad_fn: G,
        trace: &SolverTracer,
    ) -> OptimizeResult<S::Point>
    where
        F: FnMut(&S::Point) -> f64,
        G: FnMut(&S::Point, &mut S::Tangent),
    {
        self.run_with_fn_and_line_search(x, value_fn, grad_fn, &mut NoLineSearch, trace)
    }

    fn run_with_fn_and_line_search<F, G, LS>(
        &self,
        mut x: S::Point,
        mut value_fn: F,
        mut grad_fn: G,
        line_search: &mut LS,
        trace: &SolverTracer,
    ) -> OptimizeResult<S::Point>
    where
        F: FnMut(&S::Point) -> f64,
        G: FnMut(&S::Point, &mut S::Tangent),
        LS: LineSearchPolicy + ?Sized,
    {
        let mut grad = self.space.zero_tangent_like(&x);
        let mut direction = self.space.zero_tangent_like(&x);
        let mut x_trial = self.space.zero_like(&x);
        let mut tmp = self.space.zero_tangent_like(&x);
        let mut nfev = 0;
        let mut n_attempts = 0;
        let mut n_ls_trials = 0;
        let mut njev = 0;
        let mut f = f64::NAN;
        let mut grad_norm = f64::NAN;
        let mut iters = 0;
        let status = if !self.step_size.is_finite()
            || self.step_size <= 0.0
            || !self.tol_grad.is_finite()
            || self.tol_grad < 0.0
        {
            GdTermination::InvalidOptions
        } else if !self.space.norm(&x).is_finite() {
            GdTermination::NonFinite
        } else {
            f = value_fn(&x);
            nfev += 1;
            trace.emit(TraceRow::iter(0).f(f).note("initial"));
            loop {
                if !f.is_finite() {
                    break GdTermination::NonFinite;
                }
                grad_fn(&x, &mut grad);
                njev += 1;
                grad_norm = self.space.tangent_norm(&grad);
                if !grad_norm.is_finite() || grad_norm < 0.0 {
                    break GdTermination::NonFinite;
                }
                // Check the returned point even after the last allowed update.
                if grad_norm <= self.tol_grad {
                    break GdTermination::Converged;
                }
                if iters == self.max_iters {
                    break GdTermination::MaxIterations;
                }
                n_attempts += 1;
                self.space.scale_into(&mut direction, &grad, -1.0);
                let ctx = LineSearchContext {
                    iter: iters,
                    alpha0: self.step_size,
                    cost0: f,
                    dphi0: Some(-grad_norm * grad_norm),
                    dx_norm: grad_norm,
                    lambda: 0.0,
                };
                let mut cached = None;
                let mut ls_trials = 0;
                let mut eval_cost = |alpha: f64| {
                    ls_trials += 1;
                    cached = None;
                    if !alpha.is_finite() || alpha <= 0.0 {
                        return None;
                    }
                    self.space
                        .retract_into(&mut x_trial, &x, &direction, alpha, &mut tmp);
                    if !self.space.norm(&x_trial).is_finite() {
                        return None;
                    }
                    let trial = value_fn(&x_trial);
                    nfev += 1;
                    cached = Some((alpha, trial));
                    trial.is_finite().then_some(trial)
                };
                let ls = line_search.search(&ctx, &mut eval_cost);
                n_ls_trials += ls_trials;
                let row = TraceRow::iter(iters)
                    .f(f)
                    .grad_norm(grad_norm)
                    .step_size(self.step_size)
                    .ls_trials(ls_trials)
                    .alpha(ls.alpha);
                if !ls.accepted {
                    trace.emit(row.note("rejected"));
                    break GdTermination::LineSearchFailed;
                }
                if !ls.alpha.is_finite() || ls.alpha <= 0.0 {
                    trace.emit(row.note("accepted_step_invalid"));
                    break GdTermination::InvalidStep;
                }
                // Reuse both the point and its cost from the last accepted trial.
                let trial = match cached {
                    Some((alpha, cost)) if alpha == ls.alpha => cost,
                    _ => {
                        self.space
                            .retract_into(&mut x_trial, &x, &direction, ls.alpha, &mut tmp);
                        if !self.space.norm(&x_trial).is_finite() {
                            trace.emit(row.note("accepted_step_invalid"));
                            break GdTermination::InvalidStep;
                        }
                        nfev += 1;
                        value_fn(&x_trial)
                    }
                };
                if !trial.is_finite() {
                    trace.emit(row.note("accepted_step_invalid"));
                    break GdTermination::InvalidStep;
                }
                trace.emit(row.note("accepted"));
                std::mem::swap(&mut x, &mut x_trial);
                f = trial;
                iters += 1;
            }
        };
        trace.emit(
            TraceRow::iter(iters)
                .f(f)
                .grad_norm(grad_norm)
                .note(status.as_str())
                .phase("final"),
        );
        OptimizeResult {
            x,
            f,
            iters,
            grad_norm,
            converged: status == GdTermination::Converged,
            status,
            nfev,
            n_attempts,
            n_ls_trials,
            njev,
            trace: None,
        }
    }

    pub fn minimize<O>(&self, obj: &O, x: S::Point) -> OptimizeResult<S::Point>
    where
        O: Objective<S>,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn(x, |p| obj.value(p), |p, g| obj.gradient(p, g), &trace);
        self.attach_trace(result, trace)
    }

    /// Minimize using user-provided value and gradient functions.
    pub fn minimize_with_fn<F, G>(
        &self,
        x: S::Point,
        value_fn: F,
        grad_fn: G,
    ) -> OptimizeResult<S::Point>
    where
        F: Fn(&S::Point) -> f64,
        G: Fn(&S::Point, &mut S::Tangent),
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn(x, |p| value_fn(p), |p, g| grad_fn(p, g), &trace);
        self.attach_trace(result, trace)
    }

    /// Minimize using an explicit line-search policy.
    pub fn minimize_with_line_search<O, LS>(
        &self,
        obj: &O,
        x: S::Point,
        line_search: &mut LS,
    ) -> OptimizeResult<S::Point>
    where
        O: Objective<S>,
        LS: LineSearchPolicy + ?Sized,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn_and_line_search(
            x,
            |p| obj.value(p),
            |p, g| obj.gradient(p, g),
            line_search,
            &trace,
        );
        self.attach_trace(result, trace)
    }

    /// Minimize callbacks using an explicit line-search policy.
    pub fn minimize_with_fn_and_line_search<F, G, LS>(
        &self,
        x: S::Point,
        value_fn: F,
        grad_fn: G,
        line_search: &mut LS,
    ) -> OptimizeResult<S::Point>
    where
        F: FnMut(&S::Point) -> f64,
        G: FnMut(&S::Point, &mut S::Tangent),
        LS: LineSearchPolicy + ?Sized,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn_and_line_search(x, value_fn, grad_fn, line_search, &trace);
        self.attach_trace(result, trace)
    }
}
