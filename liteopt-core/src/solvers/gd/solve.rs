use crate::manifolds::space::Space;
use crate::problems::objective::Objective;
use crate::solvers::common::step::retract_step;
use crate::solvers::common::step_policy::{LineSearchContext, LineSearchPolicy};
use crate::solvers::common::trace::{SolverTracer, TraceRow};

use super::types::{GradientDescent, OptimizeResult};

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
        mut x: S::Point,
        mut value_fn: F,
        mut grad_fn: G,
        trace: &SolverTracer,
    ) -> OptimizeResult<S::Point>
    where
        F: FnMut(&S::Point) -> f64,
        G: FnMut(&S::Point, &mut S::Tangent),
    {
        // In this lightweight design, gradient and search direction are
        // represented as local tangent vectors around x.
        let mut grad = self.space.zero_tangent_like(&x);
        // Pre-allocate buffers to avoid repeated allocations.
        let mut direction = self.space.zero_tangent_like(&x);
        let mut x_next = self.space.zero_like(&x);
        let mut tmp = self.space.zero_tangent_like(&x); // for retract_into

        for k in 0..self.max_iters {
            grad_fn(&x, &mut grad);

            let grad_norm = self.space.tangent_norm(&grad);
            let f_current = self.verbose.then(|| value_fn(&x));
            if let Some(f) = f_current {
                trace.emit(
                    TraceRow::iter(k)
                        .f(f)
                        .grad_norm(grad_norm)
                        .step_size(self.step_size),
                );
            }
            if grad_norm < self.tol_grad {
                let f = f_current.unwrap_or_else(|| value_fn(&x));
                trace.emit(
                    TraceRow::iter(k)
                        .f(f)
                        .grad_norm(grad_norm)
                        .note("converged"),
                );
                return OptimizeResult {
                    x,
                    f,
                    iters: k,
                    grad_norm,
                    converged: true,
                    trace: None,
                };
            }

            // direction = -grad
            self.space.scale_into(&mut direction, &grad, -1.0);

            // x <- Retr_x(step_size * direction)
            retract_step(
                &self.space,
                &mut x,
                &direction,
                self.step_size,
                &mut x_next,
                &mut tmp,
            );
        }

        let f = value_fn(&x);
        let grad_norm = self.space.tangent_norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
            trace: None,
        }
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
        LS: LineSearchPolicy,
    {
        let mut grad = self.space.zero_tangent_like(&x);
        let mut direction = self.space.zero_tangent_like(&x);
        let mut x_trial = self.space.zero_like(&x);
        let mut tmp = self.space.zero_tangent_like(&x);

        for k in 0..self.max_iters {
            grad_fn(&x, &mut grad);
            let grad_norm = self.space.tangent_norm(&grad);
            let f0 = value_fn(&x);

            trace.emit(
                TraceRow::iter(k)
                    .f(f0)
                    .grad_norm(grad_norm)
                    .step_size(self.step_size),
            );
            if grad_norm < self.tol_grad {
                trace.emit(
                    TraceRow::iter(k)
                        .f(f0)
                        .grad_norm(grad_norm)
                        .note("converged"),
                );
                return OptimizeResult {
                    x,
                    f: f0,
                    iters: k,
                    grad_norm,
                    converged: true,
                    trace: None,
                };
            }

            // direction = -grad
            self.space.scale_into(&mut direction, &grad, -1.0);

            // For steepest descent direction, directional derivative is -||grad||^2.
            let dphi0 = Some(-(grad_norm * grad_norm));
            let ls_ctx = LineSearchContext {
                iter: k,
                alpha0: self.step_size,
                cost0: f0,
                dphi0,
                dx_norm: grad_norm,
                lambda: 0.0,
            };

            let mut eval_cost = |alpha_trial: f64| {
                self.space
                    .retract_into(&mut x_trial, &x, &direction, alpha_trial, &mut tmp);
                let f_trial = value_fn(&x_trial);
                f_trial.is_finite().then_some(f_trial)
            };
            let ls = line_search.search(&ls_ctx, &mut eval_cost);

            if !ls.accepted {
                return OptimizeResult {
                    x,
                    f: f0,
                    iters: k,
                    grad_norm,
                    converged: false,
                    trace: None,
                };
            }

            self.space
                .retract_into(&mut x_trial, &x, &direction, ls.alpha, &mut tmp);
            let f_trial = value_fn(&x_trial);
            if !f_trial.is_finite() {
                return OptimizeResult {
                    x,
                    f: f0,
                    iters: k,
                    grad_norm,
                    converged: false,
                    trace: None,
                };
            }
            std::mem::swap(&mut x, &mut x_trial);
        }

        let f = value_fn(&x);
        let grad_norm = self.space.tangent_norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
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
        LS: LineSearchPolicy,
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
        LS: LineSearchPolicy,
    {
        let trace = self.make_tracer();
        let result = self.run_with_fn_and_line_search(x, value_fn, grad_fn, line_search, &trace);
        self.attach_trace(result, trace)
    }
}
