use crate::manifolds::space::Space;
use crate::problems::objective::Objective;
use crate::solvers::common::step::retract_step;

use super::types::{GradientDescent, OptimizeResult};

impl<S: Space> GradientDescent<S> {
    fn run_with_fn<F, G>(
        &self,
        mut x: S::Point,
        mut value_fn: F,
        mut grad_fn: G,
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
                println!(
                    "[gd] iter {:>6} | f {:>13.6e} | grad {:>13.6e} | step {:>+9.3e}",
                    k, f, grad_norm, self.step_size
                );
            }
            if grad_norm < self.tol_grad {
                let f = f_current.unwrap_or_else(|| value_fn(&x));
                if self.verbose {
                    println!(
                        "[gd] iter {:>6} | converged | f {:>13.6e} | grad {:>13.6e}",
                        k, f, grad_norm
                    );
                }
                return OptimizeResult {
                    x,
                    f,
                    iters: k,
                    grad_norm,
                    converged: true,
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
        }
    }

    pub fn minimize<O>(&self, obj: &O, x: S::Point) -> OptimizeResult<S::Point>
    where
        O: Objective<S>,
    {
        self.run_with_fn(x, |p| obj.value(p), |p, g| obj.gradient(p, g))
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
        self.run_with_fn(x, |p| value_fn(p), |p, g| grad_fn(p, g))
    }
}
