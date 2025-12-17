use crate::objective::Objective;
use crate::space::Space;

/// Configuration for gradient descent.
#[derive(Clone, Debug)]
pub struct GradientDescent<S: Space> {
    /// Space to operate on (MVP can fix this to EuclideanSpace).
    pub space: S,
    /// Learning rate / step size.
    pub step_size: f64,
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Considered converged when the gradient norm falls below this threshold.
    pub tol_grad: f64,
    /// If true, prints per-iteration diagnostics (f, \|grad\|, step size).
    pub verbose: bool,
}

/// Struct that holds the optimization result.
#[derive(Clone, Debug)]
pub struct OptimizeResult<P> {
    pub x: P,
    pub f: f64,
    pub iters: usize,
    pub grad_norm: f64,
    pub converged: bool,
}

impl<S: Space> GradientDescent<S> {
    pub fn minimize<O>(&self, obj: &O, mut x: S::Point) -> OptimizeResult<S::Point>
    where
        O: Objective<S>,
    {
        let mut grad = self.space.zero_like(&x);

        // pre-allocate buffers to avoid repeated allocations
        let mut direction = self.space.zero_like(&x);
        let mut x_next = self.space.zero_like(&x);
        let mut tmp = self.space.zero_like(&x); // for retract_into

        for k in 0..self.max_iters {
            // Compute gradient.
            obj.gradient(&x, &mut grad);

            let grad_norm = self.space.norm(&grad);
            let f_current = self.verbose.then(|| obj.value(&x));
            if let Some(f) = f_current {
                println!(
                    "[gd] iter {:>6} | f {:>13.6e} | grad {:>13.6e} | step {:>+9.3e}",
                    k, f, grad_norm, self.step_size
                );
            }
            if grad_norm < self.tol_grad {
                let f = f_current.unwrap_or_else(|| obj.value(&x));
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

            // x_next = Retr_x(step_size * direction)
            self.space
                .retract_into(&mut x_next, &x, &direction, self.step_size, &mut tmp);

            // x <- x_next
            std::mem::swap(&mut x, &mut x_next);
        }

        let f = obj.value(&x);
        let grad_norm = self.space.norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
        }
    }

    /// ★ Minimize using user-provided value and gradient functions.
    pub fn minimize_with_fn<F, G>(
        &self,
        mut x: S::Point,
        value_fn: F,
        grad_fn: G,
    ) -> OptimizeResult<S::Point>
    where
        F: Fn(&S::Point) -> f64,
        G: Fn(&S::Point, &mut S::Point),
    {
        let mut grad = self.space.zero_like(&x);

        // pre-allocate buffers to avoid repeated allocations
        let mut direction = self.space.zero_like(&x);
        let mut x_next = self.space.zero_like(&x);
        let mut tmp = self.space.zero_like(&x); // for retract_into

        for k in 0..self.max_iters {
            // call the user-provided gradient function
            grad_fn(&x, &mut grad);

            let grad_norm = self.space.norm(&grad);
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

            // x_next = Retr_x(step_size * direction)
            self.space
                .retract_into(&mut x_next, &x, &direction, self.step_size, &mut tmp);

            // x <- x_next
            std::mem::swap(&mut x, &mut x_next);
        }

        let f = value_fn(&x);
        let grad_norm = self.space.norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
        }
    }
}
