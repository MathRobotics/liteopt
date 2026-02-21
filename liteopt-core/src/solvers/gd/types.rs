use crate::manifolds::space::Space;

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
    /// If true, prints per-iteration diagnostics (f, |grad|, step size).
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
