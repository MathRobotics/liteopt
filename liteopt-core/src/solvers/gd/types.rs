use crate::manifolds::{space::Space, EuclideanSpace};
use crate::solvers::SolverTraceRecord;

/// Configuration for gradient descent.
#[derive(Clone, Debug)]
pub struct GradientDescent<S: Space = EuclideanSpace> {
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
    /// If true, stores per-iteration trace rows into the result.
    pub collect_trace: bool,
}

impl<S: Space> GradientDescent<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            step_size: 1e-3,
            max_iters: 100,
            tol_grad: 1e-6,
            verbose: false,
            collect_trace: false,
        }
    }
}

impl GradientDescent<EuclideanSpace> {
    /// Build a solver with Euclidean space defaults.
    pub fn new() -> Self {
        Self::with_space(EuclideanSpace)
    }
}

impl Default for GradientDescent<EuclideanSpace> {
    fn default() -> Self {
        Self::new()
    }
}

/// Struct that holds the optimization result.
#[derive(Clone, Debug)]
pub struct OptimizeResult<P> {
    pub x: P,
    pub f: f64,
    pub iters: usize,
    pub grad_norm: f64,
    pub converged: bool,
    pub trace: Option<Vec<SolverTraceRecord>>,
}
