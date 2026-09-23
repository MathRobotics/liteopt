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
    /// Considered converged when the gradient norm is at or below this threshold.
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
    pub status: GdTermination,
    /// Number of objective evaluations, including line-search trials.
    pub nfev: usize,
    pub n_attempts: usize,
    pub n_ls_trials: usize,
    /// Number of gradient evaluations, including the final point.
    pub njev: usize,
}

/// Reason gradient descent stopped. Only `Converged` is success.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdTermination {
    Converged,
    MaxIterations,
    LineSearchFailed,
    NonFinite,
    InvalidStep,
    InvalidOptions,
}
impl GdTermination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::MaxIterations => "max_iters",
            Self::LineSearchFailed => "line_search_failed",
            Self::NonFinite => "non_finite",
            Self::InvalidStep => "invalid_step",
            Self::InvalidOptions => "invalid_options",
        }
    }
}
