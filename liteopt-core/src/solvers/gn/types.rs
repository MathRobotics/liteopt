use crate::manifolds::{space::Space, EuclideanSpace};
use crate::solvers::SolverTraceRecord;
use crate::solvers::{CgOptions, LinearSolver};

#[derive(Clone, Debug)]
pub struct GaussNewtonResult<P> {
    pub x: P,
    pub cost: f64, // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    /// Norm of the last successfully computed search direction, before step scaling.
    /// Zero if no direction was computed; not a diagnostic recomputed at final x.
    pub dx_norm: f64,
    pub converged: bool,
    /// Termination reason; also emitted as the final history note.
    pub status: &'static str,
    /// Gradient norm at the returned point, if evaluated successfully.
    pub grad_norm: Option<f64>,
    pub n_linear_iters: usize,
    pub linear_status: Option<&'static str>,
    pub linear_residual_norm: Option<f64>,
    pub nfev: usize,
    pub njev: usize,
    pub n_attempts: usize,
    pub n_accepted: usize,
    pub n_retries: usize,
    /// Calls to the evaluator provided to line search; excludes acceptance verification.
    pub n_ls_trials: usize,
    pub trace: Option<Vec<SolverTraceRecord>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaussNewtonLinearSystem {
    Qr,
    LeftJjT,
    #[default]
    NormalJtJ,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaussNewtonLineSearchMethod {
    #[default]
    Armijo,
    StrictDecrease,
    None,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DirectionResult {
    pub dx_norm: f64,
    pub dphi0: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct GaussNewton<S: Space<Point = Vec<f64>, Tangent = Vec<f64>> = EuclideanSpace> {
    pub space: S,
    pub linear_solver: LinearSolver,
    pub cg: CgOptions,
    pub linear_system: GaussNewtonLinearSystem,
    pub line_search_method: GaussNewtonLineSearchMethod,
    pub step_size: f64, // alpha0 in (0,1]
    pub ls_beta: f64,
    pub ls_min_step: f64,
    pub ls_max_steps: usize,
    pub c_armijo: f64,
    pub max_iters: usize,
    /// Stationarity threshold for ||J^T r||.
    pub tol_grad: f64,
    pub tol_r: f64,          // converged if ||r|| <= tol_r
    pub tol_dq: f64,         // stalled if direction norm <= tol_dq without stationarity
    pub verbose: bool,       // print per-iteration diagnostics
    pub collect_trace: bool, // store per-iteration trace rows into the result
}

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> GaussNewton<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            linear_solver: LinearSolver::Direct,
            cg: CgOptions::default(),
            linear_system: GaussNewtonLinearSystem::NormalJtJ,
            line_search_method: GaussNewtonLineSearchMethod::Armijo,
            step_size: 1.0,
            ls_beta: 0.5,
            ls_min_step: 1e-8,
            ls_max_steps: 20,
            c_armijo: 1e-4,
            max_iters: 100,
            tol_grad: 1e-6,
            tol_r: 1e-6,
            tol_dq: 1e-6,
            verbose: false,
            collect_trace: false,
        }
    }
}

impl GaussNewton<EuclideanSpace> {
    /// Build a solver with Euclidean space defaults.
    pub fn new() -> Self {
        Self::with_space(EuclideanSpace)
    }
}

impl Default for GaussNewton<EuclideanSpace> {
    fn default() -> Self {
        Self::new()
    }
}
