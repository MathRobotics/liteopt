use crate::manifolds::{space::Space, EuclideanSpace};
use crate::solvers::SolverTraceRecord;
use crate::solvers::{CgOptions, LinearSolver};

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtResult<P> {
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

/// Built-in step acceptance/search used by the configured solve methods.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevenbergMarquardtLineSearchMethod {
    #[default]
    CostDecrease,
    Armijo,
    StrictDecrease,
    None,
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardt<S: Space<Point = Vec<f64>, Tangent = Vec<f64>> = EuclideanSpace> {
    pub space: S,
    pub linear_solver: LinearSolver,
    pub cg: CgOptions,
    pub linear_system: LevenbergMarquardtLinearSystem,
    pub damping_update: LevenbergMarquardtDampingUpdate,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub lambda: f64,      // initial damping
    pub lambda_up: f64,   // multiply lambda on rejected step
    pub lambda_down: f64, // multiply lambda on accepted step
    pub step_size: f64,   // alpha in (0, 1]
    pub line_search_method: LevenbergMarquardtLineSearchMethod,
    pub ls_beta: f64,
    pub ls_min_step: f64,
    pub ls_max_steps: usize,
    pub c_armijo: f64,
    pub max_iters: usize,
    /// Stationarity threshold for ||J^T r||.
    pub tol_grad: f64,
    pub tol_r: f64,  // converged if ||r|| <= tol_r
    pub tol_dq: f64, // stalled if direction norm <= tol_dq without stationarity
    pub verbose: bool,
    pub collect_trace: bool,
}

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> LevenbergMarquardt<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            linear_solver: LinearSolver::Direct,
            cg: CgOptions::default(),
            linear_system: LevenbergMarquardtLinearSystem::LeftJjT,
            damping_update: LevenbergMarquardtDampingUpdate::CostBased,
            lambda_min: 1e-12,
            lambda_max: f64::MAX,
            lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.5,
            step_size: 1.0,
            line_search_method: LevenbergMarquardtLineSearchMethod::CostDecrease,
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

impl LevenbergMarquardt<EuclideanSpace> {
    /// Build a solver with Euclidean space defaults.
    pub fn new() -> Self {
        Self::with_space(EuclideanSpace)
    }
}

impl Default for LevenbergMarquardt<EuclideanSpace> {
    fn default() -> Self {
        Self::new()
    }
}

/// Damped dense linear system backend. QR solves the augmented least-squares system.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevenbergMarquardtLinearSystem {
    #[default]
    LeftJjT,
    Qr,
}

/// Controls acceptance and damping adjustment after line search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LevenbergMarquardtDampingUpdate {
    #[default]
    CostBased,
    GainRatio,
}
