use crate::manifolds::{space::Space, EuclideanSpace};
use crate::solvers::SolverTraceRecord;

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtResult<P> {
    pub x: P,
    pub cost: f64, // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    pub dx_norm: f64,
    pub converged: bool,
    pub trace: Option<Vec<SolverTraceRecord>>,
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardt<S: Space<Point = Vec<f64>, Tangent = Vec<f64>> = EuclideanSpace> {
    pub space: S,
    pub lambda: f64,      // initial damping
    pub lambda_up: f64,   // multiply lambda on rejected step
    pub lambda_down: f64, // multiply lambda on accepted step
    pub step_scale: f64,  // alpha in (0, 1]
    pub max_iters: usize,
    pub tol_r: f64,  // stop if ||r|| < tol_r
    pub tol_dq: f64, // stop if ||local update|| < tol_dq
    pub verbose: bool,
    pub collect_trace: bool,
}

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> LevenbergMarquardt<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            lambda: 1e-3,
            lambda_up: 10.0,
            lambda_down: 0.5,
            step_scale: 1.0,
            max_iters: 100,
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
