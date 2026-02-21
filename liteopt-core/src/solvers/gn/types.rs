use crate::manifolds::{space::Space, EuclideanSpace};

#[derive(Clone, Debug)]
pub struct GaussNewtonResult<P> {
    pub x: P,
    pub cost: f64, // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    pub dx_norm: f64,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DirectionResult {
    pub dx_norm: f64,
    pub dphi0: f64,
    pub used_steepest_descent: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LineSearchResult {
    pub accepted: bool,
    pub alpha: f64,
}

#[derive(Clone, Debug)]
pub struct GaussNewton<S: Space<Point = Vec<f64>, Tangent = Vec<f64>> = EuclideanSpace> {
    pub space: S,
    pub lambda: f64,     // damping (initial)
    pub step_scale: f64, // alpha in (0,1]
    pub max_iters: usize,
    pub tol_r: f64,          // stop if ||r|| < tol_r
    pub tol_dq: f64,         // stop if ||local update|| < tol_dq
    pub line_search: bool,   // backtracking line search
    pub ls_beta: f64,        // e.g. 0.5
    pub ls_max_steps: usize, // e.g. 20
    pub c_armijo: f64,       // Armijo condition constant (e.g. 1e-4)
    pub verbose: bool,       // print per-iteration diagnostics
}

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> GaussNewton<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            lambda: 1e-3,
            step_scale: 1.0,
            max_iters: 100,
            tol_r: 1e-6,
            tol_dq: 1e-6,
            line_search: true,
            ls_beta: 0.5,
            ls_max_steps: 20,
            c_armijo: 1e-4,
            verbose: false,
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
