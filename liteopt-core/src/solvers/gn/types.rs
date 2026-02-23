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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaussNewtonDampingUpdate {
    #[default]
    Adaptive,
    Fixed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaussNewtonLinearSystem {
    #[default]
    LeftJjT,
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
    pub lambda: f64, // damping (initial/fixed depending on damping_update)
    pub damping_update: GaussNewtonDampingUpdate,
    pub linear_system: GaussNewtonLinearSystem,
    pub line_search_method: GaussNewtonLineSearchMethod,
    pub step_scale: f64, // alpha0 in (0,1]
    pub ls_beta: f64,
    pub ls_min_step: f64,
    pub ls_max_steps: usize,
    pub c_armijo: f64,
    pub max_iters: usize,
    pub tol_r: f64,    // stop if ||r|| < tol_r
    pub tol_dq: f64,   // stop if ||local update|| < tol_dq
    pub verbose: bool, // print per-iteration diagnostics
}

impl<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> GaussNewton<S> {
    /// Build a solver on an explicitly provided space.
    pub fn with_space(space: S) -> Self {
        Self {
            space,
            lambda: 1e-3,
            damping_update: GaussNewtonDampingUpdate::Adaptive,
            linear_system: GaussNewtonLinearSystem::LeftJjT,
            line_search_method: GaussNewtonLineSearchMethod::Armijo,
            step_scale: 1.0,
            ls_beta: 0.5,
            ls_min_step: 1e-8,
            ls_max_steps: 20,
            c_armijo: 1e-4,
            max_iters: 100,
            tol_r: 1e-6,
            tol_dq: 1e-6,
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
