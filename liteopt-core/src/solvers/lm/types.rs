use crate::manifolds::space::Space;

#[derive(Clone, Debug)]
pub struct LevenbergMarquardtResult<P> {
    pub x: P,
    pub cost: f64, // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    pub dx_norm: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct LevenbergMarquardt<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> {
    pub space: S,
    pub lambda: f64,      // initial damping
    pub lambda_up: f64,   // multiply lambda on rejected step
    pub lambda_down: f64, // multiply lambda on accepted step
    pub step_scale: f64,  // alpha in (0, 1]
    pub max_iters: usize,
    pub tol_r: f64,  // stop if ||r|| < tol_r
    pub tol_dq: f64, // stop if ||local update|| < tol_dq
    pub verbose: bool,
}
