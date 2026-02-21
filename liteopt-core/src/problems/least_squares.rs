use crate::manifolds::space::Space;

/// Problem definition for nonlinear least squares on a space.
///
/// All derivatives are with respect to the local update vector used by
/// `Space::retract_into`.
pub trait LeastSquaresProblem<S: Space<Point = Vec<f64>, Tangent = Vec<f64>>> {
    /// Residual dimension m.
    fn residual_dim(&self) -> usize;

    /// Fill residual vector r(x), len = m.
    fn residual(&self, x: &[f64], out: &mut [f64]);

    /// Fill Jacobian J(x), len = m*n, row-major.
    ///
    /// The i-th row and k-th column must be written to `out[i*n + k]`.
    fn jacobian(&self, x: &[f64], out: &mut [f64]);

    /// Optional projection/constraint hook applied after each step.
    fn project(&self, _x: &mut [f64]) {}
}
