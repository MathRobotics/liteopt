use crate::manifolds::space::Space;

/// Objective function to be minimized.
///
/// - `S::Point` represents points on the space
/// - In `gradient` the user computes a local gradient/update vector and writes
///   it into the buffer (compatible with `Space::retract_into`)
pub trait Objective<S: Space> {
    /// Function value f(x) at x.
    fn value(&self, x: &S::Point) -> f64;

    /// Write the gradient ∇f(x) at x into grad.
    ///
    /// grad is assumed to be pre-initialized, e.g., via zero_tangent_like(x).
    fn gradient(&self, x: &S::Point, grad: &mut S::Tangent);
}
