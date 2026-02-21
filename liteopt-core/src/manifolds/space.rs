//! Space abstractions with manifold-style retraction updates.

/// Trait that represents an abstract optimization space.
///
/// The interface is intentionally small. Solvers work with:
/// - points (`x`) on the space
/// - local update vectors (`direction`) used by `retract_into`
pub trait Space {
    type Point: Clone;
    type Tangent: Clone;

    fn zero_like(&self, x: &Self::Point) -> Self::Point;
    fn norm(&self, v: &Self::Point) -> f64;

    // --- core ops (allocation-free if impl does it right) ---
    fn scale_into(&self, out: &mut Self::Tangent, v: &Self::Tangent, alpha: f64);
    fn add_into(&self, out: &mut Self::Point, x: &Self::Point, v: &Self::Tangent);
    fn difference_into(&self, out: &mut Self::Tangent, x: &Self::Point, y: &Self::Point);

    /// Tangent/local zero vector at `x`.
    fn zero_tangent_like(&self, x: &Self::Point) -> Self::Tangent;

    /// Norm on the local update vector.
    fn tangent_norm(&self, v: &Self::Tangent) -> f64;

    /// out = Retr_x(alpha * direction)
    fn retract_into(
        &self,
        out: &mut Self::Point,
        x: &Self::Point,
        direction: &Self::Tangent,
        alpha: f64,
        tmp: &mut Self::Tangent,
    ) {
        self.scale_into(tmp, direction, alpha);
        self.add_into(out, x, tmp);
    }

    /// In-place step update: x <- Retr_x(alpha * direction)
    fn retract_step_into(
        &self,
        x: &mut Self::Point,
        direction: &Self::Tangent,
        alpha: f64,
        x_next: &mut Self::Point,
        tmp: &mut Self::Tangent,
    ) {
        self.retract_into(x_next, x, direction, alpha, tmp);
        std::mem::swap(x, x_next);
    }

    // --- convenience wrappers (allocate; OK for examples) ---
    fn scale(&self, v: &Self::Tangent, alpha: f64) -> Self::Tangent {
        let mut out = v.clone();
        self.scale_into(&mut out, v, alpha);
        out
    }
    fn add(&self, x: &Self::Point, v: &Self::Tangent) -> Self::Point {
        let mut out = self.zero_like(x);
        self.add_into(&mut out, x, v);
        out
    }
    fn difference(&self, x: &Self::Point, y: &Self::Point) -> Self::Tangent {
        let mut out = self.zero_tangent_like(x);
        self.difference_into(&mut out, x, y);
        out
    }
    fn retract(&self, x: &Self::Point, direction: &Self::Tangent, alpha: f64) -> Self::Point {
        let mut out = self.zero_like(x);
        let mut tmp = self.zero_tangent_like(x);
        self.retract_into(&mut out, x, direction, alpha, &mut tmp);
        out
    }
}

// Compatibility re-export: existing code can still import
// `manifolds::space::EuclideanSpace`.
pub use super::euclidean::EuclideanSpace;
