use crate::manifolds::space::Space;

/// Apply a local retraction step in-place.
pub fn retract_step<S: Space>(
    space: &S,
    x: &mut S::Point,
    direction: &S::Tangent,
    alpha: f64,
    x_next: &mut S::Point,
    tmp: &mut S::Tangent,
) {
    space.retract_step_into(x, direction, alpha, x_next, tmp);
}
