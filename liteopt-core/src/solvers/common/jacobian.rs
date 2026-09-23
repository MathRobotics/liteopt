//! Derivatives in the solver's Euclidean local coordinates.
use crate::numerics::linalg::{dot, jt_mul_vec};

/// Jacobian access for GN/LM. Products must be linear and exact adjoints at x.
/// Implementations overwrite every output element. Non-finite outputs abort.
/// For custom manifolds, the columns use the coordinates of the local displacement.
pub trait Jacobian {
    fn is_matrix_free(&self) -> bool {
        false
    }
    /// Fill row-major dense storage, or do nothing for matrix-free operators.
    fn prepare(&mut self, x: &[f64], dense: &mut [f64]);
    fn mul(&mut self, x: &[f64], dense: &[f64], v: &[f64], out: &mut [f64]);
    fn transpose_mul(&mut self, x: &[f64], dense: &[f64], w: &[f64], out: &mut [f64]);
}

/// Existing dense callbacks also implement Jacobian.
impl<F: FnMut(&[f64], &mut [f64])> Jacobian for F {
    fn prepare(&mut self, x: &[f64], dense: &mut [f64]) {
        self(x, dense);
    }
    fn mul(&mut self, _x: &[f64], dense: &[f64], v: &[f64], out: &mut [f64]) {
        for (row, value) in dense.chunks_exact(v.len()).zip(out) {
            *value = dot(row, v);
        }
    }
    fn transpose_mul(&mut self, _x: &[f64], dense: &[f64], w: &[f64], out: &mut [f64]) {
        jt_mul_vec(dense, w.len(), out.len(), w, out);
    }
}

/// Matrix-free products; passed in place of a dense callback to solve_with_fn.
/// Requires LinearSolver::Cg. No basis-vector probing or dense storage is used.
pub struct JacobianProducts<F, T> {
    pub forward: F,
    pub transpose: T,
}

impl<F, T> JacobianProducts<F, T> {
    pub fn new(forward: F, transpose: T) -> Self {
        Self { forward, transpose }
    }
}

impl<F, T> Jacobian for JacobianProducts<F, T>
where
    F: FnMut(&[f64], &[f64], &mut [f64]),
    T: FnMut(&[f64], &[f64], &mut [f64]),
{
    fn is_matrix_free(&self) -> bool {
        true
    }
    fn prepare(&mut self, _x: &[f64], dense: &mut [f64]) {
        debug_assert!(dense.is_empty());
    }
    fn mul(&mut self, x: &[f64], _dense: &[f64], v: &[f64], out: &mut [f64]) {
        (self.forward)(x, v, out);
    }
    fn transpose_mul(&mut self, x: &[f64], _dense: &[f64], w: &[f64], out: &mut [f64]) {
        (self.transpose)(x, w, out);
    }
}
