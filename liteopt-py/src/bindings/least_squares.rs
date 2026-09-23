use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bindings::{gn, lm};

/// Minimize 0.5 * ||residual(x)||^2 with Gauss-Newton or Levenberg-Marquardt.
///
/// method: "lm" (default) or "gn".
/// Provide jacobian(x), or both jacobian_vec(x, v) and jacobian_transpose_vec(x, w).
/// Product callbacks use matrix-free CG. Options are validated by the
/// selected solver; options specific to the other method are rejected.
/// Returns (x, cost, iters, r_norm, dx_norm, ok), with history appended when
/// debug={"history": True}.
#[pyfunction(signature = (
    residual,
    x0,
    *,
    method = "lm",
    jacobian = None,
    jacobian_vec = None,
        jacobian_transpose_vec = None,
    project = None,
    options = None,
    debug = None
))]
fn least_squares(
    py: Python<'_>,
    residual: Py<PyAny>,
    x0: Vec<f64>,
    method: &str,
    jacobian: Option<Py<PyAny>>,
    jacobian_vec: Option<Py<PyAny>>,
    jacobian_transpose_vec: Option<Py<PyAny>>,
    project: Option<Py<PyAny>>,
    options: Option<Py<PyAny>>,
    debug: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    match method {
        "gn" => gn::gn(
            py,
            residual,
            Some(x0),
            jacobian,
            jacobian_vec,
            jacobian_transpose_vec,
            project,
            options,
            debug,
        ),
        "lm" => lm::lm(
            py,
            residual,
            Some(x0),
            jacobian,
            jacobian_vec,
            jacobian_transpose_vec,
            project,
            options,
            debug,
        ),
        other => Err(PyValueError::new_err(format!(
            "least_squares: method must be 'gn' or 'lm', got '{other}'"
        ))),
    }
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(least_squares, module)?)?;
    Ok(())
}
