use liteopt_core::solvers::gd::GradientDescent;
use pyo3::prelude::*;

use crate::bindings::callbacks::{PyErrState, PyObjectiveCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;

/// Gradient Descent optimizer exposed to Python.
///
/// f:    callable(x: list[float]) -> float
/// grad: callable(x: list[float]) -> list[float]
/// manifold: optional object with callbacks such as
///   retract(x, direction, alpha) and tangent_norm(v)
#[pyfunction(
    signature = (
        f,
        grad,
        x0,
        step_size = None,
        max_iters = None,
        tol_grad = None,
        verbose = None,
        manifold = None,
        line_search = None
    )
)]
fn gd(
    py: Python<'_>,
    f: Py<PyAny>,
    grad: Py<PyAny>,
    x0: Vec<f64>,
    step_size: Option<f64>,
    max_iters: Option<usize>,
    tol_grad: Option<f64>,
    verbose: Option<bool>,
    manifold: Option<Py<PyAny>>,
    line_search: Option<Py<PyAny>>,
) -> PyResult<(Vec<f64>, f64, bool)> {
    let (space, manifold_err) = PyVecManifold::from_python(py, manifold)?;
    let solver = GradientDescent {
        space,
        step_size: step_size.unwrap_or(1e-3),
        max_iters: max_iters.unwrap_or(100),
        tol_grad: tol_grad.unwrap_or(1e-6),
        verbose: verbose.unwrap_or(false),
    };

    let err_state = PyErrState::default();
    let callbacks = PyObjectiveCallbacks::new(f, grad, err_state.clone());

    let result = if let Some(line_search_obj) = line_search {
        let mut policy = PyLineSearchPolicy::new(line_search_obj, err_state.clone());
        solver.minimize_with_fn_and_line_search(
            x0,
            |x| callbacks.value(py, x),
            |x, grad_out| callbacks.gradient_into(py, x, grad_out),
            &mut policy,
        )
    } else {
        solver.minimize_with_fn(
            x0,
            |x| callbacks.value(py, x),
            |x, grad_out| callbacks.gradient_into(py, x, grad_out),
        )
    };

    if let Some(e) = err_state.take() {
        return Err(e);
    }
    if let Some(e) = manifold_err.take() {
        return Err(e);
    }

    Ok((result.x, result.f, result.converged))
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gd, module)?)?;
    Ok(())
}
