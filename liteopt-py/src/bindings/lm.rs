use liteopt_core::solvers::lm::{CostDecrease, LevenbergMarquardt};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyLeastSquaresCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::trace::trace_records_to_pylist;

/// Nonlinear least squares LM solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// jacobian: callable(x: list[float]) -> list[float]           (len = m*n, row-major)
/// line_search: optional callable(ctx: dict) -> alpha or (accepted, alpha)
/// history: if true, return an additional list[dict] with per-iteration trace rows
#[pyfunction(
    signature = (
        residual,
        jacobian,
        x0,
        project = None,
        lambda_ = None,
        lambda_up = None,
        lambda_down = None,
        step_scale = None,
        max_iters = None,
        tol_r = None,
        tol_dx = None,
        verbose = None,
        manifold = None,
        line_search = None,
        history = None
    )
)]
fn lm(
    py: Python<'_>,
    residual: Py<PyAny>,
    jacobian: Py<PyAny>,
    x0: Vec<f64>,
    project: Option<Py<PyAny>>,
    lambda_: Option<f64>,
    lambda_up: Option<f64>,
    lambda_down: Option<f64>,
    step_scale: Option<f64>,
    max_iters: Option<usize>,
    tol_r: Option<f64>,
    tol_dx: Option<f64>,
    verbose: Option<bool>,
    manifold: Option<Py<PyAny>>,
    line_search: Option<Py<PyAny>>,
    history: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let want_history = history.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, manifold)?;
    let solver = LevenbergMarquardt {
        space,
        lambda: lambda_.unwrap_or(1e-3),
        lambda_up: lambda_up.unwrap_or(10.0),
        lambda_down: lambda_down.unwrap_or(0.5),
        step_scale: step_scale.unwrap_or(1.0),
        max_iters: max_iters.unwrap_or(100),
        tol_r: tol_r.unwrap_or(1e-6),
        tol_dq: tol_dx.unwrap_or(1e-6),
        verbose: verbose.unwrap_or(false),
        collect_trace: want_history,
    };

    let err_state = PyErrState::default();
    let callbacks = PyLeastSquaresCallbacks::new(residual, jacobian, project, err_state.clone());
    let m = callbacks.infer_residual_dim(py, &x0)?;

    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| callbacks.residual_into(py, x, r_out);
    let mut jacobian_fn = |x: &[f64], j_out: &mut [f64]| callbacks.jacobian_into(py, x, j_out);
    let mut project_fn = |x: &mut [f64]| callbacks.project_in_place(py, x);

    let mut result = if let Some(line_search_obj) = line_search {
        let mut policy = PyLineSearchPolicy::new(line_search_obj, err_state.clone());
        solver.solve_with_fn(
            m,
            x0,
            &mut residual_fn,
            &mut jacobian_fn,
            &mut project_fn,
            &mut policy,
        )
    } else {
        let mut policy = CostDecrease;
        solver.solve_with_fn(
            m,
            x0,
            &mut residual_fn,
            &mut jacobian_fn,
            &mut project_fn,
            &mut policy,
        )
    };

    if let Some(e) = err_state.take() {
        return Err(e);
    }
    if let Some(e) = manifold_err.take() {
        return Err(e);
    }

    if let Some(trace) = result.trace.take() {
        let history_obj = trace_records_to_pylist(py, trace)?;
        let out = PyTuple::new(
            py,
            [
                result.x.into_py_any(py)?,
                result.cost.into_py_any(py)?,
                result.iters.into_py_any(py)?,
                result.r_norm.into_py_any(py)?,
                result.dx_norm.into_py_any(py)?,
                result.converged.into_py_any(py)?,
                history_obj,
            ],
        )?;
        Ok(out.into_any().unbind())
    } else {
        let out = PyTuple::new(
            py,
            [
                result.x.into_py_any(py)?,
                result.cost.into_py_any(py)?,
                result.iters.into_py_any(py)?,
                result.r_norm.into_py_any(py)?,
                result.dx_norm.into_py_any(py)?,
                result.converged.into_py_any(py)?,
            ],
        )?;
        Ok(out.into_any().unbind())
    }
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lm, module)?)?;
    Ok(())
}
