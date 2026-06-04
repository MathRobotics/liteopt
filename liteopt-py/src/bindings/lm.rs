use liteopt_core::solvers::lm::{CostDecrease, LevenbergMarquardt};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyLeastSquaresCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::options::PyOptions;
use crate::bindings::trace::trace_records_to_pylist;
use crate::bindings::validation::{
    finite_gt, finite_nonnegative, finite_open_closed_unit, finite_open_unit,
};

const LM_OPTIONS: &[&str] = &[
    "lambda",
    "lambda_",
    "lambda_up",
    "lambda_down",
    "step_scale",
    "max_iters",
    "tol_r",
    "tol_dx",
    "manifold",
    "line_search",
];
const DEBUG_OPTIONS: &[&str] = &["history", "verbose"];

/// Nonlinear least squares LM solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// x0: initial point
/// jacobian: optional callable(x: list[float]) -> list[float]  (len = m*n, row-major)
/// jacobian_vec: optional callable(x, v) -> J(x) @ v           (len = m)
/// project: optional callable(x) -> projected x
/// options: optional dict for solver settings
/// debug: optional dict for trace and logging controls
#[pyfunction(
    signature = (
        residual,
        x0 = None,
        *,
        jacobian = None,
        jacobian_vec = None,
        project = None,
        options = None,
        debug = None
    )
)]
fn lm(
    py: Python<'_>,
    residual: Py<PyAny>,
    x0: Option<Vec<f64>>,
    jacobian: Option<Py<PyAny>>,
    jacobian_vec: Option<Py<PyAny>>,
    project: Option<Py<PyAny>>,
    options: Option<Py<PyAny>>,
    debug: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let Some(x0) = x0 else {
        return Err(PyValueError::new_err("lm: x0 must be provided"));
    };

    let options = PyOptions::from_python(py, "lm", "options", options, LM_OPTIONS)?;
    let debug = PyOptions::from_python(py, "lm", "debug", debug, DEBUG_OPTIONS)?;
    let lambda = match (options.f64("lambda")?, options.f64("lambda_")?) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "lm: options must not contain both 'lambda' and 'lambda_'",
            ));
        }
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => 1e-3,
    };
    let lambda = finite_nonnegative("lm: options.lambda", lambda)?;
    let lambda_up = finite_gt(
        "lm: options.lambda_up",
        options.f64("lambda_up")?.unwrap_or(10.0),
        1.0,
    )?;
    let lambda_down = finite_open_unit(
        "lm: options.lambda_down",
        options.f64("lambda_down")?.unwrap_or(0.5),
    )?;
    let step_scale = finite_open_closed_unit(
        "lm: options.step_scale",
        options.f64("step_scale")?.unwrap_or(1.0),
    )?;
    let tol_r = finite_nonnegative("lm: options.tol_r", options.f64("tol_r")?.unwrap_or(1e-6))?;
    let tol_dx = finite_nonnegative("lm: options.tol_dx", options.f64("tol_dx")?.unwrap_or(1e-6))?;

    let want_history = debug.bool("history")?.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, options.py("manifold")?)?;
    let solver = LevenbergMarquardt {
        space,
        lambda,
        lambda_up,
        lambda_down,
        step_scale,
        max_iters: options.usize("max_iters")?.unwrap_or(100),
        tol_r,
        tol_dq: tol_dx,
        verbose: debug.bool("verbose")?.unwrap_or(false),
        collect_trace: want_history,
    };

    let err_state = PyErrState::default();
    let callbacks =
        PyLeastSquaresCallbacks::new(residual, jacobian, jacobian_vec, project, err_state.clone());
    let m = callbacks.infer_residual_dim(py, &x0)?;

    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| callbacks.residual_into(py, x, r_out);
    let mut jacobian_fn = |x: &[f64], j_out: &mut [f64]| callbacks.jacobian_into(py, x, j_out);
    let mut project_fn = |x: &mut [f64]| callbacks.project_in_place(py, x);

    let mut result = if let Some(line_search_obj) = options.py("line_search")? {
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
