use liteopt_core::solvers::gd::GradientDescent;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyObjectiveCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::options::PyOptions;
use crate::bindings::trace::trace_records_to_pylist;

const GD_OPTIONS: &[&str] = &[
    "step_size",
    "max_iters",
    "tol_grad",
    "manifold",
    "line_search",
];
const DEBUG_OPTIONS: &[&str] = &["history", "verbose"];

/// Gradient Descent optimizer exposed to Python.
///
/// f:    callable(x: list[float]) -> float
/// grad: callable(x: list[float]) -> list[float]
/// x0: initial point
/// options: optional dict for solver settings
/// debug: optional dict for trace and logging controls
#[pyfunction(
    signature = (
        f,
        grad,
        x0,
        *,
        options = None,
        debug = None
    )
)]
fn gd(
    py: Python<'_>,
    f: Py<PyAny>,
    grad: Py<PyAny>,
    x0: Vec<f64>,
    options: Option<Py<PyAny>>,
    debug: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let options = PyOptions::from_python(py, "gd", "options", options, GD_OPTIONS)?;
    let debug = PyOptions::from_python(py, "gd", "debug", debug, DEBUG_OPTIONS)?;
    let want_history = debug.bool("history")?.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, options.py("manifold")?)?;
    let solver = GradientDescent {
        space,
        step_size: options.f64("step_size")?.unwrap_or(1e-3),
        max_iters: options.usize("max_iters")?.unwrap_or(100),
        tol_grad: options.f64("tol_grad")?.unwrap_or(1e-6),
        verbose: debug.bool("verbose")?.unwrap_or(false),
        collect_trace: want_history,
    };

    let err_state = PyErrState::default();
    let callbacks = PyObjectiveCallbacks::new(f, grad, err_state.clone());

    let mut result = if let Some(line_search_obj) = options.py("line_search")? {
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

    if let Some(trace) = result.trace.take() {
        let history_obj = trace_records_to_pylist(py, trace)?;
        let out = PyTuple::new(
            py,
            [
                result.x.into_py_any(py)?,
                result.f.into_py_any(py)?,
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
                result.f.into_py_any(py)?,
                result.converged.into_py_any(py)?,
            ],
        )?;
        Ok(out.into_any().unbind())
    }
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gd, module)?)?;
    Ok(())
}
