use liteopt_core::solvers::gd::{
    ArmijoBacktracking, CostDecrease, GradientDescent, LineSearchPolicy, NoLineSearch,
};
use liteopt_core::solvers::gn::StrictDecreaseBacktracking;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyObjectiveCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::options::PyOptions;
use crate::bindings::trace::trace_records_to_pylist;

use crate::bindings::validation::{finite_gt, finite_nonnegative, finite_open_unit, nonzero_usize};

const GD_OPTIONS: &[&str] = &[
    "step_size",
    "max_iters",
    "tol_grad",
    "manifold",
    "line_search",
    "line_search_method",
    "ls_beta",
    "ls_min_step",
    "ls_max_steps",
    "c_armijo",
];
const DEBUG_OPTIONS: &[&str] = &["history", "verbose", "info"];

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
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("gd: x0 must contain finite values"));
    }
    let method = options
        .string("line_search_method")?
        .unwrap_or_else(|| "none".into());
    let beta = finite_open_unit(
        "gd: options.ls_beta",
        options.f64("ls_beta")?.unwrap_or(0.5),
    )?;
    let min_step = finite_gt(
        "gd: options.ls_min_step",
        options.f64("ls_min_step")?.unwrap_or(1e-8),
        0.0,
    )?;
    let max_steps = nonzero_usize(
        "gd: options.ls_max_steps",
        options.usize("ls_max_steps")?.unwrap_or(20),
    )?;
    let c = finite_open_unit(
        "gd: options.c_armijo",
        options.f64("c_armijo")?.unwrap_or(1e-4),
    )?;
    let mut built_in: Box<dyn LineSearchPolicy> = match method.as_str() {
        "none" => Box::new(NoLineSearch),
        "cost_decrease" => Box::new(CostDecrease),
        "armijo" => Box::new(ArmijoBacktracking::new(beta, max_steps, c).with_min_step(min_step)),
        "strict_decrease" => Box::new(StrictDecreaseBacktracking::new(beta, min_step, max_steps)),
        _ => return Err(PyValueError::new_err("gd: line_search_method must be 'none', 'cost_decrease', 'armijo', or 'strict_decrease'")),
    };
    let want_info = debug.bool("info")?.unwrap_or(false);
    let want_history = debug.bool("history")?.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, options.py("manifold")?)?;
    let solver = GradientDescent {
        space,
        step_size: finite_gt(
            "gd: options.step_size",
            options.f64("step_size")?.unwrap_or(1e-3),
            0.0,
        )?,
        max_iters: options.usize("max_iters")?.unwrap_or(100),
        tol_grad: finite_nonnegative(
            "gd: options.tol_grad",
            options.f64("tol_grad")?.unwrap_or(1e-6),
        )?,
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
        solver.minimize_with_fn_and_line_search(
            x0,
            |x| callbacks.value(py, x),
            |x, grad_out| callbacks.gradient_into(py, x, grad_out),
            built_in.as_mut(),
        )
    };

    if let Some(e) = err_state.take() {
        return Err(e);
    }
    if let Some(e) = manifold_err.take() {
        return Err(e);
    }

    let mut out = vec![
        result.x.into_py_any(py)?,
        result.f.into_py_any(py)?,
        result.converged.into_py_any(py)?,
    ];
    if let Some(trace) = result.trace.take() {
        out.push(trace_records_to_pylist(py, trace)?);
    }
    if want_info {
        let info = PyDict::new(py);
        info.set_item("iters", result.iters)?;
        info.set_item("grad_norm", result.grad_norm)?;
        info.set_item("status", result.status.as_str())?;
        info.set_item("nfev", result.nfev)?;
        info.set_item("njev", result.njev)?;
        info.set_item("n_attempts", result.n_attempts)?;
        info.set_item("n_accepted", result.iters)?;
        info.set_item("n_retries", 0)?;
        info.set_item("n_ls_trials", result.n_ls_trials)?;
        out.push(info.into_any().unbind());
    }
    Ok(PyTuple::new(py, out)?.into_any().unbind())
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gd, module)?)?;
    Ok(())
}
