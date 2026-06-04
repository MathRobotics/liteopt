use liteopt_core::solvers::gn::{
    GaussNewton, GaussNewtonDampingUpdate, GaussNewtonLineSearchMethod, GaussNewtonLinearSystem,
    NoLineSearch,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyLeastSquaresCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::options::PyOptions;
use crate::bindings::trace::trace_records_to_pylist;
use crate::bindings::validation::{finite, finite_nonnegative, finite_open_unit, nonzero_usize};

const GN_OPTIONS: &[&str] = &[
    "lambda",
    "lambda_",
    "step_scale",
    "max_iters",
    "tol_r",
    "tol_dx",
    "damping_update",
    "linear_system",
    "line_search_method",
    "line_search",
    "ls_beta",
    "ls_min_step",
    "ls_max_steps",
    "c_armijo",
    "manifold",
];
const DEBUG_OPTIONS: &[&str] = &["history", "verbose"];

fn parse_damping_update(value: Option<String>) -> PyResult<GaussNewtonDampingUpdate> {
    match value.as_deref().unwrap_or("adaptive") {
        "adaptive" => Ok(GaussNewtonDampingUpdate::Adaptive),
        "fixed" => Ok(GaussNewtonDampingUpdate::Fixed),
        other => Err(PyValueError::new_err(format!(
            "gn: damping_update must be 'adaptive' or 'fixed', got '{other}'"
        ))),
    }
}

fn parse_linear_system(value: Option<String>) -> PyResult<GaussNewtonLinearSystem> {
    match value.as_deref().unwrap_or("left_jjt") {
        "left_jjt" => Ok(GaussNewtonLinearSystem::LeftJjT),
        "normal_jtj" => Ok(GaussNewtonLinearSystem::NormalJtJ),
        other => Err(PyValueError::new_err(format!(
            "gn: linear_system must be 'left_jjt' or 'normal_jtj', got '{other}'"
        ))),
    }
}

fn parse_line_search_method(value: Option<String>) -> PyResult<GaussNewtonLineSearchMethod> {
    match value.as_deref().unwrap_or("armijo") {
        "armijo" => Ok(GaussNewtonLineSearchMethod::Armijo),
        "strict_decrease" => Ok(GaussNewtonLineSearchMethod::StrictDecrease),
        "none" => Ok(GaussNewtonLineSearchMethod::None),
        other => Err(PyValueError::new_err(format!(
            "gn: line_search_method must be 'armijo', 'strict_decrease', or 'none', got '{other}'"
        ))),
    }
}

/// Nonlinear least squares solver exposed to Python.
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
fn gn(
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
        return Err(PyValueError::new_err("gn: x0 must be provided"));
    };

    let options = PyOptions::from_python(py, "gn", "options", options, GN_OPTIONS)?;
    let debug = PyOptions::from_python(py, "gn", "debug", debug, DEBUG_OPTIONS)?;
    let lambda = match (options.f64("lambda")?, options.f64("lambda_")?) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "gn: options must not contain both 'lambda' and 'lambda_'",
            ));
        }
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => 1e-3,
    };
    let lambda = finite_nonnegative("gn: options.lambda", lambda)?;
    let ls_beta = finite_open_unit(
        "gn: options.ls_beta",
        options.f64("ls_beta")?.unwrap_or(0.5),
    )?;
    let ls_min_step = crate::bindings::validation::finite_gt(
        "gn: options.ls_min_step",
        options.f64("ls_min_step")?.unwrap_or(1e-8),
        0.0,
    )?;
    let ls_max_steps = nonzero_usize(
        "gn: options.ls_max_steps",
        options.usize("ls_max_steps")?.unwrap_or(20),
    )?;
    let c_armijo = finite(
        "gn: options.c_armijo",
        options.f64("c_armijo")?.unwrap_or(1e-4),
    )?;
    let damping_update = parse_damping_update(options.string("damping_update")?)?;
    let linear_system = parse_linear_system(options.string("linear_system")?)?;
    let line_search_method = parse_line_search_method(options.string("line_search_method")?)?;

    enum RunMode {
        Configured,
        Disabled,
        Custom(Py<PyAny>),
    }

    let run_mode = match options.py("line_search")? {
        None => RunMode::Configured,
        Some(obj) => match obj.bind(py).extract::<bool>() {
            Ok(true) => RunMode::Configured,
            Ok(false) => RunMode::Disabled,
            Err(_) => RunMode::Custom(obj),
        },
    };

    let want_history = debug.bool("history")?.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, options.py("manifold")?)?;
    let solver = GaussNewton {
        space,
        lambda,
        damping_update,
        linear_system,
        line_search_method,
        step_scale: options.f64("step_scale")?.unwrap_or(1.0),
        ls_beta,
        ls_min_step,
        ls_max_steps,
        c_armijo,
        max_iters: options.usize("max_iters")?.unwrap_or(100),
        tol_r: options.f64("tol_r")?.unwrap_or(1e-6),
        tol_dq: options.f64("tol_dx")?.unwrap_or(1e-6),
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

    let mut result = match run_mode {
        RunMode::Configured => solver.solve_with_fn_default_line_search(
            m,
            x0,
            &mut residual_fn,
            &mut jacobian_fn,
            &mut project_fn,
        ),
        RunMode::Disabled => {
            let mut ls = NoLineSearch;
            solver.solve_with_fn(
                m,
                x0,
                &mut residual_fn,
                &mut jacobian_fn,
                &mut project_fn,
                &mut ls,
            )
        }
        RunMode::Custom(obj) => {
            let mut ls = PyLineSearchPolicy::new(obj, err_state.clone());
            solver.solve_with_fn(
                m,
                x0,
                &mut residual_fn,
                &mut jacobian_fn,
                &mut project_fn,
                &mut ls,
            )
        }
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
    module.add_function(wrap_pyfunction!(gn, module)?)?;
    Ok(())
}
