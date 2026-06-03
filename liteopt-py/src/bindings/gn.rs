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
use crate::bindings::trace::trace_records_to_pylist;
use crate::bindings::validation::{finite, finite_nonnegative, finite_open_unit, nonzero_usize};

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
/// jacobian: optional callable(x: list[float]) -> list[float]  (len = m*n, row-major)
/// jacobian_vec: optional callable(x, v) -> J(x) @ v           (len = m)
/// line_search: bool or callable(ctx: dict) -> alpha or (accepted, alpha)
/// damping_update: "adaptive" or "fixed"
/// linear_system: "left_jjt" or "normal_jtj"
/// line_search_method: "armijo", "strict_decrease", or "none"
/// history: if true, return an additional list[dict] with per-iteration trace rows
#[pyfunction(
    signature = (
        residual,
        jacobian = None,
        x0 = None,
        project = None,
        lambda_ = None,
        step_scale = None,
        max_iters = None,
        tol_r = None,
        tol_dx = None,
        damping_update = None,
        linear_system = None,
        line_search_method = None,
        line_search = None,
        ls_beta = None,
        ls_min_step = None,
        ls_max_steps = None,
        c_armijo = None,
        verbose = None,
        manifold = None,
        history = None,
        jacobian_vec = None
    )
)]
fn gn(
    py: Python<'_>,
    residual: Py<PyAny>,
    jacobian: Option<Py<PyAny>>,
    x0: Option<Vec<f64>>,
    project: Option<Py<PyAny>>,
    lambda_: Option<f64>,
    step_scale: Option<f64>,
    max_iters: Option<usize>,
    tol_r: Option<f64>,
    tol_dx: Option<f64>,
    damping_update: Option<String>,
    linear_system: Option<String>,
    line_search_method: Option<String>,
    line_search: Option<Py<PyAny>>,
    ls_beta: Option<f64>,
    ls_min_step: Option<f64>,
    ls_max_steps: Option<usize>,
    c_armijo: Option<f64>,
    verbose: Option<bool>,
    manifold: Option<Py<PyAny>>,
    history: Option<bool>,
    jacobian_vec: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let Some(x0) = x0 else {
        return Err(PyValueError::new_err("gn: x0 must be provided"));
    };

    let lambda = finite_nonnegative("gn: lambda_", lambda_.unwrap_or(1e-3))?;
    let ls_beta = finite_open_unit("gn: ls_beta", ls_beta.unwrap_or(0.5))?;
    let ls_min_step = crate::bindings::validation::finite_gt(
        "gn: ls_min_step",
        ls_min_step.unwrap_or(1e-8),
        0.0,
    )?;
    let ls_max_steps = nonzero_usize("gn: ls_max_steps", ls_max_steps.unwrap_or(20))?;
    let c_armijo = finite("gn: c_armijo", c_armijo.unwrap_or(1e-4))?;
    let damping_update = parse_damping_update(damping_update)?;
    let linear_system = parse_linear_system(linear_system)?;
    let line_search_method = parse_line_search_method(line_search_method)?;

    enum RunMode {
        Configured,
        Disabled,
        Custom(Py<PyAny>),
    }

    let run_mode = match line_search {
        None => RunMode::Configured,
        Some(obj) => match obj.bind(py).extract::<bool>() {
            Ok(true) => RunMode::Configured,
            Ok(false) => RunMode::Disabled,
            Err(_) => RunMode::Custom(obj),
        },
    };

    let want_history = history.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, manifold)?;
    let solver = GaussNewton {
        space,
        lambda,
        damping_update,
        linear_system,
        line_search_method,
        step_scale: step_scale.unwrap_or(1.0),
        ls_beta,
        ls_min_step,
        ls_max_steps,
        c_armijo,
        max_iters: max_iters.unwrap_or(100),
        tol_r: tol_r.unwrap_or(1e-6),
        tol_dq: tol_dx.unwrap_or(1e-6),
        verbose: verbose.unwrap_or(false),
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
