use liteopt_core::solvers::lm::{
    LevenbergMarquardt, LevenbergMarquardtLineSearchMethod,
    LevenbergMarquardtLinearSystem,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::bindings::callbacks::{PyErrState, PyJacobian, PyLeastSquaresCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;
use crate::bindings::options::PyOptions;
use crate::bindings::trace::trace_records_to_pylist;
use crate::bindings::validation::{
    finite_gt, finite_nonnegative, finite_open_closed_unit, finite_open_unit, nonzero_usize,
};

const LM_OPTIONS: &[&str] = &[
    "linear_system",
    "linear_solver",
    "cg_max_iters",
    "cg_rtol",
    "cg_atol",
    "lambda_min",
    "lambda_max",
    "lambda",
    "lambda_",
    "lambda_up",
    "lambda_down",
    "step_size",
    "max_iters",
    "tol_r",
    "tol_grad",
    "tol_dx",
    "manifold",
    "line_search",
    "line_search_method",
    "ls_beta",
    "ls_min_step",
    "ls_max_steps",
    "c_armijo",
];
const DEBUG_OPTIONS: &[&str] = &["history", "verbose", "info"];

fn parse_line_search_method(value: Option<String>) -> PyResult<LevenbergMarquardtLineSearchMethod> {
    match value.as_deref().unwrap_or("cost_decrease") {
        "cost_decrease" => Ok(LevenbergMarquardtLineSearchMethod::CostDecrease),
        "armijo" => Ok(LevenbergMarquardtLineSearchMethod::Armijo),
        "strict_decrease" => Ok(LevenbergMarquardtLineSearchMethod::StrictDecrease),
        "none" => Ok(LevenbergMarquardtLineSearchMethod::None),
        other => Err(PyValueError::new_err(format!(
            "lm: line_search_method must be 'cost_decrease', 'armijo', 'strict_decrease', or 'none', got '{other}'"
        ))),
    }
}

/// Nonlinear least squares LM solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// x0: initial point
/// jacobian: optional callable(x: list[float]) -> list[float]  (len = m*n, row-major)
/// jacobian_vec: optional callable(x, v) -> J(x) @ v           (len = m)
/// jacobian_transpose_vec: callable(x, w) -> J(x).T @ w (len = n); required with jacobian_vec
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
        jacobian_transpose_vec = None,
        project = None,
        options = None,
        debug = None
    )
)]
pub(super) fn lm(
    py: Python<'_>,
    residual: Py<PyAny>,
    x0: Option<Vec<f64>>,
    jacobian: Option<Py<PyAny>>,
    jacobian_vec: Option<Py<PyAny>>,
    jacobian_transpose_vec: Option<Py<PyAny>>,
    project: Option<Py<PyAny>>,
    options: Option<Py<PyAny>>,
    debug: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let Some(x0) = x0 else {
        return Err(PyValueError::new_err("lm: x0 must be provided"));
    };

    if x0.is_empty() || x0.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("lm: x0 must be nonempty and finite"));
    }
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
    let lambda_min = finite_gt(
        "lm: options.lambda_min",
        options.f64("lambda_min")?.unwrap_or(1e-12),
        0.,
    )?;
    let lambda_max = finite_gt(
        "lm: options.lambda_max",
        options.f64("lambda_max")?.unwrap_or(f64::MAX),
        0.,
    )?;
    if lambda_max < lambda_min || lambda > lambda_max {
        return Err(PyValueError::new_err(
            "lm: require lambda_min <= lambda_max and lambda <= lambda_max",
        ));
    }
    let lambda_up = finite_gt(
        "lm: options.lambda_up",
        options.f64("lambda_up")?.unwrap_or(10.0),
        1.0,
    )?;
    let lambda_down = finite_open_unit(
        "lm: options.lambda_down",
        options.f64("lambda_down")?.unwrap_or(0.5),
    )?;
    let step_size = finite_open_closed_unit(
        "lm: options.step_size",
        options.f64("step_size")?.unwrap_or(1.0),
    )?;
    let tol_r = finite_nonnegative("lm: options.tol_r", options.f64("tol_r")?.unwrap_or(1e-6))?;
    let tol_dx = finite_nonnegative("lm: options.tol_dx", options.f64("tol_dx")?.unwrap_or(1e-6))?;

    let line_search_method = parse_line_search_method(options.string("line_search_method")?)?;
    let ls_beta = finite_open_unit(
        "lm: options.ls_beta",
        options.f64("ls_beta")?.unwrap_or(0.5),
    )?;
    let ls_min_step = finite_gt(
        "lm: options.ls_min_step",
        options.f64("ls_min_step")?.unwrap_or(1e-8),
        0.0,
    )?;
    let ls_max_steps = nonzero_usize(
        "lm: options.ls_max_steps",
        options.usize("ls_max_steps")?.unwrap_or(20),
    )?;
    let c_armijo = finite_open_unit(
        "lm: options.c_armijo",
        options.f64("c_armijo")?.unwrap_or(1e-4),
    )?;

    let (linear_solver, cg, matrix_free) = crate::bindings::linear_solver::parse(
        py,
        &options,
        &jacobian,
        &jacobian_vec,
        &jacobian_transpose_vec,
    )?;
    let want_info = debug.bool("info")?.unwrap_or(false);
    let want_history = debug.bool("history")?.unwrap_or(false);
    let (space, manifold_err) = PyVecManifold::from_python(py, options.py("manifold")?)?;
    let linear_system = match options
        .string("linear_system")?
        .as_deref()
        .unwrap_or("left_jjt")
    {
        "left_jjt" => LevenbergMarquardtLinearSystem::LeftJjT,
        "qr" => LevenbergMarquardtLinearSystem::Qr,
        _ => {
            return Err(PyValueError::new_err(
                "lm: linear_system must be 'left_jjt' or 'qr'",
            ))
        }
    };
    let solver = LevenbergMarquardt {
        linear_system,
        linear_solver,
        cg,
        space,
        lambda,
        lambda_min,
        lambda_max,
        lambda_up,
        lambda_down,
        step_size,
        line_search_method,
        ls_beta,
        ls_min_step,
        ls_max_steps,
        c_armijo,
        max_iters: options.usize("max_iters")?.unwrap_or(100),
        tol_r,
        tol_grad: crate::bindings::validation::finite_nonnegative(
            "lm: options.tol_grad",
            options.f64("tol_grad")?.unwrap_or(1e-6),
        )?,
        tol_dq: tol_dx,
        verbose: debug.bool("verbose")?.unwrap_or(false),
        collect_trace: want_history,
    };

    let err_state = PyErrState::default();
    let callbacks = PyLeastSquaresCallbacks::new(
        residual,
        jacobian,
        jacobian_vec,
        jacobian_transpose_vec,
        project,
        err_state.clone(),
    );
    let m = callbacks.infer_residual_dim(py, &x0)?;

    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| callbacks.residual_into(py, x, r_out);
    let jacobian_fn = PyJacobian {
        callbacks: &callbacks,
        py,
        matrix_free,
    };
    let mut project_fn = |x: &mut [f64]| callbacks.project_in_place(py, x);

    let mut result = if let Some(line_search_obj) = options.py("line_search")? {
        let mut policy = PyLineSearchPolicy::new(line_search_obj, err_state.clone());
        solver.solve_with_derivatives(
            m,
            x0,
            &mut residual_fn,
            jacobian_fn,
            &mut project_fn,
            &mut policy,
        )
    } else {
        solver.solve_with_derivatives_default_line_search(
            m,
            x0,
            &mut residual_fn,
            jacobian_fn,
            &mut project_fn,
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
        result.cost.into_py_any(py)?,
        result.iters.into_py_any(py)?,
        result.r_norm.into_py_any(py)?,
        result.dx_norm.into_py_any(py)?,
        result.converged.into_py_any(py)?,
    ];
    if let Some(trace) = result.trace.take() {
        out.push(trace_records_to_pylist(py, trace)?);
    }
    if want_info {
        let info = PyDict::new(py);
        info.set_item("status", result.status)?;
        info.set_item("grad_norm", result.grad_norm)?;
        info.set_item("iters", result.iters)?;
        info.set_item("nfev", callbacks.nfev.get())?;
        info.set_item("njev", result.njev)?;
        info.set_item("n_jac_calls", callbacks.n_jac_calls.get())?;
        info.set_item("n_jvp", callbacks.n_jvp.get())?;
        info.set_item("n_jtvp", callbacks.n_jtvp.get())?;
        info.set_item(
            "linear_solver",
            if linear_solver == liteopt_core::solvers::LinearSolver::Cg {
                "cg"
            } else {
                "direct"
            },
        )?;
        info.set_item("matrix_free", matrix_free)?;
        info.set_item("n_linear_iters", result.n_linear_iters)?;
        info.set_item("linear_status", result.linear_status)?;
        info.set_item("linear_residual_norm", result.linear_residual_norm)?;
        info.set_item("n_attempts", result.n_attempts)?;
        info.set_item("n_accepted", result.n_accepted)?;
        info.set_item("n_retries", result.n_retries)?;
        info.set_item("n_ls_trials", result.n_ls_trials)?;
        out.push(info.into_any().unbind());
    }
    Ok(PyTuple::new(py, out)?.into_any().unbind())
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lm, module)?)?;
    Ok(())
}
