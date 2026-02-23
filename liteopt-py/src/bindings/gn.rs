use liteopt_core::solvers::gn::{ArmijoBacktracking, GaussNewton, NoLineSearch};
use pyo3::prelude::*;

use crate::bindings::callbacks::{PyErrState, PyLeastSquaresCallbacks};
use crate::bindings::line_search::PyLineSearchPolicy;
use crate::bindings::manifold::PyVecManifold;

/// Nonlinear least squares solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// jacobian: callable(x: list[float]) -> list[float]           (len = m*n, row-major)
/// line_search: bool or callable(ctx: dict) -> alpha or (accepted, alpha)
/// manifold: optional object with callbacks such as
///   retract(x, direction, alpha) and tangent_norm(v)
#[pyfunction(
    signature = (
        residual,
        jacobian,
        x0,
        project = None,
        lambda_ = None,
        step_scale = None,
        max_iters = None,
        tol_r = None,
        tol_dx = None,
        line_search = None,
        ls_beta = None,
        ls_max_steps = None,
        c_armijo = None,
        verbose = None,
        manifold = None
    )
)]
fn gn(
    py: Python<'_>,
    residual: Py<PyAny>,
    jacobian: Py<PyAny>,
    x0: Vec<f64>,
    project: Option<Py<PyAny>>,
    lambda_: Option<f64>,
    step_scale: Option<f64>,
    max_iters: Option<usize>,
    tol_r: Option<f64>,
    tol_dx: Option<f64>,
    line_search: Option<Py<PyAny>>,
    ls_beta: Option<f64>,
    ls_max_steps: Option<usize>,
    c_armijo: Option<f64>,
    verbose: Option<bool>,
    manifold: Option<Py<PyAny>>,
) -> PyResult<(Vec<f64>, f64, usize, f64, f64, bool)> {
    let (space, manifold_err) = PyVecManifold::from_python(py, manifold)?;
    let solver = GaussNewton {
        space,
        lambda: lambda_.unwrap_or(1e-3),
        step_scale: step_scale.unwrap_or(1.0),
        max_iters: max_iters.unwrap_or(100),
        tol_r: tol_r.unwrap_or(1e-6),
        tol_dq: tol_dx.unwrap_or(1e-6),
        verbose: verbose.unwrap_or(false),
    };

    let err_state = PyErrState::default();
    let callbacks = PyLeastSquaresCallbacks::new(residual, jacobian, project, err_state.clone());
    let m = callbacks.infer_residual_dim(py, &x0)?;

    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| callbacks.residual_into(py, x, r_out);
    let mut jacobian_fn = |x: &[f64], j_out: &mut [f64]| callbacks.jacobian_into(py, x, j_out);
    let mut project_fn = |x: &mut [f64]| callbacks.project_in_place(py, x);

    enum LineSearchMode {
        Armijo,
        Disabled,
        Custom(Py<PyAny>),
    }

    let line_search_mode = match line_search {
        None => LineSearchMode::Armijo,
        Some(obj) => match obj.bind(py).extract::<bool>() {
            Ok(true) => LineSearchMode::Armijo,
            Ok(false) => LineSearchMode::Disabled,
            Err(_) => LineSearchMode::Custom(obj),
        },
    };

    let result = match line_search_mode {
        LineSearchMode::Armijo => {
            let mut ls = ArmijoBacktracking::new(
                ls_beta.unwrap_or(0.5),
                ls_max_steps.unwrap_or(20),
                c_armijo.unwrap_or(1e-4),
            );
            solver.solve_with_fn(
                m,
                x0,
                &mut residual_fn,
                &mut jacobian_fn,
                &mut project_fn,
                &mut ls,
            )
        }
        LineSearchMode::Disabled => {
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
        LineSearchMode::Custom(obj) => {
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

    // If any python error occurred inside callbacks, raise it
    if let Some(e) = err_state.take() {
        return Err(e);
    }
    if let Some(e) = manifold_err.take() {
        return Err(e);
    }

    Ok((
        result.x,
        result.cost,
        result.iters,
        result.r_norm,
        result.dx_norm,
        result.converged,
    ))
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gn, module)?)?;
    Ok(())
}
