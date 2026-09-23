use super::options::PyOptions;
use super::validation::{finite_nonnegative, finite_open_unit, nonzero_usize};
use liteopt_core::solvers::{CgOptions, LinearSolver};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn parse(
    py: Python<'_>,
    options: &PyOptions<'_>,
    jacobian: &Option<Py<PyAny>>,
    forward: &Option<Py<PyAny>>,
    transpose: &Option<Py<PyAny>>,
) -> PyResult<(LinearSolver, CgOptions, bool)> {
    for (name, callback) in [
        ("jacobian", jacobian),
        ("jacobian_vec", forward),
        ("jacobian_transpose_vec", transpose),
    ] {
        if callback.as_ref().is_some_and(|f| !f.bind(py).is_callable()) {
            return Err(PyValueError::new_err(format!("{name} must be callable")));
        }
    }
    if forward.is_some() != transpose.is_some() {
        return Err(PyValueError::new_err(
            "jacobian_vec and jacobian_transpose_vec must be provided together",
        ));
    }
    if jacobian.is_none() && forward.is_none() {
        return Err(PyValueError::new_err("jacobian or jacobian_vec must be provided; jacobian_vec requires jacobian_transpose_vec"));
    }
    let backend = match options.string("linear_solver")?.as_deref() {
        None => {
            if forward.is_some() {
                LinearSolver::Cg
            } else {
                LinearSolver::Direct
            }
        }
        Some("direct") => LinearSolver::Direct,
        Some("cg") => LinearSolver::Cg,
        _ => {
            return Err(PyValueError::new_err(
                "linear_solver must be 'direct' or 'cg'",
            ))
        }
    };
    if backend == LinearSolver::Direct && jacobian.is_none() {
        return Err(PyValueError::new_err("linear_solver='direct' requires jacobian; products are never converted to a dense matrix"));
    }
    if backend == LinearSolver::Cg && options.string("linear_system")?.is_some() {
        return Err(PyValueError::new_err(
            "linear_system is only available with linear_solver='direct'",
        ));
    }
    let cg = CgOptions {
        max_iters: nonzero_usize(
            "cg_max_iters",
            options.usize("cg_max_iters")?.unwrap_or(100),
        )?,
        rtol: finite_open_unit("cg_rtol", options.f64("cg_rtol")?.unwrap_or(1e-6))?,
        atol: finite_nonnegative("cg_atol", options.f64("cg_atol")?.unwrap_or(0.))?,
    };
    Ok((
        backend,
        cg,
        backend == LinearSolver::Cg && forward.is_some(),
    ))
}
