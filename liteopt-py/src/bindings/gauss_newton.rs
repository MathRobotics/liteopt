use liteopt_core::{manifolds::EuclideanSpace, solvers::gauss_newton::GaussNewton};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::cell::RefCell;

/// Nonlinear least squares solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// jacobian: callable(x: list[float]) -> list[float]           (len = m*n, row-major)
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
        verbose = None
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
    line_search: Option<bool>,
    ls_beta: Option<f64>,
    ls_max_steps: Option<usize>,
    c_armijo: Option<f64>,
    verbose: Option<bool>,
) -> PyResult<(Vec<f64>, f64, usize, f64, f64, bool)> {
    let space = EuclideanSpace;
    let solver = GaussNewton {
        space,
        lambda: lambda_.unwrap_or(1e-3),
        step_scale: step_scale.unwrap_or(1.0),
        max_iters: max_iters.unwrap_or(100),
        tol_r: tol_r.unwrap_or(1e-6),
        tol_dq: tol_dx.unwrap_or(1e-6),
        line_search: line_search.unwrap_or(true),
        ls_beta: ls_beta.unwrap_or(0.5),
        ls_max_steps: ls_max_steps.unwrap_or(20),
        c_armijo: c_armijo.unwrap_or(1e-4),
        verbose: verbose.unwrap_or(false),
    };

    // Python 側 callable をこの GIL コンテキストに紐付けて clone
    let residual_obj = residual.clone_ref(py);
    let project_obj = project.map(|p| p.clone_ref(py));

    // ---- infer m by calling residual(x0) once ----
    let out0 = residual_obj.call1(py, (x0.clone(),))?;
    let r0: Vec<f64> = out0.extract(py)?;
    let m = r0.len();
    if m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "residual(x0) returned empty list; cannot infer residual dimension m",
        ));
    }

    // ---- error propagation from closures ----
    let err_cell: RefCell<Option<PyErr>> = RefCell::new(None);

    // residual_fn(x, r_out)
    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }

        let res: PyResult<()> = (|| {
            let out = residual.bind(py).call1((x.to_vec(),))?; // <- Bound<'py, PyAny>

            if let Ok(arr) = out.cast::<PyArray1<f64>>() {
                let owned;
                let arr_c = if arr.is_contiguous() {
                    arr
                } else {
                    owned = arr.to_owned_array().into_pyarray(py);
                    &owned
                };
                let slice = unsafe { arr_c.as_slice()? };
                if slice.len() != r_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "residual length mismatch",
                    ));
                }
                r_out.copy_from_slice(slice);
                Ok(())
            } else {
                let vec: Vec<f64> = out.extract()?;
                if vec.len() != r_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "residual length mismatch",
                    ));
                }
                r_out.copy_from_slice(&vec);
                Ok(())
            }
        })();

        if let Err(e) = res {
            *err_cell.borrow_mut() = Some(e);
        }
    };

    // jacobian_fn(x, j_out)  (row-major m*n)
    let mut jacobian_fn = |x: &[f64], j_out: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }

        let res: PyResult<()> = (|| {
            let out = jacobian.bind(py).call1((x.to_vec(),))?; // <- Bound<'py, PyAny>

            if let Ok(arr) = out.cast::<PyArray2<f64>>() {
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian must be 2D ndarray",
                    ));
                }
                let (m2, n2) = (shape[0], shape[1]);
                if m2 * n2 != j_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian size mismatch",
                    ));
                }

                let owned;
                let arr_c = if arr.is_contiguous() {
                    arr
                } else {
                    owned = arr.to_owned_array().into_pyarray(py);
                    &owned
                };

                let slice = unsafe { arr_c.as_slice()? }; // contiguous 前提
                j_out.copy_from_slice(slice);
                Ok(())
            } else {
                let vec: Vec<f64> = out.extract()?;
                if vec.len() != j_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian size mismatch",
                    ));
                }
                j_out.copy_from_slice(&vec);
                Ok(())
            }
        })();

        if let Err(e) = res {
            *err_cell.borrow_mut() = Some(e);
        }
    };

    // project(x): optional
    let mut project_fn = |x: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }
        let Some(p) = &project_obj else {
            return;
        };

        let res: PyResult<Vec<f64>> = (|| {
            let out = p.call1(py, (x.to_vec(),))?;
            out.extract::<Vec<f64>>(py)
        })();

        match res {
            Ok(x_new) => {
                if x_new.len() != x.len() {
                    *err_cell.borrow_mut() =
                        Some(pyo3::exceptions::PyValueError::new_err(format!(
                            "project length mismatch: expected {}, got {}",
                            x.len(),
                            x_new.len()
                        )));
                    return;
                }
                x.copy_from_slice(&x_new);
            }
            Err(e) => {
                *err_cell.borrow_mut() = Some(e);
            }
        }
    };

    let result = solver.solve_with_fn(m, x0, &mut residual_fn, &mut jacobian_fn, &mut project_fn);

    // If any python error occurred inside callbacks, raise it
    if let Some(e) = err_cell.into_inner() {
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
