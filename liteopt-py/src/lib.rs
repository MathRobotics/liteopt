use liteopt_core::{space::EuclideanSpace, gd::GradientDescent, nls::NonlinearLeastSquares};
use pyo3::prelude::*;
use std::cell::RefCell;

use numpy::{PyArray1, PyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods}; 

/// Gradient Descent optimizer exposed to Python.
///
/// f:    callable(x: list[float]) -> float
/// grad: callable(x: list[float]) -> list[float]
#[pyfunction]
fn gd(
    py: Python<'_>,
    f: Py<PyAny>,
    grad: Py<PyAny>,
    x0: Vec<f64>,
    step_size: f64,
    max_iters: usize,
    tol_grad: f64,
) -> PyResult<(Vec<f64>, f64, bool)> {
    let space = EuclideanSpace;
    let solver = GradientDescent {
        space,
        step_size,
        max_iters,
        tol_grad,
    };

    let f_obj = f.clone_ref(py);
    let grad_obj = grad.clone_ref(py);

    // closure for calling Python function f(x)
    let f_closure = move |x: &Vec<f64>| -> f64 {
        let arg = x.clone();
        let res = f_obj
            .call1(py, (arg,))
            .expect("failed to call objective function from Python");
        res.extract::<f64>(py)
            .expect("objective function must return float")
    };

    // closure for calling Python gradient function grad(x)
    let grad_closure = move |x: &Vec<f64>, grad_out: &mut Vec<f64>| {
        let arg = x.clone();
        let res = grad_obj
            .call1(py, (arg,))
            .expect("failed to call gradient function from Python");
        let g: Vec<f64> = res
            .extract(py)
            .expect("gradient function must return list[float]");

        assert_eq!(
            g.len(),
            grad_out.len(),
            "gradient length mismatch: expected {}, got {}",
            grad_out.len(),
            g.len()
        );

        for (o, gi) in grad_out.iter_mut().zip(g.iter()) {
            *o = *gi;
        }
    };

    let result = solver.minimize_with_fn(x0, f_closure, grad_closure);

    Ok((result.x, result.f, result.converged))
}

/// Nonlinear least squares solver exposed to Python.
///
/// residual: callable(x: list[float]) -> list[float]           (len = m)
/// jacobian: callable(x: list[float]) -> list[float]           (len = m*n, row-major)
/// project : Optional[callable(x: list[float]) -> list[float]] (len = n)  or None
#[pyfunction]
fn nls(
    residual: Py<PyAny>,
    jacobian: Py<PyAny>,
    project: Option<Py<PyAny>>,
    x0: Vec<f64>,
    lambda_: f64,
    step_scale: f64,
    max_iters: usize,
    tol_r: f64,
    tol_dx: f64,
    line_search: bool,
    ls_beta: f64,
    ls_max_steps: usize,
) -> PyResult<(Vec<f64>, f64, bool, usize, f64, f64)> {
    let space = EuclideanSpace;
    let solver = NonlinearLeastSquares {
        space,
        lambda: lambda_,
        step_scale,
        max_iters,
        tol_r,
        tol_dq: tol_dx, // Rust側フィールド名が tol_dq のままならここに入れる
        line_search,
        ls_beta,
        ls_max_steps,
    };

    let n = x0.len();

    // ---- infer m by calling residual(x0) once ----
    let r0: Vec<f64> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
        let out = residual.bind(py).call1((x0.clone(),))?;
        out.extract::<Vec<f64>>()
    })?;
    let m = r0.len();
    if m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "residual(x0) returned empty list; cannot infer residual dimension m",
        ));
    }

    // ---- error propagation from closures ----
    let err_cell: RefCell<Option<PyErr>> = RefCell::new(None);

    // residual_fn(q, r_out)
    let mut residual_fn = |x: &[f64], r_out: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }

        let res = Python::with_gil(|py| -> PyResult<()> {
            let out = residual.bind(py).call1((x.to_vec(),))?;

            // numpy.ndarray (1D)
            if let Ok(arr) = out.downcast::<PyArray1<f64>>() {
                let slice = unsafe { arr.as_slice()? };
                if slice.len() != r_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "residual length mismatch",
                    ));
                }
                r_out.copy_from_slice(slice);
                Ok(())
            } else {
                // fallback: list
                let vec: Vec<f64> = out.extract()?;
                if vec.len() != r_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "residual length mismatch",
                    ));
                }
                r_out.copy_from_slice(&vec);
                Ok(())
            }
        });

        if let Err(e) = res {
            *err_cell.borrow_mut() = Some(e);
        }
    };

    // jacobian_fn(q, j_out)  (row-major m*n)
    let mut jacobian_fn = |x: &[f64], j_out: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }

        let res = Python::with_gil(|py| -> PyResult<()> {
            let out = jacobian.bind(py).call1((x.to_vec(),))?;

            // numpy.ndarray (2D)
            if let Ok(arr) = out.downcast::<PyArray2<f64>>() {
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian must be 2D ndarray",
                    ));
                }

                let m2 = shape[0];
                let n2 = shape[1];
                if m2 * n2 != j_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian size mismatch",
                    ));
                }

                let slice = unsafe { arr.as_slice()? }; // row-major contiguous 想定
                j_out.copy_from_slice(slice);
                Ok(())
            } else {
                // fallback: flat list
                let vec: Vec<f64> = out.extract()?;
                if vec.len() != j_out.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "jacobian size mismatch",
                    ));
                }
                j_out.copy_from_slice(&vec);
                Ok(())
            }
        });

        if let Err(e) = res {
            *err_cell.borrow_mut() = Some(e);
        }
    };

    // project(q): optional
    let mut project_fn = |x: &mut [f64]| {
        if err_cell.borrow().is_some() {
            return;
        }
        let Some(p) = &project else { return; };

        let res: PyResult<Vec<f64>> = Python::with_gil(|py| {
            let out = p.bind(py).call1((x.to_vec(),))?;
            out.extract::<Vec<f64>>()
        });

        match res {
            Ok(x_new) => {
                if x_new.len() != x.len() {
                    *err_cell.borrow_mut() = Some(pyo3::exceptions::PyValueError::new_err(
                        format!(
                            "project length mismatch: expected {}, got {}",
                            x.len(),
                            x_new.len()
                        ),
                    ));
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
        result.converged,
        result.iters,
        result.r_norm,
        result.dx_norm,
    ))
}

/// Python module definition
#[pymodule]
fn liteopt(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gd, m)?)?;
    m.add_function(wrap_pyfunction!(nls, m)?)?;
    Ok(())
}