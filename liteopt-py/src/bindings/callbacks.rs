use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Default)]
pub(crate) struct PyErrState {
    inner: Rc<RefCell<Option<PyErr>>>,
}

impl PyErrState {
    pub(crate) fn has_error(&self) -> bool {
        self.inner.borrow().is_some()
    }

    pub(crate) fn set_once(&self, err: PyErr) {
        let mut slot = self.inner.borrow_mut();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    pub(crate) fn take(&self) -> Option<PyErr> {
        self.inner.borrow_mut().take()
    }
}

fn extract_vec1(py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = out.cast::<PyArray1<f64>>() {
        let owned;
        let arr_c = if arr.is_contiguous() {
            arr
        } else {
            owned = arr.to_owned_array().into_pyarray(py);
            &owned
        };
        let slice = unsafe { arr_c.as_slice()? };
        return Ok(slice.to_vec());
    }

    out.extract::<Vec<f64>>()
}

fn extract_jacobian_row_major(
    py: Python<'_>,
    out: &Bound<'_, PyAny>,
    expected_len: usize,
) -> PyResult<Vec<f64>> {
    if let Ok(arr) = out.cast::<PyArray2<f64>>() {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("jacobian must be 2D ndarray"));
        }

        let (m, n) = (shape[0], shape[1]);
        if m * n != expected_len {
            return Err(PyValueError::new_err("jacobian size mismatch"));
        }

        let owned;
        let arr_c = if arr.is_contiguous() {
            arr
        } else {
            owned = arr.to_owned_array().into_pyarray(py);
            &owned
        };
        let slice = unsafe { arr_c.as_slice()? };
        return Ok(slice.to_vec());
    }

    let vec = out.extract::<Vec<f64>>()?;
    if vec.len() != expected_len {
        return Err(PyValueError::new_err("jacobian size mismatch"));
    }
    Ok(vec)
}

pub(crate) struct PyObjectiveCallbacks {
    value_fn: Py<PyAny>,
    grad_fn: Py<PyAny>,
    err: PyErrState,
}

impl PyObjectiveCallbacks {
    pub(crate) fn new(value_fn: Py<PyAny>, grad_fn: Py<PyAny>, err: PyErrState) -> Self {
        Self {
            value_fn,
            grad_fn,
            err,
        }
    }

    pub(crate) fn value(&self, py: Python<'_>, x: &[f64]) -> f64 {
        if self.err.has_error() {
            return f64::INFINITY;
        }

        let out = self.value_fn.bind(py).call1((x.to_vec(),));
        match out.and_then(|v| v.extract::<f64>()) {
            Ok(v) if v.is_finite() => v,
            Ok(_) => {
                self.err.set_once(PyValueError::new_err(
                    "objective function must return finite float",
                ));
                f64::INFINITY
            }
            Err(e) => {
                self.err.set_once(e);
                f64::INFINITY
            }
        }
    }

    pub(crate) fn gradient_into(&self, py: Python<'_>, x: &[f64], grad_out: &mut [f64]) {
        if self.err.has_error() {
            grad_out.fill(0.0);
            return;
        }

        let result: PyResult<Vec<f64>> = (|| {
            let out = self.grad_fn.bind(py).call1((x.to_vec(),))?;
            extract_vec1(py, &out)
        })();

        match result {
            Ok(g) if g.len() == grad_out.len() => grad_out.copy_from_slice(&g),
            Ok(g) => {
                self.err.set_once(PyValueError::new_err(format!(
                    "gradient length mismatch: expected {}, got {}",
                    grad_out.len(),
                    g.len()
                )));
                grad_out.fill(0.0);
            }
            Err(e) => {
                self.err.set_once(e);
                grad_out.fill(0.0);
            }
        }
    }
}

pub(crate) struct PyLeastSquaresCallbacks {
    residual_fn: Py<PyAny>,
    jacobian_fn: Py<PyAny>,
    project_fn: Option<Py<PyAny>>,
    err: PyErrState,
}

impl PyLeastSquaresCallbacks {
    pub(crate) fn new(
        residual_fn: Py<PyAny>,
        jacobian_fn: Py<PyAny>,
        project_fn: Option<Py<PyAny>>,
        err: PyErrState,
    ) -> Self {
        Self {
            residual_fn,
            jacobian_fn,
            project_fn,
            err,
        }
    }

    pub(crate) fn infer_residual_dim(&self, py: Python<'_>, x0: &[f64]) -> PyResult<usize> {
        let out = self.residual_fn.bind(py).call1((x0.to_vec(),))?;
        let r0 = extract_vec1(py, &out)?;
        if r0.is_empty() {
            return Err(PyValueError::new_err(
                "residual(x0) returned empty list; cannot infer residual dimension m",
            ));
        }
        Ok(r0.len())
    }

    pub(crate) fn residual_into(&self, py: Python<'_>, x: &[f64], r_out: &mut [f64]) {
        if self.err.has_error() {
            r_out.fill(0.0);
            return;
        }

        let result: PyResult<Vec<f64>> = (|| {
            let out = self.residual_fn.bind(py).call1((x.to_vec(),))?;
            extract_vec1(py, &out)
        })();

        match result {
            Ok(r) if r.len() == r_out.len() => r_out.copy_from_slice(&r),
            Ok(r) => {
                self.err.set_once(PyValueError::new_err(format!(
                    "residual length mismatch: expected {}, got {}",
                    r_out.len(),
                    r.len()
                )));
                r_out.fill(0.0);
            }
            Err(e) => {
                self.err.set_once(e);
                r_out.fill(0.0);
            }
        }
    }

    pub(crate) fn jacobian_into(&self, py: Python<'_>, x: &[f64], j_out: &mut [f64]) {
        if self.err.has_error() {
            j_out.fill(0.0);
            return;
        }

        let result: PyResult<Vec<f64>> = (|| {
            let out = self.jacobian_fn.bind(py).call1((x.to_vec(),))?;
            extract_jacobian_row_major(py, &out, j_out.len())
        })();

        match result {
            Ok(j) => j_out.copy_from_slice(&j),
            Err(e) => {
                self.err.set_once(e);
                j_out.fill(0.0);
            }
        }
    }

    pub(crate) fn project_in_place(&self, py: Python<'_>, x: &mut [f64]) {
        if self.err.has_error() {
            return;
        }
        let Some(project) = &self.project_fn else {
            return;
        };

        let result: PyResult<Vec<f64>> = (|| {
            let out = project.bind(py).call1((x.to_vec(),))?;
            extract_vec1(py, &out)
        })();

        match result {
            Ok(x_new) if x_new.len() == x.len() => x.copy_from_slice(&x_new),
            Ok(x_new) => {
                self.err.set_once(PyValueError::new_err(format!(
                    "project length mismatch: expected {}, got {}",
                    x.len(),
                    x_new.len()
                )));
            }
            Err(e) => {
                self.err.set_once(e);
            }
        }
    }
}
