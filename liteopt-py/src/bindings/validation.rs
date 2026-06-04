use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn finite_nonnegative(label: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{label} must be finite and >= 0, got {value}"
        )))
    }
}

pub(crate) fn finite_gt(label: &str, value: f64, lower: f64) -> PyResult<f64> {
    if value.is_finite() && value > lower {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{label} must be finite and > {lower}, got {value}"
        )))
    }
}

pub(crate) fn finite_open_unit(label: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{label} must be finite and in (0,1), got {value}"
        )))
    }
}

pub(crate) fn finite_open_closed_unit(label: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{label} must be finite and in (0,1], got {value}"
        )))
    }
}

pub(crate) fn finite(label: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!(
            "{label} must be finite, got {value}"
        )))
    }
}

pub(crate) fn nonzero_usize(label: &str, value: usize) -> PyResult<usize> {
    if value > 0 {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{label} must be > 0")))
    }
}
