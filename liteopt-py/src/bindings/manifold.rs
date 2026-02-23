use liteopt_core::manifolds::space::Space;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bindings::callbacks::PyErrState;

/// Python-callable manifold adapter for `Vec<f64>` point/tangent spaces.
///
/// Supported optional callables on `manifold`:
/// - `retract(x, direction, alpha) -> Vec[float]`
/// - `tangent_norm(v) -> float`
/// - `scale(v, alpha) -> Vec[float]`
/// - `add(x, v) -> Vec[float]`
/// - `difference(x, y) -> Vec[float]`
///
/// If omitted, each operation falls back to Euclidean behavior.
pub(crate) struct PyVecManifold {
    retract: Option<Py<PyAny>>,
    tangent_norm: Option<Py<PyAny>>,
    scale: Option<Py<PyAny>>,
    add: Option<Py<PyAny>>,
    difference: Option<Py<PyAny>>,
    error: PyErrState,
}

impl PyVecManifold {
    pub(crate) fn from_python(
        py: Python<'_>,
        manifold: Option<Py<PyAny>>,
    ) -> PyResult<(Self, PyErrState)> {
        let error = PyErrState::default();
        let mut space = Self {
            retract: None,
            tangent_norm: None,
            scale: None,
            add: None,
            difference: None,
            error: error.clone(),
        };

        if let Some(manifold_obj) = manifold {
            let m = manifold_obj.bind(py);
            space.retract = Self::get_callable(m, "retract")?;
            space.tangent_norm = Self::get_callable(m, "tangent_norm")?;
            space.scale = Self::get_callable(m, "scale")?;
            space.add = Self::get_callable(m, "add")?;
            space.difference = Self::get_callable(m, "difference")?;
        }

        Ok((space, error))
    }

    fn get_callable(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        if !obj.hasattr(name)? {
            return Ok(None);
        }

        let f = obj.getattr(name)?;
        if !f.is_callable() {
            return Err(PyValueError::new_err(format!(
                "manifold.{} must be callable",
                name
            )));
        }
        Ok(Some(f.unbind()))
    }

    fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    fn fallback_scale(out: &mut Vec<f64>, v: &[f64], alpha: f64) {
        out.resize(v.len(), 0.0);
        for i in 0..v.len() {
            out[i] = alpha * v[i];
        }
    }

    fn fallback_add(out: &mut Vec<f64>, x: &[f64], v: &[f64]) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = x[i] + v[i];
        }
    }

    fn fallback_difference(out: &mut Vec<f64>, x: &[f64], y: &[f64]) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = y[i] - x[i];
        }
    }

    fn call_vec2(&self, f: &Py<PyAny>, a: &[f64], b: &[f64]) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let out = f.bind(py).call1((a.to_vec(), b.to_vec()))?;
            out.extract::<Vec<f64>>()
        })
    }

    fn call_vec_scale(&self, f: &Py<PyAny>, v: &[f64], alpha: f64) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let out = f.bind(py).call1((v.to_vec(), alpha))?;
            out.extract::<Vec<f64>>()
        })
    }

    fn call_retract(
        &self,
        f: &Py<PyAny>,
        x: &[f64],
        direction: &[f64],
        alpha: f64,
    ) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let out = f.bind(py).call1((x.to_vec(), direction.to_vec(), alpha))?;
            out.extract::<Vec<f64>>()
        })
    }
}

impl Space for PyVecManifold {
    type Point = Vec<f64>;
    type Tangent = Vec<f64>;

    fn zero_like(&self, x: &Self::Point) -> Self::Point {
        vec![0.0; x.len()]
    }

    fn norm(&self, v: &Self::Point) -> f64 {
        Self::l2_norm(v)
    }

    fn scale_into(&self, out: &mut Self::Tangent, v: &Self::Tangent, alpha: f64) {
        if !self.error.has_error() {
            if let Some(f) = &self.scale {
                match self.call_vec_scale(f, v, alpha) {
                    Ok(vec) if vec.len() == v.len() => {
                        out.resize(vec.len(), 0.0);
                        out.copy_from_slice(&vec);
                        return;
                    }
                    Ok(vec) => {
                        self.error.set_once(PyValueError::new_err(format!(
                            "manifold.scale length mismatch: expected {}, got {}",
                            v.len(),
                            vec.len()
                        )));
                    }
                    Err(e) => self.error.set_once(e),
                }
            }
        }
        Self::fallback_scale(out, v, alpha);
    }

    fn add_into(&self, out: &mut Self::Point, x: &Self::Point, v: &Self::Tangent) {
        if !self.error.has_error() {
            if let Some(f) = &self.add {
                match self.call_vec2(f, x, v) {
                    Ok(vec) if vec.len() == x.len() => {
                        out.resize(vec.len(), 0.0);
                        out.copy_from_slice(&vec);
                        return;
                    }
                    Ok(vec) => {
                        self.error.set_once(PyValueError::new_err(format!(
                            "manifold.add length mismatch: expected {}, got {}",
                            x.len(),
                            vec.len()
                        )));
                    }
                    Err(e) => self.error.set_once(e),
                }
            }
        }
        Self::fallback_add(out, x, v);
    }

    fn difference_into(&self, out: &mut Self::Tangent, x: &Self::Point, y: &Self::Point) {
        if !self.error.has_error() {
            if let Some(f) = &self.difference {
                match self.call_vec2(f, x, y) {
                    Ok(vec) if vec.len() == x.len() => {
                        out.resize(vec.len(), 0.0);
                        out.copy_from_slice(&vec);
                        return;
                    }
                    Ok(vec) => {
                        self.error.set_once(PyValueError::new_err(format!(
                            "manifold.difference length mismatch: expected {}, got {}",
                            x.len(),
                            vec.len()
                        )));
                    }
                    Err(e) => self.error.set_once(e),
                }
            }
        }
        Self::fallback_difference(out, x, y);
    }

    fn zero_tangent_like(&self, x: &Self::Point) -> Self::Tangent {
        vec![0.0; x.len()]
    }

    fn tangent_norm(&self, v: &Self::Tangent) -> f64 {
        if !self.error.has_error() {
            if let Some(f) = &self.tangent_norm {
                let out: PyResult<f64> = Python::attach(|py| {
                    let v = f.bind(py).call1((v.to_vec(),))?;
                    v.extract::<f64>()
                });
                match out {
                    Ok(norm) if norm.is_finite() => return norm,
                    Ok(_) => {
                        self.error.set_once(PyValueError::new_err(
                            "manifold.tangent_norm returned non-finite value",
                        ));
                    }
                    Err(e) => self.error.set_once(e),
                }
            }
        }
        Self::l2_norm(v)
    }

    fn retract_into(
        &self,
        out: &mut Self::Point,
        x: &Self::Point,
        direction: &Self::Tangent,
        alpha: f64,
        _tmp: &mut Self::Tangent,
    ) {
        if !self.error.has_error() {
            if let Some(f) = &self.retract {
                match self.call_retract(f, x, direction, alpha) {
                    Ok(vec) if vec.len() == x.len() => {
                        out.resize(vec.len(), 0.0);
                        out.copy_from_slice(&vec);
                        return;
                    }
                    Ok(vec) => {
                        self.error.set_once(PyValueError::new_err(format!(
                            "manifold.retract length mismatch: expected {}, got {}",
                            x.len(),
                            vec.len()
                        )));
                    }
                    Err(e) => self.error.set_once(e),
                }
            }
        }

        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = x[i] + alpha * direction[i];
        }
    }
}
