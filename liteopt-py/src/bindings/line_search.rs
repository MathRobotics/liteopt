use liteopt_core::solvers::gn::{LineSearchContext, LineSearchPolicy, LineSearchResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bindings::callbacks::PyErrState;

pub(crate) struct PyLineSearchPolicy {
    callback: Py<PyAny>,
    err_state: PyErrState,
}

impl PyLineSearchPolicy {
    pub(crate) fn new(callback: Py<PyAny>, err_state: PyErrState) -> Self {
        Self {
            callback,
            err_state,
        }
    }

    fn parse_result(out: &Bound<'_, PyAny>) -> PyResult<LineSearchResult> {
        if let Ok(alpha) = out.extract::<f64>() {
            if !alpha.is_finite() {
                return Err(PyValueError::new_err(
                    "line_search callback returned non-finite alpha",
                ));
            }
            return Ok(LineSearchResult {
                accepted: true,
                alpha,
            });
        }

        if let Ok((accepted, alpha)) = out.extract::<(bool, f64)>() {
            if !alpha.is_finite() {
                return Err(PyValueError::new_err(
                    "line_search callback returned non-finite alpha",
                ));
            }
            return Ok(LineSearchResult { accepted, alpha });
        }

        if let Ok(d) = out.cast::<PyDict>() {
            let accepted = d
                .get_item("accepted")?
                .and_then(|v| v.extract::<bool>().ok())
                .unwrap_or(true);
            let Some(alpha_obj) = d.get_item("alpha")? else {
                return Err(PyValueError::new_err(
                    "line_search callback dict result requires key 'alpha'",
                ));
            };
            let alpha = alpha_obj.extract::<f64>()?;
            if !alpha.is_finite() {
                return Err(PyValueError::new_err(
                    "line_search callback returned non-finite alpha",
                ));
            }
            return Ok(LineSearchResult { accepted, alpha });
        }

        Err(PyValueError::new_err(
            "line_search callback must return float alpha, (accepted, alpha), or {'alpha': ..., 'accepted': ...}",
        ))
    }
}

impl LineSearchPolicy for PyLineSearchPolicy {
    fn requires_directional_derivative(&self) -> bool {
        true
    }

    fn search(
        &mut self,
        ctx: &LineSearchContext,
        _eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        if self.err_state.has_error() {
            return LineSearchResult {
                accepted: false,
                alpha: ctx.alpha0,
            };
        }

        let res: PyResult<LineSearchResult> = Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("iter", ctx.iter)?;
            d.set_item("alpha0", ctx.alpha0)?;
            d.set_item("cost0", ctx.cost0)?;
            d.set_item("dphi0", ctx.dphi0)?;
            d.set_item("dx_norm", ctx.dx_norm)?;
            d.set_item("lambda", ctx.lambda)?;
            let out = self.callback.bind(py).call1((d,))?;
            Self::parse_result(&out)
        });

        match res {
            Ok(ls) => ls,
            Err(e) => {
                self.err_state.set_once(e);
                LineSearchResult {
                    accepted: false,
                    alpha: ctx.alpha0,
                }
            }
        }
    }
}
