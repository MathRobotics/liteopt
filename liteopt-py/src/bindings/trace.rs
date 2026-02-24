use liteopt_core::solvers::SolverTraceRecord;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn trace_records_to_pylist(
    py: Python<'_>,
    records: Vec<SolverTraceRecord>,
) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for row in records {
        let d = PyDict::new(py);
        d.set_item("solver", row.solver)?;
        d.set_item("iter", row.iter)?;
        d.set_item("f", row.f)?;
        d.set_item("cost", row.cost)?;
        d.set_item("r_norm", row.r_norm)?;
        d.set_item("grad_norm", row.grad_norm)?;
        d.set_item("dx_norm", row.dx_norm)?;
        d.set_item("step_size", row.step_size)?;
        d.set_item("alpha", row.alpha)?;
        d.set_item("lambda", row.lambda)?;
        d.set_item("dphi0", row.dphi0)?;
        d.set_item("note", row.note)?;
        list.append(d)?;
    }
    Ok(list.into_any().unbind())
}
