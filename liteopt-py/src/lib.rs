use pyo3::prelude::*;
mod bindings;

/// Python module definition
#[pymodule]
fn liteopt(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    bindings::gd::register(m)?;
    bindings::gn::register(m)?;
    bindings::lm::register(m)?;
    Ok(())
}
