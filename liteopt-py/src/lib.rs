use liteopt_core::{EuclideanSpace, GradientDescent};
use pyo3::prelude::*;

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

/// Python module definition
#[pymodule]
fn liteopt(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gd, m)?)?;
    Ok(())
}