use liteopt_core::solvers::gd::GradientDescent;
use pyo3::prelude::*;

use crate::bindings::manifold::PyVecManifold;

/// Gradient Descent optimizer exposed to Python.
///
/// f:    callable(x: list[float]) -> float
/// grad: callable(x: list[float]) -> list[float]
/// manifold: optional object with callbacks such as
///   retract(x, direction, alpha) and tangent_norm(v)
#[pyfunction(
    signature = (
        f,
        grad,
        x0,
        step_size = None,
        max_iters = None,
        tol_grad = None,
        verbose = None,
        manifold = None
    )
)]
fn gd(
    py: Python<'_>,
    f: Py<PyAny>,
    grad: Py<PyAny>,
    x0: Vec<f64>,
    step_size: Option<f64>,
    max_iters: Option<usize>,
    tol_grad: Option<f64>,
    verbose: Option<bool>,
    manifold: Option<Py<PyAny>>,
) -> PyResult<(Vec<f64>, f64, bool)> {
    let (space, manifold_err) = PyVecManifold::from_python(py, manifold)?;
    let solver = GradientDescent {
        space,
        step_size: step_size.unwrap_or(1e-3),
        max_iters: max_iters.unwrap_or(100),
        tol_grad: tol_grad.unwrap_or(1e-6),
        verbose: verbose.unwrap_or(false),
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
    if let Some(e) = manifold_err.take() {
        return Err(e);
    }

    Ok((result.x, result.f, result.converged))
}

pub(crate) fn register(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(gd, module)?)?;
    Ok(())
}
