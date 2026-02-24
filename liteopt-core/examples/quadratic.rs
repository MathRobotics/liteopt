use liteopt::{manifolds::EuclideanSpace, solvers::gd::GradientDescent};

fn main() {
    let solver = GradientDescent {
        space: EuclideanSpace,
        step_size: 0.1,
        max_iters: 100,
        tol_grad: 1e-9,
        verbose: false,
        collect_trace: false,
    };

    let result = solver.minimize_with_fn(
        vec![0.0],
        |x| (x[0] - 3.0).powi(2),
        |x, grad| {
            grad[0] = 2.0 * (x[0] - 3.0);
        },
    );

    println!(
        "converged={} x*={:.6} f(x*)={:.3e}",
        result.converged, result.x[0], result.f
    );
}
