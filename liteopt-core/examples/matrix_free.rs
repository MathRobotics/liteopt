use liteopt::solvers::lm::LevenbergMarquardt;
use liteopt::solvers::{JacobianProducts, LinearSolver};

fn main() {
    // J = 2 I, represented entirely by products.
    let solver = LevenbergMarquardt {
        linear_solver: LinearSolver::Cg,
        ..Default::default()
    };
    let products = JacobianProducts::new(
        |_x: &[f64], v: &[f64], out: &mut [f64]| {
            for (o, v) in out.iter_mut().zip(v) {
                *o = 2. * v;
            }
        },
        |_x: &[f64], w: &[f64], out: &mut [f64]| {
            for (o, w) in out.iter_mut().zip(w) {
                *o = 2. * w;
            }
        },
    );
    let result = solver.solve_with_derivatives_default_line_search(
        128,
        vec![0.; 128],
        |x, r| {
            for (r, x) in r.iter_mut().zip(x) {
                *r = 2. * x - 1.;
            }
        },
        products,
        |_| {},
    );
    assert!(result.converged);
    assert_eq!(result.njev, 0);
    println!(
        "cost={} inner_iterations={}",
        result.cost, result.n_linear_iters
    );
}
