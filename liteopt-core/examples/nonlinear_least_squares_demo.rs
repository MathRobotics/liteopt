use liteopt::{manifolds::EuclideanSpace, solvers::gn::GaussNewton};

fn main() {
    let solver = GaussNewton {
        space: EuclideanSpace,
        lambda: 1e-3,
        step_scale: 1.0,
        max_iters: 20,
        tol_r: 1e-12,
        tol_dq: 1e-12,
        line_search: true,
        ls_beta: 0.5,
        ls_max_steps: 20,
        c_armijo: 1e-4,
        verbose: false,
    };

    let target = [1.0, -2.0];
    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - target[0];
        r[1] = x[1] - target[1];
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        j[0] = 1.0;
        j[1] = 0.0;
        j[2] = 0.0;
        j[3] = 1.0;
    };
    let result = solver.solve_with_fn(2, vec![0.0, 0.0], residual, jacobian, |_x| {});
    println!(
        "converged={} x*={:?} cost={:.3e}",
        result.converged, result.x, result.cost
    );
}
