use liteopt::{space::EuclideanSpace, nls::NonlinearLeastSquares};

fn main() {
    let space = EuclideanSpace;

    let solver = NonlinearLeastSquares {
        space,
        lambda: 1e-3,
        step_scale: 1.0,
        max_iters: 200,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        line_search: true,
        ls_beta: 0.5,
        ls_max_steps: 20,
    };

    let l1 = 1.0;
    let l2 = 1.0;

    let target = [1.2, 0.6];

    let residual_fn = |q: &[f64], r: &mut [f64]| {
        let q1 = q[0];
        let q2 = q[1];
        let x = l1 * q1.cos() + l2 * (q1 + q2).cos();
        let y = l1 * q1.sin() + l2 * (q1 + q2).sin();
        r[0] = x - target[0];
        r[1] = y - target[1];
    };

    let jacobian_fn = |q: &[f64], j: &mut [f64]| {
        let q1 = q[0];
        let q2 = q[1];
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = (q1 + q2).sin();
        let c12 = (q1 + q2).cos();

        j[0 * 2 + 0] = -l1 * s1 - l2 * s12;
        j[0 * 2 + 1] = -l2 * s12;
        j[1 * 2 + 0] =  l1 * c1 + l2 * c12;
        j[1 * 2 + 1] =  l2 * c12;
    };

    let project = |_q: &mut [f64]| {};

    let x0 = vec![0.0, 0.0];
    let result = solver.solve_with_fn(2, x0, residual_fn, jacobian_fn, project);

    println!("converged: {}", result.converged);
    println!("iters    : {}", result.iters);
    println!("x*       : {:?}", result.x);
    println!("‖r‖      : {}", result.r_norm);
    println!("‖dq‖     : {}", result.dx_norm);
}