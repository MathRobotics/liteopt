use liteopt::{EuclideanSpace, GradientDescent};

fn main() {
    let space = EuclideanSpace;

    let solver = GradientDescent {
        space,
        step_size: 1e-3,
        max_iters: 200_000,
        tol_grad: 1e-4,
    };

    // initial point
    let x0 = vec![-1.2, 1.0];

    // objective function f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
    let value_fn = |x: &Vec<f64>| {
        let x0 = x[0];
        let x1 = x[1];
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    };

    // gradient of the objective function
    let grad_fn = |x: &Vec<f64>, grad: &mut Vec<f64>| {
        let x0 = x[0];
        let x1 = x[1];

        grad[0] = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
        grad[1] = 200.0 * (x1 - x0 * x0);
    };

    let result = solver.minimize_with_fn(x0, value_fn, grad_fn);

    println!("converged: {}", result.converged);
    println!("iters    : {}", result.iters);
    println!("x*       : {:?}", result.x);
    println!("f(x*)    : {}", result.f);
    println!("‖grad‖   : {}", result.grad_norm);
}