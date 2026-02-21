use liteopt::{
    manifolds::EuclideanSpace,
    problems::{
        objective::Objective,
        test_functions::{Quadratic, Rosenbrock},
    },
    solvers::gd::GradientDescent,
};

fn gd_solver(step_size: f64, max_iters: usize, tol_grad: f64) -> GradientDescent<EuclideanSpace> {
    GradientDescent {
        space: EuclideanSpace,
        step_size,
        max_iters,
        tol_grad,
        verbose: false,
    }
}

fn quadratic_value(x: &Vec<f64>) -> f64 {
    let d = x[0] - 3.0;
    d * d
}

fn quadratic_gradient(x: &Vec<f64>, grad: &mut Vec<f64>) {
    grad[0] = 2.0 * (x[0] - 3.0);
}

#[test]
fn gradient_descent_behavior_converges_on_quadratic() {
    // f(x) = 0.5 * a x^2 - b x
    // Minimizer is x* = b / a
    let obj = Quadratic { a: 2.0, b: 4.0 }; // f(x) = x^2 - 4x => x* = 2
    let space = EuclideanSpace;
    let solver = GradientDescent {
        space,
        step_size: 0.1,
        max_iters: 1000,
        tol_grad: 1e-6,
        verbose: false,
    };

    let x0 = vec![0.0];
    let f0 = obj.value(&x0);
    let result = solver.minimize(&obj, x0);

    assert!(result.converged);
    assert!((result.x[0] - 2.0).abs() < 1e-3);
    assert!(result.f < f0);
}

#[test]
fn gradient_descent_behavior_converges_on_rosenbrock() {
    let obj = Rosenbrock { a: 1.0, b: 100.0 };
    let space = EuclideanSpace;
    let solver = GradientDescent {
        space,
        step_size: 1e-3,
        max_iters: 200_000,
        tol_grad: 1e-4,
        verbose: false,
    };

    let x0 = vec![-1.2, 1.0];
    let f0 = obj.value(&x0);
    let result = solver.minimize(&obj, x0);

    // True minimizer is (1,1)
    assert!(result.converged);
    assert!(result.f < f0);
    assert!((result.x[0] - 1.0).abs() < 5e-2);
    assert!((result.x[1] - 1.0).abs() < 5e-2);
}

#[test]
fn gradient_descent_behavior_supports_function_callbacks() {
    let solver = gd_solver(1e-3, 200_000, 1e-4);

    // initial point
    let x0 = vec![0.0, 0.0];

    // objective function
    // p = [ cos(x) + cos(x+y)
    //       sin(x) + sin(x+y)]
    let p_fn = |x: &Vec<f64>| {
        let p = vec![
            f64::cos(x[0]) + f64::cos(x[0] + x[1]),
            f64::sin(x[0]) + f64::sin(x[0] + x[1]),
        ];
        p
    };
    let dp_fn = |x: &Vec<f64>| {
        let dp = vec![
            vec![
                -(f64::sin(x[0]) + f64::sin(x[0] + x[1])),
                -f64::sin(x[0] + x[1]),
            ],
            vec![
                f64::cos(x[0]) + f64::cos(x[0] + x[1]),
                -f64::sin(x[0] + x[1]),
            ],
        ];
        dp
    };
    let target = vec![0.5, (f64::sqrt(3.0) + 2.0) / 2.0];
    use std::f64::consts::PI;

    let value_fn = |x: &Vec<f64>| {
        let x0_target = target[0];
        let x1_target = target[1];
        let p = p_fn(x);
        let residual = vec![p[0] - x0_target, p[1] - x1_target];
        0.5 * (residual[0].powi(2) + residual[1].powi(2))
    };

    // gradient of the objective function
    let grad_fn = |x: &Vec<f64>, grad: &mut Vec<f64>| {
        let x0_target = target[0];
        let x1_target = target[1];
        let p = p_fn(x);
        let residual = vec![p[0] - x0_target, p[1] - x1_target];

        let dp = dp_fn(x);
        grad[0] = residual[0] * dp[0][0] + residual[1] * dp[1][0];
        grad[1] = residual[0] * dp[0][1] + residual[1] * dp[1][1];
    };

    let result = solver.minimize_with_fn(x0, value_fn, grad_fn);

    // True minimizer is (pi/3, pi/6)
    assert!((result.x[0] - PI / 3.0).abs() < 1e-3);
    assert!((result.x[1] - PI / 6.0).abs() < 1e-3);
}

#[test]
fn gradient_descent_behavior_stops_after_maximum_iterations() {
    let result =
        gd_solver(0.01, 1, 1e-12).minimize_with_fn(vec![0.0], quadratic_value, quadratic_gradient);

    assert!(!result.converged);
    assert_eq!(result.iters, 1);
}

#[test]
fn gradient_descent_behavior_larger_step_moves_farther_in_one_iteration() {
    let short_small =
        gd_solver(0.01, 1, 1e-12).minimize_with_fn(vec![0.0], quadratic_value, quadratic_gradient);
    let short_large =
        gd_solver(0.1, 1, 1e-12).minimize_with_fn(vec![0.0], quadratic_value, quadratic_gradient);

    assert!(short_small.x[0] < short_large.x[0] && short_large.x[0] < 3.0);
}

#[test]
fn gradient_descent_behavior_longer_run_reduces_cost_more_than_short_run() {
    let short_run =
        gd_solver(0.1, 1, 1e-12).minimize_with_fn(vec![0.0], quadratic_value, quadratic_gradient);
    let long_run =
        gd_solver(0.1, 200, 1e-9).minimize_with_fn(vec![0.0], quadratic_value, quadratic_gradient);

    assert!(long_run.converged);
    assert!(long_run.f < short_run.f);
}

#[test]
fn gradient_descent_behavior_default_can_omit_space_field() {
    let solver = GradientDescent {
        step_size: 0.1,
        max_iters: 500,
        tol_grad: 1e-9,
        ..Default::default()
    };

    let result = solver.minimize_with_fn(
        vec![0.0],
        |x| {
            let d = x[0] - 3.0;
            d * d
        },
        |x, grad| {
            grad[0] = 2.0 * (x[0] - 3.0);
        },
    );

    assert!(result.converged);
    assert!((result.x[0] - 3.0).abs() < 1e-6);
}
