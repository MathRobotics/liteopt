use liteopt::{
    manifolds::EuclideanSpace,
    problems::{
        objective::Objective,
        test_functions::{Quadratic, Rosenbrock},
    },
    solvers::gd::GradientDescent,
};

#[test]
fn quadratic_minimization() {
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
fn rosenbrock_minimization() {
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
fn nonlinear_minimization_with_fn() {
    let space = EuclideanSpace;
    let solver = GradientDescent {
        space,
        step_size: 1e-3,
        max_iters: 200_000,
        tol_grad: 1e-4,
        verbose: false,
    };

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
fn gd_respects_max_iters_and_step_size() {
    let space = EuclideanSpace;
    let value_fn = |x: &Vec<f64>| {
        let d = x[0] - 3.0;
        d * d
    };
    let grad_fn = |x: &Vec<f64>, grad: &mut Vec<f64>| {
        grad[0] = 2.0 * (x[0] - 3.0);
    };

    let short_small = GradientDescent {
        space,
        step_size: 0.01,
        max_iters: 1,
        tol_grad: 1e-12,
        verbose: false,
    }
    .minimize_with_fn(vec![0.0], value_fn, grad_fn);

    let short_large = GradientDescent {
        space,
        step_size: 0.1,
        max_iters: 1,
        tol_grad: 1e-12,
        verbose: false,
    }
    .minimize_with_fn(vec![0.0], value_fn, grad_fn);

    let long_run = GradientDescent {
        space,
        step_size: 0.1,
        max_iters: 200,
        tol_grad: 1e-9,
        verbose: false,
    }
    .minimize_with_fn(vec![0.0], value_fn, grad_fn);

    assert!(!short_small.converged);
    assert!(!short_large.converged);
    assert!(long_run.converged);
    assert!(short_small.x[0] < short_large.x[0] && short_large.x[0] < 3.0);
    assert!(long_run.f < short_large.f);
}
