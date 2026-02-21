use liteopt::{
    manifolds::EuclideanSpace, problems::least_squares::LeastSquaresProblem,
    solvers::lm::LevenbergMarquardt,
};

struct Planar2LinkProblem {
    l1: f64,
    l2: f64,
    target: [f64; 2],
}

impl LeastSquaresProblem<EuclideanSpace> for Planar2LinkProblem {
    fn residual_dim(&self) -> usize {
        2
    }

    fn residual(&self, q: &[f64], r: &mut [f64]) {
        let q1 = q[0];
        let q2 = q[1];
        let x = self.l1 * q1.cos() + self.l2 * (q1 + q2).cos();
        let y = self.l1 * q1.sin() + self.l2 * (q1 + q2).sin();
        r[0] = x - self.target[0];
        r[1] = y - self.target[1];
    }

    fn jacobian(&self, q: &[f64], j: &mut [f64]) {
        let q1 = q[0];
        let q2 = q[1];
        let s1 = q1.sin();
        let c1 = q1.cos();
        let s12 = (q1 + q2).sin();
        let c12 = (q1 + q2).cos();

        j[0] = -self.l1 * s1 - self.l2 * s12;
        j[1] = -self.l2 * s12;
        j[2] = self.l1 * c1 + self.l2 * c12;
        j[3] = self.l2 * c12;
    }
}

fn forward_kinematics(problem: &Planar2LinkProblem, q: &[f64]) -> [f64; 2] {
    let q1 = q[0];
    let q2 = q[1];
    [
        problem.l1 * q1.cos() + problem.l2 * (q1 + q2).cos(),
        problem.l1 * q1.sin() + problem.l2 * (q1 + q2).sin(),
    ]
}

fn target_error_norm(problem: &Planar2LinkProblem, q: &[f64]) -> f64 {
    let p = forward_kinematics(problem, q);
    let ex = p[0] - problem.target[0];
    let ey = p[1] - problem.target[1];
    (ex * ex + ey * ey).sqrt()
}

#[test]
fn levenberg_marquardt_planar_2link() {
    let space = EuclideanSpace;
    let solver = LevenbergMarquardt {
        space,
        lambda: 1e-3,
        lambda_up: 10.0,
        lambda_down: 0.5,
        step_scale: 1.0,
        max_iters: 200,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        verbose: false,
    };

    let problem = Planar2LinkProblem {
        l1: 1.0,
        l2: 1.0,
        target: [1.2, 0.6],
    };

    let q0 = vec![0.0, 0.0];
    let initial_err = target_error_norm(&problem, &q0);
    let res = solver.solve(q0, &problem);
    let final_err = target_error_norm(&problem, &res.x);

    assert!(res.converged, "did not converge: {:?}", res);
    assert!(res.r_norm < 1e-6, "residual too large: {}", res.r_norm);
    assert!(
        final_err < 1e-6,
        "forward-kinematics error too large: {final_err}"
    );
    assert!(
        final_err < initial_err * 1e-3,
        "solution did not improve enough: initial={initial_err}, final={final_err}"
    );
}

#[test]
fn levenberg_marquardt_respects_max_iters() {
    let space = EuclideanSpace;
    let solver_short = LevenbergMarquardt {
        space,
        lambda: 1e-3,
        lambda_up: 10.0,
        lambda_down: 0.5,
        step_scale: 1.0,
        max_iters: 1,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        verbose: false,
    };
    let solver_full = LevenbergMarquardt {
        space,
        lambda: 1e-3,
        lambda_up: 10.0,
        lambda_down: 0.5,
        step_scale: 1.0,
        max_iters: 200,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        verbose: false,
    };

    let problem = Planar2LinkProblem {
        l1: 1.0,
        l2: 1.0,
        target: [1.2, 0.6],
    };

    let short = solver_short.solve(vec![0.0, 0.0], &problem);
    let full = solver_full.solve(vec![0.0, 0.0], &problem);

    assert!(
        !short.converged,
        "short run should not converge: {:?}",
        short
    );
    assert!(full.converged, "full run should converge: {:?}", full);
    assert_eq!(short.iters, 1);
    assert!(full.iters <= 200);
    assert!(
        full.cost < short.cost,
        "full run should reduce cost more: short={}, full={}",
        short.cost,
        full.cost
    );
}

#[test]
fn levenberg_marquardt_stops_after_repeated_linear_solve_failure() {
    let space = EuclideanSpace;
    let solver = LevenbergMarquardt {
        space,
        lambda: 1e-3,
        lambda_up: 10.0,
        lambda_down: 0.5,
        step_scale: 1.0,
        max_iters: 3,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        verbose: false,
    };

    let residual_fn = |_x: &[f64], r: &mut [f64]| {
        r[0] = 1.0;
    };

    // Force A to contain NaN, which makes solve_linear_inplace fail.
    let jacobian_fn = |_x: &[f64], j: &mut [f64]| {
        j[0] = f64::NAN;
    };

    let project = |_x: &mut [f64]| {};
    let x0 = vec![0.0];
    let res = solver.solve_with_fn(1, x0, residual_fn, jacobian_fn, project);

    assert!(
        !res.converged,
        "solver should stop as non-converged: {:?}",
        res
    );
    assert_eq!(res.iters, 3, "solver should stop exactly at max_iters");
}

#[test]
fn levenberg_marquardt_default_uses_euclidean_space() {
    let solver = LevenbergMarquardt::default();

    let residual_fn = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 2.0;
    };
    let jacobian_fn = |_x: &[f64], j: &mut [f64]| {
        j[0] = 1.0;
    };
    let project = |_x: &mut [f64]| {};

    let res = solver.solve_with_fn(1, vec![0.0], residual_fn, jacobian_fn, project);
    assert!(res.converged, "default solver should converge: {:?}", res);
    assert!((res.x[0] - 2.0).abs() < 1e-6);
}
