use liteopt::{
    manifolds::{space::Space, EuclideanSpace},
    problems::least_squares::LeastSquaresProblem,
    solvers::gn::{
        GaussNewton, LineSearchContext, LineSearchPolicy, LineSearchResult, NoLineSearch,
    },
};

#[derive(Clone, Copy, Debug, Default)]
struct MyManifold;

fn wrap_angle(theta: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (theta + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

impl Space for MyManifold {
    type Point = Vec<f64>;
    type Tangent = Vec<f64>;

    fn zero_like(&self, x: &Self::Point) -> Self::Point {
        vec![0.0; x.len()]
    }

    fn norm(&self, v: &Self::Point) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    fn scale_into(&self, out: &mut Self::Tangent, v: &Self::Tangent, alpha: f64) {
        out.resize(v.len(), 0.0);
        for i in 0..v.len() {
            out[i] = alpha * v[i];
        }
    }

    fn add_into(&self, out: &mut Self::Point, x: &Self::Point, v: &Self::Tangent) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(x[i] + v[i]);
        }
    }

    fn difference_into(&self, out: &mut Self::Tangent, x: &Self::Point, y: &Self::Point) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(y[i] - x[i]);
        }
    }

    fn zero_tangent_like(&self, x: &Self::Point) -> Self::Tangent {
        vec![0.0; x.len()]
    }

    fn tangent_norm(&self, v: &Self::Tangent) -> f64 {
        self.norm(v)
    }

    fn retract_into(
        &self,
        out: &mut Self::Point,
        x: &Self::Point,
        direction: &Self::Tangent,
        alpha: f64,
        _tmp: &mut Self::Tangent,
    ) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(x[i] + alpha * direction[i]);
        }
    }
}

struct Planar2LinkProblem {
    l1: f64,
    l2: f64,
    target: [f64; 2],
}

impl Planar2LinkProblem {
    fn residual_eval(&self, q: &[f64], r: &mut [f64]) {
        let q1 = q[0];
        let q2 = q[1];
        let x = self.l1 * q1.cos() + self.l2 * (q1 + q2).cos();
        let y = self.l1 * q1.sin() + self.l2 * (q1 + q2).sin();
        r[0] = x - self.target[0];
        r[1] = y - self.target[1];
    }

    fn jacobian_eval(&self, q: &[f64], j: &mut [f64]) {
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

impl<S> LeastSquaresProblem<S> for Planar2LinkProblem
where
    S: Space<Point = Vec<f64>, Tangent = Vec<f64>>,
{
    fn residual_dim(&self) -> usize {
        2
    }

    fn residual(&self, q: &[f64], r: &mut [f64]) {
        self.residual_eval(q, r);
    }

    fn jacobian(&self, q: &[f64], j: &mut [f64]) {
        self.jacobian_eval(q, j);
    }
}

fn planar_two_link_problem() -> Planar2LinkProblem {
    Planar2LinkProblem {
        l1: 1.0,
        l2: 1.0,
        target: [1.2, 0.6],
    }
}

fn gauss_newton_solver<S>(space: S, max_iters: usize) -> GaussNewton<S>
where
    S: Space<Point = Vec<f64>, Tangent = Vec<f64>>,
{
    GaussNewton {
        space,
        lambda: 1e-3,
        step_scale: 1.0,
        max_iters,
        tol_r: 1e-9,
        tol_dq: 1e-12,
        verbose: false,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FixedHalfStepSearch;

impl LineSearchPolicy for FixedHalfStepSearch {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        let alpha = 0.5 * ctx.alpha0;
        let accepted = eval_cost(alpha).is_some();
        LineSearchResult { accepted, alpha }
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
fn gauss_newton_behavior_converges_on_planar_two_link_problem() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(EuclideanSpace, 200);
    let res = solver.solve_with_default_line_search(vec![0.0, 0.0], &problem);

    assert!(res.converged, "did not converge: {:?}", res);
    assert!(res.r_norm < 1e-6, "residual too large: {}", res.r_norm);
}

#[test]
fn gauss_newton_behavior_improves_forward_kinematics_error() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(EuclideanSpace, 200);
    let q0 = vec![0.0, 0.0];
    let initial_err = target_error_norm(&problem, &q0);

    let res = solver.solve_with_default_line_search(q0, &problem);
    let final_err = target_error_norm(&problem, &res.x);

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
fn gauss_newton_behavior_custom_manifold_converges_on_planar_two_link_problem() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(MyManifold, 200);
    let q0 = vec![3.0 * std::f64::consts::PI, -2.0 * std::f64::consts::PI];
    let res = solver.solve_with_default_line_search(q0, &problem);
    let final_err = target_error_norm(&problem, &res.x);

    assert!(res.converged, "did not converge: {:?}", res);
    assert!(res.r_norm < 1e-6, "residual too large: {}", res.r_norm);
    assert!(
        final_err < 1e-6,
        "forward-kinematics error too large: {final_err}"
    );
}

#[test]
fn gauss_newton_behavior_custom_manifold_wraps_solution_angles_into_primary_range() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(MyManifold, 200);
    let q0 = vec![3.0 * std::f64::consts::PI, -2.0 * std::f64::consts::PI];
    let res = solver.solve_with_default_line_search(q0, &problem);

    for qi in &res.x {
        assert!(
            *qi >= -std::f64::consts::PI && *qi < std::f64::consts::PI,
            "angle should be wrapped into [-pi, pi): {}",
            qi
        );
    }
}

#[test]
fn gauss_newton_behavior_stops_after_maximum_iterations() {
    let problem = planar_two_link_problem();
    let short = gauss_newton_solver(EuclideanSpace, 1)
        .solve_with_default_line_search(vec![0.0, 0.0], &problem);

    assert!(
        !short.converged,
        "short run should not converge: {:?}",
        short
    );
    assert_eq!(short.iters, 1);
}

#[test]
fn gauss_newton_behavior_longer_run_reduces_cost_more_than_short_run() {
    let problem = planar_two_link_problem();
    let short = gauss_newton_solver(EuclideanSpace, 1)
        .solve_with_default_line_search(vec![0.0, 0.0], &problem);
    let full = gauss_newton_solver(EuclideanSpace, 200)
        .solve_with_default_line_search(vec![0.0, 0.0], &problem);

    assert!(full.converged, "full run should converge: {:?}", full);
    assert!(full.iters <= 200);
    assert!(
        full.cost < short.cost,
        "full run should reduce cost more: short={}, full={}",
        short.cost,
        full.cost
    );
}

#[test]
fn gauss_newton_behavior_stops_after_repeated_linear_solve_failure() {
    let solver = gauss_newton_solver(EuclideanSpace, 3);

    let residual_fn = |_x: &[f64], r: &mut [f64]| {
        r[0] = 1.0;
    };

    // Force A to contain NaN, which makes solve_linear_inplace fail.
    let jacobian_fn = |_x: &[f64], j: &mut [f64]| {
        j[0] = f64::NAN;
    };

    let project = |_x: &mut [f64]| {};
    let x0 = vec![0.0];
    let mut line_search = NoLineSearch;
    let res = solver.solve_with_fn(1, x0, residual_fn, jacobian_fn, project, &mut line_search);

    assert!(
        !res.converged,
        "solver should stop as non-converged: {:?}",
        res
    );
    assert_eq!(res.iters, 3, "solver should stop exactly at max_iters");
}

#[test]
fn gauss_newton_behavior_supports_custom_line_search_policy() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(EuclideanSpace, 200);
    let mut policy = FixedHalfStepSearch;
    let m = 2;

    let res = solver.solve_with_fn(
        m,
        vec![0.0, 0.0],
        |x, r| problem.residual_eval(x, r),
        |x, j| problem.jacobian_eval(x, j),
        |_x| {},
        &mut policy,
    );

    assert!(res.converged, "custom policy should converge: {:?}", res);
    assert!(res.r_norm < 1e-6, "residual too large: {}", res.r_norm);
}

#[test]
fn gauss_newton_behavior_supports_explicit_no_line_search_policy() {
    let problem = planar_two_link_problem();
    let solver = gauss_newton_solver(EuclideanSpace, 200);
    let x0 = vec![0.0, 0.0];
    let m = 2;

    let mut r0 = vec![0.0; m];
    problem.residual_eval(&x0, &mut r0);
    let initial_cost = 0.5 * r0.iter().map(|v| v * v).sum::<f64>();

    let mut line_search = NoLineSearch;
    let res = solver.solve_with_fn(
        m,
        x0,
        |x, r| problem.residual_eval(x, r),
        |x, j| problem.jacobian_eval(x, j),
        |_x| {},
        &mut line_search,
    );

    assert!(res.cost < initial_cost);
}

#[test]
fn gauss_newton_behavior_default_uses_euclidean_space() {
    let solver = GaussNewton::default();

    let residual_fn = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 2.0;
    };
    let jacobian_fn = |_x: &[f64], j: &mut [f64]| {
        j[0] = 1.0;
    };
    let project = |_x: &mut [f64]| {};

    let res =
        solver.solve_with_fn_default_line_search(1, vec![0.0], residual_fn, jacobian_fn, project);
    assert!(res.converged, "default solver should converge: {:?}", res);
    assert!((res.x[0] - 2.0).abs() < 1e-6);
}
