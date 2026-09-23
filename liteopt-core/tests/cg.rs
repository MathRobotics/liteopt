use liteopt::numerics::cg::{CgOptions, CgWorkspace};
use liteopt::solvers::gn::GaussNewton;
use liteopt::solvers::lm::LevenbergMarquardt;
use liteopt::solvers::{JacobianProducts, LinearSolver};

#[test]
fn cg_matches_diagonal_solution_and_reports_true_residual() {
    let g = [1., -6., 12.];
    let diagonal = [1., 2., 4.];
    let mut x = [0.; 3];
    let report = CgWorkspace::new(3).solve(&g, &mut x, CgOptions::default(), |v, out| {
        for i in 0..3 {
            out[i] = diagonal[i] * v[i];
        }
    });
    assert!(report.converged());
    for i in 0..3 {
        assert!((x[i] + g[i] / diagonal[i]).abs() < 1e-12);
    }
    let actual = (0..3)
        .map(|i| (diagonal[i] * x[i] + g[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert_eq!(report.residual_norm, actual);
}
#[test]
fn cg_exhaustion_breakdown_and_nonfinite_are_not_success() {
    let mut ws = CgWorkspace::new(2);
    let mut d = [0.; 2];
    let opts = CgOptions {
        max_iters: 1,
        ..Default::default()
    };
    assert_eq!(
        ws.solve(&[1., 1.], &mut d, opts, |v, o| {
            o[0] = v[0];
            o[1] = 4. * v[1];
        })
        .status,
        "linear_max_iters"
    );
    assert_eq!(
        ws.solve(&[1., 1.], &mut d, opts, |v, o| {
            o[0] = -v[0];
            o[1] = -v[1];
        })
        .status,
        "linear_breakdown"
    );
    assert_eq!(
        ws.solve(&[1., 1.], &mut d, opts, |_, o| o.fill(f64::NAN))
            .status,
        "linear_non_finite"
    );
}

#[test]
fn both_solvers_accept_products_without_dense_storage() {
    let products = || {
        JacobianProducts::new(
            |_x: &[f64], v: &[f64], out: &mut [f64]| {
                out[0] = v[0];
                out[1] = 2. * v[1];
            },
            |_x: &[f64], w: &[f64], out: &mut [f64]| {
                out[0] = w[0];
                out[1] = 2. * w[1];
            },
        )
    };
    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 1.;
        r[1] = 2. * x[1] + 4.;
    };
    let gn = GaussNewton {
        linear_solver: LinearSolver::Cg,
        ..Default::default()
    };
    let out =
        gn.solve_with_derivatives_default_line_search(2, vec![0.; 2], residual, products(), |_| {});
    assert!(out.converged);
    assert_eq!(out.njev, 0);
    assert!(out.n_linear_iters > 0);
    let lm = LevenbergMarquardt {
        linear_solver: LinearSolver::Cg,
        ..Default::default()
    };
    let out =
        lm.solve_with_derivatives_default_line_search(2, vec![0.; 2], residual, products(), |_| {});
    assert!(out.converged);
    assert_eq!(out.njev, 0);
    assert!((out.x[0] - 1.).abs() < 1e-6 && (out.x[1] + 2.).abs() < 1e-6);
}

#[test]
fn product_input_with_direct_backend_is_rejected() {
    let gn = GaussNewton::default();
    let out = gn.solve_with_derivatives_default_line_search(
        1,
        vec![0.],
        |_, r| r[0] = 1.,
        JacobianProducts::new(
            |_: &[f64], _: &[f64], _: &mut [f64]| panic!("must not run"),
            |_: &[f64], _: &[f64], _: &mut [f64]| panic!("must not run"),
        ),
        |_| {},
    );
    assert_eq!(out.status, "invalid_options");
}

#[test]
fn convergence_uses_true_residual_not_only_the_recurrence() {
    let mut calls = 0;
    let report = CgWorkspace::new(1).solve(
        &[1.],
        &mut [0.],
        CgOptions {
            max_iters: 1,
            ..Default::default()
        },
        |v, out| {
            calls += 1;
            out[0] = if calls == 1 { v[0] } else { 2. * v[0] };
        },
    );
    // Deliberately inconsistent operator: recurrence says zero, actual residual does not.
    assert_eq!(report.status, "linear_max_iters");
    assert_eq!(report.residual_norm, 1.);
}
