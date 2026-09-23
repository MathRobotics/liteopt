use liteopt::solvers::{gn::GaussNewton, lm::LevenbergMarquardt};

#[test]
fn final_update_and_zero_budget_are_checked() {
    for budget in [0, 1] {
        let gn = GaussNewton {
            max_iters: budget,
            ..Default::default()
        };
        let lm = LevenbergMarquardt {
            max_iters: budget,
            lambda: 0.0,
            ..Default::default()
        };
        let g = gn.solve_with_fn_default_line_search(
            1,
            vec![1.],
            |x, r| r[0] = x[0],
            |_, j| j[0] = 1.,
            |_| {},
        );
        let l = lm.solve_with_fn_default_line_search(
            1,
            vec![1.],
            |x, r| r[0] = x[0],
            |_, j| j[0] = 1.,
            |_| {},
        );
        assert_eq!(g.converged, budget == 1);
        assert_eq!(l.converged, budget == 1);
        assert_eq!(g.r_norm, g.x[0].abs());
        assert_eq!(l.r_norm, l.x[0].abs());
    }
}

#[test]
fn stationary_nonzero_residual_needs_no_linear_solve() {
    let gn = GaussNewton {
        max_iters: 0,
        ..Default::default()
    };
    let lm = LevenbergMarquardt {
        max_iters: 0,
        ..Default::default()
    };
    let g = gn.solve_with_fn_default_line_search(
        2,
        vec![0.],
        |_, r| r.copy_from_slice(&[-1., 1.]),
        |_, j| j.fill(1.),
        |_| {},
    );
    let l = lm.solve_with_fn_default_line_search(
        2,
        vec![0.],
        |_, r| r.copy_from_slice(&[-1., 1.]),
        |_, j| j.fill(1.),
        |_| {},
    );
    assert_eq!(g.status, "converged_grad");
    assert_eq!(l.status, "converged_grad");
}

#[test]
fn rust_invalid_configuration_is_not_silently_corrected() {
    let gn = GaussNewton {
        ls_beta: 0.,
        ..Default::default()
    };
    let lm = LevenbergMarquardt {
        lambda_up: f64::INFINITY,
        ..Default::default()
    };
    let g = gn.solve_with_fn_default_line_search(
        1,
        vec![0.],
        |_, _| panic!("must validate first"),
        |_, _| {},
        |_| {},
    );
    let l = lm.solve_with_fn_default_line_search(
        1,
        vec![0.],
        |_, _| panic!("must validate first"),
        |_, _| {},
        |_| {},
    );
    assert_eq!(g.status, "invalid_options");
    assert_eq!(l.status, "invalid_options");
}

#[test]
fn nonfinite_initial_point_and_jacobian_stop_immediately() {
    let g = GaussNewton::default().solve_with_fn_default_line_search(
        1,
        vec![f64::NAN],
        |_, _| panic!("invalid point"),
        |_, _| {},
        |_| {},
    );
    assert_eq!(g.status, "non_finite_initial");
    let l = LevenbergMarquardt::default().solve_with_fn_default_line_search(
        1,
        vec![0.],
        |_, r| r[0] = 1.,
        |_, j| j[0] = f64::NAN,
        |_| {},
    );
    assert_eq!(l.status, "non_finite_jacobian");
    assert_eq!(l.iters, 0);
}
