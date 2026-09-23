use liteopt::solvers::{
    gn::{GaussNewton, GaussNewtonLinearSystem},
    lm::{LevenbergMarquardt, LevenbergMarquardtLineSearchMethod},
};

#[test]
fn gn_default_is_undamped_normal_equations() {
    let solver = GaussNewton {
        max_iters: 2,
        collect_trace: true,
        ..Default::default()
    };
    assert_eq!(solver.linear_system, GaussNewtonLinearSystem::NormalJtJ);
    let result = solver.solve_with_fn_default_line_search(
        2,
        vec![0.0],
        |x, r| {
            r[0] = x[0] - 1.0;
            r[1] = 2.0 * x[0] - 3.0;
        },
        |_, j| {
            j[0] = 1.0;
            j[1] = 2.0;
        },
        |_| {},
    );
    assert!(result.converged);
    assert!((result.x[0] - 1.4).abs() < 1e-12);
    assert!(result.trace.unwrap().iter().all(|row| row.lambda.is_none()));
}

#[test]
fn gn_singular_system_fails_without_regularization() {
    let solver = GaussNewton {
        collect_trace: true,
        ..Default::default()
    };
    let result = solver.solve_with_fn_default_line_search(
        1,
        vec![0.0, 0.0],
        |x, r| r[0] = x[0] + x[1] - 1.0,
        |_, j| j.fill(1.0),
        |_| {},
    );
    assert!(!result.converged);
    assert_eq!(result.x, vec![0.0, 0.0]);
    assert_eq!(
        result.trace.unwrap().last().unwrap().note,
        Some("linear_solve_failed")
    );
}

#[test]
fn gn_explicit_left_system_solves_full_row_rank_problem() {
    let solver = GaussNewton {
        linear_system: GaussNewtonLinearSystem::LeftJjT,
        ..Default::default()
    };
    let result = solver.solve_with_fn_default_line_search(
        1,
        vec![0.0, 0.0],
        |x, r| r[0] = x[0] + x[1] - 1.0,
        |_, j| j.fill(1.0),
        |_| {},
    );
    assert!(result.converged);
    assert_eq!(result.x, vec![0.5, 0.5]);
}

#[test]
fn lm_configured_backtracking_accepts_a_shorter_step() {
    for method in [
        LevenbergMarquardtLineSearchMethod::Armijo,
        LevenbergMarquardtLineSearchMethod::StrictDecrease,
    ] {
        let solver = LevenbergMarquardt {
            line_search_method: method,
            max_iters: 1,
            collect_trace: true,
            ..Default::default()
        };
        let result = solver.solve_with_fn_default_line_search(
            1,
            vec![0.1],
            |x, r| r[0] = x[0] * x[0] - 1.0,
            |x, j| j[0] = 2.0 * x[0],
            |_| {},
        );
        assert!(result.cost < 0.49005);
        let trace = result.trace.unwrap();
        let row = trace
            .iter()
            .find(|row| row.note == Some("accepted"))
            .unwrap();
        assert!(row.alpha.unwrap() > 0.0 && row.alpha.unwrap() < 1.0);
    }
}
