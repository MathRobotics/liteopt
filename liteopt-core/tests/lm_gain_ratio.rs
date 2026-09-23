use liteopt::solvers::lm::{LevenbergMarquardt, LevenbergMarquardtDampingUpdate};
#[test]
fn linear_model_has_unit_gain_with_partial_step() {
    let result = LevenbergMarquardt {
        damping_update: LevenbergMarquardtDampingUpdate::GainRatio,
        lambda: 1.,
        step_size: 0.25,
        max_iters: 1,
        collect_trace: true,
        ..Default::default()
    }
    .solve_with_fn_default_line_search(
        1,
        vec![0.],
        |x, r| r[0] = x[0] - 1.,
        |_, j| j[0] = 1.,
        |_| {},
    );
    let row = result
        .trace
        .unwrap()
        .into_iter()
        .find(|r| r.phase == "search")
        .unwrap();
    assert!((row.gain_ratio.unwrap() - 1.).abs() < 1e-12);
    assert_eq!(row.lambda_next, Some(0.5));
    assert_eq!(result.n_accepted, 1);
}
#[test]
fn invalid_bounds_are_rejected_before_callbacks() {
    let result = LevenbergMarquardt {
        lambda_min: 2.,
        lambda_max: 1.,
        ..Default::default()
    }
    .solve_with_fn_default_line_search(
        1,
        vec![0.],
        |_, _| panic!("invalid options"),
        |_, _| {},
        |_| {},
    );
    assert_eq!(result.status, "invalid_options");
}
