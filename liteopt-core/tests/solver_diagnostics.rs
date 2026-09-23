use liteopt::solvers::{gn::GaussNewton, lm::LevenbergMarquardt};
use std::cell::Cell;

#[test]
fn residual_and_jacobian_counts_match_calls() {
    let r_calls = Cell::new(0);
    let j_calls = Cell::new(0);
    let result = GaussNewton {
        max_iters: 1,
        step_size: 0.25,
        collect_trace: true,
        ..Default::default()
    }
    .solve_with_fn_default_line_search(
        1,
        vec![1.],
        |x, r| {
            r_calls.set(r_calls.get() + 1);
            r[0] = x[0];
        },
        |_, j| {
            j_calls.set(j_calls.get() + 1);
            j[0] = 1.;
        },
        |_| {},
    );
    assert_eq!(result.nfev, r_calls.get());
    assert_eq!(result.njev, j_calls.get());
    assert_eq!(result.grad_norm, Some(0.75));
    assert_eq!(
        (
            result.iters,
            result.n_attempts,
            result.n_accepted,
            result.n_retries
        ),
        (1, 1, 1, 0)
    );
    assert_eq!(result.n_ls_trials, 1);
    let rows = result.trace.unwrap();
    assert_eq!(rows[0].phase, "initial");
    assert_eq!(rows[1].cost, Some(0.5));
    assert_eq!(rows[1].accepted, Some(true));
    assert_eq!(rows.last().unwrap().phase, "final");
}

#[test]
fn lm_retries_and_damping_are_distinct_from_updates() {
    let result = LevenbergMarquardt {
        max_iters: 1,
        collect_trace: true,
        ..Default::default()
    }
    .solve_with_fn_default_line_search(
        1,
        vec![0.1],
        |x, r| r[0] = x[0] * x[0] - 1.,
        |x, j| j[0] = 2. * x[0],
        |_| {},
    );
    assert_eq!(
        (result.n_attempts, result.n_accepted, result.n_retries),
        (1, 0, 1)
    );
    let rows = result.trace.unwrap();
    let row = rows.iter().find(|r| r.phase == "search").unwrap();
    assert_eq!(row.lambda, Some(0.001));
    assert_eq!(row.lambda_next, Some(0.01));
    assert_eq!(row.ls_trials, Some(1));
    assert_eq!(row.accepted, Some(false));
}
