use liteopt::solvers::{
    gn::{GaussNewton, LineSearchContext, LineSearchPolicy, LineSearchResult},
    lm::LevenbergMarquardt,
    SolverTraceRecord,
};

struct FixedDecision(bool);

impl LineSearchPolicy for FixedDecision {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        _eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        LineSearchResult {
            accepted: self.0,
            alpha: ctx.alpha0,
        }
    }
}

fn run<R, J>(
    lm: bool,
    residual: R,
    jacobian: J,
    accept: bool,
) -> (bool, Vec<f64>, Vec<SolverTraceRecord>)
where
    R: FnMut(&[f64], &mut [f64]),
    J: FnMut(&[f64], &mut [f64]),
{
    let mut policy = FixedDecision(accept);
    if lm {
        let solver = LevenbergMarquardt {
            collect_trace: true,
            ..Default::default()
        };
        let result = solver.solve_with_fn(1, vec![0.0], residual, jacobian, |_| {}, &mut policy);
        (result.converged, result.x, result.trace.unwrap())
    } else {
        let solver = GaussNewton {
            collect_trace: true,
            ..Default::default()
        };
        let result = solver.solve_with_fn(1, vec![0.0], residual, jacobian, |_| {}, &mut policy);
        (result.converged, result.x, result.trace.unwrap())
    }
}

#[test]
fn rejected_steps_do_not_converge_when_damping_shrinks_the_direction() {
    for lm in [false, true] {
        let (converged, x, trace) = run(lm, |x, r| r[0] = x[0] - 1.0, |_, j| j[0] = 1.0, false);
        assert!(!converged);
        assert_eq!(x, vec![0.0]);
        assert_eq!(
            trace.last().unwrap().note,
            Some(if lm { "stalled" } else { "rejected" })
        );
    }
}

#[test]
fn invalid_accepted_trials_do_not_converge_when_damping_shrinks_the_direction() {
    for lm in [false, true] {
        let (converged, x, trace) = run(
            lm,
            |x, r| r[0] = if x[0] == 0.0 { -1.0 } else { f64::INFINITY },
            |_, j| j[0] = 1.0,
            true,
        );
        assert!(!converged);
        assert_eq!(x, vec![0.0]);
        assert!(trace
            .iter()
            .any(|row| row.note == Some("accepted_step_invalid")));
        assert_eq!(
            trace.last().unwrap().note,
            Some(if lm {
                "stalled"
            } else {
                "accepted_step_invalid"
            })
        );
    }
}

#[test]
fn nonfinite_jacobian_stops_without_damping_retries() {
    for lm in [false, true] {
        let mut calls = 0;
        let (converged, x, trace) = run(
            lm,
            |x, r| r[0] = x[0] - 1.0,
            |_, j| {
                calls += 1;
                j[0] = if calls <= 9 { f64::NAN } else { 1.0 };
            },
            true,
        );
        assert!(!converged);
        assert_eq!(x, vec![0.0]);
        assert_eq!(trace.last().unwrap().note, Some("non_finite_jacobian"));
    }
}
