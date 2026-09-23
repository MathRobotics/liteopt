use liteopt::solvers::gn::{
    ArmijoBacktracking, CostDecrease, LineSearchContext, LineSearchPolicy, NoLineSearch,
    StrictDecreaseBacktracking,
};

fn context() -> LineSearchContext {
    LineSearchContext {
        iter: 0,
        alpha0: 1.0,
        cost0: 1.0,
        dphi0: Some(-1.0),
        dx_norm: 1.0,
        lambda: 0.0,
    }
}

#[test]
fn policies_never_accept_nonfinite_costs() {
    let mut policies: Vec<Box<dyn LineSearchPolicy>> = vec![
        Box::new(NoLineSearch),
        Box::new(CostDecrease),
        Box::new(ArmijoBacktracking::default()),
        Box::new(StrictDecreaseBacktracking::default()),
    ];
    for policy in &mut policies {
        for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!policy.search(&context(), &mut |_| Some(cost)).accepted);
        }
    }
}

#[test]
fn armijo_requires_a_finite_descent_slope() {
    for slope in [
        None,
        Some(0.0),
        Some(1.0),
        Some(f64::NAN),
        Some(f64::NEG_INFINITY),
    ] {
        let mut ctx = context();
        ctx.dphi0 = slope;
        let result =
            ArmijoBacktracking::default().search(&ctx, &mut |_| panic!("invalid slope evaluated"));
        assert!(!result.accepted);
    }
}

#[test]
fn invalid_backtracking_configuration_does_not_evaluate_candidates() {
    for beta in [0.0, 1.0, -0.5, f64::NAN] {
        assert!(
            !ArmijoBacktracking::new(beta, 20, 1e-4)
                .search(&context(), &mut |_| panic!("invalid beta evaluated"))
                .accepted
        );
        assert!(
            !StrictDecreaseBacktracking::new(beta, 1e-8, 20)
                .search(&context(), &mut |_| panic!("invalid beta evaluated"))
                .accepted
        );
    }
}

#[test]
fn armijo_recovers_from_nonfinite_trial() {
    let mut attempts = Vec::new();
    let result = ArmijoBacktracking::default().search(&context(), &mut |alpha| {
        attempts.push(alpha);
        Some(if alpha == 1.0 { f64::NEG_INFINITY } else { 0.5 })
    });
    assert!(result.accepted);
    assert_eq!(result.alpha, 0.5);
    assert_eq!(attempts, vec![1.0, 0.5]);
}
