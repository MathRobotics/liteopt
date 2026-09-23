use liteopt::numerics::cg::{CgOptions, CgWorkspace};

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
