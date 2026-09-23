use liteopt::numerics::qr::{QrError, QrWorkspace};

#[test]
fn scaled_and_pivoted_system_recovers_known_solution() {
    // Non-orthogonal, differently scaled columns; forces a trailing-column pivot.
    let a = [1e-8, 1., 1e8, 2e-8, 1., 0., -1e-8, 2., 1e8, 3e-8, -1., 2e8];
    let target = [1e8, 2., 1e-8];
    let r: Vec<_> = a
        .chunks(3)
        .map(|row| -row.iter().zip(target).map(|(a, x)| a * x).sum::<f64>())
        .collect();
    let mut x = [0.; 3];
    QrWorkspace::new(4, 3)
        .solve(&a, &r, 4, 3, 0., &mut x)
        .unwrap();
    for i in 0..3 {
        assert!((x[i] - target[i]).abs() <= 1e-12 * target[i].abs());
    }
}

#[test]
fn near_dependent_columns_do_not_require_normal_equations() {
    let e = 1e-8;
    let a = [1., 1. + e, 1., 1. - e, 2., 2. + e, 2., 2. - e];
    let r: Vec<_> = a.chunks(2).map(|v| -(v[0] + 2. * v[1])).collect();
    let mut x = [0.; 2];
    QrWorkspace::new(4, 2)
        .solve(&a, &r, 4, 2, 0., &mut x)
        .unwrap();
    assert!((x[0] - 1.).abs() < 1e-6 && (x[1] - 2.).abs() < 1e-6);
}

#[test]
fn rank_failure_does_not_mutate_output_and_workspace_is_reusable() {
    let mut qr = QrWorkspace::new(2, 2);
    let mut x = [42.; 2];
    assert_eq!(
        qr.solve(&[1., 1., 2., 2.], &[-1., -2.], 2, 2, 0., &mut x),
        Err(QrError::RankDeficient)
    );
    assert_eq!(x, [42.; 2]);
    qr.solve(&[0., 1., 1., 0.], &[-2., -1.], 2, 2, 0., &mut x)
        .unwrap();
    assert!((x[0] - 1.).abs() < 1e-14 && (x[1] - 2.).abs() < 1e-14);
    assert_eq!(qr.last_error, None);
}

#[test]
fn augmented_system_handles_underdetermined_and_rank_deficient_jacobians() {
    let mut x = [0.; 2];
    QrWorkspace::new(3, 2)
        .solve(&[1., 1.], &[-3.], 1, 2, 1., &mut x)
        .unwrap();
    assert!((x[0] - 1.).abs() < 1e-14 && (x[1] - 1.).abs() < 1e-14);
    assert_eq!(
        QrWorkspace::new(1, 2).solve(&[1., 1.], &[-3.], 1, 2, 0., &mut x),
        Err(QrError::InvalidShape)
    );
}

#[test]
fn extreme_scale_and_invalid_values() {
    let mut x = [0.];
    for scale in [1e-200, 1e200] {
        QrWorkspace::new(1, 1)
            .solve(&[scale], &[-scale], 1, 1, 0., &mut x)
            .unwrap();
        assert!((x[0] - 1.).abs() < 1e-14);
    }
    assert_eq!(
        QrWorkspace::new(1, 1).solve(&[f64::NAN], &[-1.], 1, 1, 0., &mut x),
        Err(QrError::NonFinite)
    );
}
