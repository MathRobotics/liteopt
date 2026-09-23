use super::types::LevenbergMarquardtLinearSystem;
use crate::numerics::cg::{CgReport, CgWorkspace, LinearSolver};
use crate::numerics::qr::QrWorkspace;
pub(super) struct LmWorkspace {
    pub(super) cg: Option<CgWorkspace>,
    pub(super) linear_report: Option<CgReport>,
    pub(super) jv: Vec<f64>,
    pub(super) qr: Option<QrWorkspace>,
    pub(super) r: Vec<f64>,
    pub(super) j: Vec<f64>, // row-major (m x n)
    pub(super) a: Vec<f64>, // A = J J^T + lambda I, shape (m x m)
    pub(super) y: Vec<f64>,
    pub(super) dx: Vec<f64>,
    pub(super) g: Vec<f64>,
    pub(super) x_trial: Vec<f64>,
    pub(super) r_trial: Vec<f64>,
    pub(super) tmp: Vec<f64>,
}

impl LmWorkspace {
    pub(super) fn new(
        m: usize,
        n: usize,
        method: LevenbergMarquardtLinearSystem,
        backend: LinearSolver,
        matrix_free: bool,
    ) -> Self {
        Self {
            cg: (backend == LinearSolver::Cg).then(|| CgWorkspace::new(n)),
            linear_report: None,
            jv: vec![0.; m],
            qr: (backend == LinearSolver::Direct && method == LevenbergMarquardtLinearSystem::Qr)
                .then(|| QrWorkspace::new(m + n, n)),
            r: vec![0.0f64; m],
            j: vec![0.0f64; if matrix_free { 0 } else { m * n }],
            a: vec![
                0.0f64;
                if backend == LinearSolver::Direct
                    && method == LevenbergMarquardtLinearSystem::LeftJjT
                {
                    m * m
                } else {
                    0
                }
            ],
            y: vec![0.0f64; m],
            dx: vec![0.0f64; n],
            g: vec![0.0f64; n],
            x_trial: vec![0.0f64; n],
            r_trial: vec![0.0f64; m],
            tmp: vec![0.0f64; n],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_free_workspace_has_no_dense_matrices() {
        let ws = LmWorkspace::new(
            20_000,
            20_000,
            LevenbergMarquardtLinearSystem::Qr,
            LinearSolver::Cg,
            true,
        );
        assert!(ws.j.is_empty() && ws.a.is_empty() && ws.qr.is_none());
        assert!(ws.cg.is_some());
    }
}
