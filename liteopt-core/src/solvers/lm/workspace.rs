pub(super) struct LmWorkspace {
    pub(super) r: Vec<f64>,
    pub(super) j: Vec<f64>, // row-major (m x n)
    pub(super) a: Vec<f64>, // A = J J^T + lambda I, shape (m x m)
    pub(super) y: Vec<f64>,
    pub(super) dx: Vec<f64>,
    pub(super) x_trial: Vec<f64>,
    pub(super) r_trial: Vec<f64>,
    pub(super) tmp: Vec<f64>,
}

impl LmWorkspace {
    pub(super) fn new(m: usize, n: usize) -> Self {
        Self {
            r: vec![0.0f64; m],
            j: vec![0.0f64; m * n],
            a: vec![0.0f64; m * m],
            y: vec![0.0f64; m],
            dx: vec![0.0f64; n],
            x_trial: vec![0.0f64; n],
            r_trial: vec![0.0f64; m],
            tmp: vec![0.0f64; n],
        }
    }
}
