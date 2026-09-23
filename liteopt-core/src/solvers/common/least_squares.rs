use crate::numerics::linalg::{
    dot, jj_t_plus_lambda, jt_j_plus_lambda, jt_mul_vec, norm2, solve_linear_inplace,
};

pub(crate) fn residual_cost(r: &[f64]) -> f64 {
    0.5 * dot(r, r)
}

pub(crate) fn residual_norm(r: &[f64]) -> f64 {
    norm2(r)
}

pub(crate) fn solve_left_jjt_direction(
    j: &[f64],
    r: &[f64],
    m: usize,
    n: usize,
    lambda: f64,
    a: &mut [f64],
    y: &mut [f64],
    dx: &mut [f64],
) -> bool {
    jj_t_plus_lambda(j, m, n, lambda, a);

    y.copy_from_slice(r);
    if !solve_linear_inplace(a, y, m) {
        return false;
    }

    jt_mul_vec(j, m, n, y, dx);
    negate_in_place(dx);
    true
}

pub(crate) fn solve_normal_jtj_direction(
    j: &[f64],
    r: &[f64],
    m: usize,
    n: usize,
    lambda: f64,
    a: &mut [f64],
    dx: &mut [f64],
) -> bool {
    jt_j_plus_lambda(j, m, n, lambda, a);

    jt_mul_vec(j, m, n, r, dx);
    negate_in_place(dx);
    solve_linear_inplace(a, dx, n)
}

pub(crate) fn commit_trial_state(
    x: &mut [f64],
    r: &mut [f64],
    cost: &mut f64,
    r_norm: &mut f64,
    x_trial: &[f64],
    r_trial: &[f64],
    cost_trial: f64,
) {
    x.copy_from_slice(x_trial);
    r.copy_from_slice(r_trial);
    *cost = cost_trial;
    *r_norm = residual_norm(r);
}

fn negate_in_place(v: &mut [f64]) {
    for vi in v {
        *vi = -*vi;
    }
}
