use liteopt::{manifolds::space::Space, solvers::gn::GaussNewton};

#[derive(Clone, Copy, Debug, Default)]
struct MyManifold;

fn wrap_angle(theta: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (theta + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

impl Space for MyManifold {
    type Point = Vec<f64>;
    type Tangent = Vec<f64>;

    fn zero_like(&self, x: &Self::Point) -> Self::Point {
        vec![0.0; x.len()]
    }

    fn norm(&self, v: &Self::Point) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    fn scale_into(&self, out: &mut Self::Tangent, v: &Self::Tangent, alpha: f64) {
        out.resize(v.len(), 0.0);
        for i in 0..v.len() {
            out[i] = alpha * v[i];
        }
    }

    fn add_into(&self, out: &mut Self::Point, x: &Self::Point, v: &Self::Tangent) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(x[i] + v[i]);
        }
    }

    fn difference_into(&self, out: &mut Self::Tangent, x: &Self::Point, y: &Self::Point) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(y[i] - x[i]);
        }
    }

    fn zero_tangent_like(&self, x: &Self::Point) -> Self::Tangent {
        vec![0.0; x.len()]
    }

    fn tangent_norm(&self, v: &Self::Tangent) -> f64 {
        self.norm(v)
    }

    fn retract_into(
        &self,
        out: &mut Self::Point,
        x: &Self::Point,
        direction: &Self::Tangent,
        alpha: f64,
        _tmp: &mut Self::Tangent,
    ) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = wrap_angle(x[i] + alpha * direction[i]);
        }
    }
}

fn main() {
    let mut solver = GaussNewton::with_space(MyManifold);
    solver.lambda = 1e-3;
    solver.max_iters = 20;
    solver.tol_r = 1e-12;
    solver.tol_dq = 1e-12;

    let target_angle = 2.8;
    let x0 = vec![3.0 * std::f64::consts::PI];

    let result = solver.solve_with_fn_default_line_search(
        1,
        x0,
        |x, r| {
            r[0] = wrap_angle(x[0] - target_angle);
        },
        |_x, j| {
            j[0] = 1.0;
        },
        |x| {
            x[0] = wrap_angle(x[0]);
        },
    );

    println!(
        "converged={} theta={:.6} residual_norm={:.3e}",
        result.converged, result.x[0], result.r_norm
    );
}
