//! liteopt: A tiny, lightweight optimization toolbox
//!
//! - `Space`: an abstraction of vector spaces
//! - `EuclideanSpace` (`Vec<f64>`): its concrete implementation
//! - `Objective`: a generic objective function interface
//! - `GradientDescent`: a gradient descent solver
//!
//! Start with simple optimization on R^n.

/// Trait that represents an abstract "space".
///
/// MVP implements only EuclideanSpace (Vec<f64>), leaving room for
/// future manifolds such as SO(3) or SE(3).
pub trait Space {
    /// Type representing points/vectors on the space.
    type Point: Clone;

    /// Return a zero-like vector with the same shape as x.
    fn zero_like(&self, x: &Self::Point) -> Self::Point;

    /// Vector norm.
    fn norm(&self, v: &Self::Point) -> f64;

    /// Return v scaled by the scalar alpha.
    fn scale(&self, v: &Self::Point, alpha: f64) -> Self::Point;

    /// Compute x + v (result is a point).
    fn add(&self, x: &Self::Point, v: &Self::Point) -> Self::Point;

    /// Compute y - x (result is a vector).
    fn difference(&self, x: &Self::Point, y: &Self::Point) -> Self::Point;

    /// Return the point reached by moving from x along direction by step alpha.
    ///
    /// By default this matches the Euclidean update
    ///   x_{k+1} = x_k + alpha * direction
    /// in Euclidean space.
    fn retract(&self, x: &Self::Point, direction: &Self::Point, alpha: f64) -> Self::Point {
        let step = self.scale(direction, alpha);
        self.add(x, &step)
    }
}

/// Simple Euclidean space representing R^n as Vec<f64>.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanSpace;

impl Space for EuclideanSpace {
    type Point = Vec<f64>;

    fn zero_like(&self, x: &Self::Point) -> Self::Point {
        vec![0.0; x.len()]
    }

    fn norm(&self, v: &Self::Point) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    fn scale(&self, v: &Self::Point, alpha: f64) -> Self::Point {
        v.iter().map(|vi| alpha * vi).collect()
    }

    fn add(&self, x: &Self::Point, v: &Self::Point) -> Self::Point {
        x.iter().zip(v.iter()).map(|(xi, vi)| xi + vi).collect()
    }

    fn difference(&self, x: &Self::Point, y: &Self::Point) -> Self::Point {
        y.iter().zip(x.iter()).map(|(yi, xi)| yi - xi).collect()
    }
}

/// Objective function to be minimized.
///
/// - `S::Point` represents points on the space
/// - In `gradient` the user computes the gradient and writes into the buffer
pub trait Objective<S: Space> {
    /// Function value f(x) at x.
    fn value(&self, x: &S::Point) -> f64;

    /// Write the gradient ∇f(x) at x into grad.
    ///
    /// grad is assumed to be pre-initialized, e.g., via zero_like(x).
    fn gradient(&self, x: &S::Point, grad: &mut S::Point);
}

/// Configuration for gradient descent.
#[derive(Clone, Debug)]
pub struct GradientDescent<S: Space> {
    /// Space to operate on (MVP can fix this to EuclideanSpace).
    pub space: S,
    /// Learning rate / step size.
    pub step_size: f64,
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Considered converged when the gradient norm falls below this threshold.
    pub tol_grad: f64,
}

/// Struct that holds the optimization result.
#[derive(Clone, Debug)]
pub struct OptimizeResult<P> {
    pub x: P,
    pub f: f64,
    pub iters: usize,
    pub grad_norm: f64,
    pub converged: bool,
}

impl<S: Space> GradientDescent<S> {
    pub fn minimize<O>(&self, obj: &O, mut x: S::Point) -> OptimizeResult<S::Point>
    where
        O: Objective<S>,
    {
        let mut grad = self.space.zero_like(&x);

        for k in 0..self.max_iters {
            // Compute gradient.
            obj.gradient(&x, &mut grad);

            let grad_norm = self.space.norm(&grad);
            if grad_norm < self.tol_grad {
                let f = obj.value(&x);
                return OptimizeResult {
                    x,
                    f,
                    iters: k,
                    grad_norm,
                    converged: true,
                };
            }

            // x_{k+1} = Retr_x( - step_size * grad )
            // direction = -grad
            let direction = self.space.scale(&grad, -1.0);
            x = self.space.retract(&x, &direction, self.step_size);
        }

        let f = obj.value(&x);
        let grad_norm = self.space.norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
        }
    }

    /// ★ Minimize using user-provided value and gradient functions.
    pub fn minimize_with_fn<F, G>(
        &self,
        mut x: S::Point,
        value_fn: F,
        grad_fn: G,
    ) -> OptimizeResult<S::Point>
    where
        F: Fn(&S::Point) -> f64,
        G: Fn(&S::Point, &mut S::Point),
    {
        let mut grad = self.space.zero_like(&x);

        for k in 0..self.max_iters {
            // call the user-provided gradient function
            grad_fn(&x, &mut grad);

            let grad_norm = self.space.norm(&grad);
            if grad_norm < self.tol_grad {
                let f = value_fn(&x);
                return OptimizeResult {
                    x,
                    f,
                    iters: k,
                    grad_norm,
                    converged: true,
                };
            }

            let direction = self.space.scale(&grad, -1.0);
            x = self.space.retract(&x, &direction, self.step_size);
        }

        let f = value_fn(&x);
        let grad_norm = self.space.norm(&grad);
        OptimizeResult {
            x,
            f,
            iters: self.max_iters,
            grad_norm,
            converged: false,
        }
    }
}

// --------------------------------------
// NonlinearLeastSquares
// --------------------------------------
#[derive(Clone, Debug)]
pub struct LeastSquaresResult<P> {
    pub x: P,
    pub cost: f64,        // 0.5 * ||r||^2
    pub iters: usize,
    pub r_norm: f64,
    pub dx_norm: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct NonlinearLeastSquares<S: Space<Point = Vec<f64>>> {
    pub space: S,
    pub lambda: f64,          // damping
    pub step_scale: f64,      // alpha in (0,1]
    pub max_iters: usize,
    pub tol_r: f64,           // stop if ||r|| < tol_r
    pub tol_dq: f64,          // stop if ||dq|| < tol_dq
    pub line_search: bool,    // simple backtracking on cost
    pub ls_beta: f64,         // e.g. 0.5
    pub ls_max_steps: usize,  // e.g. 20
}

impl<S: Space<Point = Vec<f64>>> NonlinearLeastSquares<S> {
    /// Solve the nonlinear least squares problem.
    ///
    /// - m: residual dimension (e.g. 2 for 2D position, 3 for 3D position, 6 for SE(3) pose error)
    /// - q0: initial guess (len = n)
    /// - residual_fn(q, r): fill r (len = m)
    /// - jacobian_fn(q, J): fill J (len = m*n), row-major (i-th row, k-th col => J[i*n+k])
    /// - project(q): optional projection (joint limits etc). If not needed, pass |q| {}
    pub fn solve_with_fn<R, JF, P>(
        &self,
        m: usize,
        mut x: Vec<f64>,
        mut residual_fn: R,
        mut jacobian_fn: JF,
        mut project: P,
    ) -> LeastSquaresResult<Vec<f64>>
    where
        R: FnMut(&[f64], &mut [f64]),
        JF: FnMut(&[f64], &mut [f64]),
        P: FnMut(&mut [f64]),
    {
        let n = x.len();
        assert!(m > 0 && n > 0);

        let mut r = vec![0.0f64; m];
        let mut j = vec![0.0f64; m * n];  // row-major
        let mut a = vec![0.0f64; m * m];  // A = J J^T + lambda I
        let mut y = vec![0.0f64; m];
        let mut dx = vec![0.0f64; n];

        // for line search
        let mut x_trial = vec![0.0f64; n];
        let mut r_trial = vec![0.0f64; m];

        // initial
        residual_fn(&x, &mut r);
        let mut cost = 0.5 * dot(&r, &r);
        let mut r_norm = norm2(&r);

        for it in 0..self.max_iters {
            if r_norm <= self.tol_r {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: 0.0,
                    converged: true,
                };
            }

            // J(q)
            jacobian_fn(&x, &mut j);

            // A = J J^T + lambda I
            jj_t_plus_lambda(&j, m, n, self.lambda, &mut a);

            // y = A^{-1} r
            y.copy_from_slice(&r);
            let ok = solve_linear_inplace(&mut a, &mut y, m);
            if !ok {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm: f64::NAN,
                    converged: false,
                };
            }

            // dx = - J^T y
            jt_mul_vec(&j, m, n, &y, &mut dx);
            for v in dx.iter_mut() {
                *v = -*v;
            }

            let dx_norm = norm2(&dx);
            if dx_norm <= self.tol_dq {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: true,
                };
            }

            let mut alpha = self.step_scale.clamp(0.0, 1.0);
            if alpha == 0.0 {
                return LeastSquaresResult {
                    x,
                    cost,
                    iters: it,
                    r_norm,
                    dx_norm,
                    converged: false,
                };
            }

            if self.line_search {
                let mut accepted = false;
                for _ in 0..self.ls_max_steps {
                    // q_trial = Retr_q(alpha * dq)
                    x_trial = self.space.retract(&x, &dx, alpha);
                    project(&mut x_trial);

                    residual_fn(&x_trial, &mut r_trial);
                    let cost_trial = 0.5 * dot(&r_trial, &r_trial);

                    if cost_trial.is_finite() && cost_trial <= cost {
                        x.copy_from_slice(&x_trial);
                        r.copy_from_slice(&r_trial);
                        cost = cost_trial;
                        r_norm = norm2(&r);
                        accepted = true;
                        break;
                    }
                    alpha *= self.ls_beta;
                }
                if !accepted {
                    return LeastSquaresResult {
                        x,
                        cost,
                        iters: it + 1,
                        r_norm,
                        dx_norm,
                        converged: false,
                    };
                }
            } else {
                // x = Retr_x(alpha * dx)
                x = self.space.retract(&x, &dx, alpha);
                project(&mut x);

                residual_fn(&x, &mut r);
                cost = 0.5 * dot(&r, &r);
                r_norm = norm2(&r);
            }
        }

        LeastSquaresResult {
            x,
            cost,
            iters: self.max_iters,
            r_norm,
            dx_norm: f64::NAN,
            converged: false,
        }
    }
}

// ----------------- helpers (dependency-free) -----------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// A = J J^T + lambda I, J: (m x n) row-major
fn jj_t_plus_lambda(j: &[f64], m: usize, n: usize, lambda: f64, a: &mut [f64]) {
    a.fill(0.0);
    for i in 0..m {
        for k in 0..n {
            let jik = j[i * n + k];
            for jrow in 0..m {
                a[i * m + jrow] += jik * j[jrow * n + k];
            }
        }
    }
    for i in 0..m {
        a[i * m + i] += lambda;
    }
}

/// out = J^T v
fn jt_mul_vec(j: &[f64], m: usize, n: usize, v: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..m {
        let vi = v[i];
        for k in 0..n {
            out[k] += j[i * n + k] * vi;
        }
    }
}

/// Gaussian elimination with partial pivoting (in-place).
/// Solves A x = b, overwriting A and b (b becomes x).
fn solve_linear_inplace(a: &mut [f64], b: &mut [f64], dim: usize) -> bool {
    const EPS: f64 = 1e-12;

    for col in 0..dim {
        // pivot
        let mut piv = col;
        let mut max_abs = a[col * dim + col].abs();
        for row in (col + 1)..dim {
            let v = a[row * dim + col].abs();
            if v > max_abs {
                max_abs = v;
                piv = row;
            }
        }
        if max_abs < EPS || !max_abs.is_finite() {
            return false;
        }

        if piv != col {
            for k in col..dim {
                a.swap(col * dim + k, piv * dim + k);
            }
            b.swap(col, piv);
        }

        let diag = a[col * dim + col];
        for row in (col + 1)..dim {
            let f = a[row * dim + col] / diag;
            a[row * dim + col] = 0.0;
            for k in (col + 1)..dim {
                a[row * dim + k] -= f * a[col * dim + k];
            }
            b[row] -= f * b[col];
        }
    }

    for i_rev in 0..dim {
        let i = dim - 1 - i_rev;
        let mut s = b[i];
        for k in (i + 1)..dim {
            s -= a[i * dim + k] * b[k];
        }
        let diag = a[i * dim + i];
        if diag.abs() < EPS || !diag.is_finite() {
            return false;
        }
        b[i] = s / diag;
    }
    true
}

/// Tests and examples: quadratic / Rosenbrock
///

/// Example quadratic of the form f(x) = 0.5 * x^T A x - b^T x.
pub struct Quadratic {
    pub a: f64,
    pub b: f64,
}

impl Objective<EuclideanSpace> for Quadratic {
    fn value(&self, x: &Vec<f64>) -> f64 {
        // Treat as 1D for simplicity:
        // f(x) = 0.5 * a * x^2 - b * x
        let x0 = x[0];
        0.5 * self.a * x0 * x0 - self.b * x0
    }

    fn gradient(&self, x: &Vec<f64>, grad: &mut Vec<f64>) {
        let x0 = x[0];
        // Gradient: df/dx = a * x - b
        grad[0] = self.a * x0 - self.b;
    }
}

/// Example 2D Rosenbrock function.
/// f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
pub struct Rosenbrock;

impl Objective<EuclideanSpace> for Rosenbrock {
    fn value(&self, x: &Vec<f64>) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
    }

    fn gradient(&self, x: &Vec<f64>, grad: &mut Vec<f64>) {
        let x0 = x[0];
        let x1 = x[1];

        // df/dx = -2(1 - x) - 400x(y - x^2)
        grad[0] = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
        // df/dy = 200(y - x^2)
        grad[1] = 200.0 * (x1 - x0 * x0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_minimization() {
        // f(x) = 0.5 * a x^2 - b x
        // Minimizer is x* = b / a
        let obj = Quadratic { a: 2.0, b: 4.0 }; // f(x) = x^2 - 4x => x* = 2
        let space = EuclideanSpace;
        let solver = GradientDescent {
            space,
            step_size: 0.1,
            max_iters: 1000,
            tol_grad: 1e-6,
        };

        let x0 = vec![0.0];
        let result = solver.minimize(&obj, x0);

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn rosenbrock_minimization() {
        let obj = Rosenbrock;
        let space = EuclideanSpace;
        let solver = GradientDescent {
            space,
            step_size: 1e-3,
            max_iters: 200_000,
            tol_grad: 1e-4,
        };

        let x0 = vec![-1.2, 1.0];
        let result = solver.minimize(&obj, x0);

        // True minimizer is (1,1)
        assert!((result.x[0] - 1.0).abs() < 5e-2);
        assert!((result.x[1] - 1.0).abs() < 5e-2);
    }

    #[test]
    fn nonlinear_minimization_with_fn() {
        let space = EuclideanSpace;
        let solver = GradientDescent {
            space,
            step_size: 1e-3,
            max_iters: 200_000,
            tol_grad: 1e-4,
        };

        // initial point
        let x0 = vec![0.0, 0.0];

        // objective function 
        // p = [ cos(x) + cos(x+y)
        //       sin(x) + sin(x+y)] 
        let p_fn = |x: &Vec<f64>| {
            let p = vec![
                f64::cos(x[0]) + f64::cos(x[0] + x[1]),
                f64::sin(x[0]) + f64::sin(x[0] + x[1]),
            ];
            p
        };
        let dp_fn = |x: &Vec<f64>| {
            let dp = vec![
                vec![
                    -(f64::sin(x[0]) + f64::sin(x[0] + x[1])),
                    -f64::sin(x[0] + x[1]),
                ],
                vec![
                    f64::cos(x[0]) + f64::cos(x[0] + x[1]),
                    -f64::sin(x[0] + x[1]),
                ],
            ];
            dp
        };
        let target = vec![0.5, (f64::sqrt(3.0) + 2.0) / 2.0];
        use std::f64::consts::PI;


        let value_fn = |x: &Vec<f64>| {
            let x0_target = target[0];
            let x1_target = target[1];
            let p = p_fn(x);
            let residual = vec![p[0] - x0_target, p[1] - x1_target];
            0.5 * (residual[0].powi(2) + residual[1].powi(2))
        };

        // gradient of the objective function
        let grad_fn = |x: &Vec<f64>, grad: &mut Vec<f64>| {
            let x0_target = target[0];
            let x1_target = target[1];
            let p = p_fn(x);
            let residual = vec![p[0] - x0_target, p[1] - x1_target];

            let dp = dp_fn(x);
            grad[0] = residual[0] * dp[0][0] + residual[1] * dp[1][0];
            grad[1] = residual[0] * dp[0][1] + residual[1] * dp[1][1];
        };

        let result = solver.minimize_with_fn(x0, value_fn, grad_fn);

        // True minimizer is (pi/3, pi/6)
        assert!((result.x[0] - PI/3.0).abs() < 1e-3);
        assert!((result.x[1] - PI/6.0).abs() < 1e-3);
    }

        #[test]
    fn nonlinear_least_squares_planar_2link() {
        let space = EuclideanSpace;
        let ik = NonlinearLeastSquares {
            space,
            lambda: 1e-3,
            step_scale: 1.0,
            max_iters: 200,
            tol_r: 1e-9,
            tol_dq: 1e-12,
            line_search: true,
            ls_beta: 0.5,
            ls_max_steps: 20,
        };

        // 2-link planar arm
        let l1 = 1.0;
        let l2 = 1.0;

        // target (reachable)
        let target = [1.2, 0.6];

        // residual r(q) = p(q) - target  (m=2)
        let residual_fn = |q: &[f64], r: &mut [f64]| {
            let q1 = q[0];
            let q2 = q[1];
            let x = l1 * q1.cos() + l2 * (q1 + q2).cos();
            let y = l1 * q1.sin() + l2 * (q1 + q2).sin();
            r[0] = x - target[0];
            r[1] = y - target[1];
        };

        // Jacobian J = dr/dq, row-major (2x2)
        let jacobian_fn = |q: &[f64], j: &mut [f64]| {
            let q1 = q[0];
            let q2 = q[1];
            let s1 = q1.sin();
            let c1 = q1.cos();
            let s12 = (q1 + q2).sin();
            let c12 = (q1 + q2).cos();

            // x = l1 c1 + l2 c12
            // y = l1 s1 + l2 s12
            // dx/dq1 = -l1 s1 - l2 s12
            // dx/dq2 = -l2 s12
            // dy/dq1 =  l1 c1 + l2 c12
            // dy/dq2 =  l2 c12

            // row 0: d(rx)/dq
            j[0 * 2 + 0] = -l1 * s1 - l2 * s12;
            j[0 * 2 + 1] = -l2 * s12;

            // row 1: d(ry)/dq
            j[1 * 2 + 0] =  l1 * c1 + l2 * c12;
            j[1 * 2 + 1] =  l2 * c12;
        };

        // no joint limits for this test
        let project = |_q: &mut [f64]| {};

        let q0 = vec![0.0, 0.0];
        let res = ik.solve_with_fn(2, q0, residual_fn, jacobian_fn, project);

        assert!(res.converged, "IK did not converge: {:?}", res);
        assert!(res.r_norm < 1e-6, "residual too large: {}", res.r_norm);
    }
}
