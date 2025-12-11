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

//
// Tests and examples: quadratic / Rosenbrock
//

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
}
