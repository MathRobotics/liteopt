use crate::space::{EuclideanSpace, Space};

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
/// quadratic of the form f(x) = 0.5 * x^T A x - b^T x.
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

/// 2D Rosenbrock function.
/// f(x, y) = (a - x)^2 + b (y - x^2)^2
pub struct Rosenbrock {
    pub a : f64,
    pub b : f64,
}

impl Objective<EuclideanSpace> for Rosenbrock {
    fn value(&self, x: &Vec<f64>) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (self.a - x0).powi(2) + self.b * (x1 - x0 * x0).powi(2)
    }

    fn gradient(&self, x: &Vec<f64>, grad: &mut Vec<f64>) {
        let x0 = x[0];
        let x1 = x[1];

        // df/dx = -2a(1 - x) - 4bx(y - x^2)
        grad[0] = -2.0 * self.a * (1.0 - x0) - 4.0 * self.b * x0 * (x1 - x0 * x0);
        // df/dy = 2b(y - x^2)
        grad[1] = 2.0 * self.b * (x1 - x0 * x0);
    }
}