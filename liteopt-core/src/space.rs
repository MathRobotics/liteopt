//! Space abstractions (Euclidean now, manifolds later)

/// Trait that represents an abstract "space".
pub trait Space {
    type Point: Clone;

    fn zero_like(&self, x: &Self::Point) -> Self::Point;
    fn norm(&self, v: &Self::Point) -> f64;

    // --- core ops (allocation-free if impl does it right) ---
    fn scale_into(&self, out: &mut Self::Point, v: &Self::Point, alpha: f64);
    fn add_into(&self, out: &mut Self::Point, x: &Self::Point, v: &Self::Point);
    fn difference_into(&self, out: &mut Self::Point, x: &Self::Point, y: &Self::Point);

    /// out = Retr_x(alpha * direction)
    fn retract_into(
        &self,
        out: &mut Self::Point,
        x: &Self::Point,
        direction: &Self::Point,
        alpha: f64,
        tmp: &mut Self::Point,
    ) {
        self.scale_into(tmp, direction, alpha);
        self.add_into(out, x, tmp);
    }

    // --- convenience wrappers (allocate; OK for examples) ---
    fn scale(&self, v: &Self::Point, alpha: f64) -> Self::Point {
        let mut out = self.zero_like(v);
        self.scale_into(&mut out, v, alpha);
        out
    }
    fn add(&self, x: &Self::Point, v: &Self::Point) -> Self::Point {
        let mut out = self.zero_like(x);
        self.add_into(&mut out, x, v);
        out
    }
    fn difference(&self, x: &Self::Point, y: &Self::Point) -> Self::Point {
        let mut out = self.zero_like(x);
        self.difference_into(&mut out, x, y);
        out
    }
    fn retract(&self, x: &Self::Point, direction: &Self::Point, alpha: f64) -> Self::Point {
        let mut out = self.zero_like(x);
        let mut tmp = self.zero_like(direction);
        self.retract_into(&mut out, x, direction, alpha, &mut tmp);
        out
    }
}

/// Simple Euclidean space representing R^n as Vec<f64>.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanSpace;

impl Space for EuclideanSpace {
    type Point = Vec<f64>;

    fn zero_like(&self, x: &Vec<f64>) -> Vec<f64> {
        vec![0.0; x.len()]
    }

    fn norm(&self, v: &Vec<f64>) -> f64 {
        v.iter().map(|vi| vi * vi).sum::<f64>().sqrt()
    }

    fn scale_into(&self, out: &mut Vec<f64>, v: &Vec<f64>, alpha: f64) {
        out.resize(v.len(), 0.0);
        for i in 0..v.len() {
            out[i] = alpha * v[i];
        }
    }

    fn add_into(&self, out: &mut Vec<f64>, x: &Vec<f64>, v: &Vec<f64>) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = x[i] + v[i];
        }
    }

    fn difference_into(&self, out: &mut Vec<f64>, x: &Vec<f64>, y: &Vec<f64>) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = y[i] - x[i];
        }
    }

    fn retract_into(
        &self,
        out: &mut Vec<f64>,
        x: &Vec<f64>,
        direction: &Vec<f64>,
        alpha: f64,
        _tmp: &mut Vec<f64>,
    ) {
        out.resize(x.len(), 0.0);
        for i in 0..x.len() {
            out[i] = x[i] + alpha * direction[i];
        }
    }
}
