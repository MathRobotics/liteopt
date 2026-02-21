use super::space::Space;

/// Simple Euclidean space representing R^n as `Vec<f64>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanSpace;

impl Space for EuclideanSpace {
    type Point = Vec<f64>;
    type Tangent = Vec<f64>;

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

    fn zero_tangent_like(&self, x: &Vec<f64>) -> Vec<f64> {
        vec![0.0; x.len()]
    }

    fn tangent_norm(&self, v: &Vec<f64>) -> f64 {
        self.norm(v)
    }
}
