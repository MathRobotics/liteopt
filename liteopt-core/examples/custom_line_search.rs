use liteopt::{
    manifolds::EuclideanSpace,
    solvers::gn::{GaussNewton, LineSearchContext, LineSearchPolicy, LineSearchResult},
};

#[derive(Clone, Copy, Debug)]
struct MyLineSearch {
    beta: f64,
    max_steps: usize,
    min_alpha: f64,
}

impl Default for MyLineSearch {
    fn default() -> Self {
        Self {
            beta: 0.5,
            max_steps: 12,
            min_alpha: 1e-3,
        }
    }
}

impl LineSearchPolicy for MyLineSearch {
    fn requires_directional_derivative(&self) -> bool {
        true
    }

    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        let Some(dphi0) = ctx.dphi0 else {
            return LineSearchResult {
                accepted: false,
                alpha: ctx.alpha0,
            };
        };

        // If damping is already large, start more conservatively.
        let mut alpha = if ctx.lambda > 1.0 {
            0.5 * ctx.alpha0
        } else {
            ctx.alpha0
        };
        for _ in 0..self.max_steps {
            let Some(cost_trial) = eval_cost(alpha) else {
                alpha *= self.beta;
                continue;
            };

            let rhs = ctx.cost0 + 1e-4 * alpha * dphi0;
            if cost_trial <= rhs {
                return LineSearchResult {
                    accepted: true,
                    alpha,
                };
            }
            alpha *= self.beta;
            if alpha < self.min_alpha {
                break;
            }
        }

        LineSearchResult {
            accepted: false,
            alpha,
        }
    }
}

fn main() {
    let solver = GaussNewton {
        space: EuclideanSpace,
        max_iters: 40,
        tol_r: 1e-12,
        tol_dq: 1e-12,
        ..Default::default()
    };

    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 1.0;
        r[1] = x[1] + 2.0;
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        j[0] = 1.0;
        j[1] = 0.0;
        j[2] = 0.0;
        j[3] = 1.0;
    };

    let mut line_search = MyLineSearch::default();
    let result = solver.solve_with_fn(
        2,
        vec![0.0, 0.0],
        residual,
        jacobian,
        |_x| {},
        &mut line_search,
    );

    println!("converged={} x={:?}", result.converged, result.x);
}
