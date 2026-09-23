pub use crate::solvers::common::step_policy::{
    ArmijoBacktracking, CostDecrease, LineSearchContext, LineSearchPolicy, LineSearchResult,
    NoLineSearch,
};

/// Backtracking policy that accepts the first strictly improving trial.
#[derive(Clone, Copy, Debug)]
pub struct StrictDecreaseBacktracking {
    pub beta: f64,
    pub min_step: f64,
    pub max_steps: usize,
}

impl StrictDecreaseBacktracking {
    pub fn new(beta: f64, min_step: f64, max_steps: usize) -> Self {
        Self {
            beta,
            min_step,
            max_steps,
        }
    }
}

impl Default for StrictDecreaseBacktracking {
    fn default() -> Self {
        Self {
            beta: 0.5,
            min_step: 1e-8,
            max_steps: 12,
        }
    }
}

impl LineSearchPolicy for StrictDecreaseBacktracking {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        if !ctx.cost0.is_finite()
            || !crate::solvers::common::step_policy::valid_backtracking(self.beta, self.min_step)
        {
            return LineSearchResult {
                accepted: false,
                alpha: ctx.alpha0,
            };
        }
        let mut alpha = ctx.alpha0;
        for _ in 0..self.max_steps {
            if !alpha.is_finite() || alpha <= 0.0 || alpha < self.min_step {
                break;
            }
            let Some(cost_trial) = eval_cost(alpha) else {
                alpha *= self.beta;
                if alpha < self.min_step {
                    break;
                }
                continue;
            };
            if cost_trial.is_finite() && cost_trial < ctx.cost0 {
                return LineSearchResult {
                    accepted: true,
                    alpha,
                };
            }
            alpha *= self.beta;
            if alpha < self.min_step {
                break;
            }
        }
        LineSearchResult {
            accepted: false,
            alpha,
        }
    }
}
