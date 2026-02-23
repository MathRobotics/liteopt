/// Outcome of a step search policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSearchResult {
    pub accepted: bool,
    pub alpha: f64,
}

/// Per-iteration context passed to step search policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSearchContext {
    pub iter: usize,
    pub alpha0: f64,
    pub cost0: f64,
    pub dphi0: Option<f64>,
    pub dx_norm: f64,
    pub lambda: f64,
}

/// Policy interface for selecting a step size.
///
/// `eval_cost(alpha)` must return trial cost at step size `alpha`.
/// Returning `None` means the trial is invalid (e.g., non-finite).
pub trait LineSearchPolicy {
    /// Whether this policy needs directional derivative at alpha = 0.
    fn requires_directional_derivative(&self) -> bool {
        false
    }

    /// Pick a step size using the provided trial-cost evaluator.
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult;
}

/// Policy that accepts a step if trial point/cost is finite.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoLineSearch;

impl LineSearchPolicy for NoLineSearch {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        let accepted = eval_cost(ctx.alpha0).is_some();
        LineSearchResult {
            accepted,
            alpha: ctx.alpha0,
        }
    }
}

/// Policy that accepts only if objective strictly decreases.
#[derive(Clone, Copy, Debug, Default)]
pub struct CostDecrease;

impl LineSearchPolicy for CostDecrease {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        let accepted = eval_cost(ctx.alpha0)
            .map(|cost_trial| cost_trial.is_finite() && cost_trial < ctx.cost0)
            .unwrap_or(false);
        LineSearchResult {
            accepted,
            alpha: ctx.alpha0,
        }
    }
}

/// Armijo backtracking policy.
#[derive(Clone, Copy, Debug)]
pub struct ArmijoBacktracking {
    pub beta: f64,
    pub max_steps: usize,
    pub c_armijo: f64,
}

impl ArmijoBacktracking {
    pub fn new(beta: f64, max_steps: usize, c_armijo: f64) -> Self {
        Self {
            beta,
            max_steps,
            c_armijo,
        }
    }
}

impl Default for ArmijoBacktracking {
    fn default() -> Self {
        Self {
            beta: 0.5,
            max_steps: 20,
            c_armijo: 1e-4,
        }
    }
}

impl LineSearchPolicy for ArmijoBacktracking {
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

        let mut alpha = ctx.alpha0;
        for _ in 0..self.max_steps {
            let Some(cost_trial) = eval_cost(alpha) else {
                alpha *= self.beta;
                continue;
            };

            let rhs = ctx.cost0 + self.c_armijo * alpha * dphi0;
            if rhs.is_finite() && cost_trial <= rhs {
                return LineSearchResult {
                    accepted: true,
                    alpha,
                };
            }
            alpha *= self.beta;
        }

        LineSearchResult {
            accepted: false,
            alpha,
        }
    }
}
