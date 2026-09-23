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
        let accepted = ctx.alpha0.is_finite()
            && ctx.alpha0 > 0.0
            && eval_cost(ctx.alpha0).is_some_and(f64::is_finite);
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
        let accepted = ctx.alpha0.is_finite()
            && ctx.alpha0 > 0.0
            && ctx.cost0.is_finite()
            && eval_cost(ctx.alpha0)
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
    pub min_step: f64,
    pub beta: f64,
    pub max_steps: usize,
    pub c_armijo: f64,
}

impl ArmijoBacktracking {
    pub fn new(beta: f64, max_steps: usize, c_armijo: f64) -> Self {
        Self {
            min_step: 1e-8,
            beta,
            max_steps,
            c_armijo,
        }
    }
}

impl ArmijoBacktracking {
    pub fn with_min_step(mut self, min_step: f64) -> Self {
        self.min_step = min_step;
        self
    }
}

impl Default for ArmijoBacktracking {
    fn default() -> Self {
        Self {
            min_step: 1e-8,
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

        if !ctx.cost0.is_finite()
            || !dphi0.is_finite()
            || dphi0 >= 0.0
            || !valid_backtracking(self.beta, self.min_step)
            || !self.c_armijo.is_finite()
            || self.c_armijo <= 0.0
            || self.c_armijo >= 1.0
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
                continue;
            };

            let rhs = ctx.cost0 + self.c_armijo * alpha * dphi0;
            if cost_trial.is_finite() && rhs.is_finite() && cost_trial <= rhs {
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

// Shared by the public backtracking policies, including direct Rust callers.
pub(crate) fn valid_backtracking(beta: f64, min_step: f64) -> bool {
    beta.is_finite() && beta > 0.0 && beta < 1.0 && min_step.is_finite() && min_step > 0.0
}
