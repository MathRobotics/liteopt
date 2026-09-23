mod solve;
mod types;

pub use crate::solvers::common::step_policy::{
    ArmijoBacktracking, CostDecrease, LineSearchContext, LineSearchPolicy, LineSearchResult,
    NoLineSearch,
};
pub use types::{GdTermination, GradientDescent, OptimizeResult};
