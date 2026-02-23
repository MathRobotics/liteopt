mod line_search;
mod solve;
mod types;
mod workspace;

pub use line_search::{
    ArmijoBacktracking, CostDecrease, LineSearchContext, LineSearchPolicy, LineSearchResult,
    NoLineSearch, StrictDecreaseBacktracking,
};
pub use types::{
    GaussNewton, GaussNewtonDampingUpdate, GaussNewtonLineSearchMethod, GaussNewtonLinearSystem,
    GaussNewtonResult,
};
