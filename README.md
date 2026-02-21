# liteopt

Lightweight optimization toolbox with a small Rust core and Python bindings.

## Workspace Structure

- `liteopt-core/`: solver/manifold/problem definitions
- `liteopt-py/`: PyO3 bindings (`liteopt.gd`, `liteopt.gn`)

## `liteopt-core` Module Policy

- `manifolds/`
  - `space.rs`: minimal `Space` trait (point/tangent abstraction)
  - `euclidean.rs`: `EuclideanSpace` implementation
- `problems/`
  - `objective.rs`: generic objective trait
  - `least_squares.rs`: nonlinear least-squares problem trait
  - `test_functions.rs`: sample objectives (`Quadratic`, `Rosenbrock`)
- `numerics/`
  - `linalg.rs`: small dependency-free linear algebra helpers
- `solvers/`
  - `gd/`: gradient descent (`types.rs`, `solve.rs`)
  - `gauss_newton/`: Gauss-Newton (`types.rs`, `workspace.rs`, `solve.rs`)
  - `lm/`: Levenberg-Marquardt (`types.rs`, `workspace.rs`, `solve.rs`)
  - `common/`: shared solver utilities

## Current Design Direction

- Keep trait surfaces small and explicit.
- Separate `Point` and `Tangent` in `Space` to keep manifold extensions possible.
- Keep least-squares solvers currently vector-based (`Vec<f64>`) for a lite implementation.

## API Notes

- Canonical Euclidean import: `liteopt::manifolds::EuclideanSpace`
- Gauss-Newton solver import: `liteopt::solvers::gauss_newton::GaussNewton`
- LM solver import: `liteopt::solvers::lm::LevenbergMarquardt`
- Sample objective import: `liteopt::problems::test_functions::{Quadratic, Rosenbrock}`
