# liteopt

Lightweight optimization toolbox with a small Rust core and Python bindings.

## Workspace Structure

- `liteopt-core/`: solver/manifold/problem definitions
- `liteopt-py/`: PyO3 bindings (`liteopt.gd`, `liteopt.gn`)

## Quick Examples

Rust (Gradient Descent):

```rust
use liteopt::solvers::gd::GradientDescent;

let solver = GradientDescent {
    step_size: 0.1,
    max_iters: 100,
    tol_grad: 1e-9,
    ..Default::default() // space is EuclideanSpace
};
let res = solver.minimize_with_fn(vec![0.0], |x| (x[0] - 3.0).powi(2), |x, g| g[0] = 2.0 * (x[0] - 3.0));
println!("{:?}", res.x);
```

Rust (Gauss-Newton):

```rust
use liteopt::solvers::gn::GaussNewton;

let solver = GaussNewton {
    lambda: 1e-3,
    step_scale: 1.0,
    max_iters: 20,
    tol_r: 1e-12,
    tol_dq: 1e-12,
    line_search: true,
    ls_beta: 0.5,
    ls_max_steps: 20,
    c_armijo: 1e-4,
    ..Default::default() // space is EuclideanSpace
};
let res = solver.solve_with_fn(2, vec![0.0, 0.0], |x, r| { r[0] = x[0] - 1.0; r[1] = x[1] + 2.0; }, |_x, j| { j[0] = 1.0; j[1] = 0.0; j[2] = 0.0; j[3] = 1.0; }, |_x| {});
println!("{:?}", res.x);
```

Python examples are in `liteopt-py/README.md`.

Bundled Rust examples in `liteopt-core/examples/` can be run with:

```bash
cargo run -p liteopt --example quadratic
cargo run -p liteopt --example nonlinear_least_squares_demo
cargo run -p liteopt --example my_manifold
```

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
  - `gn/`: Gauss-Newton (`types.rs`, `workspace.rs`, `solve.rs`)
  - `lm/`: Levenberg-Marquardt (`types.rs`, `workspace.rs`, `solve.rs`)
  - `common/`: shared solver utilities

## Current Design Direction

- Keep trait surfaces small and explicit.
- Separate `Point` and `Tangent` in `Space` to keep manifold extensions possible.
- Keep least-squares solvers currently vector-based (`Vec<f64>`) for a lite implementation.

## API Notes

- Canonical Euclidean import: `liteopt::manifolds::EuclideanSpace`
- If `space` is omitted, `GradientDescent::default()`, `GaussNewton::default()`, and `LevenbergMarquardt::default()` use `EuclideanSpace`.
- Explicit manifold selection is available via `GradientDescent::with_space(...)`, `GaussNewton::with_space(...)`, and `LevenbergMarquardt::with_space(...)`.
- Custom manifold sample: `liteopt-core/tests/gn.rs` (`MyManifold`)
- Gauss-Newton solver import: `liteopt::solvers::gn::GaussNewton`
- LM solver import: `liteopt::solvers::lm::LevenbergMarquardt`
- Sample objective import: `liteopt::problems::test_functions::{Quadratic, Rosenbrock}`
- Python custom manifold callbacks: see `liteopt-py/README.md`
