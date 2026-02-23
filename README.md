# liteopt

Lightweight optimization toolbox with a small Rust core and Python bindings.

## Installation

Python package (PyPI):

```bash
uv venv
source .venv/bin/activate
uv pip install liteopt
```

Python package from source (development):

```bash
cd liteopt-py
uv sync --extra dev
uv run --extra dev maturin develop --manifest-path Cargo.toml
uv run python -c "import liteopt; print(liteopt.__file__)"
```

Rust core in this workspace:

```bash
cargo test -p liteopt
```

## End-to-End Setup (Clone -> Build -> Python Example)

Prerequisites:
- Rust toolchain (`cargo`)
- Python 3.8+
- `uv`

1. Clone and move into this repository:

```bash
git clone https://github.com/MathRobotics/liteopt.git
cd liteopt
```

2. Build `liteopt-core`:

```bash
cargo build -p liteopt
```

3. Build and install Python bindings (`liteopt-py`) into the uv-managed environment:

```bash
uv sync --project liteopt-py --extra dev
uv run --project liteopt-py maturin develop --manifest-path liteopt-py/Cargo.toml
```

4. Run a `liteopt-py` example (gradient descent):

```bash
uv run --project liteopt-py python - <<'PY'
import liteopt

f = lambda x: (x[0] - 3.0) ** 2
grad = lambda x: [2.0 * (x[0] - 3.0)]

x_star, f_star, ok = liteopt.gd(f, grad, x0=[0.0], step_size=0.1)
print(ok, x_star, f_star)
PY
```

If setup succeeded, `ok` is `True` and `x_star[0]` is close to `3.0`.

## Workspace Structure

- `liteopt-core/`: solver/manifold/problem definitions
- `liteopt-py/`: PyO3 bindings (`liteopt.gd`, `liteopt.gn`, `liteopt.lm`)

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

Rust (Gauss-Newton + custom line search):

```rust
use liteopt::solvers::gn::{GaussNewton, LineSearchContext, LineSearchPolicy, LineSearchResult};

#[derive(Default)]
struct MyLineSearch;

impl LineSearchPolicy for MyLineSearch {
    fn search(
        &mut self,
        ctx: &LineSearchContext,
        eval_cost: &mut dyn FnMut(f64) -> Option<f64>,
    ) -> LineSearchResult {
        let alpha = 0.5 * ctx.alpha0;
        LineSearchResult {
            accepted: eval_cost(alpha).is_some(),
            alpha,
        }
    }
}

let solver = GaussNewton {
    lambda: 1e-3,
    step_scale: 1.0,
    max_iters: 20,
    tol_r: 1e-12,
    tol_dq: 1e-12,
    ..Default::default() // space is EuclideanSpace
};
let mut line_search = MyLineSearch::default();
let res = solver.solve_with_fn(
    2,
    vec![0.0, 0.0],
    |x, r| { r[0] = x[0] - 1.0; r[1] = x[1] + 2.0; },
    |_x, j| { j[0] = 1.0; j[1] = 0.0; j[2] = 0.0; j[3] = 1.0; },
    |_x| {},
    &mut line_search,
);
println!("{:?}", res.x);
```

Python examples are in `liteopt-py/README.md`.

Bundled Rust examples in `liteopt-core/examples/` can be run with:

```bash
cargo run -p liteopt --example quadratic
cargo run -p liteopt --example nonlinear_least_squares_demo
cargo run -p liteopt --example my_manifold
cargo run -p liteopt --example custom_line_search
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
- Custom line search for Gauss-Newton: implement `liteopt::solvers::gn::LineSearchPolicy`
- LM solver import: `liteopt::solvers::lm::LevenbergMarquardt`
- Sample objective import: `liteopt::problems::test_functions::{Quadratic, Rosenbrock}`
- Python custom manifold callbacks: see `liteopt-py/README.md`
