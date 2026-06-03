# liteopt

Lightweight optimization toolbox with a small Rust core and Python bindings.

## Scope

`liteopt` targets small dense optimization problems with minimal dependencies.
It is intended to be easy to build, easy to inspect, and practical for compact
Rust/Python workflows such as examples, prototypes, simple robotics problems,
and small nonlinear least-squares fits.

The library intentionally keeps the numerical backend small:

- dense `Vec<f64>` least-squares data
- basic GD/GN/LM solvers
- simple convergence tolerances and maximum-iteration limits
- simple step control policies
- optional trace history for debugging

Non-goals:

- large-scale sparse optimization
- a SciPy replacement
- matrix-free large-scale solvers
- BLAS/LAPACK-backed production linear algebra
- automatic differentiation
- broad constrained-optimization support
- a large set of termination and globalization strategies

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

4. Run bundled `liteopt-py` examples:

```bash
uv run --project liteopt-py python liteopt-py/example/run.py all
```

Run a single example:

```bash
uv run --project liteopt-py python liteopt-py/example/run.py gd
uv run --project liteopt-py python liteopt-py/example/run.py gn
uv run --project liteopt-py python liteopt-py/example/run.py lm
```

## Workspace Structure

- `liteopt-core/`: solver/manifold/problem definitions
- `liteopt-py/`: PyO3 bindings (`liteopt.gd`, `liteopt.gn`, `liteopt.lm`)

## Convergence and Step Control

Convergence behavior is intentionally simple. Gradient descent uses
`max_iters` and `tol_grad`; Gauss-Newton and Levenberg-Marquardt use
`max_iters`, `tol_r`, and `tol_dx`/`tol_dq`.

Line search is available as a small safety/debug tool, not as a broad
globalization framework. The default examples avoid making line-search policy
the main API surface. Advanced users can still provide custom policies when
they need tighter control.

Trace history is optional and disabled by default. Use `history=True` in Python
or `collect_trace = true` in Rust when inspecting solver behavior.

## Development Checks

Run the same checks used by CI from the repository root:

```bash
cargo test --workspace
uv sync --project liteopt-py --extra dev
uv run --project liteopt-py maturin develop --manifest-path liteopt-py/Cargo.toml --release
uv run --project liteopt-py pytest liteopt-py/tests
```

If the uv environment becomes inconsistent, recreate it:

```bash
rm -rf liteopt-py/.venv
uv sync --project liteopt-py --extra dev
```

## Version Policy

`liteopt-py/pyproject.toml` is the canonical version for Python package
releases. The Rust crate versions in `liteopt-core/Cargo.toml` and
`liteopt-py/Cargo.toml` are internal workspace metadata unless those crates are
published separately. See `RELEASE.md` for the release checklist.

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
    max_iters: 20,
    tol_r: 1e-12,
    tol_dq: 1e-12,
    ..Default::default()
};
let res = solver.solve_with_fn_default_line_search(
    2,
    vec![0.0, 0.0],
    |x, r| { r[0] = x[0] - 1.0; r[1] = x[1] + 2.0; },
    |_x, j| { j[0] = 1.0; j[1] = 0.0; j[2] = 0.0; j[3] = 1.0; },
    |_x| {},
);
println!("converged={} x={:?}", res.converged, res.x);
```

Rust (advanced: custom line search):

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

Rust (Gauss-Newton simple loop mode):

```rust
use liteopt::solvers::gn::{
    GaussNewton, GaussNewtonDampingUpdate, GaussNewtonLineSearchMethod, GaussNewtonLinearSystem,
};

let solver = GaussNewton {
    lambda: 1e-8,
    damping_update: GaussNewtonDampingUpdate::Fixed,
    linear_system: GaussNewtonLinearSystem::NormalJtJ,
    line_search_method: GaussNewtonLineSearchMethod::StrictDecrease,
    ls_beta: 0.5,
    ls_min_step: 1e-8,
    ls_max_steps: 12,
    max_iters: 20,
    tol_r: 1e-10,
    tol_dq: 1e-12,
    ..Default::default()
};
let res = solver.solve_with_fn_default_line_search(
    2,
    vec![0.0, 0.0],
    |x, r| { r[0] = x[0] - 1.0; r[1] = x[1] + 2.0; },
    |_x, j| { j[0] = 1.0; j[1] = 0.0; j[2] = 0.0; j[3] = 1.0; },
    |_x| {},
);
println!("converged={} x={:?}", res.converged, res.x);
```

Python examples are in `liteopt-py/example/` (`run.py`) and documented in `liteopt-py/README.md`.

Python `gn(...)` and `lm(...)` can use either a dense `jacobian(x)` callback or
`jacobian_vec(x, v) -> J(x) @ v`. The `jacobian_vec` path reconstructs the dense
Jacobian internally from basis-vector products, so it is a convenience API
rather than a fully matrix-free large-scale solver.

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
- Keep convergence and step-control behavior intentionally simple.
- Treat trace history and custom line search as optional diagnostics/advanced controls.

## API Notes

- Canonical Euclidean import: `liteopt::manifolds::EuclideanSpace`
- If `space` is omitted, `GradientDescent::default()`, `GaussNewton::default()`, and `LevenbergMarquardt::default()` use `EuclideanSpace`.
- Explicit manifold selection is available via `GradientDescent::with_space(...)`, `GaussNewton::with_space(...)`, and `LevenbergMarquardt::with_space(...)`.
- Custom manifold sample: `liteopt-core/tests/gn.rs` (`MyManifold`)
- Gauss-Newton solver import: `liteopt::solvers::gn::GaussNewton`
- Custom line search for Gauss-Newton (advanced): implement `liteopt::solvers::gn::LineSearchPolicy`
- GN algorithm selection: `damping_update`, `linear_system`, `line_search_method`
- LM solver import: `liteopt::solvers::lm::LevenbergMarquardt`
- Sample objective import: `liteopt::problems::test_functions::{Quadratic, Rosenbrock}`
- Python custom manifold callbacks: see `liteopt-py/README.md`
