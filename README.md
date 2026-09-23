# liteopt

Lightweight optimization toolbox with a small Rust core and Python bindings.

`liteopt` is for small dense optimization problems where low dependency cost,
readable implementation, and quick debugging matter.

It currently provides:

- gradient descent
- Gauss-Newton
- Levenberg-Marquardt
- optional trace/debug history
- Rust core and Python bindings

It is not intended to be a SciPy replacement, a large-scale sparse optimizer,
or a BLAS/LAPACK-backed production linear algebra backend.

## Python Solver Selection

| Problem | Usage |
|---|---|
| General differentiable objective | `liteopt.gd(f, grad, x0, ...)` |
| Least squares with Gauss-Newton | `liteopt.least_squares(residual, x0, method="gn", ...)` |
| Least squares with Levenberg-Marquardt | `liteopt.least_squares(residual, x0, method="lm", ...)` |

`least_squares` defaults to `method="lm"`. Both methods share the residual,
Jacobian, and result interface; method-specific settings go in `options`.
The direct `gn()` and `lm()` functions remain available. GN uses an undamped
direction with Armijo backtracking. LM uses adaptive damping and defaults to
a single cost-decrease check; set `options={"line_search_method": "armijo"}`
or `"strict_decrease"` to enable LM backtracking. GN requires a nonsingular
linear system; choose LM for problems needing damping.

## Quick Start

Install the Python package:

```bash
uv add liteopt
```

or:

```bash
pip install liteopt
```

Copy and run a small optimization:

```bash
python - <<'PY'
import liteopt

def f(x):
    return (x[0] - 3.0) ** 2

def grad(x):
    return [2.0 * (x[0] - 3.0)]

x_star, f_star, ok = liteopt.gd(
    f,
    grad,
    x0=[0.0],
    options={"step_size": 0.1},
)

print(ok, x_star, f_star)
PY
```

Run a small least-squares optimization with Gauss-Newton:

```bash
python - <<'PY'
import liteopt

target = [1.0, -2.0]

def residual(x):
    return [x[0] - target[0], x[1] - target[1]]

def jacobian(_x):
    return [1.0, 0.0, 0.0, 1.0]

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={"max_iters": 100, "tol_r": 1e-10},
)

print(ok, x_star, cost, iters, r_norm, dx_norm)
PY
```

## From Source

Prerequisites:

- Rust toolchain (`cargo`)
- Python 3.8+
- `uv`

Clone and build the Python bindings:

```bash
git clone https://github.com/MathRobotics/liteopt.git
cd liteopt/liteopt-py
uv sync --extra dev
uv run maturin develop --manifest-path Cargo.toml
```

Example commands are documented in
[`liteopt-py/example/README.md`](liteopt-py/example/README.md).

## Python API Shape

The Python API keeps solver settings separate from debugging controls:

```python
_, _, _, history = liteopt.gd(
    f,
    grad,
    x0=[0.0],
    options={"step_size": 0.1, "max_iters": 200},
    debug={"history": True},
)
```

Least-squares solvers keep Jacobian callbacks explicit:

```python
liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={"max_iters": 100, "tol_r": 1e-10},
    debug={"history": True},
)
```

- `options`: numerical settings such as tolerances, iteration limits, manifold,
  and line-search policy
- `debug`: trace/logging settings such as `history` and `verbose`
- `jacobian`, or `jacobian_vec` + `jacobian_transpose_vec`: least-squares derivatives

For full Python usage, see [`liteopt-py/README.md`](liteopt-py/README.md).

## Scope

`liteopt` intentionally keeps the numerical backend small:

- dense `Vec<f64>` least-squares data
- basic GD/GN/LM solvers
- simple convergence tolerances and maximum-iteration limits
- simple step-control policies
- optional trace history for debugging

Non-goals:

- large-scale sparse optimization
- preconditioners and specialized large-scale solvers
- automatic differentiation
- broad constrained-optimization support
- a large set of termination and globalization strategies

## Development Checks

From the repository root:

```bash
cargo test --workspace
cd liteopt-py
uv sync --extra dev
uv run maturin develop --manifest-path Cargo.toml --release
uv run pytest tests
```

Run `maturin` commands from inside `liteopt-py`; this keeps `uv` and `maturin`
using the Python package's `pyproject.toml`.

## Repository Layout

- `liteopt-core/`: Rust solver, manifold, problem, and numerics code
- `liteopt-py/`: PyO3 bindings and Python tests
- `RELEASE.md`: release checklist and migration notes
- [benchmarks/README.md](benchmarks/README.md): dense solver accuracy and timing comparison

## Version Policy

`liteopt-py/pyproject.toml` is the canonical version for Python package
releases. The Rust crate versions in `liteopt-core/Cargo.toml` and
`liteopt-py/Cargo.toml` are internal workspace metadata unless those crates are
published separately.

GD supports built-in line searches through `options={"line_search_method": "armijo"}`
(`none`, `cost_decrease`, and `strict_decrease` are also available; default: `none`).
Use `debug={"info": True}` to return `(x, f, ok, info)`, where `info` includes
`iters`, `grad_norm`, `status`, `nfev`, and `njev`. With history enabled, the
return is `(x, f, ok, history, info)`. See [GD diagnostics](liteopt-py/README.md#gd-diagnostics).

GN and LM also support `debug={"info": True}` (appended after optional history).
See [common diagnostics and search settings](liteopt-py/README.md#common-result-diagnostics)
for iteration, evaluation-count, and history definitions.


GN/LM now support `options={"linear_solver": "direct"}` or
`options={"linear_solver": "cg"}`. The direct route keeps the existing dense
backends including QR. CG accepts either a dense Jacobian or a pair of
`jacobian_vec(x, v)` / `jacobian_transpose_vec(x, w)` callbacks; the pair
uses O(m+n) workspace without constructing J. A lone `jacobian_vec` is no
longer converted to a dense matrix. See the
[Python linear solver guide](liteopt-py/README.md#direct-and-iterative-linear-solvers)
for selection rules, tolerances, and diagnostics.

Rust keeps the dense `solve_with_fn*` methods. Set
`solver.linear_solver = LinearSolver::Cg` and pass
`JacobianProducts::new(forward, transpose)` to
`solve_with_derivatives*` for matrix-free solves. These types and
`CgOptions` are exported from `liteopt::solvers`.
