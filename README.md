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

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
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
liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={"max_iters": 100, "tol_r": 1e-10},
    debug={"history": True},
)
```

- `options`: numerical settings such as tolerances, iteration limits, manifold,
  and line-search policy
- `debug`: trace/logging settings such as `history` and `verbose`
- `jacobian`, `jacobian_vec`: least-squares problem callbacks

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
- matrix-free large-scale solvers
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

## Version Policy

`liteopt-py/pyproject.toml` is the canonical version for Python package
releases. The Rust crate versions in `liteopt-core/Cargo.toml` and
`liteopt-py/Cargo.toml` are internal workspace metadata unless those crates are
published separately.
