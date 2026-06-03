# liteopt

A lightweight optimization library written in Rust with Python bindings.

## Scope

`liteopt` is aimed at small dense optimization problems where low dependency
cost and readable implementation matter. It provides basic GD/GN/LM solvers,
simple tolerances, simple step control, and optional debug traces.

It is not intended to be a large-scale sparse optimizer, a SciPy replacement, a
matrix-free numerical backend, or a BLAS/LAPACK-backed production solver.

## Installation

Install from PyPI:

```bash
uv venv
source .venv/bin/activate
uv pip install liteopt
```

Install from source (development):

Requirements:
- Rust toolchain (`cargo`)
- Python 3.8+
- `uv`

```bash
cd liteopt-py
uv sync --extra dev
uv run --extra dev maturin develop --manifest-path Cargo.toml
uv run python -c "import liteopt; print(liteopt.__file__)"
```

## Examples

Run bundled examples from `liteopt-py/example/`:

```bash
cd liteopt-py
uv run python example/run.py all
```

Or from repository root:

```bash
uv run --project liteopt-py python liteopt-py/example/run.py all
```

Run a single example:

```bash
uv run python example/run.py gd
uv run python example/run.py gn
uv run python example/run.py lm
```

## Quick Start

Gradient Descent:

```python
import liteopt

f = lambda x: (x[0] - 3.0) ** 2
grad = lambda x: [2.0 * (x[0] - 3.0)]

x_star, f_star, ok = liteopt.gd(
    f,
    grad,
    x0=[0.0],
    options={"step_size": 0.1},
)
print(ok, x_star, f_star)
```

Gauss-Newton (least squares):

```python
import liteopt

target = [1.0, -2.0]

def residual(x):
    return [x[0] - target[0], x[1] - target[1]]

def jacobian(_x):
    # If you return a Python list, it must be row-major 1D (m*n elements).
    # `[[1.0, 0.0], [0.0, 1.0]]` raises TypeError.
    return [1.0, 0.0, 0.0, 1.0]

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
)
print(ok, x_star, cost)
```

`jacobian` must be either:
- row-major 1D list (`list[float]`, length = `m * n`)
- 2D `numpy.ndarray` (`shape = (m, n)`)

Alternatively, provide a Jacobian-vector product callback and omit `jacobian`:

```python
def jacobian_vec(x, v):
    return jacobian(x) @ v

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian_vec=jacobian_vec,
)
```

`jacobian_vec(x, v)` must return `J(x) @ v` with length `m`. The current
implementation reconstructs the dense Jacobian internally by applying
`jacobian_vec` to basis vectors, so it avoids writing a dense Jacobian callback
but is not a fully matrix-free large-scale solver.

Levenberg-Marquardt (least squares):

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.lm(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
)
print(ok, x_star, cost)
```

`lm(...)` also accepts `jacobian_vec=...` with the same `J(x) @ v` contract as
`gn(...)`.

## Convergence and Debug Options

Convergence control is intentionally small:

- `gd(...)`: `max_iters`, `tol_grad`
- `gn(...)` and `lm(...)`: `max_iters`, `tol_r`, `tol_dx`

Line search is available, but it is treated as optional step control rather
than a broad globalization framework. The default API is meant for small dense
problems; advanced users can pass a custom callback when they need direct
control over step acceptance.

Trace history is disabled by default. Enable it only when inspecting solver
behavior:

```python
x_star, cost, iters, r_norm, dx_norm, ok, history = liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    debug={"history": True},
)
```

Gauss-Newton exposes a compact configured line-search mode for users who need
fixed damping and strict decrease checks:

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={
        "lambda": 1e-8,
        "damping_update": "fixed",
        "linear_system": "normal_jtj",
        "line_search_method": "strict_decrease",
        "line_search": True,
        "ls_max_steps": 12,
    },
)
```

Custom line search callbacks receive a small context dictionary and return an
alpha or an `{ "accepted": bool, "alpha": float }` style result:

```python
def half_step(ctx):
    return {"accepted": True, "alpha": 0.5 * ctx["alpha0"]}

x_star, cost, *_ = liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={"line_search": half_step},
)
```

## Release History

### 0.1.9

The Python solver APIs were simplified while keeping debugging explicit.
Solver settings moved into an `options` dict, and trace/logging controls moved
into a separate `debug` dict.

Before:

```python
liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    max_iters=100,
    tol_r=1e-10,
    line_search=True,
    history=True,
)
```

After:

```python
liteopt.gn(
    residual,
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={
        "max_iters": 100,
        "tol_r": 1e-10,
        "line_search": True,
    },
    debug={"history": True},
)
```

For `lm(...)`, move `lambda_`/`lambda`, `lambda_up`, `lambda_down`, `step_scale`,
`max_iters`, `tol_r`, `tol_dx`, `line_search`, and `manifold` into `options`;
move `verbose` and `history` into `debug`.

For `gn(...)`, move `lambda_`/`lambda`, `step_scale`, `max_iters`, `tol_r`,
`tol_dx`, `damping_update`, `linear_system`, `line_search_method`,
`line_search`, `ls_beta`, `ls_min_step`, `ls_max_steps`, `c_armijo`,
and `manifold` into `options`; move `verbose` and `history` into `debug`.

For `gd(...)`, move `step_size`, `max_iters`, `tol_grad`, `line_search`, and
`manifold` into `options`; move `verbose` and `history` into `debug`.

Optional manifold callbacks:

`gd(...)`, `gn(...)`, and `lm(...)` accept `options={"manifold": ...}` with
these methods:
- `retract(x, direction, alpha) -> list[float]`
- `tangent_norm(v) -> float`
- `scale(v, alpha) -> list[float]`
- `add(x, v) -> list[float]`
- `difference(x, y) -> list[float]`
