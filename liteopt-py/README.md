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
uv add liteopt
```

or:

```bash
pip install liteopt
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

Bundled examples are documented in [`example/README.md`](example/README.md).

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

## Optional Manifold Callbacks

`gd(...)`, `gn(...)`, and `lm(...)` accept `options={"manifold": ...}` with
these methods:
- `retract(x, direction, alpha) -> list[float]`: apply the local update
  `alpha * direction` at point `x` and return the next point.
- `tangent_norm(v) -> float`: return the norm used for convergence checks on a
  tangent/update vector.
- `scale(v, alpha) -> list[float]`: scale a tangent/update vector.
- `add(x, v) -> list[float]`: add a tangent/update vector to a point.
- `difference(x, y) -> list[float]`: return the local update vector from point
  `x` to point `y`.

Use `manifold.retract(...)` when the optimizer should update points on a
manifold instead of using the default Euclidean step. For `gn(...)` and
`lm(...)`, `project=...` is a lighter post-step projection hook for simple
constraints.

Minimal angle-wrapping example:

```python
import math
import liteopt

def wrap_angle(theta):
    return (theta + math.pi) % (2.0 * math.pi) - math.pi

class WrappedAngles:
    def retract(self, x, direction, alpha):
        return [wrap_angle(xi + alpha * di) for xi, di in zip(x, direction)]

    def difference(self, x, y):
        return [wrap_angle(yi - xi) for xi, yi in zip(x, y)]

    def tangent_norm(self, v):
        return math.sqrt(sum(vi * vi for vi in v))

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
    x0=[3.0 * math.pi, -2.0 * math.pi],
    jacobian=jacobian,
    options={"manifold": WrappedAngles()},
)
```

The bundled `manifold` example in [`example/run.py`](example/run.py) shows the
same pattern in a complete inverse-kinematics problem.

## Migration Notes

For release notes and breaking-change migration examples, see
[`../RELEASE.md`](../RELEASE.md).
