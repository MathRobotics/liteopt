# liteopt

A lightweight optimization library written in Rust with Python bindings.

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

## Quick Start

Gradient Descent:

```python
import liteopt

f = lambda x: (x[0] - 3.0) ** 2
grad = lambda x: [2.0 * (x[0] - 3.0)]

x_star, f_star, ok = liteopt.gd(f, grad, x0=[0.0], step_size=0.1)
print(ok, x_star, f_star)
```

Custom line search callback (GD):

```python
def half_step(ctx):
    return {"accepted": True, "alpha": 0.5 * ctx["alpha0"]}

x_star, f_star, ok = liteopt.gd(f, grad, x0=[0.0], line_search=half_step)
```

Gauss-Newton (least squares):

```python
import liteopt

target = [1.0, -2.0]

def residual(x):
    return [x[0] - target[0], x[1] - target[1]]

def jacobian(_x):
    return [[1.0, 0.0], [0.0, 1.0]]

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(residual, jacobian, x0=[0.0, 0.0])
print(ok, x_star, cost)
```

Custom line search callback (GN):

```python
def half_step(ctx):
    return {"accepted": True, "alpha": 0.5 * ctx["alpha0"]}

x_star, cost, *_ = liteopt.gn(residual, jacobian, x0=[0.0, 0.0], line_search=half_step)
```

Levenberg-Marquardt (least squares):

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.lm(residual, jacobian, x0=[0.0, 0.0])
print(ok, x_star, cost)
```

Optional manifold callbacks:

`gd(...)`, `gn(...)`, and `lm(...)` accept `manifold=...` with these methods:
- `retract(x, direction, alpha) -> list[float]`
- `tangent_norm(v) -> float`
- `scale(v, alpha) -> list[float]`
- `add(x, v) -> list[float]`
- `difference(x, y) -> list[float]`
