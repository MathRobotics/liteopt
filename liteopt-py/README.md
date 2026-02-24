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

x_star, f_star, ok = liteopt.gd(f, grad, x0=[0.0], step_size=0.1)
print(ok, x_star, f_star)

# Collect per-iteration history (list[dict]) with an option:
x_star, f_star, ok, history = liteopt.gd(
    f, grad, x0=[0.0], step_size=0.1, history=True
)
print("history rows:", len(history))
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
    # If you return a Python list, it must be row-major 1D (m*n elements).
    # `[[1.0, 0.0], [0.0, 1.0]]` raises TypeError.
    return [1.0, 0.0, 0.0, 1.0]

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(residual, jacobian, x0=[0.0, 0.0])
print(ok, x_star, cost)

# Optional trace history:
x_star, cost, iters, r_norm, dx_norm, ok, history = liteopt.gn(
    residual, jacobian, x0=[0.0, 0.0], history=True
)
```

`jacobian` must be either:
- row-major 1D list (`list[float]`, length = `m * n`)
- 2D `numpy.ndarray` (`shape = (m, n)`)

Gauss-Newton simple loop (fixed damping + strict-decrease line search):

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
    residual,
    jacobian,
    x0=[0.0, 0.0],
    lambda_=1e-8,
    damping_update="fixed",
    linear_system="normal_jtj",
    line_search_method="strict_decrease",
    line_search=True,
    ls_beta=0.5,
    ls_min_step=1e-8,
    ls_max_steps=12,
)
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

# Optional trace history:
x_star, cost, iters, r_norm, dx_norm, ok, history = liteopt.lm(
    residual, jacobian, x0=[0.0, 0.0], history=True
)
```

Optional manifold callbacks:

`gd(...)`, `gn(...)`, and `lm(...)` accept `manifold=...` with these methods:
- `retract(x, direction, alpha) -> list[float]`
- `tangent_norm(v) -> float`
- `scale(v, alpha) -> list[float]`
- `add(x, v) -> list[float]`
- `difference(x, y) -> list[float]`
