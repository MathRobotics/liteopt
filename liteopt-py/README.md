# liteopt

A lightweight optimization library written in Rust with Python bindings.

## Installation

```bash
pip install liteopt
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

Optional manifold callbacks:

`gd(...)` and `gn(...)` accept `manifold=...` with these methods:
- `retract(x, direction, alpha) -> list[float]`
- `tangent_norm(v) -> float`
- `scale(v, alpha) -> list[float]`
- `add(x, v) -> list[float]`
- `difference(x, y) -> list[float]`
