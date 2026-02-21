# liteopt

A lightweight optimization library written in Rust with Python bindings.

## Installation

```bash
pip install liteopt
```

## Usage

```python
import liteopt

def f(x):
    x0, x1 = x
    return (1.0 - x0)**2 + 100.0 * (x1 - x0**2)**2

def grad(x):
    x0, x1 = x
    df_dx = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0**2)
    df_dy = 200.0 * (x1 - x0**2)
    return [df_dx, df_dy]

x0 = [-1.2, 1.0]
x_star, f_star, converged = liteopt.gd(f, grad, x0, step_size=1e-3, max_iters=200_000, tol_grad=1e-4)
print(converged, x_star, f_star)
```

Gauss-Newton for nonlinear least squares:

```python
import liteopt

def residual(x):
    # returns m-dimensional residual
    ...

def jacobian(x):
    # returns m x n Jacobian
    ...

x0 = [0.0, 0.0]
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(residual, jacobian, x0=x0)
print(ok, x_star, cost)
```

Custom manifold callbacks (optional):

```python
import math
import liteopt

def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class MyManifold:
    def retract(self, x, direction, alpha):
        # x_next = Retr_x(alpha * direction)
        return [wrap(xi + alpha * di) for xi, di in zip(x, direction)]

    def tangent_norm(self, v):
        return math.sqrt(sum(vi * vi for vi in v))

def residual(x):
    return [x[0] - 0.3, x[1] + 0.2]

def jacobian(x):
    return [[1.0, 0.0], [0.0, 1.0]]

x0 = [3.5, -4.0]
x_star, cost, *_ = liteopt.gn(residual, jacobian, x0=x0, manifold=MyManifold())
print(x_star, cost)
```

`gd(...)` and `gn(...)` accept `manifold=...`.
Supported methods on `manifold` are:
- `retract(x, direction, alpha) -> list[float]`
- `tangent_norm(v) -> float`
- `scale(v, alpha) -> list[float]`
- `add(x, v) -> list[float]`
- `difference(x, y) -> list[float]`
