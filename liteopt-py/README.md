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