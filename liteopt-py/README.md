# liteopt

A lightweight optimization library written in Rust with Python bindings.

## Scope

`liteopt` is aimed at small dense optimization problems where low dependency
cost and readable implementation matter. It provides basic GD/GN/LM solvers,
simple tolerances, simple step control, and optional debug traces.

It is not intended to be a large-scale sparse optimizer, a SciPy replacement, a
general sparse-factorization backend, or a BLAS/LAPACK-backed production solver.

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

Bundled examples are documented in [`example/README.md`](https://github.com/MathRobotics/liteopt/blob/main/liteopt-py/example/README.md).

## Choosing a Solver

| Problem | Python API | Required derivatives |
|---|---|---|
| General differentiable objective `f(x)` | `liteopt.gd(f, grad, x0, ...)` | Gradient |
| Nonlinear least squares, Gauss-Newton | `liteopt.least_squares(residual, x0, method="gn", ...)` | Jacobian or Jacobian-vector product |
| Nonlinear least squares, Levenberg-Marquardt | `liteopt.least_squares(residual, x0, method="lm", ...)` | Jacobian or Jacobian-vector product |

`least_squares` minimizes `0.5 * ||residual(x)||^2`. The default method is
`"lm"`. Change `method` to switch algorithms while keeping the same problem
callbacks and result handling. Existing `liteopt.gn(...)` and `liteopt.lm(...)`
remain available as direct entry points.

Both methods return `(x, cost, iters, r_norm, dx_norm, ok)`. With
`debug={"history": True}`, a seventh item, `history`, is appended.

| Settings in `options` | Availability |
|---|---|
| `step_size`, `max_iters`, `tol_r`, `tol_grad`, `tol_dx`, `manifold` | Both methods |
| `line_search_method`, `ls_beta`, `ls_min_step`, `ls_max_steps`, `c_armijo` | Both methods |
| `linear_system` | GN: `normal_jtj` (default), `left_jjt`, `qr`; LM: `left_jjt` (default), `qr` |
| `lambda` (alias `lambda_`), `lambda_up`, `lambda_down`, `lambda_min`, `lambda_max`, `damping_update` | `method="lm"` only |

Options are validated for the selected method. For example, `lambda_up` with
`method="gn"` raises `ValueError`. `method` is a keyword argument, not an entry
in `options`. Both methods accept `project`, `jacobian_vec`, `jacobian_transpose_vec`, and `debug`.
`line_search` retains each method's existing behavior: GN accepts a boolean or
callback, while LM accepts a callback. GN defaults to Armijo; LM defaults to
CostDecrease. Select built-in algorithms through `line_search_method`.

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

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian=jacobian,
)
print(ok, x_star, cost)
```

`jacobian` must be either:
- row-major 1D list (`list[float]`, length = `m * n`)
- 2D `numpy.ndarray` (`shape = (m, n)`)

Alternatively, provide both product callbacks and omit `jacobian`:

```python
def jacobian_vec(x, v):
    # The example above has an identity Jacobian.
    return [v[0], v[1]]

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian_vec=jacobian_vec,
    jacobian_transpose_vec=lambda x, w: [w[0], w[1]],
    options={"linear_solver": "cg"},
)
```

`jacobian_vec(x, v)` returns `J(x) @ v` (length `m`) for a vector of
length `n`. `jacobian_transpose_vec(x, w)` returns `J(x).T @ w` (length
`n`) for a vector of length `m`. Both must be linear in the vector argument,
represent the same Jacobian at the current `x`, and be adjoints in Euclidean
local coordinates. Shape, finite values, and callback exceptions are checked;
linearity and the adjoint identity are the caller's responsibility.

The product path never constructs a dense Jacobian or normal-equation matrix.
Unlike earlier versions, `jacobian_vec` alone is rejected; supply the transpose
product too, or provide a dense `jacobian`.

Levenberg-Marquardt (least squares):

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="lm",
    x0=[0.0, 0.0],
    jacobian=jacobian,
)
print(ok, x_star, cost)
```

Both methods accept the same pair of product callbacks.

## Direct and iterative linear solvers

GN and LM share their outer convergence checks, line search, and damping logic
across both backends. Select the internal linear solver through `options`:

| Input and selection | Behavior |
|---|---|
| `jacobian`, omitted `linear_solver` | Existing direct solver |
| Product pair, omitted `linear_solver` | CG, without dense matrices |
| `linear_solver="direct"` | Requires `jacobian`; `linear_system` selects the existing direct backend |
| `linear_solver="cg"` with product pair | Uses products; does not call an optional dense `jacobian` |
| `linear_solver="cg"` with only `jacobian` | Evaluates dense J, then applies CG without forming J.T @ J |

When all three callbacks are supplied, the default is CG using products.
Specify `direct` to use the dense callback instead. A partial product pair is
always an error. Explicit `linear_system` with CG is rejected, since QR and
the other direct backends do not participate in CG.

CG solves `(J.T @ J + lambda * I) d = -J.T @ r`, with zero damping for GN.
It starts from zero on every direction solve and requires the true linear
residual norm to satisfy `max(cg_atol, cg_rtol * ||J.T @ r||)`.
This residual is recomputed before declaring success.

| Option | Default | Meaning |
|---|---|---|
| `cg_max_iters` | `100` | Positive inner iteration limit per direction attempt |
| `cg_rtol` | `1e-6` | Relative linear residual tolerance, strictly between 0 and 1 |
| `cg_atol` | `0.0` | Finite nonnegative absolute linear residual tolerance |

Inner tolerances are separate from outer `tol_grad`, `tol_r`, and `tol_dx`.
An overly loose absolute tolerance can return a zero direction and cause
outer stagnation. Failed inner solves are never committed: GN terminates;
LM increases damping and retries. Non-finite operator results terminate
immediately (Python callback violations raise exceptions).

`debug.info` includes `linear_solver`, `matrix_free`, cumulative
`n_linear_iters`, and the last `linear_status` / `linear_residual_norm`
(`None` if CG never ran). CG history records have `phase="linear"`,
`linear_iters`, `linear_residual_norm`, and a status in `note`:
`linear_converged`, `linear_max_iters`, `linear_breakdown`, or
`linear_non_finite`. The outer termination reason remains in `status`.

This is unpreconditioned CG, not a rank-revealing factorization. GN's
positive-definite guarantee requires a full-column-rank J; singular problems
can break down. LM's positive damping regularizes the system, but very small
damping and poor scaling can still slow convergence. QR remains available
through the dense route. No automatic conversion to dense storage or fallback
to a direct solve occurs.

The product route uses O(m+n) solver workspace. Runtime depends on product
cost and inner iterations; it is not always faster than direct factorization.
The CG algorithm follows the standard
[Netlib Templates description](https://www.netlib.org/templates/templates.html).

## Step Size

All solvers use `options={"step_size": ...}`. It is the multiplier applied to
the search direction, or the initial trial multiplier when using backtracking.
GD defaults to `1e-3`; GN and LM default to `1.0`.

GD requires a finite, positive value. GN clamps it to `[0, 1]` (zero stops
without taking a step), and LM requires a finite value in `(0, 1]`.

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    x0=[0.0, 0.0],
    method="lm",
    jacobian=jacobian,
    options={"step_size": 0.5},
)
```

## Convergence and Debug Options

Convergence control is intentionally small:

- `gd(...)`: `max_iters`, `tol_grad`
- `least_squares(...)`: `max_iters`, `tol_r`, `tol_grad`, `tol_dx`

GN computes an undamped Gauss-Newton direction. By default it solves
`(J.T @ J) @ d = -J.T @ residual` (`linear_system="normal_jtj"`) and uses
Armijo backtracking. It terminates with `ok=False` if the linear system or
line search fails. The dense elimination backend requires a nonsingular
normal matrix: singular initial configurations no longer receive automatic
regularization. Use a full-column-rank Jacobian or choose LM. The optional
`linear_system="left_jjt"` computes `d = -J.T @ solve(J @ J.T, residual)`;
this requires full row rank and can be used for underdetermined problems.

LM supports `damping_update="cost_based"` (default) and `"gain_ratio"`.
The former multiplies damping by `lambda_up` after rejection and `lambda_down`
after acceptance. The latter additionally checks the ratio of actual to
predicted objective reduction and adjusts damping according to model quality;
see [LM damping updates](#lm-damping-updates).
The effective initial damping is `max(lambda, lambda_min)`. Defaults are
`lambda=1e-3`, `lambda_min=1e-12`, and `lambda_max=f64::MAX`.
An overflowing rejected-step increase terminates with `damping_overflow`;
an increase above the configured upper bound terminates with `damping_limit`.

GN and LM check convergence at the initial point and the final returned point,
including `max_iters=0` and the last allowed update. Success means either
`||residual|| <= tol_r` or `||J.T @ residual|| <= tol_grad` (both default to
`1e-6`). The latter detects stationarity even when the residual is nonzero;
it does not guarantee a global minimum. For stricter accuracy, set both
tolerances explicitly. `tol_dx` (Rust: `tol_dq`) now detects stagnation:
a small direction without either success condition returns `ok=False` with
`stalled`, including with a large initial LM damping value.

The return tuple remains `(x, cost, iters, r_norm, dx_norm, ok)` plus optional
history. `iters` counts completed outer iterations, including LM retries.
`dx_norm` is the last successfully computed direction norm before applying
`alpha` or projection, or zero if no direction was computed. It is not a
new direction evaluated at the returned point. `cost` and `r_norm` describe
the returned point. The final history row records the termination reason;
its `grad_norm` is evaluated at that point, including residual-based success
(otherwise `None` if evaluation fails). This requires a final Jacobian
evaluation even when the residual is already below tolerance.
Rust result structs also expose `status`.

Invalid Python options, empty/non-finite initial points, and non-finite
Jacobians/Jacobian-vector products raise `ValueError`. Non-finite initial
residuals terminate with `non_finite_residual`; non-finite trial residuals or
projected points are rejected and may be recovered by backtracking or LM
retries. Python exceptions propagate. Rust rejects invalid configurations
with `invalid_options` and zero dimensions with `invalid_dimensions`, rather
than silently replacing invalid search/damping settings. GN still clamps a
finite `step_size` to `[0, 1]`; LM requires `(0, 1]`. Both require finite,
nonnegative tolerances. Custom manifold tangent norms must be finite and
nonnegative.

| `line_search_method` | GD | GN | LM | Search acceptance |
|---|---|---|---|---|
| `"armijo"` | Available | Default | Available | Backtrack until sufficient decrease |
| `"strict_decrease"` | Available | Available | Available | Backtrack until cost decreases |
| `"cost_decrease"` | Available | Not exposed | Default | One trial; accept only if cost decreases |
| `"none"` | Default | Available | Available | One trial; accept any finite cost, even if it increases |

For backtracking, `step_size` is the initial trial multiplier. `ls_beta`
(default `0.5`) controls shrinking and `ls_max_steps` (default `20`) caps the
number of trials. `ls_min_step` (default `1e-8`) bounds both Armijo and
StrictDecrease trials for every solver, including the initial trial.
`c_armijo` (default `1e-4`) is used by Armijo. A custom `line_search` callback overrides the selected built-in policy.

Enable LM backtracking as follows:

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    x0=[0.0, 0.0],
    method="lm",
    jacobian=jacobian,
    options={"line_search_method": "armijo", "ls_beta": 0.5, "ls_max_steps": 20},
)
```

Trace history is disabled by default. Enable it only when inspecting solver
behavior:

```python
x_star, cost, iters, r_norm, dx_norm, ok, history = liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian=jacobian,
    debug={"history": True},
)
```

Gauss-Newton also supports strict decrease backtracking:

```python
x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="gn",
    x0=[0.0, 0.0],
    jacobian=jacobian,
    options={
        "linear_system": "normal_jtj",
        "line_search_method": "strict_decrease",
        "line_search": True,
        "ls_max_steps": 12,
    },
)
```

## Optional Manifold Callbacks

`gd(...)` and `least_squares(...)` accept `options={"manifold": ...}` with
these methods:
- `retract(x, direction, alpha) -> list[float]`: apply the local update
  `alpha * direction` at point `x` and return the next point.
- `tangent_norm(v) -> float`: return the norm used for convergence checks on a
  tangent/update vector.
- `scale(v, alpha) -> list[float]`: scale a tangent/update vector.
- `add(x, v) -> list[float]`: add a tangent/update vector to a point.
- `difference(x, y) -> list[float]`: return the local update vector from point
  `x` to point `y`.

When `retract` is omitted, the update composes `scale(direction, alpha)`
and `add(x, scaled_direction)`; omitted primitive operations use Euclidean
behavior. An explicit `retract` takes precedence for point updates.
GD also uses `scale(gradient, -1)` to construct its search direction.
These hooks must preserve the point/tangent dimensions used by the solver.

Use `manifold.retract(...)` when the optimizer should update points on a
manifold instead of using the default Euclidean step. For `least_squares(...)`,
`project=...` is a lighter post-step projection hook for simple constraints.

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

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.least_squares(
    residual,
    method="gn",
    x0=[3.0 * math.pi, -2.0 * math.pi + 0.3],
    jacobian=jacobian,
    options={"manifold": WrappedAngles()},
)
```

The bundled `manifold` example in [`example/run.py`](https://github.com/MathRobotics/liteopt/blob/main/liteopt-py/example/run.py) shows the
same pattern in a complete inverse-kinematics problem.

## Migration Notes

For release notes and breaking-change migration examples, see
[`../RELEASE.md`](https://github.com/MathRobotics/liteopt/blob/main/RELEASE.md).

## GD diagnostics

GD checks `grad_norm <= tol_grad` at the initial point and after every accepted
update, including the last allowed update. `tol_grad` must be finite and
nonnegative; `max_iters=0` evaluates the initial point without updating it.
Non-finite gradients raise `ValueError` in Python. Non-finite objective values
at trial points are rejected, allowing backtracking to try a smaller step;
Python callback exceptions propagate. A non-finite initial objective terminates
with `ok=False` and status `non_finite`.

The default return remains `(x, f, ok)`. `debug={"history": True}` appends history
independently of `verbose`. History begins with an initial row. Accepted/rejected search rows record the pre-step
objective and gradient, attempted `alpha`, and `note`; a final row records the
returned point and termination reason. `debug={"info": True}` appends a dictionary
containing `iters` (accepted updates), `grad_norm` at the returned point, `status`,
`nfev` (objective calls, including trials), and `njev` (gradient calls). When both
are enabled the return is `(x, f, ok, history, info)`.

```python
x, f, ok, history, info = liteopt.gd(
    lambda x: x[0] ** 2,
    lambda x: [2 * x[0]],
    [1.0],
    options={"step_size": 2.0, "line_search_method": "armijo"},
    debug={"history": True, "info": True},
)
print(info["status"], info["grad_norm"], next(row["alpha"] for row in history if row["accepted"] is True))
```

Statuses are `converged`, `max_iters`, `line_search_failed`, `non_finite`,
`invalid_step`, and (Rust configuration validation) `invalid_options`.
Only `converged` sets `ok=True`. Invalid Python options raise `ValueError`.
A custom search's acceptance does not add an Armijo/decrease check; its accepted
step must still be positive and finite and produce a finite point and cost.

## Common result diagnostics

All solvers accept `debug={"info": True}` without enabling history or logging.
The info dictionary is appended **after** optional history:

| API | Default return | Both `history` and `info` enabled |
|---|---|---|
| `gd` | `(x, f, ok)` | `(x, f, ok, history, info)` |
| `least_squares`, `gn`, `lm` | `(x, cost, iters, r_norm, dx_norm, ok)` | `(x, cost, iters, r_norm, dx_norm, ok, history, info)` |

| Info key | Meaning |
|---|---|
| `status` | Termination reason, available without history |
| `grad_norm` | Gradient norm at the returned point; least-squares uses `J.T @ residual`; `None` for GN/LM if unavailable |
| `iters` | Completed outer iterations: accepted updates plus completed LM damping-retry transitions |
| `n_attempts` | Update attempts entered after convergence/budget checks, including an attempt that fails immediately |
| `n_accepted` | Updates actually committed |
| `n_retries` | LM damping increases scheduling a retry, including at the iteration limit; zero for GD/GN |
| `nfev` | Actual objective (GD) or residual (GN/LM) callback calls, including trial costs and acceptance verification |
| `njev` | Gradient evaluations for GD; dense Jacobian evaluations for GN/LM; zero in the product route |
| `n_ls_trials` | Calls to the trial evaluator *inside* line search; excludes final acceptance verification |
| `n_jac_calls` | GN/LM only: actual calls to the supplied `jacobian` callback |
| `n_jvp` | GN/LM only: actual calls to `jacobian_vec` |
| `n_jtvp` | GN/LM only: actual calls to `jacobian_transpose_vec` |

Python GN/LM `nfev` includes the initial call used to infer residual dimension.
Rust receives the dimension explicitly and does not make that extra call.
Product calls include gradients, inner iterations, true-residual checks, and
LM gain-ratio predictions where needed. Acceptance verification can repeat a cost
already evaluated during search; `nfev` includes those calls. A Python custom
search has no trial evaluator, so its `n_ls_trials` is zero, but its proposed
point is still evaluated before acceptance. Retraction/projection/line-search
callback calls themselves are not included in `nfev` or `njev`.

`iters` can be zero while `n_attempts` is one if the first attempt fails.
For completed runs, `iters = n_accepted + n_retries`; a failed attempt that
cannot increase damping does not add a retry. No count is inferred from the
length of history.

```python
x, cost, iters, r_norm, dx_norm, ok, history, info = liteopt.least_squares(
    lambda x: [x[0] ** 2 - 1.0],
    [0.1],
    jacobian=lambda x: [2.0 * x[0]],
    options={"line_search_method": "armijo"},
    debug={"history": True, "info": True},
)
print(info["status"], info["grad_norm"], info["nfev"], info["n_retries"])
for row in history:
    if row["phase"] == "search":
        print(row["alpha"], row["accepted"], row["ls_trials"])
```

### History contract

| `phase` | Meaning |
|---|---|
| `initial` | Initial objective/residual evaluation; omitted when configuration/initial point validation fails first |
| `search` | Result of one search: pre-step cost/residual and, when present, pre-step gradient/direction |
| `iteration` | Other iteration diagnostics, such as linear-solve failure or direction fallback |
| `final` | Returned point's cost/residual and termination reason; `dx_norm` retains its documented last-direction meaning |

`iter` is the number of completed outer iterations before the event.
`accepted` is `True` only for committed search steps, `False` for rejected or
invalid proposed steps, and `None` on other rows. `ls_trials` is the per-search
counter; it is `None` outside search rows. `alpha` on an accepted row is the
actual multiplier used. On failure it is the policy's returned multiplier,
which may be below the minimum and need not have been evaluated.
LM `lambda` records damping used for that attempt; `lambda_next` records the
updated damping for the next attempt. GN does not record damping fields.
History collection remains independent of `verbose`. Full candidate-point
histories are not collected.

### Line-search option compatibility

| Option | GD | GN | LM |
|---|---|---|---|
| Default `line_search_method` | `none` | `armijo` | `cost_decrease` |
| `line_search` callable | Overrides built-in policy | Overrides built-in policy | Overrides built-in policy |
| `line_search` bool | Not supported | `True`: configured policy; `False`: no search | Not supported |
| `ls_beta`, `ls_max_steps`, `ls_min_step` | Armijo / StrictDecrease | Armijo / StrictDecrease | Armijo / StrictDecrease |
| `c_armijo` | Armijo only | Armijo only | Armijo only |

All supplied numerical settings are validated even when a custom policy
supersedes them. Custom acceptance still requires a positive finite multiplier
and a finite candidate/cost, but does not add an Armijo check. LM with
`damping_update="gain_ratio"` additionally enforces its reduction-ratio check,
even with `line_search_method="none"` or a custom callback.
`ls_min_step` bounds built-in backtracking only; it does not constrain fixed
steps or custom callbacks.

## QR backend for dense least squares

Both GN and LM accept `options={"linear_system": "qr"}`. The defaults remain
GN `normal_jtj` and LM `left_jjt`. QR uses column equilibration and pivoted
Householder transformations without adding a runtime dependency.

```python
x, cost, iters, r_norm, dx_norm, ok, info = liteopt.least_squares(
    lambda x: [x[0] - 1.0, 2.0 * x[0] - 3.0],
    [0.0],
    method="gn",
    jacobian=lambda x: [1.0, 2.0],
    options={"linear_system": "qr"},
    debug={"info": True},
)
print(x, info["status"])
```

GN QR requires at least as many residuals as unknowns and numerical full
column rank. It reports `qr_invalid_shape` for underdetermined systems and
`qr_rank_deficient` when the normalized rank test fails. It does not silently
regularize or return a pseudoinverse solution. For full-row-rank wide problems,
use GN `left_jjt`; LM QR handles wide/rank-deficient Jacobians through the
explicit damping rows `[J; sqrt(lambda) I]`.

QR can improve accuracy for nearly dependent columns, but cannot guarantee
accurate small components with extreme scaling and weak LM damping. It also
has a different time/memory tradeoff from the existing backends. See the
[reproducible comparison](https://github.com/MathRobotics/liteopt/blob/main/benchmarks/README.md) for errors, timings, numerical
rank policy, and limitations. A QR factorization failure does not affect the
last accepted point; LM may increase damping and retry.

## LM damping updates

```python
x, cost, iters, r_norm, dx_norm, ok, history, info = liteopt.least_squares(
    lambda x: [x[0] ** 2 - 1.0],
    [0.1],
    jacobian=lambda x: [2.0 * x[0]],
    options={"damping_update": "gain_ratio", "line_search_method": "armijo"},
    debug={"history": True, "info": True},
)
for row in history:
    if row["gain_ratio"] is not None:
        print(row["gain_ratio"], row["lambda"], row["lambda_next"])
```

For a proposed local displacement `s`, the undamped residual model is
`m(s) = 0.5 * ||r + J s||²`. Define:

```text
predicted_reduction = -(Jᵀr)ᵀs - 0.5 * ||J s||²
actual_reduction    = cost(x) - cost(candidate)
gain_ratio         = actual_reduction / predicted_reduction
```

The prediction models reduction of the original objective, not the objective
plus damping penalty. The step is the actual displacement after the selected
`alpha`, retraction and projection. It is obtained with `Space::difference`
(Python `manifold.difference(x, candidate)`), defaulting to `candidate - x`.
Custom manifolds must provide a difference in the same local coordinates as
`J`; angle wrapping or other nonlinear retractions may require an explicit
implementation. This is a model-based LM update, not a general constrained
trust-region solver.

After the chosen line search accepts a candidate, `gain_ratio` applies:

| Ratio | Accept candidate? | Next damping |
|---|---|---|
| Invalid prediction/ratio, or `rho <= 1e-4` | No | `lambda * lambda_up` |
| `1e-4 < rho < 0.25` | Yes | `min(lambda * lambda_up, lambda_max)` |
| `0.25 <= rho <= 0.75` | Yes | Unchanged |
| `rho > 0.75` | Yes | `max(lambda * lambda_down, lambda_min)` |

The thresholds are fixed policy constants. Defaults remain `lambda_up=10`
and `lambda_down=0.5`. Rejection by line search or failure to solve the linear
system also increases damping. Rejected points are never committed.
Zero/negative/non-finite predictions and non-finite ratios are recorded as
`invalid_prediction`; insufficient finite ratios as `gain_ratio_rejected`.
Both cause retries unless the damping bound or iteration budget is exhausted.
An accepted step whose next damping is capped still counts as accepted, not
as a retry. A small direction without convergence still reports `stalled`.

`lambda_min` must be positive and finite; `lambda_max` must be finite and at
least `lambda_min`. The initial `lambda` may be zero but may not exceed
`lambda_max`. Both policies honor these bounds. Damping is still isotropic
(`lambda I`) in the supplied coordinates; no automatic variable scaling is
introduced. Normalize variables when their characteristic scales differ.
QR column equilibration preserves the specified damping problem and is not
an automatic choice of physical parameter scales.

History adds `predicted_reduction`, `actual_reduction`, and `gain_ratio`.
Predictions/ratios are computed only for candidates accepted by the search
when using `gain_ratio`; unavailable/non-finite values appear as `None`.
The terminal row retains the returned-point semantics. These diagnostics do
not require extra residual/Jacobian evaluations.

A [fixed comparison](https://github.com/MathRobotics/liteopt/blob/main/benchmarks/lm_damping.md) covers nonlinear problems,
poor initial guesses, nonzero optimal residuals, variable scaling, and Armijo
combinations. Neither policy dominates all evaluation counts, so the default
remains `cost_based` and `gain_ratio` is explicit.
