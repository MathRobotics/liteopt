# Release Notes and Checklist

## Release Notes

### Unreleased

#### Breaking changes and migration

- **Step size:** rename GN/LM `step_scale` to `step_size` in Python `options`
  and Rust solver fields. No compatibility alias is provided. The rename
  preserves GN/LM defaults and range handling. The GN trace note
  `zero_step_scale` is now `zero_step_size`.
- **Undamped GN:** remove Python `lambda`, `lambda_`, and `damping_update`,
  Rust `lambda` and `damping_update`, and `GaussNewtonDampingUpdate`.
  GN now defaults to `normal_jtj` with Armijo backtracking. Singular systems
  and failed searches terminate without damping retries; use LM when damping
  is needed, or choose a nonsingular initial configuration.
- **Python validation:** GD requires a positive finite `step_size`, a finite
  nonnegative `tol_grad`, and finite initial points and gradients.
  Backtracking factors and Armijo constants must be strictly between zero
  and one.

#### Direct / iterative linear solver selection

- Add `linear_solver="direct" | "cg"` to GN/LM. Dense inputs keep the existing
  direct defaults; a product pair defaults to matrix-free CG. Explicit CG also
  supports dense input without forming normal equations.
- **Migration:** `jacobian_vec` now requires `jacobian_transpose_vec`. Remove
  implicit dense reconstruction. Direct solves require `jacobian`; QR remains
  a direct backend. Explicit `linear_system` is invalid with CG.
- Add `cg_max_iters`, `cg_rtol`, `cg_atol`, true-residual verification,
  inner-solve status/history and iteration counts, and `n_jtvp`.
  CG failure never commits a partial direction: GN stops, LM retries with
  increased damping. Python product shape/finiteness errors and exceptions propagate.
- Rust adds `LinearSolver`, `CgOptions`, `Jacobian`, `JacobianProducts`,
  `solve_with_derivatives*`, and result diagnostics. Explicit solver struct
  literals need `linear_solver` and `cg`; default values preserve direct solves.

#### Python API and line searches

- Add `liteopt.least_squares(residual, x0, method="lm", ...)` as the common
  nonlinear least-squares entry point. LM is the default; select `method="gn"`
  for GN. Existing `gn()` and `lm()` calls remain supported. The wrapper uses
  the selected method's options, callbacks, and return tuple, including
  optional history; options exclusive to the other method are rejected.
  README tables and examples use the common entry point.
- Expose built-in GD and LM searches through `line_search_method`:
  `none`, `cost_decrease`, `armijo`, and `strict_decrease`. GD defaults to
  `none`; LM defaults to `cost_decrease`. Backtracking settings are `ls_beta`,
  `ls_min_step`, `ls_max_steps`, and `c_armijo`.
- LM increases damping after a rejected search and retries on the next outer
  iteration. Rust's configured solve methods honor
  `LevenbergMarquardtLineSearchMethod` and its settings.

#### GN/LM termination and numerical checks

- Add `tol_grad` (default `1e-6`) for stationarity based on `||J.T @ residual||`.
  Check initial and final points even with zero or exhausted iteration budgets.
  Nonzero residuals can now terminate successfully at stationary points.
- Small directions alone no longer imply success: `tol_dx` (`tol_dq` in Rust)
  reports `stalled` unless a residual or gradient criterion is satisfied.
  Set both `tol_r` and `tol_grad` when requesting stricter accuracy.
- Validate initial points, Jacobians, tolerances, and numerical settings.
  Reject non-finite trial residuals/projected points; Python exceptions propagate.
  Rust invalid configurations return `invalid_options` instead of silently
  replacing settings. GN retains finite step-size clamping to `[0, 1]`.
- Guard LM damping increases against overflow. A zero initial `lambda` uses
  the existing `1e-12` floor so rejection can increase damping.
- Rust GN/LM configuration structs gain `tol_grad`; result structs gain
  `status`. Explicit Rust struct literals must include the new fields or use
  defaults. Python return tuple shapes are unchanged.
- GN/LM append a terminal history row with the reason. `dx_norm` denotes the
  last computed direction before scaling/projection (zero if none), while
  cost and residual norm describe the returned point.

#### Callback and boundary fixes

- Reject invalid Python line-search `accepted` values instead of silently
  treating them as `True`.
- Compose Python manifold `scale` and `add` hooks when `retract` is omitted,
  matching Rust's default `Space` behavior.
- Built-in Rust line-search policies reject non-finite trial costs and invalid
  step sizes; backtracking validates its settings and Armijo requires a finite
  descent slope even when used directly.
- Preserve rejected LM attempt history when a damping bound prevents retrying.
  Share the damping retry logic across rejection paths.

#### GD correctness fixes

- Evaluate convergence and gradient norm at the returned point, including
  zero-iteration runs and the last permitted update.
- Evaluate the objective at every fixed-step candidate to reject non-finite
  costs. Backtracking can recover from non-finite trial costs by reducing the
  step; Python callback exceptions still propagate.
- Save history independently of logging, including accepted/rejected step
  sizes and termination reasons.

#### LM gain-ratio damping

- Add `damping_update="gain_ratio"` alongside the existing default `cost_based`.
  Use actual/predicted reduction to accept steps and increase, retain, or
  decrease damping. The reduction model uses the selected step width and
  local displacement after projection/retraction.
- Gain-ratio acceptance is enforced after built-in or custom line search,
  including `line_search_method="none"`. Invalid predictions and insufficient
  ratios reject the candidate without modifying the accepted point.
- Add `lambda_min`/`lambda_max` to Python and Rust and
  `LevenbergMarquardtDampingUpdate` to Rust. Defaults preserve the previous
  initial damping/floor; bounds and overflow have explicit termination reasons.
- History adds actual/predicted reduction and gain ratio. Comparison data and
  the decision to retain the default are in `benchmarks/lm_damping.md`.

#### Dense QR backend

- Add opt-in `linear_system="qr"` to Python GN/LM; defaults remain GN
  `normal_jtj` and LM `left_jjt`. Rust gains `GaussNewtonLinearSystem::Qr`
  and `LevenbergMarquardtLinearSystem` with a `linear_system` configuration field.
- Use dependency-free column-equilibrated, pivoted Householder QR. LM solves
  an augmented system with damping rows. Allocate only the selected backend's
  dense matrix workspace and reuse QR buffers across iterations.
- GN QR reports numerical rank deficiency/unsupported wide shape explicitly,
  without implicit regularization. LM retains damping retries on solve failure.
- Record reproducible precision/timing comparisons and the extreme-scaling,
  weak-damping limitation in `benchmarks/README.md` and raw JSON results.

#### Results and diagnostics

- Add GN/LM `debug.info` with termination reason, final gradient norm, actual
  residual/Jacobian/JVP callback counts, update attempts, accepted updates,
  damping retries, and line-search trial counts. Add matching counters to Rust
  results and extend GD info with attempt/search counters. Default tuples remain unchanged.
- History gains `phase`, `accepted`, `ls_trials`, and `lambda_next`. GD now
  emits an initial row. GN/LM search rows report pre-step cost/residual;
  terminal rows report the returned point. LM `lambda` now means damping used
  for the attempt; the updated value moves to `lambda_next`.
- Evaluate the final GN/LM Jacobian even on residual-based convergence so
  final gradient diagnostics are available. Callback exceptions still propagate.
- Apply `ls_min_step` to both Armijo and StrictDecrease for every solver,
  including the first trial. Rust `ArmijoBacktracking` gains `min_step`
  (default `1e-8`) and `with_min_step`.

- GD accepts `debug={"info": True}` to append `iters`, `grad_norm`, `status`,
  `nfev`, and `njev` in an info dictionary. The default `(x, f, ok)` return is
  unchanged; enabling both history and info returns `(x, f, ok, history, info)`.
  Rust `OptimizeResult` gains `status`, `nfev`, and `njev`.
- GN failure trace notes are `linear_solve_failed`, `rejected`, and
  `accepted_step_invalid`, without the old `_fixed` suffix. GN no longer
  records lambda in history; custom Rust/Python search contexts still expose
  `lambda=0`.

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

For `gd(...)`, move `step_size`, `max_iters`, `tol_grad`, `line_search`, and
`manifold` into `options`; move `verbose` and `history` into `debug`.

For `gn(...)`, move `lambda_`/`lambda`, `step_scale`, `max_iters`, `tol_r`,
`tol_dx`, `damping_update`, `linear_system`, `line_search_method`,
`line_search`, `ls_beta`, `ls_min_step`, `ls_max_steps`, `c_armijo`, and
`manifold` into `options`; move `verbose` and `history` into `debug`.

For `lm(...)`, move `lambda_`/`lambda`, `lambda_up`, `lambda_down`,
`step_scale`, `max_iters`, `tol_r`, `tol_dx`, `line_search`, and `manifold`
into `options`; move `verbose` and `history` into `debug`.

## Release Checklist

Use `liteopt-py/pyproject.toml` as the canonical package version for Python
releases. The Rust crate versions in `liteopt-core/Cargo.toml` and
`liteopt-py/Cargo.toml` are internal workspace metadata unless the Rust crates
are published separately.

1. Update `liteopt-py/pyproject.toml` `[project].version`.
2. Run the Rust test suite:

```bash
cargo test --workspace
```

3. Recreate or refresh the Python development environment:

```bash
cd liteopt-py
uv sync --extra dev --reinstall
```

4. Build and install the Python bindings into the uv environment:

```bash
uv run maturin develop --manifest-path Cargo.toml --release
```

5. Run the Python tests:

```bash
uv run pytest tests
```

6. Build distribution artifacts:

```bash
uv run maturin build --manifest-path Cargo.toml --release
```

7. Inspect the wheel filename and metadata version, then create the release tag
   using the Python package version, for example `v0.1.8`.

If the local uv environment starts importing namespace-only packages such as
`pytest` or `numpy`, remove `liteopt-py/.venv` and run the sync step again.
