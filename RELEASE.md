# Release Notes and Checklist

## Release Notes

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
