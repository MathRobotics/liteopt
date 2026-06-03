# Release Checklist

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
uv sync --project liteopt-py --extra dev --reinstall
```

4. Build and install the Python bindings into the uv environment:

```bash
uv run --project liteopt-py maturin develop --manifest-path liteopt-py/Cargo.toml --release
```

5. Run the Python tests:

```bash
uv run --project liteopt-py pytest liteopt-py/tests
```

6. Build distribution artifacts:

```bash
uv run --project liteopt-py maturin build --manifest-path liteopt-py/Cargo.toml --release
```

7. Inspect the wheel filename and metadata version, then create the release tag
   using the Python package version, for example `v0.1.8`.

If the local uv environment starts importing namespace-only packages such as
`pytest` or `numpy`, remove `liteopt-py/.venv` and run the sync step again.
