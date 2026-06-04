# liteopt Python Examples

Run these commands from `liteopt-py` after installing the local bindings:

```bash
uv sync --extra dev
uv run maturin develop --manifest-path Cargo.toml
```

Run all bundled examples:

```bash
uv run python example/run.py all
```

Run one solver example:

```bash
uv run python example/run.py gd
uv run python example/run.py gn
uv run python example/run.py lm
uv run python example/run.py manifold
```

The examples cover:

- `gd`: gradient descent on a one-dimensional quadratic
- `gn`: Gauss-Newton on a small least-squares problem
- `lm`: Levenberg-Marquardt on a two-link inverse-kinematics problem
- `manifold`: Gauss-Newton with angle wrapping via `manifold.retract`
