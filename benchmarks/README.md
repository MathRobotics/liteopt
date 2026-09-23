# Dense backend comparison

Recorded 2026-09-23. Decision: add opt-in `linear_system="qr"` for GN and LM;
keep GN `normal_jtj` and LM `left_jjt` as defaults.

## Reproduction

Build and install the **release** wheel, then run from the repository root:

```bash
liteopt-py/.venv/bin/python benchmarks/dense_solvers.py --output benchmarks/dense_results.json
```

The script fixes seed 723 and six matrices (well-conditioned 32×6, nearly
collinear 8×2, diagonal 3×3 with scales 1e-8/1/1e8, tall 128×8, wide 4×12,
and rank-deficient 8×2). Targets are generated deterministically. All solves
start at zero, take at most one step, disable backtracking, and use zero
stopping tolerances. LM damping is tested at 1e-3 and 1e-12.

Reference solutions use 80-digit Decimal elimination on the normal equations
formed from the exact input floats. At this precision the reference retains
the digits lost by double-precision normal equations in these cases. Singular
and wide GN cases instead use NumPy's SVD least-squares reference; these are
not supported by the full-column-rank QR backend. Reference computation is
excluded from timings. Independent SVD comparisons also appear in the tests.

Timings are the median of three batches of ten calls, in microseconds per
whole Python solve: callbacks, workspace allocations, diagnostic checks, and
one update are included. This is not an isolated factorization benchmark.
Environment: macOS 15.7.4 arm64, Python 3.13.1, NumPy 2.3.5, Rust release build.
These small samples measure this workload and machine, not general performance.
Raw data: [dense_results.json](dense_results.json).

## Results

Relative error is `||x - reference|| / max(1, ||reference||)`.
`max_iters` is expected after one step with zero tolerances and is not a
linear-solve failure. A useful step is defined here as an accepted update
with relative error at most 1e-6; it is not the nonlinear convergence flag.
Failure timings must not be compared as successful solve speedups.

| Case | Method / damping | Backend | Relative error | Residual norm | μs | Accepted |
|---|---|---|---:|---:|---:|---|
| well_conditioned | gn / 0e+00 | normal_jtj | 3.48e-16 | 4.17e-15 | 17.7 | yes |
| well_conditioned | gn / 0e+00 | qr | 6.72e-16 | 9.19e-15 | 33.6 | yes |
| well_conditioned | lm / 1e-03 | left_jjt | 5.22e-16 | 4.21e-04 | 33.6 | yes |
| well_conditioned | lm / 1e-03 | qr | 5.61e-16 | 4.21e-04 | 41.9 | yes |
| well_conditioned | lm / 1e-12 | left_jjt | 2.61e-16 | 4.21e-13 | 32.1 | yes |
| well_conditioned | lm / 1e-12 | qr | 5.29e-16 | 4.21e-13 | 36.6 | yes |
| near_dependent | gn / 0e+00 | normal_jtj | 1.00e+00 | 2.86e+01 | 10.0 | linear_solve_failed |
| near_dependent | gn / 0e+00 | qr | 9.24e-09 | 3.14e-15 | 15.3 | yes |
| near_dependent | lm / 1e-03 | left_jjt | 7.85e-17 | 7.00e-05 | 16.2 | yes |
| near_dependent | lm / 1e-03 | qr | 2.48e-16 | 7.00e-05 | 15.9 | yes |
| near_dependent | lm / 1e-12 | left_jjt | 1.84e-07 | 1.41e-08 | 15.5 | yes |
| near_dependent | lm / 1e-12 | qr | 5.14e-12 | 1.41e-08 | 15.8 | yes |
| mixed_units | gn / 0e+00 | normal_jtj | 1.00e+00 | 1.50e+08 | 10.0 | linear_solve_failed |
| mixed_units | gn / 0e+00 | qr | 0.00e+00 | 0.00e+00 | 15.2 | yes |
| mixed_units | lm / 1e-03 | left_jjt | 1.38e-16 | 9.99e-04 | 15.4 | yes |
| mixed_units | lm / 1e-03 | qr | 4.25e-14 | 9.99e-04 | 15.5 | yes |
| mixed_units | lm / 1e-12 | left_jjt | 1.38e-16 | 3.02e-08 | 15.2 | yes |
| mixed_units | lm / 1e-12 | qr | 2.77e-05 | 5.00e-09 | 15.8 | yes |
| tall | gn / 0e+00 | normal_jtj | 5.25e-16 | 1.79e-14 | 47.1 | yes |
| tall | gn / 0e+00 | qr | 9.24e-16 | 2.99e-14 | 147.5 | yes |
| tall | lm / 1e-03 | left_jjt | 6.40e-16 | 2.63e-04 | 345.5 | yes |
| tall | lm / 1e-03 | qr | 4.25e-16 | 2.63e-04 | 158.3 | yes |
| tall | lm / 1e-12 | left_jjt | 1.00e+00 | 3.45e+01 | 207.4 | max_iters |
| tall | lm / 1e-12 | qr | 8.73e-16 | 2.56e-13 | 157.2 | yes |
| wide | gn / 0e+00 | normal_jtj | 1.00e+00 | 8.88e+00 | 10.8 | linear_solve_failed |
| wide | gn / 0e+00 | left_jjt | 3.54e-16 | 2.18e-15 | 16.2 | yes |
| wide | gn / 0e+00 | qr | 1.00e+00 | 8.88e+00 | 10.0 | qr_invalid_shape |
| wide | lm / 1e-03 | left_jjt | 2.28e-16 | 5.03e-04 | 16.1 | yes |
| wide | lm / 1e-03 | qr | 1.79e-15 | 5.03e-04 | 22.1 | yes |
| rank_deficient | gn / 0e+00 | normal_jtj | 1.00e+00 | 2.86e+01 | 8.7 | linear_solve_failed |
| rank_deficient | gn / 0e+00 | qr | 1.00e+00 | 2.86e+01 | 8.9 | qr_rank_deficient |
| rank_deficient | lm / 1e-03 | left_jjt | 1.11e-16 | 7.00e-05 | 13.3 | yes |
| rank_deficient | lm / 1e-03 | qr | 4.23e-16 | 7.00e-05 | 13.4 | yes |

For the four full-column-rank GN matrices, QR produces useful steps in 4/4,
versus 2/4 for normal equations. GN's left system is only appropriate for
full-row-rank problems; its success on the wide case and failure on tall
cases are expected. Both LM backends produce useful steps in 11/12, failing
the accuracy criterion on different cases.

QR improves the near-dependent GN case and weakly damped tall LM case.
It is slower for the well-conditioned GN examples, but faster than the
m×m LM left system on the tall example. In the mixed-units, weak-damping LM
case, augmented QR has relative error about 2.77e-5 while the left system
remains accurate. Column equilibration does not eliminate loss of small
components when applying transformations to an extremely scaled right-hand
side. This limitation is retained in the data, rather than dropping the case.

## Implementation and scope

The new backend uses Householder transformations, column equilibration, and
column pivoting, with trailing norms recomputed using `hypot`. GN solves
`J d ≈ -r`; LM solves `[J; sqrt(lambda) I] d ≈ [-r; 0]`. Scaling is applied
to the entire augmented matrix and undone in the direction, so it does not
change the specified isotropic damping penalty. Workspaces are reused across
iterations and only the selected backend's matrix storage is allocated.
There are no new runtime dependencies. QR workspace is O(mn) for GN and
O((m+n)n) for LM; the existing LM left matrix alone is O(m²).

The rank threshold is `f64::EPSILON * max(rows, columns)` after unit-column
normalization. `qr_rank_deficient` denotes numerical rank deficiency at this
threshold, not proof of exact algebraic rank. GN QR requires m >= n and full
column rank; it supplies neither a pseudoinverse nor hidden regularization.
Use GN `left_jjt` for full-row-rank wide problems, or explicit LM damping.
LM can increase damping and retry after a QR failure. Other QR failures are
`qr_invalid_shape` and `qr_non_finite`. Existing normal/left backends retain
`linear_solve_failed` and their existing absolute pivot threshold (1e-12).

LAPACK's [DGELSY documentation](https://www.netlib.org/lapack/explore-html/dc/d8b/group__gelsy_ga6d1d46ead18df76e993cd4eda6dc1bbb.html)
describes column-pivoted QR followed by additional transformations for a
minimum-norm solution. This implementation uses only a full-rank QR solve;
it does not implement DGELSY's rank-deficient minimum-norm algorithm or its
condition-estimation rank test.

The evidence supports making QR available explicitly, not replacing every
default. Further default changes require broader nonlinear benchmarks and
analysis of the weak-damping/scaling limitation. This work does not change
LM's damping update algorithm (roadmap item 4).

For the subsequent damping-policy comparison, see [LM damping updates](lm_damping.md).
