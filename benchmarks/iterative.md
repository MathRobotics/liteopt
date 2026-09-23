# Direct / CG comparison

Fixed seed 732; release wheel; Python 3.13.1; NumPy 2.3.5.
Platform: macOS-15.7.4-arm64-arm-64bit-Mach-O.

One outer iteration, five timed runs after warm-up; median whole Python-call time.
GN uses the default normal-equation direct solver; LM uses the default left system.
CG uses rtol=1e-10. The diagonal product callbacks multiply elementwise without building J.
These small synthetic cases are measurements, not a general speed guarantee.

| Case | Method | Route | Median ms | CG iterations | Direction error vs direct |
|---|---|---|---:|---:|---:|
| dense_64x16 | gn | direct | 0.062 | 0 | 0.00e+00 |
| dense_64x16 | gn | dense_cg | 0.062 | 16 | 1.62e-15 |
| dense_64x16 | gn | products_cg | 0.241 | 16 | 2.14e-15 |
| dense_64x16 | lm | direct | 0.143 | 0 | 0.00e+00 |
| dense_64x16 | lm | dense_cg | 0.061 | 16 | 1.41e-15 |
| dense_64x16 | lm | products_cg | 0.231 | 16 | 1.11e-15 |
| diagonal_128 | gn | direct | 1.347 | 0 | 0.00e+00 |
| diagonal_128 | gn | dense_cg | 0.604 | 21 | 7.71e-10 |
| diagonal_128 | gn | products_cg | 0.765 | 21 | 7.71e-10 |
| diagonal_128 | lm | direct | 4.060 | 0 | 0.00e+00 |
| diagonal_128 | lm | dense_cg | 1.032 | 21 | 7.62e-10 |
| diagonal_128 | lm | products_cg | 1.012 | 21 | 7.62e-10 |

The product route is slower on the small dense case because it crosses the Python
callback boundary repeatedly. CG is faster on this diagonal case. Dense-input CG
can also be competitive; it still stores J. Only the product route avoids dense J.
No preconditioner is implemented. Poor conditioning can exhaust the inner budget.

Reproduce: `python benchmarks/iterative_solvers.py --output benchmarks/iterative_results.json`.
