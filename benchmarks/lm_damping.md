# LM damping update comparison

Recorded 2026-09-23 using the release build. Reproduce from the repository root:

```bash
liteopt-py/.venv/bin/python benchmarks/lm_damping.py
```

Raw data: [lm_damping_results.json](lm_damping_results.json).
The 12 cases comprise Rosenbrock at three initial guesses with two coordinate
scales (unit and 1e-3/1e3), a scalar square-root residual at three initial guesses,
and a nonlinear fit with a nonzero optimal residual at three initial guesses.
Each case runs both damping policies with single-trial cost decrease and Armijo.
All use the QR backend, identical initial damping (1e-3), multipliers (10, 0.5),
300 outer iterations, tol_r/tol_grad=1e-8 and tol_dx=1e-14. Times are not measured;
the comparison focuses on success and actual function/Jacobian work.

Success requires the solver success flag and cost < 1e-12 for zero-residual
problems, or gradient norm <= 1e-8 for the nonzero-residual fit.
Counts include the Python residual-dimension inference call and all failed
trials and acceptance verification. Totals include all runs, not just successes.

| Search | Damping update | Successes | Iterations | Residual calls | Jacobian calls | Retries |
|---|---|---:|---:|---:|---:|---:|
| cost_decrease | cost_based | 12/12 | 173 | 347 | 185 | 23 |
| cost_decrease | gain_ratio | 12/12 | 182 | 373 | 194 | 15 |
| armijo | cost_based | 12/12 | 127 | 378 | 139 | 0 |
| armijo | gain_ratio | 12/12 | 149 | 349 | 161 | 0 |

The gain-ratio policy reduces rejected-step retries with cost-decrease search,
but increases total residual/Jacobian evaluations. With Armijo it reduces
residual evaluations but requires more iterations and Jacobian evaluations.
The small fixed suite does not establish a universally better default.
Decision: retain `cost_based`, expose `gain_ratio` explicitly. No automatic
variable scaling is added: lambda I remains defined in the supplied coordinates.
Both policies succeeded here under mixed scales; this is not scale invariance.

The ratio compares reduction of the original least-squares objective with its
linearized residual model. It includes alpha and the actual local displacement
after projection/retraction; it does not include the damping penalty in the
predicted objective decrease. Nonpositive/nonfinite predictions or nonfinite
ratios are rejected. The chosen acceptance threshold (1e-4), model-quality
thresholds (0.25/0.75), and multiplicative lambda changes are explicit design
choices, rather than a claim to reproduce a particular external solver.

The model-quality approach is described in the
[GSL nonlinear least-squares documentation](https://www.gnu.org/software/gsl/doc/html/nls.html).
This implementation adjusts LM damping directly; it does not solve a general
constrained trust-region subproblem. Unit tests separately cover shortened
Armijo steps, projection, custom manifold differences, invalid predictions,
custom acceptance, lower/upper damping bounds, and non-finite arithmetic.
