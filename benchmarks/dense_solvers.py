"""Reproducible one-step backend comparison; requires NumPy and local liteopt.
Run: python benchmarks/dense_solvers.py --output benchmarks/dense_results.json
Times include Python callbacks, solver/workspace setup, checks and one update.
"""
from decimal import Decimal, localcontext
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import liteopt


def cases():
    rng = np.random.default_rng(723)
    a = rng.normal(size=(32, 6))
    u = np.arange(1., 9.)
    v = np.array([1., -1.] * 4)
    return {
        'well_conditioned': a,
        'near_dependent': np.column_stack((u, u + 1e-8*v)),
        'mixed_units': np.diag([1e-8, 1., 1e8]),
        'tall': rng.normal(size=(128, 8)),
        'wide': rng.normal(size=(4, 12)),
        'rank_deficient': np.column_stack((u, u)),
    }


def high_precision_reference(a, b, damping):
    # Independent 80-digit normal-equation reference on the exact input floats.
    # Used only for full-column-rank GN or positive-damping LM.
    with localcontext() as ctx:
        ctx.prec = 80
        m,n = a.shape
        d = [[Decimal.from_float(float(v)) for v in row] for row in a]
        rhs = [Decimal.from_float(float(v)) for v in b]
        lam = Decimal.from_float(damping)
        h = [[sum(d[k][i]*d[k][j] for k in range(m)) + (lam if i==j else 0)
              for j in range(n)] for i in range(n)]
        g = [sum(d[k][i]*rhs[k] for k in range(m)) for i in range(n)]
        for i in range(n):
            pivot = max(range(i,n),key=lambda row: abs(h[row][i]))
            h[i],h[pivot] = h[pivot],h[i]
            g[i],g[pivot] = g[pivot],g[i]
            if h[i][i] == 0: raise ValueError('singular reference')
            for row in range(i+1,n):
                f = h[row][i]/h[i][i]
                for col in range(i+1,n): h[row][col] -= f*h[i][col]
                g[row] -= f*g[i]
        x = [Decimal(0)]*n
        for i in range(n-1,-1,-1):
            x[i] = (g[i]-sum(h[i][j]*x[j] for j in range(i+1,n)))/h[i][i]
        return np.array([float(v) for v in x])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    results = []
    for name, a in cases().items():
        m, n = a.shape
        target = np.linspace(.5, 1.5, n)
        b = a @ target
        for method, damping, backends in [('gn', 0., ['normal_jtj', 'left_jjt', 'qr']),
                ('lm', 1e-3, ['left_jjt', 'qr']), ('lm', 1e-12, ['left_jjt', 'qr'])]:
            aug = np.vstack((a, np.sqrt(damping)*np.eye(n))) if damping else a
            rhs = np.r_[b, np.zeros(n)] if damping else b
            # Equilibrate before the SVD reference to retain differently scaled columns.
            scales = np.linalg.norm(aug, axis=0)
            scaled = aug / scales
            z, _, rank, _ = np.linalg.lstsq(scaled, rhs, rcond=np.finfo(float).eps*max(scaled.shape))
            reference = z / scales
            # GN QR deliberately has no minimum-norm/rank-deficient mode.
            full_rank = rank == n
            if full_rank or damping:
                reference = high_precision_reference(a,b,damping)
            for backend in backends:
                opts = dict(linear_system=backend, max_iters=1, tol_r=0., tol_grad=0., tol_dx=0.,
                            line_search_method='none')
                if method == 'lm': opts['lambda'] = damping
                def run():
                    return liteopt.least_squares(lambda x: a@x-b, [0.]*n, method=method,
                        jacobian=lambda x: a, options=opts, debug={'info': True})
                out = run()
                elapsed = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    for _ in range(args.repeats): run()
                    elapsed.append((time.perf_counter_ns()-start)/args.repeats/1000)
                x = np.asarray(out[0])
                # For wide GN use NumPy's unscaled minimum-norm reference instead.
                ref = np.linalg.lstsq(a, b, rcond=None)[0] if m < n and not damping else reference
                error = float(np.linalg.norm(x-ref)/max(1., np.linalg.norm(ref)))
                results.append(dict(case=name, shape=[m,n], method=method, damping=damping,
                    backend=backend, status=out[-1]['status'], accepted=out[-1]['n_accepted'],
                    relative_error=error, residual=float(np.linalg.norm(a@x-b)),
                    microseconds=statistics.median(elapsed), reference_full_column_rank=bool(full_rank)))
    payload = dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        seed=723, repeats=args.repeats, build="release", reference="80-digit Decimal normal equations on input floats; NumPy minimum-norm SVD for singular/wide GN", timing='median of 3 batches; microseconds per full one-step Python solve', results=results)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    for row in results:
        print(f"{row['case']:17} {row['method']} {row['damping']:.0e} {row['backend']:10} "
              f"err={row['relative_error']:.2e} us={row['microseconds']:.1f} {row['status']}")

if __name__ == '__main__': main()
