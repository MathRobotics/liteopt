"""Fixed direct/CG comparisons. Run with a local release wheel installed."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import liteopt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(732)
    cases = []
    a = rng.normal(size=(64, 16))
    cases.append(("dense_64x16", 16, lambda v: a @ v, lambda w: a.T @ w, lambda: a))
    diagonal = np.linspace(1., 2., 128)
    cases.append(("diagonal_128", 128, lambda v: diagonal * v,
                  lambda w: diagonal * w, lambda: np.diag(diagonal)))
    results = []
    for name, n, forward, transpose, dense in cases:
        target = np.linspace(-1., 1., n)
        b = forward(target)
        for method in ("gn", "lm"):
            reference = None
            for route in ("direct", "dense_cg", "products_cg"):
                options = dict(linear_solver="direct" if route == "direct" else "cg",
                               max_iters=1, tol_r=0., tol_grad=0., tol_dx=0., cg_rtol=1e-10)
                callbacks = dict(jacobian=lambda x: dense())
                if route == "products_cg":
                    callbacks = dict(jacobian_vec=lambda x, v: forward(v),
                                     jacobian_transpose_vec=lambda x, w: transpose(w))
                def run():
                    return liteopt.least_squares(lambda x: forward(x)-b, [0.]*n,
                        method=method, options=options, debug={"info": True}, **callbacks)
                run()  # warm up
                timings = []
                for _ in range(5):
                    start = time.perf_counter()
                    out = run()
                    timings.append(time.perf_counter()-start)
                if reference is None:
                    reference = np.asarray(out[0])
                error = float(np.linalg.norm(np.asarray(out[0])-reference))
                assert out[-1]["n_accepted"] == 1
                assert error < 1e-7
                results.append(dict(case=name, method=method, route=route,
                    median_seconds=statistics.median(timings), direction_error=error, **out[-1]))
    args.output.write_text(json.dumps(dict(platform=platform.platform(),
        python=platform.python_version(), numpy=np.__version__, build="release",
        seed=732, results=results), indent=2)+"\n")
    for row in results:
        print(row["case"], row["method"], row["route"],
              f'{1e3*row["median_seconds"]:.3f} ms',
              "inner", row["n_linear_iters"], "error", f'{row["direction_error"]:.2e}')


if __name__ == "__main__":
    main()
