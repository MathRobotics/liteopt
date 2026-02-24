#!/usr/bin/env python3
"""Run small `liteopt` examples from one entry point.

Usage:
  python example/run.py           # run all examples
  python example/run.py gd        # run one example
  python example/run.py gn
  python example/run.py lm
"""

from __future__ import annotations

import argparse
import math

try:
    import liteopt
except ImportError as exc:
    raise SystemExit(
        "Could not import `liteopt`. "
        "Run `uv run --extra dev maturin develop --manifest-path Cargo.toml` first."
    ) from exc


def run_gd() -> None:
    print("[gd] Minimize f(x) = (x - 3)^2 from x0 = [0.0]")

    def f(x):
        return (x[0] - 3.0) ** 2

    def grad(x):
        return [2.0 * (x[0] - 3.0)]

    x_star, f_star, ok = liteopt.gd(
        f,
        grad,
        x0=[0.0],
        step_size=0.1,
        max_iters=200,
        tol_grad=1e-9,
    )
    print(f"  converged={ok}, x*={x_star}, f*={f_star:.3e}")


def run_gn() -> None:
    print("[gn] Solve residual(x) = x - target with target = [1, -2]")
    target = [1.0, -2.0]

    def residual(x):
        return [x[0] - target[0], x[1] - target[1]]

    def jacobian(_x):
        # row-major: [J00, J01, J10, J11]
        return [1.0, 0.0, 0.0, 1.0]

    x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(
        residual,
        jacobian,
        x0=[0.0, 0.0],
        max_iters=100,
        tol_r=1e-12,
        tol_dx=1e-12,
        verbose=False,
    )
    print(
        "  "
        f"converged={ok}, x*={x_star}, cost={cost:.3e}, "
        f"iters={iters}, ||r||={r_norm:.3e}, ||dx||={dx_norm:.3e}"
    )


def run_lm() -> None:
    print("[lm] 2-link inverse kinematics toward target = [1.2, 0.6]")
    target = [1.2, 0.6]
    l1 = 1.0
    l2 = 1.0

    def forward_kinematics(q):
        q1, q2 = q
        return [
            l1 * math.cos(q1) + l2 * math.cos(q1 + q2),
            l1 * math.sin(q1) + l2 * math.sin(q1 + q2),
        ]

    def residual(q):
        p = forward_kinematics(q)
        return [p[0] - target[0], p[1] - target[1]]

    def jacobian(q):
        q1, q2 = q
        s1 = math.sin(q1)
        c1 = math.cos(q1)
        s12 = math.sin(q1 + q2)
        c12 = math.cos(q1 + q2)
        # row-major 2x2 jacobian
        return [
            -l1 * s1 - l2 * s12,
            -l2 * s12,
            l1 * c1 + l2 * c12,
            l2 * c12,
        ]

    x_star, cost, iters, r_norm, dx_norm, ok = liteopt.lm(
        residual,
        jacobian,
        x0=[0.0, 0.0],
        max_iters=200,
        tol_r=1e-12,
        tol_dx=1e-12,
        verbose=False,
    )
    p_star = forward_kinematics(x_star)
    err = math.hypot(p_star[0] - target[0], p_star[1] - target[1])
    print(
        "  "
        f"converged={ok}, q*={x_star}, cost={cost:.3e}, "
        f"iters={iters}, ||r||={r_norm:.3e}, ||dx||={dx_norm:.3e}, "
        f"target_error={err:.3e}"
    )


EXAMPLES = {
    "gd": run_gd,
    "gn": run_gn,
    "lm": run_lm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run liteopt Python examples")
    parser.add_argument(
        "name",
        nargs="?",
        default="all",
        choices=["all", *EXAMPLES],
        help="example to run",
    )
    args = parser.parse_args()

    if args.name == "all":
        for name in ("gd", "gn", "lm"):
            EXAMPLES[name]()
            print("")
        return

    EXAMPLES[args.name]()


if __name__ == "__main__":
    main()
