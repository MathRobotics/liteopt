import numpy as np
import pytest
import liteopt


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def rosenbrock_grad(x):
    x = np.asarray(x, dtype=float)
    return [
        -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
        200.0 * (x[1] - x[0] ** 2),
    ]


def quadratic_1d(x):
    x0 = float(np.asarray(x, dtype=float)[0])
    return (x0 - 3.0) ** 2


def quadratic_1d_grad(x):
    x0 = float(np.asarray(x, dtype=float)[0])
    return [2.0 * (x0 - 3.0)]


def test_gradient_descent_rosenbrock_converges_and_reduces_objective():
    x0 = [-1.2, 1.0]
    f0 = rosenbrock(x0)

    x_star, f_star, converged = liteopt.gd(
        rosenbrock,
        rosenbrock_grad,
        x0,
        step_size=1e-3,
        max_iters=200_000,
        tol_grad=1e-4,
    )

    x_star = np.asarray(x_star, dtype=float)
    assert converged
    assert f_star < 1e-6
    assert f_star < f0 * 1e-4
    assert np.allclose(x_star, [1.0, 1.0], atol=5e-2)


def test_gradient_descent_respects_maximum_iterations():
    x0 = [0.0]

    _, f_one_step, converged_one_step = liteopt.gd(
        quadratic_1d,
        quadratic_1d_grad,
        x0,
        step_size=0.1,
        max_iters=1,
        tol_grad=1e-12,
    )
    _, f_many_steps, converged_many_steps = liteopt.gd(
        quadratic_1d,
        quadratic_1d_grad,
        x0,
        step_size=0.1,
        max_iters=200,
        tol_grad=1e-9,
    )

    assert not converged_one_step
    assert converged_many_steps
    assert f_many_steps < f_one_step


def test_gradient_descent_step_size_changes_single_step_result():
    x0 = [0.0]

    x_small, _, _ = liteopt.gd(
        quadratic_1d,
        quadratic_1d_grad,
        x0,
        step_size=0.01,
        max_iters=1,
        tol_grad=0.0,
    )
    x_large, _, _ = liteopt.gd(
        quadratic_1d,
        quadratic_1d_grad,
        x0,
        step_size=0.1,
        max_iters=1,
        tol_grad=0.0,
    )

    x_small = float(np.asarray(x_small, dtype=float)[0])
    x_large = float(np.asarray(x_large, dtype=float)[0])
    assert 0.0 < x_small < x_large < 3.0
