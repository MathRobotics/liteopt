import numpy as np
import pytest
import liteopt


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    df_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    df_dy = 200.0 * (x[1] - x[0] ** 2)
    return np.array([df_dx, df_dy])


def test_gd_rosenbrock_reaches_minimum():
    x0 = [-1.2, 1.0]
    x_star, f_star, converged = liteopt.gd(
        rosenbrock,
        rosenbrock_grad,
        x0,
        step_size=1e-3,
        max_iters=200_000,
        tol_grad=1e-6,
    )

    assert converged is True
    assert np.allclose(x_star, [1.0, 1.0], atol=1e-3)
    assert f_star == pytest.approx(0.0, abs=1e-6)
