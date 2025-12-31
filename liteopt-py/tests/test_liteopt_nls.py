import numpy as np
import pytest
import liteopt


l1 = 1.0
l2 = 1.0
# Fixed target for determinism
_target = np.array([0.25, 1.35])


def residual(x: np.ndarray) -> np.ndarray:
    r = np.zeros(2)
    px = l1 * np.cos(x[0]) + l2 * np.cos(x[0] + x[1])
    py = l1 * np.sin(x[0]) + l2 * np.sin(x[0] + x[1])
    r[0] = px - _target[0]
    r[1] = py - _target[1]
    return r


def jacobian(x: np.ndarray) -> np.ndarray:
    J = np.zeros((2, 2))
    s1 = np.sin(x[0])
    c1 = np.cos(x[0])
    s12 = np.sin(x[0] + x[1])
    c12 = np.cos(x[0] + x[1])

    J[0, 0] = -l1 * s1 - l2 * s12
    J[0, 1] = -l2 * s12
    J[1, 0] = l1 * c1 + l2 * c12
    J[1, 1] = l2 * c12
    return J


def _forward_kinematics(x: np.ndarray) -> np.ndarray:
    return np.array([
        l1 * np.cos(x[0]) + l2 * np.cos(x[0] + x[1]),
        l1 * np.sin(x[0]) + l2 * np.sin(x[0] + x[1]),
    ])


def test_nls_two_link_reaches_target():
    x0 = [-1.2, 1.0]
    x_star, cost, iters, rnorm, dxnorm, converged = liteopt.nls(
        residual,
        jacobian,
        x0=x0,
        max_iters=200,
        tol_r=1e-10,
        tol_dx=1e-10,
    )

    assert converged is True
    assert rnorm == pytest.approx(0.0, abs=1e-6)
    assert dxnorm == pytest.approx(0.0, abs=1e-6)
    end_effector = _forward_kinematics(np.asarray(x_star))
    assert end_effector == pytest.approx(_target, abs=1e-6)
    assert cost == pytest.approx(0.0, abs=1e-6)
    assert iters <= 200
