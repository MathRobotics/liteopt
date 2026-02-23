import numpy as np
import pytest
import liteopt


TARGET = np.array([1.2, 0.6], dtype=float)
L1 = 1.0
L2 = 1.0


def forward_kinematics(q):
    q = np.asarray(q, dtype=float)
    q1, q2 = q
    return np.array(
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L1 * np.sin(q1) + L2 * np.sin(q1 + q2),
        ],
        dtype=float,
    )


def residual(q):
    return forward_kinematics(q) - TARGET


def jacobian(q):
    q = np.asarray(q, dtype=float)
    q1, q2 = q
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    return np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ],
        dtype=float,
    )


def test_levenberg_marquardt_planar_two_link_converges_and_reaches_target():
    x_star, cost, _, rnorm, _, ok = liteopt.lm(residual, jacobian, x0=[0.0, 0.0], verbose=False)

    x_star = np.asarray(x_star, dtype=float)
    p_star = forward_kinematics(x_star)
    err = np.linalg.norm(p_star - TARGET)

    assert ok
    assert cost < 1e-12
    assert rnorm < 1e-6
    assert err < 1e-6


def test_levenberg_marquardt_supports_custom_line_search_callback():
    calls = {"n": 0}

    def custom_policy(ctx):
        calls["n"] += 1
        return (True, 0.5 * float(ctx["alpha0"]))

    x_star, cost, _, rnorm, _, ok = liteopt.lm(
        residual,
        jacobian,
        x0=[0.0, 0.0],
        max_iters=200,
        verbose=False,
        line_search=custom_policy,
    )

    x_star = np.asarray(x_star, dtype=float)
    p_star = forward_kinematics(x_star)
    err = np.linalg.norm(p_star - TARGET)

    assert ok
    assert cost < 1e-12
    assert rnorm < 1e-6
    assert err < 1e-6
    assert calls["n"] > 0


def test_levenberg_marquardt_raises_for_invalid_jacobian_size():
    def bad_jacobian(_x):
        return np.zeros((1, 1), dtype=float)

    with pytest.raises(ValueError, match="jacobian size mismatch"):
        liteopt.lm(residual, bad_jacobian, x0=[0.0, 0.0], verbose=False)
