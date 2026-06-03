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


def jacobian_vec(q, v):
    return jacobian(q) @ np.asarray(v, dtype=float)


def test_levenberg_marquardt_planar_two_link_converges_and_reaches_target():
    x_star, cost, _, rnorm, _, ok = liteopt.lm(residual, jacobian, x0=[0.0, 0.0], verbose=False)

    x_star = np.asarray(x_star, dtype=float)
    p_star = forward_kinematics(x_star)
    err = np.linalg.norm(p_star - TARGET)

    assert ok
    assert cost < 1e-12
    assert rnorm < 1e-6
    assert err < 1e-6


def test_levenberg_marquardt_accepts_jacobian_vec_without_dense_jacobian():
    x_star, cost, _, rnorm, _, ok = liteopt.lm(
        residual,
        x0=[0.0, 0.0],
        jacobian_vec=jacobian_vec,
        verbose=False,
    )

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


def test_levenberg_marquardt_can_return_history_with_option():
    x_star, cost, _, rnorm, _, ok, history = liteopt.lm(
        residual,
        jacobian,
        x0=[0.0, 0.0],
        max_iters=200,
        verbose=False,
        history=True,
    )

    x_star = np.asarray(x_star, dtype=float)
    p_star = forward_kinematics(x_star)
    err = np.linalg.norm(p_star - TARGET)

    assert ok
    assert cost < 1e-12
    assert rnorm < 1e-6
    assert err < 1e-6
    assert isinstance(history, list)
    assert len(history) > 0
    assert history[0]["solver"] == "lm"
    assert history[0]["note"] == "initial"
    assert any(row["note"] == "accepted" for row in history)


def test_levenberg_marquardt_history_option_works_with_custom_line_search():
    def custom_policy(ctx):
        return (True, 0.5 * float(ctx["alpha0"]))

    x_star, cost, _, rnorm, _, ok, history = liteopt.lm(
        residual,
        jacobian,
        x0=[0.0, 0.0],
        max_iters=200,
        verbose=False,
        history=True,
        line_search=custom_policy,
    )

    x_star = np.asarray(x_star, dtype=float)
    p_star = forward_kinematics(x_star)
    err = np.linalg.norm(p_star - TARGET)

    assert ok
    assert cost < 1e-12
    assert rnorm < 1e-6
    assert err < 1e-6
    assert len(history) > 0
    assert history[0]["solver"] == "lm"


def test_levenberg_marquardt_raises_for_invalid_jacobian_size():
    def bad_jacobian(_x):
        return np.zeros((1, 1), dtype=float)

    with pytest.raises(ValueError, match="jacobian size mismatch"):
        liteopt.lm(residual, bad_jacobian, x0=[0.0, 0.0], verbose=False)


def test_levenberg_marquardt_requires_jacobian_or_jacobian_vec():
    with pytest.raises(ValueError, match="jacobian or jacobian_vec must be provided"):
        liteopt.lm(residual, x0=[0.0, 0.0], verbose=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lambda_": -1.0}, "lambda_ must be finite and >= 0"),
        ({"lambda_": float("nan")}, "lambda_ must be finite and >= 0"),
        ({"lambda_up": 1.0}, "lambda_up must be finite and > 1"),
        ({"lambda_down": 1.0}, "lambda_down must be finite and in \\(0,1\\)"),
        ({"step_scale": 0.0}, "step_scale must be finite and in \\(0,1\\]"),
        ({"step_scale": 1.5}, "step_scale must be finite and in \\(0,1\\]"),
        ({"tol_r": -1.0}, "tol_r must be finite and >= 0"),
        ({"tol_dx": float("inf")}, "tol_dx must be finite and >= 0"),
    ],
)
def test_levenberg_marquardt_raises_for_invalid_solver_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        liteopt.lm(residual, jacobian, x0=[0.0, 0.0], verbose=False, **kwargs)
