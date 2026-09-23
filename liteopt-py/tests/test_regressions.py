"""Regressions for runtime dependencies, array layouts, and damping retries."""
from importlib.metadata import requires

import liteopt
import numpy as np
import pytest
from packaging.requirements import Requirement


def test_numpy_is_an_unconditional_runtime_dependency():
    numpy_requirements = [
        Requirement(value) for value in requires("liteopt")
        if Requirement(value).name.lower() == "numpy"
    ]
    assert any(requirement.marker is None for requirement in numpy_requirements)


@pytest.mark.parametrize("solver", [liteopt.gn, liteopt.lm])
@pytest.mark.parametrize("layout", ["C", "F", "transpose", "slice", "reverse"])
def test_jacobian_layout_preserves_the_linear_step(solver, layout):
    matrix = np.array([[1., 2.], [3., 4.], [2., -1.]])
    if layout in ("C", "F"):
        jacobian = np.array(matrix, order=layout)
    elif layout == "transpose":
        jacobian = matrix.T.copy().T
    elif layout == "slice":
        storage = np.zeros((6, 4))
        storage[::2, ::2] = matrix
        jacobian = storage[::2, ::2]
    else:
        jacobian = matrix[::-1, ::-1].copy()[::-1, ::-1]
    target = matrix @ np.array([1., -2.])
    damping = 1e-3 if solver is liteopt.lm else 0.0
    expected = np.linalg.solve(matrix.T @ matrix + damping * np.eye(2), matrix.T @ target)
    x, cost, *_ = solver(
        lambda x: matrix @ x - target,
        [0., 0.],
        jacobian=lambda x: jacobian,
        options={"max_iters": 1},
    )
    np.testing.assert_allclose(x, expected, rtol=1e-9, atol=1e-10)
    assert cost == pytest.approx(0.5 * np.linalg.norm(matrix @ expected - target) ** 2)


@pytest.mark.parametrize("solver", [liteopt.gn, liteopt.lm])
@pytest.mark.parametrize("accept_first", [False, True])
def test_rejected_steps_report_stagnation(solver, accept_first):
    result = solver(
        lambda x: [x[0] - 1.],
        [0.],
        jacobian=lambda x: [1.],
        options={"line_search": lambda ctx: (accept_first and ctx["iter"] == 0, 0.5)},
        debug={"history": True},
    )
    x, cost, _, r_norm, dx_norm, ok, history = result
    assert not ok
    assert x[0] == pytest.approx((0.5 / 1.001 if solver is liteopt.lm else 0.5) if accept_first else 0.)
    assert r_norm > 0.4
    assert cost > 0.1
    if solver is liteopt.lm:
        assert dx_norm <= 1e-6
        assert history[-1]["note"] == "stalled"
    else:
        assert history[-1]["note"] == "rejected"


@pytest.mark.parametrize("solver", [liteopt.lm])
def test_solver_can_recover_after_rejection(solver):
    # This problem has a nonzero residual at its least-squares minimum, so
    # success must still be possible through the gradient criterion.
    x, _, _, r_norm, _, ok = solver(
        lambda x: [x[0] - 1., 1.],
        [0.],
        jacobian=lambda x: [1., 0.],
        options={"line_search": lambda ctx: (ctx["iter"] > 0, 1.)},
    )
    assert ok
    assert x[0] == pytest.approx(1., abs=1e-6)
    assert r_norm == pytest.approx(1.)


@pytest.mark.parametrize("solver", [liteopt.gn, liteopt.lm])
def test_stationary_initial_point_with_nonzero_residual_still_converges(solver):
    x, _, iters, r_norm, _, ok = solver(
        lambda x: [x[0] - 1., 1.], [1.], jacobian=lambda x: [1., 0.]
    )
    assert ok
    assert x == [1.]
    assert iters == 0
    assert r_norm == 1.
