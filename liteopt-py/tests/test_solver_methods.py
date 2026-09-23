"""GN is undamped; LM can backtrack while retaining adaptive damping."""
import liteopt
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_linear_step_matches_method_equation(method):
    j = np.array([[1., 2.], [3., -1.], [2., 1.]])
    target = np.array([1., 2., -1.])
    damping = 0. if method == "gn" else 0.5
    options = {"max_iters": 1}
    if method == "lm":
        options["lambda"] = damping
    x, cost, *_ = liteopt.least_squares(
        lambda x: j @ x - target, [0., 0.], method=method,
        jacobian=lambda x: j, options=options,
    )
    expected = np.linalg.solve(j.T @ j + damping * np.eye(2), j.T @ target)
    np.testing.assert_allclose(x, expected, atol=1e-12)
    assert cost == pytest.approx(0.5 * np.linalg.norm(j @ expected - target) ** 2)


@pytest.mark.parametrize("options", [{"lambda": 0.}, {"lambda_": 1e-3}, {"damping_update": "fixed"}])
def test_gn_rejects_damping_options(options):
    with pytest.raises(ValueError, match="unknown options key"):
        liteopt.least_squares(lambda x: [x[0]-1.], [0.], method="gn",
                             jacobian=lambda x: [1.], options=options)


def test_singular_gn_stops_but_lm_can_solve():
    def residual(x):
        return [x[0] + x[1] - 1., 2. * (x[0] + x[1] - 1.)]

    kwargs = dict(jacobian=lambda x: [1., 1., 2., 2.], debug={"history": True})
    gn = liteopt.least_squares(residual, [0., 0.], method="gn", **kwargs)
    assert not gn[5]
    assert gn[0] == [0., 0.]
    assert gn[6][-1]["note"] == "linear_solve_failed"
    assert not any(row["lambda"] is not None for row in gn[6])
    lm = liteopt.least_squares(residual, [0., 0.], method="lm", **kwargs)
    assert lm[5]
    assert sum(lm[0]) == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize("method", ["gn", "lm"])
@pytest.mark.parametrize("search", ["armijo", "strict_decrease"])
@pytest.mark.parametrize("use_jvp", [False, True])
def test_backtracking_reduces_step_and_cost(method, search, use_jvp):
    derivatives = {"jacobian_vec": lambda x, v: [2. * x[0] * v[0]],
                   "jacobian_transpose_vec": lambda x, w: [2. * x[0] * w[0]]} if use_jvp else {
        "jacobian": lambda x: [2. * x[0]]}
    result = liteopt.least_squares(
        lambda x: [x[0] ** 2 - 1.], [0.1], method=method, **derivatives,
        options={"line_search_method": search, "max_iters": 1}, debug={"history": True},
    )
    x, cost, _, _, _, _, history = result
    row = next(row for row in history if row["note"] == "accepted")
    assert 0. < row["alpha"] < 1.
    assert cost < 0.5 * (0.1 ** 2 - 1.) ** 2
    direction = 0.198 / (0.04 + (0.001 if method == "lm" else 0.))
    assert x[0] == pytest.approx(0.1 + row["alpha"] * direction)


@pytest.mark.parametrize("search, note", [("cost_decrease", "rejected"), ("none", "accepted")])
def test_lm_non_backtracking_policies(search, note):
    result = liteopt.least_squares(
        lambda x: [x[0] ** 2 - 1.], [0.1], method="lm", jacobian=lambda x: [2. * x[0]],
        options={"line_search_method": search, "max_iters": 1}, debug={"history": True},
    )
    row = next(row for row in result[6] if row["note"] == note)
    assert result[6][-1]["note"] == "max_iters"
    if note == "rejected":
        assert result[0] == [0.1]
        assert row["lambda"] == pytest.approx(0.001)
        assert row["lambda_next"] == pytest.approx(0.01)
    else:
        assert result[1] > 0.49005


def test_lm_default_remains_cost_decrease():
    kwargs = dict(residual=lambda x: [x[0] ** 2 - 1.], x0=[0.1], method="lm",
                  jacobian=lambda x: [2. * x[0]], debug={"history": True})
    default = liteopt.least_squares(**kwargs)
    explicit = liteopt.least_squares(**kwargs, options={"line_search_method": "cost_decrease"})
    assert default == explicit
    assert default[5]


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_failed_backtracking_has_solver_specific_behavior(method):
    result = liteopt.least_squares(
        lambda x: [x[0] ** 2 - 1.], [0.1], method=method, jacobian=lambda x: [2. * x[0]],
        options={"line_search_method": "armijo", "ls_max_steps": 1, "max_iters": 100},
        debug={"history": True},
    )
    if method == "gn":
        assert not result[5]
        assert result[0] == [0.1]
        assert result[6][-1]["note"] == "rejected"
    else:
        assert result[5]
        assert result[0][0] == pytest.approx(1., abs=1e-6)
        assert any(row["note"] == "rejected" for row in result[6])
        assert any(row["note"] == "accepted" for row in result[6])


@pytest.mark.parametrize("options", [
    {"line_search_method": "unknown"}, {"ls_beta": 0.}, {"ls_beta": 1.},
    {"ls_beta": float("nan")}, {"ls_min_step": 0.}, {"ls_max_steps": 0},
    {"c_armijo": 0.}, {"c_armijo": 1.},
])
def test_lm_validates_builtin_search_options(options):
    with pytest.raises(ValueError):
        liteopt.least_squares(lambda x: [x[0]-1.], [0.], method="lm", jacobian=lambda x: [1.],
                             options=options)
