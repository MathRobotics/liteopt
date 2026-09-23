"""Public callback contracts shared by GD, GN and LM."""
import pytest
import liteopt


def solve(method, options):
    if method == "gd":
        return liteopt.gd(lambda x: .5 * x[0] ** 2, lambda x: [x[0]], [1.],
                          options=options)
    return liteopt.least_squares(lambda x: [x[0]], [1.], method=method,
                                jacobian=lambda x: [1.], options=options)


@pytest.mark.parametrize("method", ["gd", "gn", "lm"])
@pytest.mark.parametrize("accepted", ["false", None, []])
def test_invalid_acceptance_flag_raises(method, accepted):
    with pytest.raises(TypeError):
        solve(method, {"line_search": lambda ctx: {"accepted": accepted, "alpha": .5}})


@pytest.mark.parametrize("method", ["gd", "gn", "lm"])
def test_retraction_composes_scale_and_add_when_not_overridden(method):
    calls = []

    class Manifold:
        def scale(self, v, alpha):
            calls.append("scale")
            return [alpha * v[0]]

        def add(self, x, v):
            calls.append("add")
            # A simple retraction into the nonnegative half-line.
            return [max(0.75, x[0] + v[0])]

    result = solve(method, {"manifold": Manifold(), "step_size": 1., "max_iters": 1})
    if method == "gd":
        assert calls.pop(0) == "scale"  # GD constructs -gradient first.
    assert calls
    assert all(calls[i:i + 2] == ["scale", "add"] for i in range(0, len(calls), 2))
    assert result[0] == [0.75]


@pytest.mark.parametrize("method", ["gd", "gn", "lm"])
def test_explicit_retraction_overrides_primitive_hooks(method):
    class Manifold:
        def retract(self, x, direction, alpha):
            return [0.75]

        def scale(self, v, alpha):
            assert method == "gd" and alpha == -1.
            return [-v[0]]

        def add(self, x, v):
            raise AssertionError("retract must take precedence")

    assert solve(method, {"manifold": Manifold(), "max_iters": 1})[0] == [0.75]


def test_damping_limit_preserves_rejected_search_history():
    result = liteopt.least_squares(
        lambda x: [x[0]], [1.], jacobian=lambda x: [1.],
        options={"lambda": 1., "lambda_max": 1., "line_search": lambda ctx: (False, 1.)},
        debug={"history": True, "info": True})
    history, info = result[-2:]
    assert info["status"] == "damping_limit"
    assert info["n_attempts"] == 1 and info["n_retries"] == 0
    rejected = [row for row in history if row["phase"] == "search"]
    assert len(rejected) == 1
    assert rejected[0]["accepted"] is False
    assert rejected[0]["lambda"] == 1.
    assert rejected[0]["lambda_next"] is None
