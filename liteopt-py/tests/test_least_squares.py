"""The common least-squares API preserves each solver's behavior."""
import liteopt
import pytest


@pytest.mark.parametrize("method", ["gn", "lm"])
@pytest.mark.parametrize("use_jvp", [False, True])
@pytest.mark.parametrize("history", [False, True])
def test_common_api_matches_direct_solver(method, use_jvp, history):
    options = {"max_iters": 100, "tol_r": 1e-10, "tol_dx": 1e-10, "tol_grad": 1e-10}
    options.update(
        {"linear_system": "normal_jtj", "line_search_method": "strict_decrease"}
        if method == "gn" else {"lambda_up": 5., "lambda_down": 0.25}
    )
    kwargs = dict(options=options, debug={"history": history})
    if use_jvp:
        kwargs["jacobian_vec"] = lambda x, v: [v[0], 2. * v[1]]
        kwargs["jacobian_transpose_vec"] = lambda x, w: [w[0], 2. * w[1]]
        options.pop("linear_system", None)
    else:
        kwargs["jacobian"] = lambda x: [1., 0., 0., 2.]

    def residual(x):
        return [x[0] - 1., 2. * (x[1] + 2.)]

    actual = liteopt.least_squares(residual, [0., 0.], method=method, **kwargs)
    expected = getattr(liteopt, method)(residual, [0., 0.], **kwargs)
    assert actual == expected
    assert actual[5]
    assert actual[0] == pytest.approx([1., -2.], abs=1e-8)
    assert len(actual) == (7 if history else 6)
    if history:
        assert actual[6]
        assert all(row["solver"] == method for row in actual[6])


def test_default_method_is_levenberg_marquardt():
    kwargs = dict(residual=lambda x: [x[0] - 1.], x0=[0.], jacobian=lambda x: [1.],
                  options={"lambda": 0.5, "max_iters": 1}, debug={"history": True})
    actual = liteopt.least_squares(**kwargs)
    assert actual == liteopt.lm(**kwargs)
    assert actual == liteopt.least_squares(method="lm", **kwargs)
    assert actual[0][0] == pytest.approx(1. / 1.5)
    assert all(row["solver"] == "lm" for row in actual[6])


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_projection_and_manifold_are_forwarded(method):
    def run(solver):
        calls = {"retract": 0, "project": 0}

        class Manifold:
            def retract(self, x, direction, alpha):
                calls["retract"] += 1
                return [x[0] + alpha * direction[0]]

        def project(x):
            calls["project"] += 1
            return [min(x[0], 0.5)]

        result = solver(
            lambda x: [x[0] - 1.], [0.], jacobian=lambda x: [1.],
            project=project, options={"manifold": Manifold(), "max_iters": 2},
        )
        return result, calls

    actual, calls = run(lambda *args, **kwargs: liteopt.least_squares(*args, method=method, **kwargs))
    expected, expected_calls = run(getattr(liteopt, method))
    assert actual[0] == expected[0] == [0.5]
    assert actual[1] == expected[1]
    assert actual[5] == expected[5]
    assert calls == expected_calls
    assert calls["retract"] > 0 and calls["project"] > 0


@pytest.mark.parametrize("method", ["gd", "unknown", "GN", ""])
def test_unknown_method_is_rejected_before_evaluating_problem(method):
    def residual(x):
        pytest.fail("Invalid methods must not evaluate callbacks")

    with pytest.raises(ValueError, match="method must be 'gn' or 'lm'"):
        liteopt.least_squares(residual, [0.], method=method)


@pytest.mark.parametrize("method, options", [
    ("gn", {"lambda_up": 10.}),
    ("lm", {"linear_system": "normal_jtj"}),
    ("gn", {"method": "lm"}),
])
def test_options_are_validated_for_selected_method(method, options):
    with pytest.raises(ValueError, match="linear_system must be" if method == "lm" else "unknown options key"):
        liteopt.least_squares(lambda x: [x[0] - 1.], [0.], method=method,
                             jacobian=lambda x: [1.], options=options)


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_callback_errors_are_preserved(method):
    def jacobian(x):
        raise RuntimeError("jacobian failed")

    with pytest.raises(RuntimeError, match="jacobian failed"):
        liteopt.least_squares(lambda x: [x[0] - 1.], [0.], method=method, jacobian=jacobian)


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_derivative_is_required(method):
    with pytest.raises(ValueError, match="jacobian or jacobian_vec must be provided"):
        liteopt.least_squares(lambda x: [x[0] - 1.], [0.], method=method)
