"""Direct and CG routes share the outer GN/LM algorithm."""
import numpy as np
import pytest
import liteopt


@pytest.mark.parametrize("method,shape", [
    ("gn", (8, 3)), ("gn", (3, 3)),
    ("lm", (8, 3)), ("lm", (3, 3)), ("lm", (3, 8)),
])
@pytest.mark.parametrize("products", [False, True])
def test_cg_step_matches_independent_linear_solution(method, shape, products):
    rng = np.random.default_rng(71)
    a = rng.normal(size=shape)
    b = rng.normal(size=shape[0])
    damping = .3 if method == "lm" else 0.
    expected = np.linalg.solve(a.T @ a + damping * np.eye(shape[1]), a.T @ b)
    callbacks = {"jacobian": lambda x: a}
    if products:
        def forbidden(x):
            raise AssertionError("dense callback must not be evaluated")
        callbacks = dict(jacobian=forbidden, jacobian_vec=lambda x, v: a @ v,
                         jacobian_transpose_vec=lambda x, w: a.T @ w)
    options = dict(linear_solver="cg", cg_rtol=1e-11, max_iters=1,
                   tol_r=0., tol_grad=0., tol_dx=0., line_search_method="none")
    if method == "lm":
        options["lambda"] = damping
    result = liteopt.least_squares(lambda x: a @ x - b, [0.] * shape[1],
        method=method, **callbacks, options=options, debug={"info": True, "history": True})
    np.testing.assert_allclose(result[0], expected, rtol=1e-9, atol=1e-10)
    info = result[-1]
    assert info["n_accepted"] == 1 and info["linear_status"] == "linear_converged"
    assert info["matrix_free"] is products
    assert info["n_jac_calls"] == (0 if products else 2)
    assert info["linear_residual_norm"] <= 1e-11 * np.linalg.norm(a.T @ b)
    rows = [row for row in result[-2] if row["phase"] == "linear"]
    assert len(rows) == 1
    assert info["n_linear_iters"] == rows[0]["linear_iters"]


@pytest.mark.parametrize("method,gain_ratio", [("gn", False), ("lm", False)])
@pytest.mark.parametrize("search", ["none", "armijo", "strict_decrease"])
def test_nonlinear_direct_and_matrix_free_paths_agree(method, search, gain_ratio):
    residual = lambda x: [x[0] ** 2 - 1.]
    options = {"line_search_method": search, "max_iters": 100, "tol_grad": 1e-9, "tol_r": 1e-9, "tol_dx": 1e-14}
    if method == "lm" and gain_ratio:
        options["damping_update"] = "gain_ratio"
    common = dict(method=method, project=lambda x: [max(.01, x[0])],
                  options=options, debug={"info": True, "history": True})
    dense = liteopt.least_squares(residual, [.1], jacobian=lambda x: [2*x[0]], **common)
    free = liteopt.least_squares(residual, [.1],
        jacobian_vec=lambda x, v: [2*x[0]*v[0]],
        jacobian_transpose_vec=lambda x, w: [2*x[0]*w[0]], **common)
    assert dense[5] and free[5]
    np.testing.assert_allclose(free[0], dense[0], atol=1e-8)
    assert free[-1]["n_jac_calls"] == free[-1]["njev"] == 0
    assert free[-1]["n_jvp"] > 0 and free[-1]["n_jtvp"] > 0
    if gain_ratio:
        assert any(row["gain_ratio"] is not None for row in free[-2])


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_large_identity_needs_no_basis_probes(method):
    n = 5000
    calls = {"v": 0, "t": 0}
    def forward(x, v):
        calls["v"] += 1
        # A basis-probing implementation would trigger this.
        assert np.count_nonzero(v) == n
        return v
    def transpose(x, w):
        calls["t"] += 1
        return w
    out = liteopt.least_squares(lambda x: np.asarray(x)-1., [0.]*n, method=method,
        jacobian_vec=forward, jacobian_transpose_vec=transpose,
        options={"max_iters": 1}, debug={"info": True})
    assert out[-1]["n_linear_iters"] == 1
    assert out[-1]["n_jvp"] == calls["v"] == 2  # product and true-residual check
    assert out[-1]["n_jtvp"] == calls["t"] == 4
    assert out[-1]["n_jac_calls"] == out[-1]["njev"] == 0


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_inner_exhaustion_does_not_commit_partial_direction(method):
    out = liteopt.least_squares(lambda x: [x[0]-1., 2*x[1]-1.], [0.,0.], method=method,
        jacobian_vec=lambda x, v: [v[0], 2*v[1]],
        jacobian_transpose_vec=lambda x, w: [w[0], 2*w[1]],
        options={"cg_max_iters": 1, "max_iters": 1}, debug={"info": True, "history": True})
    assert not out[5] and out[0] == [0., 0.]
    info = out[-1]
    assert info["linear_status"] == "linear_max_iters" and info["n_accepted"] == 0
    assert info["n_retries"] == (1 if method == "lm" else 0)
    assert info["status"] == ("max_iters" if method == "lm" else "linear_max_iters")


@pytest.mark.parametrize("method", ["gn", "lm"])
@pytest.mark.parametrize("callback", ["jacobian_vec", "jacobian_transpose_vec"])
@pytest.mark.parametrize("bad", ["shape", "nan", "exception"])
def test_product_contract_errors_propagate(method, callback, bad):
    def invalid(x, v):
        if bad == "exception":
            raise RuntimeError("product failed")
        return [] if bad == "shape" else [float("nan")]
    derivatives = dict(jacobian_vec=lambda x, v: v, jacobian_transpose_vec=lambda x, w: w)
    derivatives[callback] = invalid
    error = RuntimeError if bad == "exception" else ValueError
    with pytest.raises(error, match="product failed" if bad == "exception" else callback):
        liteopt.least_squares(lambda x: [x[0]-1.], [0.], method=method, **derivatives)


@pytest.mark.parametrize("options", [
    {"linear_solver": "unknown"}, {"cg_max_iters": 0}, {"cg_rtol": 0.},
    {"cg_rtol": 1.}, {"cg_rtol": float("nan")}, {"cg_atol": -1.},
    {"cg_atol": float("inf")}, {"linear_solver": "cg", "linear_system": "qr"},
])
@pytest.mark.parametrize("method", ["gn", "lm"])
def test_invalid_linear_solver_options(method, options):
    with pytest.raises(ValueError):
        liteopt.least_squares(lambda x: x, [1.], method=method,
                             jacobian=lambda x: [1.], options=options)


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_incomplete_products_and_implicit_dense_conversion_are_rejected(method):
    common = dict(method=method, jacobian_vec=lambda x, v: v)
    with pytest.raises(ValueError, match="provided together"):
        liteopt.least_squares(lambda x: x, [1.], **common)
    with pytest.raises(ValueError, match="requires jacobian"):
        liteopt.least_squares(lambda x: x, [1.], **common,
            jacobian_transpose_vec=lambda x, w: w, options={"linear_solver": "direct"})


@pytest.mark.parametrize("method", ["gn", "lm"])
def test_explicit_switch_when_both_representations_are_available(method):
    for backend in ["direct", "cg"]:
        out = liteopt.least_squares(lambda x: [x[0]-1.], [0.], method=method,
            jacobian=lambda x: [1.], jacobian_vec=lambda x, v: v,
            jacobian_transpose_vec=lambda x, w: w,
            options={"linear_solver": backend}, debug={"info": True})
        assert out[5]
        assert out[-1]["matrix_free"] == (backend == "cg")
        assert out[-1]["n_jac_calls"] > 0 if backend == "direct" else out[-1]["n_jac_calls"] == 0


@pytest.mark.parametrize("method", ["gn", "lm"])
@pytest.mark.parametrize("initial", [0., 1.])
def test_zero_budget_and_stationary_initial_point(method, initial):
    out = liteopt.least_squares(lambda x: [x[0], 1.], [initial], method=method,
        jacobian_vec=lambda x, v: [v[0], 0.], jacobian_transpose_vec=lambda x, w: [w[0]],
        options={"max_iters": 0}, debug={"info": True})
    assert out[5] == (initial == 0.)
    assert out[-1]["n_jvp"] == out[-1]["n_linear_iters"] == 0
    assert out[-1]["n_jtvp"] == 1 and out[-1]["linear_status"] is None

@pytest.mark.parametrize("method", ["gn", "lm"])
def test_custom_retraction_and_search_work_with_products(method):
    class Retraction:
        def retract(self, x, direction, alpha):
            return [max(.5, x[0]+alpha*direction[0])]
    out = liteopt.least_squares(lambda x: [x[0]], [1.], method=method,
        jacobian_vec=lambda x, v: v, jacobian_transpose_vec=lambda x, w: w,
        options={"manifold": Retraction(), "max_iters": 1,
                 "line_search": lambda ctx: (True, 1.)},
        debug={"info": True})
    assert out[0] == [.5] and out[-1]["n_accepted"] == 1


def test_lm_recovers_from_inner_exhaustion_by_increasing_damping():
    out = liteopt.least_squares(lambda x: [x[0]-1., 2*x[1]-1.], [0.,0.],
        jacobian_vec=lambda x, v: [v[0], 2*v[1]],
        jacobian_transpose_vec=lambda x, w: [w[0], 2*w[1]],
        options={"cg_max_iters": 1, "cg_rtol": .01, "max_iters": 10},
        debug={"info": True, "history": True})
    assert out[-1]["n_retries"] > 0
    assert out[-1]["n_accepted"] > 0
    assert out[1] < 1.
    assert any(row["note"] == "linear_max_iters" for row in out[-2])
    assert any(row["note"] == "linear_converged" for row in out[-2])


