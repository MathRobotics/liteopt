import math
import pytest
import liteopt


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('budget,x0,expected', [(0, [0.], True), (0, [1.], False), (1, [1.], True)])
def test_final_and_initial_convergence(method, budget, x0, expected):
    out = liteopt.least_squares(lambda x: [x[0]], x0, method=method, jacobian=lambda x: [1.],
        options={'max_iters': budget, **({'lambda': 0.} if method == 'lm' else {})}, debug={'history': True})
    assert out[5] == expected
    assert out[3] == pytest.approx(abs(out[0][0]))
    assert out[-1][-1]['note'] == ('converged_r' if expected else 'max_iters')


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_nonzero_residual_stationary_point(method):
    out = liteopt.least_squares(lambda x: [x[0]-1, x[0]+1], [0.], method=method,
        jacobian=lambda x: [1., 1.], options={'max_iters': 0}, debug={'history': True})
    assert out[5] and out[3] == pytest.approx(math.sqrt(2))
    assert out[-1][-1]['note'] == 'converged_grad'


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_small_direction_with_large_gradient_is_stalled(method):
    out = liteopt.least_squares(lambda x: [1e9*x[0]-1], [0.], method=method,
        jacobian=lambda x: [1e9], debug={'history': True})
    assert not out[5] and out[0] == [0.]
    assert out[-1][-1]['note'] == 'stalled'


def test_large_initial_damping_is_not_success():
    out = liteopt.least_squares(lambda x: [x[0]-1], [0.], jacobian=lambda x: [1.],
        options={'lambda': 1e15}, debug={'history': True})
    assert not out[5] and out[-1][-1]['note'] == 'stalled'


def test_damping_overflow_preserves_point():
    out = liteopt.least_squares(lambda x: [x[0]-1], [0.], jacobian=lambda x: [1.],
        options={'lambda': 1e100, 'lambda_up': 1e300, 'tol_dx': 0.,
                 'line_search': lambda ctx: (False, ctx['alpha0'])}, debug={'history': True})
    assert not out[5] and out[0] == [0.]
    assert out[-1][-1]['note'] == 'damping_overflow'


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('options', [{'tol_grad': -1.}, {'tol_r': math.inf}, {'tol_dx': math.nan}, {'step_size': math.nan}])
def test_invalid_options(method, options):
    with pytest.raises(ValueError):
        liteopt.least_squares(lambda x: [1.], [0.], method=method, jacobian=lambda x: [1.], options=options)


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('x0', [[], [math.inf], [math.nan]])
def test_invalid_initial_point(method, x0):
    with pytest.raises(ValueError, match='x0'):
        liteopt.least_squares(lambda x: [1.], x0, method=method, jacobian=lambda x: [1.])


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_initial_nonfinite_residual(method):
    out = liteopt.least_squares(lambda x: [math.inf], [0.], method=method,
        jacobian=lambda x: [1.], debug={'history': True})
    assert not out[5] and out[-1][-1]['note'] == 'non_finite_residual'


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('callback', ['jacobian', 'jacobian_vec'])
def test_nonfinite_jacobian(method, callback):
    derivatives = {callback: lambda *args: [math.nan]}
    if callback == 'jacobian_vec':
        derivatives['jacobian_transpose_vec'] = lambda x, w: w
    with pytest.raises(ValueError, match=callback):
        liteopt.least_squares(lambda x: [1.], [0.], method=method,
            **derivatives)


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_trial_domain_failure_can_backtrack(method):
    out = liteopt.least_squares(lambda x: [math.inf if x[0] > 1. else x[0]**2-1], [.1],
        method=method, jacobian=lambda x: [2*x[0]],
        options={'line_search_method': 'armijo', 'max_iters': 1}, debug={'history': True})
    assert out[1] < .49005 and math.isfinite(out[0][0])
    assert any(row['note'] == 'accepted' and row['alpha'] < 1 for row in out[-1])


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_invalid_projection_never_committed(method):
    out = liteopt.least_squares(lambda x: [x[0]-1], [0.], method=method,
        jacobian=lambda x: [1.], project=lambda x: [math.inf], debug={'history': True})
    assert not out[5] and out[0] == [0.]


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_callback_exception_propagates(method):
    def fail(x):
        raise RuntimeError('bad projection')
    with pytest.raises(RuntimeError, match='bad projection'):
        liteopt.least_squares(lambda x: [x[0]-1], [0.], method=method,
            jacobian=lambda x: [1.], project=fail)


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_terminal_gradient_matches_returned_point(method):
    out = liteopt.least_squares(lambda x: [x[0]**2-1], [.5], method=method,
        jacobian=lambda x: [2*x[0]], options={'max_iters': 1, 'step_size': .25}, debug={'history': True})
    x = out[0][0]
    assert out[-1][-1]['grad_norm'] == pytest.approx(abs(2*x*(x*x-1)))


def test_zero_damping_can_increase_after_rejection():
    lambdas = []
    def reject(ctx):
        lambdas.append(ctx['lambda'])
        return False, ctx['alpha0']
    out = liteopt.least_squares(lambda x: [x[0]-1], [0.], jacobian=lambda x: [1.],
        options={'lambda': 0., 'max_iters': 2, 'line_search': reject})
    assert not out[5] and 0 < lambdas[0] < lambdas[1]


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_negative_manifold_norm_is_invalid(method):
    class BadNorm:
        def tangent_norm(self, v):
            return -1.
    with pytest.raises(ValueError, match='nonnegative'):
        liteopt.least_squares(lambda x: [1.], [0.], method=method,
            jacobian=lambda x: [1.], options={'manifold': BadNorm()})
