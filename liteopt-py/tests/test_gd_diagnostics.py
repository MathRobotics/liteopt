import math
import pytest
import liteopt


def f(x):
    return x[0] ** 2


def grad(x):
    return [2 * x[0]]


@pytest.mark.parametrize('method', ['none', 'armijo', 'strict_decrease', 'cost_decrease'])
def test_final_step_convergence_and_counts(method):
    calls = {'f': 0, 'g': 0}
    def value(x):
        calls['f'] += 1
        return f(x)
    def gradient(x):
        calls['g'] += 1
        return grad(x)
    x, cost, ok, history, info = liteopt.gd(
        value, gradient, [1.], options={'step_size': .5, 'max_iters': 1, 'tol_grad': 0., 'line_search_method': method},
        debug={'history': True, 'info': True})
    assert x == [0.] and cost == 0. and ok
    assert {k: info[k] for k in ['iters', 'grad_norm', 'status', 'nfev', 'njev']} == {'iters': 1, 'grad_norm': 0., 'status': 'converged', 'nfev': calls['f'], 'njev': calls['g']}
    assert calls == {'f': 2, 'g': 2}
    assert history[1]['alpha'] == .5
    assert history[-1]['note'] == 'converged'


@pytest.mark.parametrize('budget', [0, 1, 3])
def test_budget_reports_gradient_at_returned_point(budget):
    x, _, ok, history, info = liteopt.gd(f, grad, [1.], options={'max_iters': budget}, debug={'history': True, 'info': True})
    assert not ok
    assert info['grad_norm'] == pytest.approx(abs(2 * x[0]))
    assert info['iters'] == budget and info['status'] == 'max_iters'
    assert len(history) == budget + 2


@pytest.mark.parametrize('options', [
    {'step_size': -1.}, {'step_size': 0.}, {'step_size': float('nan')},
    {'tol_grad': float('inf')}, {'tol_grad': -1.}, {'ls_beta': 0.},
    {'ls_beta': 1.}, {'ls_min_step': 0.}, {'ls_max_steps': 0},
    {'c_armijo': 1.}, {'line_search_method': 'unknown'},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        liteopt.gd(f, grad, [1.], options=options)


def test_nonfinite_gradient_and_initial_point():
    calls = []
    def bad_grad(x):
        calls.append(x)
        return [float('nan')]
    with pytest.raises(ValueError, match='gradient.*finite'):
        liteopt.gd(f, bad_grad, [1.])
    assert len(calls) == 1
    with pytest.raises(ValueError, match='x0.*finite'):
        liteopt.gd(f, grad, [float('inf')])


@pytest.mark.parametrize('method', ['armijo', 'strict_decrease'])
def test_backtracking_recovers_from_nonfinite_trial_cost(method):
    def domain_value(x):
        return f(x) if x[0] >= 0 else float('inf')
    x, _, ok, history, info = liteopt.gd(domain_value, grad, [1.],
        options={'step_size': 2., 'line_search_method': method}, debug={'history': True, 'info': True})
    assert ok and x == [0.]
    assert history[1]['alpha'] == .5
    assert info['nfev'] == 4


def test_rejection_and_custom_step_diagnostics():
    _, _, ok, history, info = liteopt.gd(f, grad, [1.],
        options={'step_size': 2., 'line_search_method': 'cost_decrease'}, debug={'history': True, 'info': True})
    assert not ok and info['status'] == 'line_search_failed'
    assert history[1]['note'] == 'rejected'
    x, _, _, history = liteopt.gd(f, grad, [1.],
        options={'max_iters': 1, 'line_search': lambda ctx: .1}, debug={'history': True})
    assert x == pytest.approx([.8]) and history[1]['alpha'] == .1


def test_nonfinite_initial_cost_and_exceptions():
    _, _, ok, info = liteopt.gd(lambda x: math.inf, grad, [1.], debug={'info': True})
    assert not ok and info['status'] == 'non_finite'
    def fail(x):
        raise RuntimeError('objective failed')
    with pytest.raises(RuntimeError, match='objective failed'):
        liteopt.gd(fail, grad, [1.])


def test_armijo_min_step_can_prevent_acceptance():
    _, _, ok, info = liteopt.gd(f, grad, [1.], options={
        'step_size': 2., 'line_search_method': 'armijo', 'ls_min_step': .75}, debug={'info': True})
    assert not ok and info['status'] == 'line_search_failed'


@pytest.mark.parametrize('alpha', [0., -1., float('nan'), float('inf')])
def test_custom_search_cannot_accept_invalid_step(alpha):
    if not math.isfinite(alpha):
        with pytest.raises(ValueError):
            liteopt.gd(f, grad, [1.], options={'line_search': lambda ctx: alpha})
    else:
        x, _, ok, info = liteopt.gd(f, grad, [1.],
            options={'line_search': lambda ctx: alpha}, debug={'info': True})
        assert not ok and x == [1.] and info['status'] == 'invalid_step'
        assert info['iters'] == 0
