import pytest
import liteopt


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('jvp', [False, True])
@pytest.mark.parametrize('history', [False, True])
def test_info_counts_real_callbacks_and_final_gradient(method, jvp, history):
    counts = dict(r=0, j=0, v=0, t=0)
    def residual(x):
        counts['r'] += 1
        return [x[0]-1., 2*x[1]+2.]
    def jacobian(x):
        counts['j'] += 1
        return [1., 0., 0., 2.]
    def jacobian_vec(x, v):
        counts['v'] += 1
        return [v[0], 2*v[1]]
    def transpose(x, w):
        counts['t'] += 1
        return [w[0], 2*w[1]]
    derivatives = {'jacobian_vec': jacobian_vec, 'jacobian_transpose_vec': transpose} if jvp else {'jacobian': jacobian}
    out = liteopt.least_squares(residual, [0., 0.], method=method, **derivatives,
        options={'max_iters': 1, 'step_size': .25}, debug={'info': True, 'history': history})
    info = out[-1]
    assert len(out) == (8 if history else 7)
    assert info['nfev'] == counts['r']
    assert info['n_jac_calls'] == counts['j']
    assert info['n_jvp'] == counts['v']
    assert info['njev'] == counts['j']
    assert info['n_jtvp'] == counts['t']
    x = out[0]
    assert info['grad_norm'] == pytest.approx(((x[0]-1)**2 + (4*x[1]+4)**2)**.5)
    assert info['iters'] == info['n_attempts'] == info['n_accepted'] == 1
    assert info['n_retries'] == 0 and info['n_ls_trials'] == 1
    if history:
        rows = out[-2]
        assert rows[0]['phase'] == 'initial'
        assert rows[-1]['phase'] == 'final' and rows[-1]['accepted'] is None
        assert rows[-1]['note'] == info['status']
        row = next(row for row in rows if row['phase'] == 'search')
        assert row['accepted'] is True and row['alpha'] == .25
        assert row['cost'] == 2.5  # pre-step cost
        assert row['ls_trials'] == 1


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_info_without_history_explains_rejection(method):
    out = liteopt.least_squares(lambda x: [x[0]-1], [0.], method=method,
        jacobian=lambda x: [1.], options={'max_iters': 2, 'line_search': lambda ctx: (False, .1)},
        debug={'info': True})
    info = out[-1]
    assert not out[5] and info['n_accepted'] == 0 and info['n_ls_trials'] == 0
    assert info['n_attempts'] == (1 if method == 'gn' else 2)
    assert info['n_retries'] == (0 if method == 'gn' else 2)
    assert info['status'] == ('rejected' if method == 'gn' else 'max_iters')


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_converged_initial_point_has_gradient_and_counts(method):
    out = liteopt.least_squares(lambda x: [x[0]], [0.], method=method,
        jacobian=lambda x: [1.], options={'max_iters': 0}, debug={'info': True})
    info = out[-1]
    assert out[5] and info['grad_norm'] == 0.
    assert info['nfev'] == 2  # dimension inference plus initial residual
    assert info['njev'] == 1
    assert info['n_attempts'] == info['n_accepted'] == info['n_retries'] == 0


@pytest.mark.parametrize('solver', ['gd', 'gn', 'lm'])
@pytest.mark.parametrize('search', ['armijo', 'strict_decrease'])
def test_min_step_applies_before_first_trial(solver, search):
    options = {'step_size': .1, 'ls_min_step': .2, 'line_search_method': search, 'max_iters': 1}
    if solver == 'gd':
        out = liteopt.gd(lambda x: x[0]**2, lambda x: [2*x[0]], [1.], options=options,
            debug={'info': True, 'history': True})
    else:
        out = liteopt.least_squares(lambda x: [x[0]], [1.], method=solver,
            jacobian=lambda x: [1.], options=options, debug={'info': True, 'history': True})
    assert out[-1]['n_ls_trials'] == 0 and out[-1]['n_accepted'] == 0
    row = next(row for row in out[-2] if row['phase'] == 'search')
    assert row['accepted'] is False and row['ls_trials'] == 0


def test_gd_counts_and_initial_search_final_rows():
    out = liteopt.gd(lambda x: x[0]**2, lambda x: [2*x[0]], [1.],
        options={'step_size': 2., 'line_search_method': 'armijo'}, debug={'info': True, 'history': True})
    info = out[-1]
    assert info['n_attempts'] == info['n_accepted'] == 1
    assert info['n_ls_trials'] == 3 and info['n_retries'] == 0
    assert [row['phase'] for row in out[-2]] == ['initial', 'search', 'final']
    assert out[-2][1]['ls_trials'] == 3


@pytest.mark.parametrize('method', ['gn', 'lm'])
def test_custom_search_overrides_builtin_but_verifies_cost(method):
    calls = []
    out = liteopt.least_squares(lambda x: calls.append(x) or [x[0]], [1.], method=method,
        jacobian=lambda x: [1.], options={'max_iters': 1, 'line_search_method': 'armijo',
            'ls_min_step': .9, 'line_search': lambda ctx: .1}, debug={'info': True})
    assert out[-1]['n_accepted'] == 1 and out[-1]['n_ls_trials'] == 0
    assert out[-1]['nfev'] == len(calls) == 3
