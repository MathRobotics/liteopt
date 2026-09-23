import numpy as np
import pytest
import liteopt


@pytest.mark.parametrize('method', ['gn', 'lm'])
@pytest.mark.parametrize('shape', [(8,3), (3,3), (32,6)])
@pytest.mark.parametrize('scaled', [False, True])
def test_qr_direction_matches_independent_svd_reference(method, shape, scaled):
    m,n = shape
    rng = np.random.default_rng(934+m+n)
    a = rng.normal(size=shape)
    if scaled: a *= np.geomspace(1e-6,1e6,n)
    b = rng.normal(size=m)
    damping = .1 if method=='lm' else 0.
    aug = np.vstack([a, np.sqrt(damping)*np.eye(n)]) if damping else a
    rhs = np.r_[b,np.zeros(n)] if damping else b
    scales = np.linalg.norm(aug,axis=0)
    expected = np.linalg.lstsq(aug/scales,rhs,rcond=None)[0]/scales
    out = liteopt.least_squares(lambda x: a@x-b, [0.]*n, method=method, jacobian=lambda x: a,
        options={'linear_system':'qr','max_iters':1,'tol_grad':0.,'tol_dx':0.,'tol_r':0.,
            'line_search_method':'none', **({'lambda':damping} if method=='lm' else {})}, debug={'info':True})
    assert out[-1]['n_accepted']==1
    assert np.allclose(out[0],expected,rtol=1e-8,atol=1e-10)


def test_near_dependent_gn_system_can_use_qr():
    a = np.array([[1.,1.+1e-8],[1.,1.-1e-8],[2.,2.+1e-8],[2.,2.-1e-8]])
    b = a@np.array([1.,2.])
    derivative = {'jacobian': lambda x:a}
    out = liteopt.least_squares(lambda x:a@x-b,[0.,0.],method='gn',**derivative,
        options={'linear_system':'qr','max_iters':1,'tol_grad':0.,'tol_dx':0.},debug={'info':True})
    assert out[-1]['n_accepted']==1
    assert np.allclose(out[0],[1.,2.],atol=1e-6)


@pytest.mark.parametrize('shape,status', [((2,2),'qr_rank_deficient'),((1,2),'qr_invalid_shape')])
def test_gn_qr_does_not_silently_regularize(shape,status):
    a = np.ones(shape)
    out = liteopt.least_squares(lambda x:a@x-1,[0.]*shape[1],method='gn',jacobian=lambda x:a,
        options={'linear_system':'qr'},debug={'info':True,'history':True})
    assert not out[5] and out[-1]['status']==status
    assert out[-1]['n_accepted']==0 and out[0]==[0.]*shape[1]


def test_augmented_lm_qr_supports_wide_jacobian():
    out = liteopt.least_squares(lambda x:[x[0]+x[1]-3],[0.,0.],jacobian=lambda x:[1.,1.],
        options={'linear_system':'qr','lambda':1.,'max_iters':1},debug={'info':True})
    assert np.allclose(out[0],[1.,1.]) and out[-1]['n_accepted']==1


def test_qr_lm_backtracking_and_projection():
    out = liteopt.least_squares(lambda x:[x[0]**2-1],[.1],jacobian=lambda x:[2*x[0]],
        project=lambda x:[max(0.,x[0])],options={'linear_system':'qr','line_search_method':'armijo'},
        debug={'info':True,'history':True})
    assert out[5] and abs(out[0][0]-1)<1e-5
    assert any(row['accepted'] and row['alpha']<1 for row in out[-2])
