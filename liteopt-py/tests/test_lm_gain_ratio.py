import math
import pytest
import liteopt


def run(residual, jacobian, **options):
    return liteopt.least_squares(residual, [0.], jacobian=jacobian,
        options={'damping_update':'gain_ratio','lambda':1.,'max_iters':1,
                 'tol_r':0.,'tol_grad':0.,'tol_dx':0., **options},debug={'history':True,'info':True})


@pytest.mark.parametrize('rho,next_lambda',[(.1,10.),(.5,1.),(1.,.5)])
def test_accepted_step_uses_model_quality(rho,next_lambda):
    predicted = .375
    end_r = -math.sqrt(1.-2*rho*predicted)
    c = (end_r+.5)/.25
    out=run(lambda x:[-1+x[0]+c*x[0]**2],lambda x:[1+2*c*x[0]])
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['accepted'] and out[-1]['n_accepted']==1
    assert row['predicted_reduction']==pytest.approx(predicted)
    assert row['gain_ratio']==pytest.approx(rho)
    assert row['lambda_next']==pytest.approx(next_lambda)
    assert out[-1]['n_retries']==0


def test_small_actual_reduction_is_rejected_despite_cost_decrease():
    rho=1e-5
    c=(-math.sqrt(1-.75*rho)+.5)/.25
    out=run(lambda x:[-1+x[0]+c*x[0]**2],lambda x:[1+2*c*x[0]])
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['note']=='gain_ratio_rejected' and row['actual_reduction']>0
    assert out[0]==[0.] and out[-1]['n_retries']==1


@pytest.mark.parametrize('alpha',[.25,1.])
@pytest.mark.parametrize('backend',['left_jjt','qr'])
def test_prediction_uses_actual_step_width(alpha,backend):
    out=run(lambda x:[x[0]-1],lambda x:[1.],step_size=alpha,linear_system=backend)
    row=next(row for row in out[-2] if row['phase']=='search')
    step=out[0][0]
    assert row['predicted_reduction']==pytest.approx(step-.5*step**2)
    assert row['gain_ratio']==pytest.approx(1.)


def test_armijo_backtracking_prediction():
    out=liteopt.least_squares(lambda x:[x[0]**2-1],[.1],jacobian=lambda x:[2*x[0]],
        options={'damping_update':'gain_ratio','line_search_method':'armijo','max_iters':1},debug={'history':True})
    row=next(row for row in out[-1] if row['phase']=='search')
    assert row['alpha']<1 and row['accepted']
    s=out[0][0]-.1
    predicted=.198*s-.5*(.2*s)**2
    assert row['predicted_reduction']==pytest.approx(predicted)
    assert row['gain_ratio']==pytest.approx((.49005-out[1])/predicted)


@pytest.mark.parametrize('target,note',[(0.,'invalid_prediction'),(-.1,'invalid_prediction'),(.2,'accepted')])
def test_projection_is_included_in_model(target,note):
    out=liteopt.least_squares(lambda x:[x[0]-1],[0.],jacobian=lambda x:[1.],project=lambda x:[target],
        options={'damping_update':'gain_ratio','lambda':1.,'max_iters':1,'line_search_method':'none'},debug={'history':True})
    row=next(row for row in out[-1] if row['phase']=='search')
    assert row['note']==note
    assert row['predicted_reduction']==pytest.approx(target-.5*target**2)
    assert out[0]==([target] if note=='accepted' else [0.])


def test_nonfinite_model_prediction_is_rejected():
    out=run(lambda x:[-1.],lambda x:[1.],line_search=lambda ctx:1e200)
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['note']=='invalid_prediction' and row['predicted_reduction'] is None
    assert out[0]==[0.]


def test_gain_ratio_cannot_be_bypassed_by_custom_acceptance():
    out=run(lambda x:[x[0]-1],lambda x:[1.],line_search=lambda ctx:10.)
    assert out[0]==[0.] and out[-1]['n_accepted']==0


def test_lambda_bounds_and_initial_floor():
    out=run(lambda x:[x[0]-1],lambda x:[1.],lambda_min=2.,lambda_max=2.)
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['lambda']==row['lambda_next']==2.
    out=run(lambda x:[x[0]-1],lambda x:[1.],lambda_max=1.,line_search=lambda ctx:(False,1.))
    assert out[-1]['status']=='damping_limit' and out[0]==[0.]


@pytest.mark.parametrize('options',[{'damping_update':'bad'},{'lambda_min':0.},{'lambda_max':math.inf},
    {'lambda_min':2.,'lambda_max':1.},{'lambda':3.,'lambda_max':2.}])
def test_invalid_options(options):
    with pytest.raises(ValueError):run(lambda x:[x[0]-1],lambda x:[1.],**options)


def test_manifold_difference_supplies_local_displacement():
    class Chart:
        def retract(self,x,d,alpha):return [x[0]+2*alpha*d[0]]
        def difference(self,x,y):return [(y[0]-x[0])/2]
    out=run(lambda x:[x[0]-1],lambda x:[2.],manifold=Chart())
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['predicted_reduction']==pytest.approx(.48)
    assert row['gain_ratio']==pytest.approx(1.)


def test_accepted_poor_prediction_caps_damping_without_discarding_step():
    rho=.1
    c=(-math.sqrt(1-.75*rho)+.5)/.25
    out=run(lambda x:[-1+x[0]+c*x[0]**2],lambda x:[1+2*c*x[0]],lambda_max=2.)
    row=next(row for row in out[-2] if row['phase']=='search')
    assert row['accepted'] and row['lambda_next']==2.
