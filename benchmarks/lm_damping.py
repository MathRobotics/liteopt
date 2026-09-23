"""Compare LM damping policies on fixed nonlinear problems and initial guesses.
Requires a release build of local liteopt. Run from the repository root.
"""
import json
import math
import platform
from pathlib import Path
import liteopt


def main():
    rows=[]
    problems=[]
    for start in [[-1.2,1.],[-3.,-4.],[2.,2.]]:
        for scale in [[1.,1.],[1e-3,1e3]]:
            def residual(z,s=scale):
                x,y=[a*b for a,b in zip(z,s)]
                return [10*(y-x*x),1-x]
            def jacobian(z,s=scale):
                x=z[0]*s[0]
                return [-20*x*s[0],10*s[1],-s[0],0.]
            problems.append(('rosenbrock',start,scale,residual,jacobian))
    for start in [[.1],[10.],[-.1]]:
        problems.append(('square_root',start,[1.],lambda x:[x[0]*x[0]-1],lambda x:[2*x[0]]))
    # A nonlinear residual that cannot reach zero: tests stationary fitting.
    for start in [[0.],[2.],[-2.]]:
        problems.append(('nonzero_residual',start,[1.],lambda x:[x[0]-1,math.exp(x[0])-2],lambda x:[1.,math.exp(x[0])]))
    for name,start,scale,residual,jacobian in problems:
        for search in ['cost_decrease','armijo']:
            for policy in ['cost_based','gain_ratio']:
                out=liteopt.least_squares(residual,[v/s for v,s in zip(start,scale)],jacobian=jacobian,
                    options={'damping_update':policy,'line_search_method':search,'linear_system':'qr',
                        'max_iters':300,'tol_r':1e-8,'tol_grad':1e-8,'tol_dx':1e-14},debug={'info':True})
                info=out[-1]
                # The nonzero-residual fit is evaluated by its stationarity condition.
                # All other problems have a zero-residual solution.
                accurate=bool(out[5] and (info['grad_norm']<=1e-8 if name=='nonzero_residual' else out[1]<1e-12))
                rows.append(dict(problem=name,start=start,scale=scale,search=search,policy=policy,
                    cost=out[1],success=accurate,**info))
    data=dict(platform=platform.platform(),python=platform.python_version(),build='release',
        limits={'max_iters':300,'tol_r':1e-8,'tol_grad':1e-8,'tol_dx':1e-14},results=rows)
    Path('benchmarks/lm_damping_results.json').write_text(json.dumps(data,indent=2)+'\n')
    for search in ['cost_decrease','armijo']:
        for policy in ['cost_based','gain_ratio']:
            selected=[r for r in rows if r['search']==search and r['policy']==policy]
            print(search,policy,'success',sum(r['success'] for r in selected),'/',len(selected),
                'iterations',sum(r['iters'] for r in selected),'nfev',sum(r['nfev'] for r in selected),
                'njev',sum(r['njev'] for r in selected),'retries',sum(r['n_retries'] for r in selected))
    for r in rows:
        if not r['success']: print('failed',r['problem'],r['start'],r['scale'],r['search'],r['policy'],r['status'],r['cost'])

if __name__=='__main__':main()
