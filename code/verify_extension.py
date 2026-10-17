"""Independent numerical checks for the extension; python code/verify_extension.py."""
import unittest, json
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import dirichlet_multinomial, multinomial, betabinom, beta
from extended_model import (objective, initial, mean_model, unpack, point,
    interval_from_nuisance, beta_mean_bounds, log_probability)
from extension import load_data, folds

ROOT=Path(__file__).resolve().parents[1];AUDIT={}


class ExtensionChecks(unittest.TestCase):
    def test_gradient(self):
        ids,t,y,p,pl=load_data();x=initial(y,p,pl)
        for family in ('mn','dm'):
            xx=x if family=='dm' else x[:-1]
            _,g=objective(xx,y,p,pl,family)
            numerical=approx_derivative(lambda z:objective(z,y,p,pl,family)[0],xx).ravel()
            err=np.max(np.abs(g-numerical)/(1+np.abs(g)))
            AUDIT['gradient_relative_'+family]=float(err);self.assertLess(err,2e-5)

    def test_log_probability_independent_library(self):
        y=np.array([3,5,7]);mu=np.array([.2,.3,.5]);kap=20.
        self.assertAlmostEqual(log_probability(y,mu,kap),dirichlet_multinomial.logpmf(y,kap*mu,15),places=10)
        self.assertAlmostEqual(log_probability(y,mu),multinomial.logpmf(y,15,mu),places=10)

    def test_exact_small_coverage(self):
        failures=[]
        for n in (5,20):
            yy=np.array([[k,n-k] for k in range(n+1)])
            for kap in (2.,30.):
                for eta in (.01,.1):
                    for pp in (0.,.2,.8,1.):
                        p=np.array([pp,1-pp]);b=np.array([2.,.5]);h=np.array([.3,.7])
                        mu=(1-eta)*(b*p)/(b*p).sum()+eta*h
                        lo,hi,empty=interval_from_nuisance(yy,b,eta,h,kap,.2)
                        covered=np.all((lo<=p)&(p<=hi),axis=1)&~empty
                        prob=betabinom.pmf(np.arange(n+1),n,kap*mu[0],kap*mu[1])
                        self.assertAlmostEqual(prob.sum(),1.,places=11)
                        failures.append(float(prob[~covered].sum()))
        AUDIT['exact_parameter_cases']=len(failures)
        AUDIT['exact_max_failure_alpha_point2']=max(failures)
        self.assertLessEqual(max(failures),.2+1e-10)

    def test_beta_inversion_tails(self):
        zlo=np.array([0,.001,.2,.5,.9]);zhi=np.array([.001,.2,.5,.9,1.])
        for kap in (2,100,10000):
            lo,hi=beta_mean_bounds(zlo,zhi,kap,.001)
            self.assertEqual(lo[0],0);self.assertEqual(hi[-1],1)
            np.testing.assert_allclose(beta.sf(zlo[1:],kap*lo[1:],kap*(1-lo[1:])),.001,atol=1e-9)
            np.testing.assert_allclose(beta.cdf(zhi[:-1],kap*hi[:-1],kap*(1-hi[:-1])),.001,atol=1e-9)

    def test_inverse_and_scaling(self):
        rng=np.random.default_rng(42)
        for i in range(100):
            p=rng.dirichlet(np.ones(7));p[0]=0;p/=p.sum()
            b=np.exp(rng.normal(size=7));eta=.03;h=rng.dirichlet(np.ones(7))
            mu=(1-eta)*(p*b)/(p*b).sum()+eta*h
            np.testing.assert_allclose(point(mu,b,eta,h),p,atol=1e-14)
            y=rng.multinomial(1000,mu)
            lo,hi,e=interval_from_nuisance(y,b,eta,h,100.)
            ll,hh,ee=interval_from_nuisance(y,13*b,eta,h,100.)
            np.testing.assert_allclose(lo,ll,atol=1e-13);np.testing.assert_allclose(hi,hh,atol=1e-13)

    def test_empty_set_is_exposed(self):
        lo,hi,empty=interval_from_nuisance(np.array([10000,0]),np.ones(2),.1,np.array([0,1.]))
        self.assertTrue(empty);self.assertTrue(np.isnan(lo).all());self.assertTrue(np.isnan(hi).all())

    def test_exact_simplex_endpoint(self):
        # Regression: no numerical tolerance may turn a returned 0.99999...
        # upper endpoint into literal containment of the singleton p_i=1.
        rng=np.random.default_rng(17)
        for _ in range(200):
            b=np.exp(rng.normal(size=7));h=np.ones(7)/7
            y=np.zeros(7,int);y[0]=20000
            lo,hi,empty=interval_from_nuisance(y,b,.003,h,350.)
            self.assertFalse(empty);self.assertEqual(hi[0],1.)
            np.testing.assert_array_equal(lo[1:],0.)

    def test_grouped_folds_and_design(self):
        ids,t,y,p,pl=load_data();a=folds(p)
        self.assertEqual(sum(a==-1),2);self.assertEqual(sum(a>=0),78)
        for i in range(len(p)):
            for j in range(len(p)):
                if np.array_equal(p[i]>0,p[j]>0):self.assertEqual(a[i],a[j])
        ranks=[]
        for f in range(5):
            train=a!=f;test=a==f
            self.assertFalse(set(np.where(train)[0])&set(np.where(test)[0]))
            for plate in range(2):self.assertTrue(np.any(train&(pl==plate)&np.all(p>0,axis=1)))
            theta=initial(y[train],p[train],pl[train])[:-1]
            jac=approx_derivative(lambda z:mean_model(p[train],pl[train],z,'mn').ravel(),theta)
            rank=int(np.linalg.matrix_rank(jac,tol=1e-8));ranks.append(rank)
            self.assertEqual(rank,len(theta))
        AUDIT['mean_jacobian_ranks_at_start']=ranks
        AUDIT['identification_limit']='Local numerical rank only, not proof of global identifiability.'


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    AUDIT.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    out=ROOT/'results/extension';out.mkdir(exist_ok=True,parents=True)
    (out/'verification.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
