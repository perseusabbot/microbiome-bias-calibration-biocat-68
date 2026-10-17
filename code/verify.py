"""Independent checks of formulas, exhaustive probability sums, and boundaries."""
import itertools, json, math, unittest
from pathlib import Path
import numpy as np
from calibration import *

EVIDENCE={}
ROOT=Path(__file__).resolve().parents[1]


def binomial_prob(n,k,p):
    return math.comb(n,k)*p**k*(1-p)**(n-k)


class Verification(unittest.TestCase):
    def test_inverse_identity(self):
        rng=np.random.default_rng(73)
        for _ in range(200):
            p=normalize(rng.uniform(.01,1,5)); r=normalize(rng.uniform(.01,1,5))
            b=np.exp(rng.uniform(-4,4,5))
            rho=normalize(b*p);sigma=normalize(b*r)
            np.testing.assert_allclose(normalize(r*rho/sigma),p,atol=2e-15)
        EVIDENCE['inverse_identity_trials']=200

    def test_kl_boundary_closed_forms(self):
        for n in [1,3,100,100000]:
            tail=.0025
            lo,hi=binomial_kl_interval(np.array([0,n]),n,tail)
            np.testing.assert_allclose(lo,[0,tail**(1/n)],atol=2e-15)
            np.testing.assert_allclose(hi,[1-tail**(1/n),1],atol=2e-15)

    def test_exact_binomial_tail_probabilities(self):
        max_failure=0;checked=0
        grid=np.unique(np.r_[0,.00001,.0001,.001,.01,np.linspace(.02,.98,49),.99,.999,.9999,1])
        for n in [1,2,5,10,20,50,100]:
            k=np.arange(n+1);tail=.01
            lo,hi=binomial_kl_interval(k,n,tail)
            for p in grid:
                probs=np.array([binomial_prob(n,int(x),p) for x in k])
                self.assertAlmostEqual(float(probs.sum()),1,places=12)
                left=float(probs[lo>p+1e-14].sum());right=float(probs[hi<p-1e-14].sum())
                self.assertLessEqual(left,tail+1e-12);self.assertLessEqual(right,tail+1e-12)
                max_failure=max(max_failure,left+right);checked+=1
        EVIDENCE['exact_binomial_parameter_cases']=checked
        EVIDENCE['max_exact_binomial_failure_at_nominal_0p02']=max_failure

    def test_exact_paired_count_coverage(self):
        # Enumerate ALL possible count pairs, weighted by independent exact
        # binomial PMFs. This is not a Monte Carlo check.
        n,m=25,20;alpha=.2;r=np.array([.3,.7]);checked=0;minimum=1
        ys=np.array([[i,n-i] for i in range(n+1)])
        cs=np.array([[j,m-j] for j in range(m+1)])
        yy=np.repeat(ys,m+1,axis=0);cc=np.tile(cs,(n+1,1))
        lo,hi=calibration_intervals(yy,cc,r,alpha)
        point=point_estimate(yy,cc,r);rad=l1_radius(point,lo,hi)
        for p0 in [.01,.1,.3,.5,.8,.99]:
            for ratio in [.1,1,10]:
                p=np.array([p0,1-p0]);b=np.array([ratio,1])
                sy,sc=normalize(b*p)[0],normalize(b*r)[0]
                probs=np.array([binomial_prob(n,int(y[0]),sy)*binomial_prob(m,int(c[0]),sc)
                                for y,c in zip(yy,cc)])
                self.assertAlmostEqual(float(probs.sum()),1,places=11)
                cover=np.all((lo<=p+1e-14)&(p<=hi+1e-14),axis=1)
                cov=float(probs[cover].sum());self.assertGreaterEqual(cov,1-alpha-1e-12)
                l1cov=float(probs[np.abs(point-p).sum(1)<=rad+1e-12].sum())
                self.assertGreaterEqual(l1cov,cov-1e-12)
                minimum=min(minimum,cov);checked+=1
        EVIDENCE['exact_joint_parameter_cases']=checked
        EVIDENCE['outcomes_per_joint_case']=len(yy)
        EVIDENCE['minimum_exact_joint_coverage_at_nominal_0p80']=minimum

    def test_normalized_box_against_corners(self):
        rng=np.random.default_rng(84)
        for _ in range(100):
            a=rng.uniform(.01,1,4);b=a+rng.uniform(0,2,4)
            lo,hi=normalized_box(a,b)
            pts=np.array([normalize(np.where(bits,b,a)) for bits in itertools.product([0,1],repeat=4)])
            np.testing.assert_allclose(lo,pts.min(0),atol=1e-14)
            np.testing.assert_allclose(hi,pts.max(0),atol=1e-14)
        EVIDENCE['normalization_corner_trials']=100

    def test_l1_formula_against_vertices(self):
        rng=np.random.default_rng(23)
        for _ in range(100):
            true=normalize(rng.uniform(.1,1,4));point=normalize(rng.uniform(.1,1,4))
            lo=true*rng.uniform(0,1,4);hi=true+(1-true)*rng.uniform(0,1,4)
            vertices=[]
            for free in range(4):
                other=[j for j in range(4) if j!=free]
                for bits in itertools.product([0,1],repeat=3):
                    v=np.zeros(4);v[other]=np.where(bits,hi[other],lo[other]);v[free]=1-v.sum()
                    if lo[free]-1e-12<=v[free]<=hi[free]+1e-12:vertices.append(v)
            expected=max(float(np.abs(v-point).sum()) for v in vertices)
            self.assertAlmostEqual(float(l1_radius(point,lo,hi)),expected,places=10)
        EVIDENCE['l1_vertex_enumeration_trials']=100

    def test_drift_bound_sharpness(self):
        errors=[]
        for R in [.01,.1,.5,1,2,4,10]:
            a=math.exp(R/2);p=np.array([a/(a+1),1/(a+1)])
            shifted=normalize(p*np.array([1,math.exp(R)]))
            actual=float(np.abs(shifted-p).sum());bound=drift_l1_bound(R)
            errors.append(abs(actual-bound));self.assertAlmostEqual(actual,bound,places=13)
        EVIDENCE['drift_sharpness_max_error']=max(errors)

    def test_drift_random_and_interval_expansion(self):
        rng=np.random.default_rng(42)
        for _ in range(1000):
            p=normalize(rng.uniform(.001,1,5));d=np.exp(rng.uniform(-2,2,5))
            star=normalize(p*d);R=float(np.ptp(np.log(d)))
            self.assertLessEqual(np.abs(star-p).sum(),drift_l1_bound(R)+1e-12)
            lo,hi=expand_for_drift(star,star,R)
            self.assertTrue(np.all((lo<=p+1e-12)&(p<=hi+1e-12)))
        EVIDENCE['random_drift_trials']=1000

    def test_zero_read_semantics(self):
        lo,hi=calibration_intervals([100,0],[50,50],[.5,.5])
        self.assertEqual(lo[1],0);self.assertGreater(hi[1],0)
        lo,hi=calibration_intervals([50,50],[100,0],[.5,.5])
        self.assertEqual(hi[1],1);self.assertEqual(lo[0],0)
        self.assertTrue(np.all(np.isfinite([lo,hi])))

    def test_input_contracts(self):
        for bad in ([0,0],[-1,2],[1.5,2],[float('nan'),2]):
            with self.assertRaises(ValueError):calibration_intervals(bad,[3,4],[.5,.5])
        with self.assertRaises(ValueError):calibration_intervals([1,2],[3,4],[1,0])
        with self.assertRaises(ValueError):calibration_intervals([1,2],[3,4],[.5,.6])
        with self.assertRaises(ValueError):calibration_intervals([1,2],[3,4],[.5,.5],0)

    def test_single_taxon(self):
        lo,hi=calibration_intervals([1],[2],[1])
        np.testing.assert_array_equal(lo,[1]);np.testing.assert_array_equal(hi,[1])

    def test_batch_matches_scalar(self):
        y=np.array([[5,3,2],[10,0,0]]);c=np.array([[3,4,3],[0,5,5]]);r=np.ones(3)/3
        lo,hi=calibration_intervals(y,c,r)
        for i in range(2):
            x,z=calibration_intervals(y[i],c[i],r)
            np.testing.assert_array_equal(x,lo[i]);np.testing.assert_array_equal(z,hi[i])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Verification))
    EVIDENCE.update(test_methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                    passed=result.wasSuccessful(),scope='Independent numerical checks, not machine-checked proofs')
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/verification.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
