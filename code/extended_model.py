"""Contaminated multinomial / Dirichlet-multinomial calibration.

The interval theorem is conditional on KNOWN nuisance parameters. Plug-in
and bootstrap-envelope results are exploratory, not finite-sample guarantees.
"""
import numpy as np
from scipy.special import expit, softmax, gammaln, digamma, betainc, betaincc
from scipy.optimize import minimize
from calibration import binomial_kl_interval, normalize, normalized_box


def unpack(theta, q=7, plates=2, family='dm'):
    k=plates*(q-1)
    b=np.exp(np.c_[np.asarray(theta[:k]).reshape(plates,q-1),np.zeros(plates)])
    eta=.2*expit(theta[k])
    h=softmax(np.r_[theta[k+1:k+q],0.])
    kap=np.exp(theta[-1]) if family=='dm' else np.inf
    return b,eta,h,kap


def mean_model(p, plate, theta, family='dm', plates=2):
    p=np.asarray(p,float); b,eta,h,kap=unpack(theta,p.shape[-1],plates,family)
    clean=normalize(p*b[np.asarray(plate)])
    return (1-eta)*clean+eta*h


def objective(theta, y, p, plate, family='dm', plates=2):
    """Mean negative log likelihood (count constants omitted) and gradient."""
    q=p.shape[1]; k=plates*(q-1); b,eta,h,kap=unpack(theta,q,plates,family)
    clean=normalize(p*b[plate]); mu=(1-eta)*clean+eta*h; n=y.sum(1)
    if family=='dm':
        a=kap*mu; diff=digamma(y+a)-digamma(a)
        ll=gammaln(kap)-gammaln(n+kap)+np.sum(gammaln(y+a)-gammaln(a),axis=1)
        score=kap*diff
        dk=kap*np.sum(digamma(kap)-digamma(n+kap)+np.sum(mu*diff,axis=1))
    else:
        ll=np.sum(y*np.log(mu),axis=1); score=y/mu
    db=(1-eta)*clean*(score-np.sum(score*clean,axis=1,keepdims=True))
    gb=np.concatenate([db[plate==i,:-1].sum(0) for i in range(plates)])
    ge=np.sum(score*(h-clean))*eta*(1-eta/.2)
    sh=score.sum(0); gh=eta*h*(sh-np.dot(sh,h))
    g=np.r_[gb,ge,gh[:-1]]
    if family=='dm':g=np.r_[g,dk]
    return -float(np.mean(ll)),-g/len(y)


def initial(y,p,plate,plates=2):
    q=p.shape[1]; bs=[]
    for z in range(plates):
        idx=np.where((plate==z)&np.all(p>0,axis=1))[0]
        if len(idx): b=(y[idx].sum(0)+.5)/p[idx].mean(0)
        else: b=np.ones(q)
        bs.extend(np.log(b[:-1]/b[-1]))
    return np.r_[bs,np.log(.002/(.2-.002)),np.zeros(q-1),np.log(500.)]


def fit(y,p,plate,family='dm',start=None,multistart=False,plates=2):
    y=np.asarray(y,float);p=np.asarray(p,float);plate=np.asarray(plate,int)
    q=p.shape[1];k=plates*(q-1)
    if start is None:
        start=initial(y,p,plate,plates)
        if family=='mn':start=start[:-1]
    bounds=[(-12,12)]*k+[(-16,8)]+[(-12,12)]*(q-1)
    if family=='dm':bounds+=[(0,np.log(1e7))]
    starts=[np.asarray(start).copy()]
    if multistart:
        alt=starts[0].copy();alt[:k]=0
        if family=='dm':alt[-1]=np.log(2000)
        starts.append(alt)
    results=[minimize(objective,s,args=(y,p,plate,family,plates),jac=True,
                      method='L-BFGS-B',bounds=bounds,
                      options={'maxiter':1500,'ftol':1e-12,'gtol':1e-6,'maxls':40}) for s in starts]
    best=min(results,key=lambda r:r.fun)
    # Retry failed optimizations; never silently discard a replicate.
    if not best.success:
        best=minimize(objective,best.x,args=(y,p,plate,family,plates),jac=True,
                      method='L-BFGS-B',bounds=bounds,
                      options={'maxiter':3000,'ftol':1e-11,'gtol':1e-5,'maxls':60})
    return best.x,dict(success=bool(best.success),objective=float(best.fun),
        iterations=int(best.nit),gradient_max=float(np.max(np.abs(best.jac))),
        message=str(best.message),starts=len(starts),
        start_objective_spread=float(np.ptp([r.fun for r in results])),
        bound_hits=[int(i) for i,(x,bd) in enumerate(zip(best.x,bounds)) if min(x-bd[0],bd[1]-x)<1e-5])


def log_probability(y,mu,kappa=np.inf):
    y=np.asarray(y); n=y.sum(-1)
    constant=gammaln(n+1)-gammaln(y+1).sum(-1)
    if np.isinf(kappa):return constant+(y*np.log(mu)).sum(-1)
    a=kappa*mu
    return constant+gammaln(kappa)-gammaln(n+kappa)+(gammaln(y+a)-gammaln(a)).sum(-1)


def draw(rng, depths, mu, kappa=np.inf):
    mu=np.atleast_2d(mu);depths=np.broadcast_to(depths,len(mu))
    if np.isfinite(kappa):
        # Gamma underflow can occur for tiny contamination means: zero draws
        # are permissible, normalize explicitly instead of clipping them.
        g=rng.gamma(kappa*mu); mu=normalize(g)
    return np.array([rng.multinomial(int(n),x) for n,x in zip(depths,mu)])


def beta_mean_bounds(zlo,zhi,kappa,tail):
    """Invert Beta(kappa*mu,kappa*(1-mu)) tails with outer bisection."""
    zlo=np.asarray(zlo); zhi=np.asarray(zhi)
    l=np.zeros(zlo.shape);u=np.ones(zlo.shape)
    for _ in range(52):
        m=(l+u)/2;sf=betaincc(kappa*m,kappa*(1-m),zlo)
        l=np.where(sf<tail,m,l);u=np.where(sf<tail,u,m)
    lower=np.where(zlo<=0,0,l)
    l=np.zeros(zhi.shape);u=np.ones(zhi.shape)
    for _ in range(52):
        m=(l+u)/2;cdf=betainc(kappa*m,kappa*(1-m),zhi)
        l=np.where(cdf>tail,m,l);u=np.where(cdf>tail,u,m)
    upper=np.where(zhi>=1,1,u)
    return lower,upper


def interval_from_nuisance(y,b,eta,h,kappa=np.inf,alpha=.05):
    """Conservative simultaneous intervals for one sample, known nuisance.

    DM: split alpha equally over count and latent-Beta events. MN: all alpha
    goes to counts. Probability boxes are intersected with contamination
    bounds, then propagated through normalized inverse efficiency. An empty
    box explicitly signals a model rejection, never a vacuous interval.
    """
    y=np.asarray(y);q=y.shape[-1];n=y.sum(-1,keepdims=True)
    tail=alpha/((4 if np.isfinite(kappa) else 2)*q)
    lo,hi=binomial_kl_interval(y,n,tail)
    if np.isfinite(kappa):lo,hi=beta_mean_bounds(lo,hi,kappa,tail)
    rawlo=(lo-eta*h)/(1-eta);rawhi=(hi-eta*h)/(1-eta)
    cl=np.maximum(0,rawlo);cu=np.minimum(1,rawhi)
    empty=np.any(cl>cu,axis=-1)|(cl.sum(-1)>1)|(cu.sum(-1)<1)
    # Explicit sums over OTHER coordinates preserve exact endpoints 0 and 1.
    # Subtracting the own coordinate from a total can round U=1 down by an ulp.
    a=np.where(empty[...,None],0,cl/b)
    A=np.where(empty[...,None],1,cu/b)
    low,high=normalized_box(a,A)
    low=np.where(empty[...,None],np.nan,low);high=np.where(empty[...,None],np.nan,high)
    return low,high,empty


def point(y,b,eta,h):
    # Constrained nonnegative subtraction then normalization, not a likelihood MLE.
    x=np.maximum(normalize(y)-eta*h,0)/b
    return normalize(x)
