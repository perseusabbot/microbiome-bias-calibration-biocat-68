"""Reproduce the failure diagnosis, grouped validation, and model simulations."""
from pathlib import Path
import csv, json, argparse, platform
import numpy as np
import scipy
from scipy.stats import chi2
from calibration import normalize
from experiments import read_dict_csv, write_csv, wilson
from extended_model import fit, unpack, mean_model, interval_from_nuisance, point, draw, log_probability

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/extension'
PLAN=json.loads((ROOT/'data/extension_plan.json').read_text())


def load_data():
    meta=read_dict_csv(ROOT/'data/brooks2015-sample-data.csv')
    obs={r['Sample']:r for r in read_dict_csv(ROOT/'data/brooks2015-observed.csv')}
    act={r['Sample']:r for r in read_dict_csv(ROOT/'data/brooks2015-actual.csv')}
    taxa=[k for k in next(iter(act.values())) if k!='Sample']
    ids=[r['Sample'] for r in meta]
    y=np.array([[int(obs[s][t]) for t in taxa] for s in ids])
    p=normalize([[float(act[s][t]) for t in taxa] for s in ids])
    plate=np.array([int(r['Plate'])-1 for r in meta])
    return ids,taxa,y,p,plate


def folds(p):
    groups={}
    for i,row in enumerate(p):
        if np.all(row>0):continue
        groups.setdefault(tuple(np.where(row>0)[0]),[]).append(i)
    keys=list(sorted(groups));rng=np.random.default_rng(PLAN['seed']);rng.shuffle(keys)
    sizes=np.zeros(PLAN['folds'],int);assign=np.full(len(p),-1)
    for key in keys:
        f=int(np.argmin(sizes));assign[groups[key]]=f;sizes[f]+=len(groups[key])
    return assign


def diagnosis(ids,taxa,y,p,plate):
    old={r['sample']:r for r in read_dict_csv(ROOT/'results/empirical_samples.csv')}
    rows=[];residuals=[]
    for s in range(len(y)):
        if ids[s] not in old:continue
        cal=np.where((plate==plate[s])&np.all(p>0,axis=1))[0][0]
        present=p[s]>0;pred=normalize(y[cal,present]/p[cal,present]*p[s,present])
        count=y[s,present];depth=count.sum();expected=depth*pred
        pearson=float(((count-expected)**2/expected).sum())
        row=dict(sample=ids[s],plate=int(plate[s])+1,richness=int(present.sum()),
            absent_reads=int(y[s,~present].sum()),absent_fraction=float(y[s,~present].sum()/y[s].sum()),
            all_nominal=int(old[ids[s]]['all_taxa_cover_nominal']),
            present_nominal=int(old[ids[s]]['present_taxa_cover_nominal']),
            conditional_present_pearson=pearson,df=int(present.sum())-1,
            pearson_p_ignoring_calibration=float(chi2.sf(pearson,present.sum()-1)) if present.sum()>1 else 1.)
        rows.append(row)
        ix=np.where(present)[0]
        for i in ix:
            for j in ix:
                if i>=j:continue
                if min(y[s,i],y[s,j],y[cal,i],y[cal,j])==0:continue
                logerr=float(np.log(y[s,i]/y[s,j])-np.log(p[s,i]/p[s,j])-np.log(y[cal,i]/y[cal,j])+np.log(p[cal,i]/p[cal,j]))
                # Delta-method variance includes both target and calibrator counts.
                se=float(np.sqrt(1/y[s,i]+1/y[s,j]+1/y[cal,i]+1/y[cal,j]))
                residuals.append(dict(sample=ids[s],taxon_i=taxa[i],taxon_j=taxa[j],
                    logratio_error=logerr,count_only_se=se,z=logerr/se))
    tab={f'present_{a}_absent_reads_{b}':sum((r['present_nominal']==a) and ((r['absent_reads']>0)==b) for r in rows) for a in (0,1) for b in (0,1)}
    summary=dict(n=len(rows),cross_table=tab,
        n_stray_reads=sum(r['absent_reads']>0 for r in rows),
        total_stray_reads=sum(r['absent_reads'] for r in rows),
        fraction_all_reads_stray=float(y[p==0].sum()/y[~np.all(p>0,axis=1)].sum()),
        samples_only_absence_prevents_containment=sum(r['present_nominal'] and not r['all_nominal'] for r in rows),
        samples_present_fail=sum(not r['present_nominal'] for r in rows),
        sample_median_absent_fraction=float(np.median([r['absent_fraction'] for r in rows])),
        n_logratios=len(residuals),fraction_logratios_abs_z_gt3=float(np.mean([abs(r['z'])>3 for r in residuals])),
        median_abs_logratio_error=float(np.median([abs(r['logratio_error']) for r in residuals])),
        interpretation='Descriptive dependent residuals against nominal truths; delta approximation includes calibration counts. No independent hypothesis tests or causal attribution.')
    write_csv(OUT/'diagnostic_samples.csv',rows);write_csv(OUT/'diagnostic_logratios.csv',residuals)
    (OUT/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def record_intervals(rows,intervals,ids,taxa,indices,p,y,plate,fold,model,theta,family,lo,hi,empty):
    b,eta,h,kappa=unpack(theta,family=family)
    estimates=point(y[indices],b[plate[indices]],eta,h)
    logp=log_probability(y[indices],mean_model(p[indices],plate[indices],theta,family),kappa)
    for z,i in enumerate(indices):
        rows.append(dict(sample=ids[i],fold=fold,plate=int(plate[i])+1,model=model,
            contains_nominal=int(not empty[z] and np.all((lo[z]<=p[i])&(p[i]<=hi[z]))),
            empty=int(empty[z]),mean_width=float(np.mean(hi[z]-lo[z])) if not empty[z] else '',
            l1_error=float(np.abs(estimates[z]-p[i]).sum()),log_probability=float(logp[z])))
        for j,t in enumerate(taxa):
            intervals.append(dict(sample=ids[i],fold=fold,model=model,taxon=t,nominal=float(p[i,j]),
                estimate=float(estimates[z,j]),lower=float(lo[z,j]),upper=float(hi[z,j]),empty=int(empty[z])))


def validation(ids,taxa,y,p,plate,assign,boots):
    rows=[];intervals=[];fits=[];bootstrap=[]
    for fold in range(PLAN['folds']):
        train=np.where(assign!=fold)[0];test=np.where(assign==fold)[0]
        for family in ('mn','dm'):
            theta,audit=fit(y[train],p[train],plate[train],family,multistart=True)
            b,eta,h,kappa=unpack(theta,family=family)
            fitrow=dict(fold=fold,family=family,train=[ids[i] for i in train],test=[ids[i] for i in test],
                theta=theta.tolist(),efficiency=b.tolist(),eta=float(eta),h=h.tolist(),
                kappa=None if family=='mn' else float(kappa),audit=audit)
            fits.append(fitrow)
            lo,hi,empty=interval_from_nuisance(y[test],b[plate[test]],eta,h,kappa)
            record_intervals(rows,intervals,ids,taxa,test,p,y,plate,fold,family,theta,family,lo,hi,empty)
            print('fold',fold,family,'eta',round(eta,6),'kappa',round(kappa,2),
                  'contained',sum(np.all((lo<=p[test])&(p[test]<=hi),axis=1)), '/',len(test),flush=True)
            if family=='dm':
                # Envelope is a sensitivity summary, not a confidence theorem.
                elo=lo.copy();ehi=hi.copy();valid=(~empty).copy()
                rng=np.random.default_rng(PLAN['seed']+100+fold)
                trainmean=mean_model(p[train],plate[train],theta)
                for rep in range(boots):
                    yy=draw(rng,y[train].sum(1),trainmean,kappa)
                    tb,ab=fit(yy,p[train],plate[train],'dm',start=theta)
                    if not ab['success']:raise RuntimeError('Bootstrap optimizer failed: '+str(ab))
                    bb,ee,hh,kk=unpack(tb)
                    lb,ub,eb=interval_from_nuisance(y[test],bb[plate[test]],ee,hh,kk)
                    elo=np.fmin(elo,lb);ehi=np.fmax(ehi,ub);valid|=~eb
                    bootstrap.append(dict(fold=fold,rep=rep,theta=tb.tolist(),audit=ab,
                        counts=yy.astype(int).tolist()))
                record_intervals(rows,intervals,ids,taxa,test,p,y,plate,fold,'dm_bootstrap_envelope',theta,'dm',elo,ehi,~valid)
    summary=[]
    for model in ('mn','dm','dm_bootstrap_envelope'):
        rr=[r for r in rows if r['model']==model]
        summary.append(dict(model=model,n=len(rr),contains_nominal=sum(r['contains_nominal'] for r in rr),
            empty=sum(r['empty'] for r in rr),
            mean_width_nonempty=float(np.mean([r['mean_width'] for r in rr if not r['empty']])),
            median_l1=float(np.median([r['l1_error'] for r in rr])),
            total_log_probability=float(sum(r['log_probability'] for r in rr))))
    write_csv(OUT/'validation_samples.csv',rows);write_csv(OUT/'validation_intervals.csv',intervals)
    (OUT/'fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    (OUT/'bootstrap_refits.json').write_text(json.dumps(bootstrap,separators=(',',':'))+'\n')
    (OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return fits,summary


def simulations(y,p,plate,assign,fits,knownreps,refitreps):
    reference=next(z for z in fits if z['fold']==0 and z['family']=='dm')
    theta=np.array(reference['theta']);b,eta,h,kappa=unpack(theta)
    # Chosen compositions are fixed in source before fitting; the nuisance
    # generator is deliberately fitted to fold 0, not asserted to be truth.
    compositions=[np.ones(7)/7,np.array([1/3,1/3,1/3,0,0,0,0]),np.array([1.,0,0,0,0,0,0])]
    summaries=[];raw=[]
    for idx,ptrue in enumerate(compositions):
        rng=np.random.default_rng(PLAN['seed']+1000+idx)
        mu=mean_model(ptrue,0,theta)
        yy=draw(rng,20000,np.broadcast_to(mu,(knownreps,7)),kappa)
        for infer in ('mn','dm'):
            lo,hi,empty=interval_from_nuisance(yy,b[0],eta,h,np.inf if infer=='mn' else kappa)
            covered=np.all((lo<=ptrue)&(ptrue<=hi),axis=1)&~empty
            n=int(covered.sum());cl,ch=wilson(n,knownreps)
            summaries.append(dict(kind='known_nuisance',composition=idx,inference=infer,n=knownreps,
                covered=n,coverage=n/knownreps,mc_low=cl,mc_high=ch,empty=int(empty.sum()),
                mean_width=float(np.nanmean(hi-lo))))
            for z in range(knownreps):
                rr=dict(kind='known_nuisance',composition=idx,inference=infer,rep=z,
                    covered=int(covered[z]),empty=int(empty[z]),width=float(np.mean(hi[z]-lo[z])))
                rr.update({f'Y{j+1}':int(yy[z,j]) for j in range(7)});raw.append(rr)
    train=np.where(assign!=0)[0];mu_train=mean_model(p[train],plate[train],theta)
    refits=[]
    for idx,ptrue in enumerate(compositions[:2]):
        rng=np.random.default_rng(PLAN['seed']+2000+idx)
        cv=[];width=[];emp=[]
        for rep in range(refitreps):
            yy=draw(rng,y[train].sum(1),mu_train,kappa)
            yt=draw(rng,20000,mean_model(ptrue,0,theta),kappa)[0]
            th,audit=fit(yy,p[train],plate[train],start=theta)
            if not audit['success']:raise RuntimeError('Refit simulation failed '+str(audit))
            bb,ee,hh,kk=unpack(th)
            lo,hi,empty=interval_from_nuisance(yt,bb[0],ee,hh,kk)
            covered=bool(not empty and np.all((lo<=ptrue)&(ptrue<=hi)))
            cv.append(covered);width.append(float(np.mean(hi-lo)));emp.append(bool(empty))
            refits.append(dict(composition=idx,rep=rep,counts=yy.tolist(),target_counts=yt.tolist(),
                theta=th.tolist(),audit=audit,covered=covered,empty=bool(empty),width=width[-1]))
        n=sum(cv);cl,ch=wilson(n,refitreps)
        summaries.append(dict(kind='refitted_nuisance',composition=idx,inference='dm',n=refitreps,
            covered=n,coverage=n/refitreps,mc_low=cl,mc_high=ch,empty=sum(emp),mean_width=float(np.nanmean(width))))
        print('refit simulation',idx,n,'/',refitreps,flush=True)
    write_csv(OUT/'simulation_summary.csv',summaries);write_csv(OUT/'simulation_known_counts.csv',raw)
    (OUT/'simulation_refits.json').write_text(json.dumps(refits,separators=(',',':'))+'\n')
    return summaries


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=100)
    ap.add_argument('--known-reps',type=int,default=10000);ap.add_argument('--refit-reps',type=int,default=300)
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    ids,taxa,y,p,plate=load_data();assign=folds(p)
    write_csv(OUT/'folds.csv',[dict(sample=s,plate=int(plate[i])+1,fold=int(assign[i]),
        support=';'.join(taxa[j] for j in np.where(p[i]>0)[0])) for i,s in enumerate(ids)])
    diagnosis(ids,taxa,y,p,plate)
    fits,summary=validation(ids,taxa,y,p,plate,assign,a.bootstrap)
    print(json.dumps(summary,indent=2),flush=True)
    simulations(y,p,plate,assign,fits,a.known_reps,a.refit_reps)
    (OUT/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,seed=PLAN['seed'],bootstrap=a.bootstrap,known_reps=a.known_reps,
        refit_reps=a.refit_reps,split='grouped nominal support',optimizer='L-BFGS-B analytic gradient'),indent=2)+'\n')

if __name__=='__main__':main()
