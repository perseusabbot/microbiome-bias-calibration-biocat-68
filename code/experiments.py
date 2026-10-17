"""Regenerate all scientific tables: python code/experiments.py [--reps 10000]."""
from pathlib import Path
import argparse, csv, json, math, platform
import numpy as np
from calibration import (normalize, calibration_intervals, fixed_bias_intervals,
    point_estimate, l1_radius, expand_for_drift, drift_l1_bound)

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wilson(k, n):
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius, center+radius


def scenario_list():
    p = [.4,.3,.15,.1,.05]; r = [.2]*5; b = [.25,.5,1,2,4]
    out = []
    for m,n in [(500,500),(2000,2000),(20000,20000),(200,50000),
                (1000,50000),(50000,200),(50000,1000)]:
        out.append(dict(name=f'depth_M{m}_N{n}', p=p,r=r,b=b,M=m,N=n,
                        kind='multinomial', drift=[1]*5))
    out.extend([
        dict(name='rare_sample',p=[.7,.2,.09,.009,.001],r=r,b=[.5,1,2,4,.1],
             M=2000,N=2000,kind='multinomial',drift=[1]*5),
        dict(name='true_zero_sample',p=[.55,.25,.2,0,0],r=r,b=b,
             M=2000,N=2000,kind='multinomial',drift=[1]*5),
        dict(name='calibration_dropout',p=p,r=[.24995]*4+[.0002],b=[.5,1,2,4,.1],
             M=2000,N=2000,kind='multinomial',drift=[1]*5),
        dict(name='drift_1p5x',p=p,r=r,b=b,M=20000,N=20000,
             kind='multinomial',drift=[1.5,1,1,1,1]),
        dict(name='drift_4x',p=p,r=r,b=b,M=20000,N=20000,
             kind='multinomial',drift=[4,1,1,1,1]),
        dict(name='contamination_5pct',p=p,r=r,b=b,M=20000,N=20000,
             kind='contamination',contamination=.05,drift=[1]*5),
        dict(name='overdispersion_kappa200',p=p,r=r,b=b,M=20000,N=20000,
             kind='dirichlet_multinomial',kappa=200,drift=[1]*5),
    ])
    return out


def simulate(config, reps, base_seed, alpha):
    summaries, raw_rows = [], []
    for idx, sc in enumerate(config):
        rng = np.random.Generator(np.random.PCG64(base_seed+idx))
        p,r,b,d = (np.asarray(sc[k],float) for k in ('p','r','b','drift'))
        m,n = sc['M'],sc['N']; q=len(p)
        rho_c = normalize(b*r); rho_y = normalize(b*d*p)
        if sc['kind']=='contamination':
            h = np.zeros(q);h[-1]=1
            rho_y = (1-sc['contamination'])*rho_y+sc['contamination']*h
        if sc['kind']=='dirichlet_multinomial':
            cc = rng.dirichlet(sc['kappa']*rho_c, size=reps)
            yy = rng.dirichlet(sc['kappa']*rho_y, size=reps)
            c = np.array([rng.multinomial(m,z) for z in cc])
            y = np.array([rng.multinomial(n,z) for z in yy])
        else:
            c = rng.multinomial(m,rho_c,size=reps)
            y = rng.multinomial(n,rho_y,size=reps)
        point=point_estimate(y,c,r)
        low,high=calibration_intervals(y,c,r,alpha)
        rad=l1_radius(point,low,high)
        covered=np.all((low<=p)&(p<=high),axis=1)
        err=np.abs(point-p).sum(1)
        estimated_bias=(c+.5)/r  # common scalar cancels
        flo,fhi=fixed_bias_intervals(y,estimated_bias,alpha)
        fcovered=np.all((flo<=p)&(p<=fhi),axis=1)
        log_range=float(np.ptp(np.log(d)))
        dlo,dhi=expand_for_drift(low,high,log_range)
        dcovered=np.all((dlo<=p)&(p<=dhi),axis=1)
        drift_rad=np.minimum(2,rad+drift_l1_bound(log_range))
        cov_lo,cov_hi=wilson(int(covered.sum()),reps)
        summary=dict(scenario=sc['name'],kind=sc['kind'],M=m,N=n,reps=reps,
            coverage=float(covered.mean()),coverage_mc_lo=cov_lo,coverage_mc_hi=cov_hi,
            frozen_calibration_coverage=float(fcovered.mean()),
            drift_adjusted_coverage=float(dcovered.mean()),
            l1_certificate_coverage=float((err<=rad+1e-10).mean()),
            drift_l1_certificate_coverage=float((err<=drift_rad+1e-10).mean()),
            mean_interval_width=float((high-low).mean()),
            median_l1_radius=float(np.median(rad)),
            median_drift_l1_radius=float(np.median(drift_rad)),
            median_l1_corrected=float(np.median(err)),
            median_l1_raw=float(np.median(np.abs(y/n-p).sum(1))),
            calibration_any_zero_rate=float(np.any(c==0,axis=1).mean()),
            sample_last_taxon_zero_rate=float((y[:,-1]==0).mean()),
            median_last_taxon_upper=float(np.median(high[:,-1])),log_drift_range=log_range)
        summaries.append(summary)
        # All per-replicate results plus input counts: no hidden draws needed.
        for z in range(reps):
            row=dict(scenario=sc['name'],rep=z,covered=int(covered[z]),
                     frozen_covered=int(fcovered[z]),drift_covered=int(dcovered[z]),
                     l1_error=float(err[z]),l1_radius=float(rad[z]),
                     mean_width=float(np.mean(high[z]-low[z])))
            row.update({f'C{i+1}':int(c[z,i]) for i in range(q)})
            row.update({f'Y{i+1}':int(y[z,i]) for i in range(q)})
            raw_rows.append(row)
        print(sc['name'], 'coverage',round(summary['coverage'],4),
              'width',round(summary['mean_interval_width'],4),flush=True)
    write_csv(ROOT/'results/simulation_summary.csv',summaries)
    write_csv(ROOT/'results/simulation_replicates.csv',raw_rows)
    return summaries


def read_dict_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def empirical(alpha):
    meta=read_dict_csv(ROOT/'data/brooks2015-sample-data.csv')
    obs=read_dict_csv(ROOT/'data/brooks2015-observed.csv')
    act=read_dict_csv(ROOT/'data/brooks2015-actual.csv')
    taxa=[k for k in act[0] if k!='Sample']
    observed={row['Sample']:np.array([int(row[t]) for t in taxa]) for row in obs}
    actual={row['Sample']:np.array([float(row[t]) for t in taxa]) for row in act}
    omap={row['Sample']:int(row['Other']) for row in obs}
    calibrators={}
    for row in meta:
        if row['Mixture_type']=='Cells' and np.all(actual[row['Sample']]>0):
            if row['Plate'] in calibrators:
                raise RuntimeError('Selection rule assumed one full-support calibrator per plate')
            calibrators[row['Plate']]=row['Sample']
    samples,intervals=[],[]
    for row in meta:
        sid=row['Sample'];cal=calibrators[row['Plate']]
        if sid==cal or row['Mixture_type']!='Cells':continue
        y,c=observed[sid],observed[cal];p=normalize(actual[sid]);r=normalize(actual[cal])
        lo,hi=calibration_intervals(y,c,r,alpha)
        point=point_estimate(y,c,r)
        absent=p==0;present=~absent
        samples.append(dict(sample=sid,plate=row['Plate'],calibrator=cal,
            retained_reads=int(y.sum()),excluded_other_reads=omap[sid],
            absent_taxon_reads=int(y[absent].sum()),
            absent_taxa_with_reads=int((y[absent]>0).sum()),
            all_taxa_cover_nominal=int(np.all((lo<=p)&(p<=hi))),
            present_taxa_cover_nominal=int(np.all((lo[present]<=p[present])&(p[present]<=hi[present]))),
            l1_raw=float(np.abs(normalize(y)-p).sum()),
            l1_corrected=float(np.abs(point-p).sum()),
            l1_radius=float(l1_radius(point,lo,hi)),mean_width=float(np.mean(hi-lo))))
        for i,taxon in enumerate(taxa):
            intervals.append(dict(sample=sid,calibrator=cal,taxon=taxon,counts=int(y[i]),
                nominal=float(p[i]),estimate=float(point[i]),lower=float(lo[i]),upper=float(hi[i])))
    write_csv(ROOT/'results/empirical_samples.csv',samples)
    write_csv(ROOT/'results/empirical_intervals.csv',intervals)
    summary=dict(dataset='Brooks2015 cells from metacal',calibrators=calibrators,
        n_held_out=len(samples),taxa=taxa,
        n_all_taxa_cover_nominal=sum(z['all_taxa_cover_nominal'] for z in samples),
        n_present_taxa_cover_nominal=sum(z['present_taxa_cover_nominal'] for z in samples),
        n_samples_with_absent_taxon_reads=sum(z['absent_taxon_reads']>0 for z in samples),
        absent_taxon_reads=sum(z['absent_taxon_reads'] for z in samples),
        retained_reads=sum(z['retained_reads'] for z in samples),
        excluded_other_reads=sum(z['excluded_other_reads'] for z in samples),
        median_l1_raw=float(np.median([z['l1_raw'] for z in samples])),
        median_l1_corrected=float(np.median([z['l1_corrected'] for z in samples])),
        mean_interval_width=float(np.mean([z['mean_width'] for z in samples])),
        n_l1_certificates_cover_nominal=sum(z['l1_corrected']<=z['l1_radius'] for z in samples),
        interpretation='Model stress test against nominal mixtures; not a coverage estimate from independent replications.')
    (ROOT/'results/empirical_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20261017);args=ap.parse_args()
    if args.reps<1:raise ValueError('reps must be positive')
    config=json.loads((ROOT/'data/experiment_plan.json').read_text())
    (ROOT/'results').mkdir(exist_ok=True)
    simulate(config['scenarios'],args.reps,args.seed,config['alpha'])
    print(json.dumps(empirical(config['alpha']),indent=2))
    provenance=dict(python=platform.python_version(),numpy=np.__version__,
        rng='NumPy PCG64',seed=args.seed,reps=args.reps,
        alpha=config['alpha'],scenario_seed_rule='base seed + zero-based scenario index',
        command='python code/experiments.py --reps %d --seed %d'%(args.reps,args.seed))
    (ROOT/'results/run_environment.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
