"""Version 2 report pages, generated from reproducible extension outputs."""
import json,csv
import numpy as np


def extension_content(api):
    root=api['ROOT'];out=root/'results/extension'
    p,h,page,table,eq=(api[k] for k in ('p','h','page','table','eq'))
    d=json.loads((out/'diagnosis.json').read_text())
    v={r['model']:r for r in json.loads((out/'validation_summary.json').read_text())}
    f=[z for z in json.loads((out/'fits.json').read_text()) if z['family']=='dm']
    s=list(csv.DictReader((out/'simulation_summary.csv').open()))
    audit=json.loads((out/'verification.json').read_text())
    env=json.loads((out/'environment.json').read_text())
    if (env['bootstrap'],env['known_reps'],env['refit_reps'])!=(100,10000,300):
        raise ValueError('The extension report requires the complete reference run.')
    plt=api['plt'];blue=api['BLUE'];red=api['RED'];gray=api['GRAY']
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.2))
    labels=['Original','Contamination','+ sample variation','+ bootstrap envelope']
    vals=[4,v['mn']['contains_nominal'],v['dm']['contains_nominal'],v['dm_bootstrap_envelope']['contains_nominal']]
    yy=np.arange(4);ax[0].barh(yy,vals,color=[gray,red,blue,blue]);ax[0].invert_yaxis()
    ax[0].set_yticks(yy,labels);ax[0].set_xlim(0,84);ax[0].set_xlabel('Nominal compositions contained / 78')
    for j,x in enumerate(vals):ax[0].text(x+.8,j,str(x),va='center',fontsize=9)
    rr=list(csv.DictReader((out/'diagnostic_logratios.csv').open()))
    zz=[float(r['z']) for r in rr]
    ax[1].hist(zz,bins=25,color=blue,alpha=.8)
    for z in [-3,3]:ax[1].axvline(z,color=red,linestyle='--')
    ax[1].set(xlabel='Log-ratio residual / count-only SE',ylabel='Dependent taxon-pair comparisons')
    fig.subplots_adjust(left=.22,right=.97,wspace=.6,bottom=.23,top=.95)
    fig.savefig(root/'figures/extension_diagnosis.png',dpi=220);plt.close(fig)

    page('10. Why the original intervals failed')
    p('The 78 original targets, seven taxa, raw counts, and nominal cell-mixture compositions are retained without deleting reads at nominally absent taxa. The original simultaneous guarantee assumed those read probabilities were exactly zero. Any positive count then produces a positive lower bound, excluding the nominal zero by construction.')
    table(['Original diagnostic','Samples'],[
        ['All nominal proportions contained','4'],
        ['Present taxa contained; nominally absent taxa cause failure','31'],
        ['Present taxa fail; reads also occur at nominally absent taxa','40'],
        ['Present taxa fail; no reads at nominally absent taxa','3'],
    ],[425,95])
    p(f"The {d['total_stray_reads']:,} reads assigned to nominally absent taxa represent {100*d['fraction_all_reads_stray']:.3f}% of retained target reads, but occur in {d['n_stray_reads']}/78 targets. Their small total fraction does not make them harmless to an exact-zero hypothesis. Cross-contamination, misclassification, preparation errors, and incorrect nominal truth cannot be separated by these processed counts.")
    h('Contamination alone cannot explain the failures')
    p('Even when the diagnostic is restricted to taxa declared present, 43/78 original interval vectors miss at least one nominal proportion. For each co-present pair, compare its observed log ratio with the nominal ratio and the plate calibrator ratio. The approximate count-only variance is 1/Y_i + 1/Y_j + 1/C_i + 1/C_j, including calibration noise.')
    p(f"Among {d['n_logratios']} dependent pair comparisons, {100*d['fraction_logratios_abs_z_gt3']:.1f}% exceed three such standard errors; the median absolute log-ratio discrepancy is {d['median_abs_logratio_error']:.3f}. This is descriptive evidence of excess discrepancy, not 160 independent hypothesis tests. Nominal preparation error, sample-specific efficiency, and other process effects remain confounded.")
    h('Interpretation')
    p('The theorem was applied outside its model. More sequencing reduces multinomial noise but cannot eliminate systematic mean errors or variation in sample-level read probabilities. The extension explicitly gives those probabilities a distribution and allows background reads.')

    page('11. Contamination plus sample variation')
    eq(r'\mu_s=(1-\eta)\,\frac{b_{\ell(s)}\odot p_s}{\sum_j b_{\ell(s),j}p_{s,j}}+\eta h')
    eq(r'Z_s\sim\mathrm{Dirichlet}(\kappa\mu_s),\qquad Y_s\mid Z_s\sim\mathrm{Mult}(N_s,Z_s)')
    p('The latent Z_s represents variation in read probabilities between samples around the mean mu_s. This Dirichlet-multinomial model is a phenomenological measurement model, not a separation of biological variation from preparation noise. Its count variance is N mu_i(1-mu_i)(N+kappa)/(1+kappa). The contamination-only comparator instead sets Z_s=mu_s and uses multinomial counts.')
    table(['Parameter','Fit and restriction'],[
        ['b by plate','Six log efficiency ratios per plate; Streptococcus is the reference (b_7=1).'],
        ['eta and h','One shared background-read fraction and one seven-taxon profile, shared across plates and specimens. Estimated from known-composition training controls.'],
        ['kappa','One concentration shared across samples; lower values mean greater variation.'],
        ['Numerical bounds','Log efficiency/profile ratios in [-12,12]; eta=0.2 sigmoid(t), t in [-16,8]; kappa in [1,10^7]. No primary fit hit a bound.'],
    ],[102,418])
    p('Fit the marginal count likelihood using L-BFGS-B with analytic gradients. Each primary fit uses two starts, and optimizer diagnostics are retained. This is not a proof of global optimization. Each training design has numerical mean-Jacobian rank 19 at the checked interior start, supporting local mean identifiability there; global identifiability is not established.')
    h('Validation and uncertainty protocol')
    p('Five fixed folds group identical nominal support sets across plates. The two full-support controls are always in training; each fold trains on 64 or 65 samples and predicts 15 or 16. The split uses mixture metadata. Target nominal values are used for scoring, never as a prediction support mask or fitted outcome. All seven taxa remain candidates. The new design was fixed before model fitting, after the original outcomes had already been examined; it is exploratory, not an external preregistration.')
    p('For each fold, 100 parametric bootstrap datasets retain the training compositions, plates, and depths, resample counts from the fitted DM model, and refit every parameter. The envelope is the coordinatewise union of target intervals across those refits and the original fit. It is a sensitivity envelope; it is not a proved 95% confidence procedure for fitted nuisance parameters.')

    page('12. Conditional intervals and their limits')
    h('A guarantee when b, eta, h, and kappa are known')
    p('Allocate alpha/(4q) to each tail of each of two stages. First bound each latent Z_i with the original binomial KL interval using Y_i and N. Conditional on Z, the probability that any coordinate interval fails is at most alpha/2. Second, use the Beta(kappa mu_i, kappa(1-mu_i)) marginal of Z_i to bound its mean mu_i.')
    eq(r'\Pr_{\mu=L_\mu}\{Z_i\geq L_Z\}=\frac{\alpha}{4q},\qquad \Pr_{\mu=U_\mu}\{Z_i\leq U_Z\}=\frac{\alpha}{4q}')
    p('Both inverse endpoints are increasing in their observed argument. The Beta family is stochastically increasing in mu for fixed kappa: the density ratio at larger mu is proportional to [z/(1-z)] raised to a positive power. On the count event, the interval obtained from L_Z and U_Z contains the interval that would use the actual Z_i. The two Beta tails fail with probability at most alpha/(2q) per coordinate. A union bound over both stages and q coordinates gives simultaneous failure probability at most alpha. Coordinate independence is unnecessary.')
    eq(r'l_i=\max\{0,(L_{\mu,i}-\eta h_i)/(1-\eta)\},\quad u_i=\min\{1,(U_{\mu,i}-\eta h_i)/(1-\eta)\}')
    p('These bound the clean mean composition in read space. Divide both endpoints by b_i and apply the original normalized-weight bounds. If any l_i>u_i, sum l_i>1, or sum u_i<1, the candidate set is empty: flag model rejection, record noncontainment, and return NaN endpoints. Never replace rejection with [0,1]. For the multinomial comparator, omit the Beta stage and allocate alpha/(2q) to the count tails.')
    h('Boundary behavior and computation')
    p('Zero true biological abundance is permitted because eta h can still generate positive counts. Zero counts get a zero latent lower bound; endpoint bisection preserves outer brackets. Explicit sums over other coordinates preserve an exact upper endpoint of one. Verification caught a floating-point cancellation issue in an earlier implementation; the delivered code and all final runs use the corrected calculation.')
    h('What is not guaranteed')
    p('Inserting estimated b, eta, h, and kappa invalidates a direct application of the conditional theorem. The bootstrap envelope explores training-estimation sensitivity but lacks a nuisance-coverage theorem. A rigorous extension could take a union over a valid simultaneous nuisance confidence region and allocate a separate failure budget; this study does not construct that region. A simplex point estimate is produced by truncating negative contamination-subtracted read proportions, dividing by b, and normalizing. It is not claimed to be an unbiased estimator or a likelihood MLE.')

    page('13. Grouped held-out results')
    rows=[['Original one-control construction','4/78','0.056','0.126','0']]
    names={'mn':'Contamination only','dm':'Contamination + variation','dm_bootstrap_envelope':'DM bootstrap envelope'}
    for key in names:
        r=v[key];rows.append([names[key],f"{r['contains_nominal']}/78",f"{r['mean_width_nonempty']:.3f}",f"{r['median_l1']:.3f}",str(r['empty'])])
    table(['Method','Contains nominal','Mean width','Median L1','Empty'],rows,[180,95,80,90,75])
    p('Containment requires all seven nominal proportions to be inside their intervals. Width is in abundance units; 0.163 is 16.3 percentage points. Width for contamination-only is averaged over its 45 nonempty cases; 33 rejected cases have no interval width. Empty cases count as failures. The original method uses one full-support control per plate; the new models use many training controls, so improvement is not attributable solely to the new likelihood.','small')
    api['story'].append(api['Image'](str(root/'figures/extension_diagnosis.png'),width=520,height=520*3.2/9.5))
    p(f"Across folds, fitted eta ranges from {100*min(z['eta'] for z in f):.3f}% to {100*max(z['eta'] for z in f):.3f}%, and kappa from {min(z['kappa'] for z in f):.1f} to {max(z['kappa'] for z in f):.1f}. At 20,000 reads this implies approximately {(20000+max(z['kappa'] for z in f))/(1+max(z['kappa'] for z in f)):.0f}-{(20000+min(z['kappa'] for z in f))/(1+min(z['kappa'] for z in f)):.0f} times the multinomial count variance. Eta includes background reads at both present and absent taxa and need not equal the observed absent-taxon fraction.")
    p(f"Total held-out log count probability at nominal compositions improves from {v['mn']['total_log_probability']:.1f} for contamination-only to {v['dm']['total_log_probability']:.1f} for DM (higher is better). Overlapping training sets and dependent designs preclude treating this aggregate as an independent-test significance result.")
    h('The retained failure')
    p('The envelope still misses s1-12. Its nominal Atopobium/Prevotella composition is 0.5/0.5, but the corrected estimate is 0.291/0.709. The Atopobium envelope is [0.106, 0.469]. No additional tuning was performed to force this sample inside. The cause is unresolved; a shared concentration and contamination profile do not capture every possible sample effect.')

    page('14. Synthetic and numerical verification')
    p('Synthetic nuisance values use the DM fit from fold 0; they are a controlled test generator, not established biological truth. Draw 10,000 independent target samples of depth 20,000 for each fixed composition: uniform seven taxa, equal first three taxa, and a singleton first taxon. Compare known-parameter MN and DM intervals on the same 30,000 draws.')
    simrows=[]
    labels=['Seven equal taxa','Three equal taxa','Singleton']
    for idx in range(3):
        mn=next(z for z in s if z['kind']=='known_nuisance' and z['composition']==str(idx) and z['inference']=='mn')
        dm=next(z for z in s if z['kind']=='known_nuisance' and z['composition']==str(idx) and z['inference']=='dm')
        simrows.append([labels[idx],f"{100*float(mn['coverage']):.2f}%",f"{100*float(dm['coverage']):.2f}%",f"{float(dm['mean_width']):.3f}"])
    table(['True composition','MN coverage','DM coverage','DM mean width'],simrows,[160,120,120,120])
    p('For each of the first two compositions, 300 additional runs independently resample the complete training counts, refit every DM parameter, and draw a new target. Plug-in intervals contain truth in 300/300 runs in each case; the individual Wilson lower bound is 98.74%. This limited, model-conforming test does not cover arbitrary nuisance values, contamination mechanisms, small calibration sets, or the bootstrap envelope as a procedure. All counts and fitted parameters are retained.')
    h('Verification beyond simulations')
    table(['Check','Result'],[
        ['Analytic likelihood gradients','Relative discrepancy < 4e-7 against numerical differentiation for both models.'],
        ['Likelihood implementation','Agrees with independent SciPy multinomial and Dirichlet-multinomial log-PMF APIs.'],
        ['Exact finite enumeration',f"32 two-taxon parameter cases; maximal failure probability {audit['exact_max_failure_alpha_point2']:.5f} at alpha=0.2, below the allowed 0.2."],
        ['Numerical and design checks','Beta-tail inversion, noiseless inverse, efficiency-scale invariance, empty sets, exact simplex endpoints, grouped folds, and local rank checks pass.'],
        ['Audit totals','8 extension test methods pass, in addition to 12 original methods. All 10 primary fits, 500 bootstrap fits, and 600 simulation refits converged.'],
    ],[160,360])
    p('The elementary concentration and distributional inversions are not claimed to be new methods. Prior count-aware and contamination-aware approaches include fido and the broader model of Clausen and Willis [S3, S4, S7]. No head-to-head package benchmark was conducted.')

    page('15. Reproduce, interpret, and publish')
    h('Reproduction from the package root')
    p('<font name="Code">python -m pip install -r requirements.txt<br/>python code/verify.py<br/>python code/experiments.py --reps 10000 --seed 20261017<br/>python code/verify_extension.py<br/>python code/extension.py<br/>python -m pip install -r requirements-report.txt<br/>python code/build_report.py</font>')
    p('The reference extension uses Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3, seed 20261018, 100 bootstrap refits per fold, 10,000 known-parameter draws per composition, and 300 training refits per selected composition. Default command arguments reproduce these settings. The model plan is in data/extension_plan.json. See results/extension for folds, diagnostics, all individual predictions, fits, resampled counts, audit results, and the environment. Clean-copy agreement is recorded in results/extension/reproduction_check.json.')
    h('What the extension establishes')
    p('On these processed mock-community counts, a model allowing background reads and sample-level variation is substantially more compatible with nominal compositions than a count-noise-only model. Conditional intervals have a written known-nuisance coverage argument. Numerical and synthetic checks support the implementation, while real-data held-out results expose a remaining failure.')
    h('What remains open')
    p('There are only two plates from one experimental system, nominal rather than independently remeasured composition truths, no blank controls or dilution series in this subset, and dependent cross-validation predictions. These data cannot establish the biological source of stray reads, separate all error mechanisms, validate matrix or laboratory transfer, identify absolute abundance, or validate a universal fitted-parameter 95% guarantee. One shared contamination profile and one concentration may be too restrictive. Wider intervals alone do not establish useful calibration; both width and failures are reported.')
    p('A next independent study should use replicated known-composition controls, blanks and dilution levels, multiple matrices/laboratories, and a locked prediction procedure. It should evaluate fitted-nuisance coverage and width with composition uncertainty carried into scoring. No full solution to BIOCAT-68 (displayed as BIO-300), new biological finding, or research-priority claim is made.')
    h('Publication record and attribution')
    p('The GitHub submission text uses the we-solve issue form and stable handle BIOCAT-68. Its claim is a partial methodological result and computational evidence. The PDF is for reading; the submission includes text and reproducible source because the issue form does not accept PDF as the write-up. publication.json records actual publication state and URLs. The human user confirmed all three required declarations. Approved licenses are CC BY 4.0 for new report text and Apache-2.0 for new software; upstream data retain their included MIT notice and attribution. Contributor and model disclosure are supplied.')
