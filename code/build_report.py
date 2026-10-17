"""Build scientific figures and the PDF from the bundled analysis outputs."""
from pathlib import Path
import csv, json, math, os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab import rl_config

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
rl_config.invariant=1
S=list(csv.DictReader((ROOT/'results/simulation_summary.csv').open()))
D={r['scenario']:r for r in S}
E=json.loads((ROOT/'results/empirical_summary.json').read_text())
V=json.loads((ROOT/'results/verification.json').read_text())
R=json.loads((ROOT/'results/run_environment.json').read_text())
if R['seed'] != 20261017 or R['reps'] != 10000:
    raise ValueError('This report describes the reference run. Regenerate experiments with --reps 10000 --seed 20261017 first.')
refs=json.loads((ROOT/'data/sources.json').read_text())['references']
NREPS=int(R['reps'])
BLUE='#176B87';RED='#BF5B45';INK='#172B3A';GRAY='#62727E'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold'})


def figures():
    selected=['depth_M500_N500','depth_M2000_N2000','depth_M20000_N20000',
              'depth_M200_N50000','rare_sample','calibration_dropout']
    labels=['M=N=500','M=N=2,000','M=N=20,000','M=200; N=50,000','Rare sample taxon','Rare calibration taxon']
    fig,ax=plt.subplots(1,2,figsize=(10.2,4.1),gridspec_kw={'width_ratios':[1.1,1]})
    yy=np.arange(len(selected))
    ax[0].barh(yy-.16,[100*float(D[s]['coverage']) for s in selected],height=.3,color=BLUE,label='Propagate calibration noise')
    ax[0].barh(yy+.16,[100*float(D[s]['frozen_calibration_coverage']) for s in selected],height=.3,color=RED,label='Freeze estimated bias')
    ax[0].set_yticks(yy,labels);ax[0].invert_yaxis();ax[0].set_title('Shared multinomial model')
    ax[0].legend(loc='lower left',bbox_to_anchor=(-.4,-.3),frameon=False,fontsize=8)
    bad=['drift_1p5x','drift_4x','contamination_5pct','overdispersion_kappa200']
    yy=np.arange(4)
    ax[1].barh(yy,[100*float(D[s]['coverage']) for s in bad],height=.48,color=RED)
    ax[1].scatter([100*float(D[s]['drift_adjusted_coverage']) for s in bad[:2]],yy[:2],
                  marker='D',s=30,color=BLUE,zorder=4,label='External drift bound supplied')
    ax[1].set_yticks(yy,['1.5x bias drift','4x bias drift','5% read contamination','Overdispersion']);ax[1].invert_yaxis()
    ax[1].set_title('Deliberate model violations')
    ax[1].legend(loc='lower left',bbox_to_anchor=(-.4,-.3),frameon=False,fontsize=8)
    for a in ax:
        a.axvline(95,color=GRAY,linestyle='--',linewidth=1);a.set_xlim(0,105)
        a.set_xlabel('Simultaneous coverage (%)');a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
    fig.subplots_adjust(left=.18,right=.98,wspace=.85,bottom=.24,top=.88)
    fig.savefig(FIG/'simulation_coverage.png',dpi=220);plt.close(fig)
    rows=list(csv.DictReader((ROOT/'results/empirical_samples.csv').open()))
    x=np.array([float(z['l1_raw']) for z in rows]);y=np.array([float(z['l1_corrected']) for z in rows])
    covered=np.array([int(z['all_taxa_cover_nominal']) for z in rows],bool)
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.7),gridspec_kw={'width_ratios':[1.2,1]})
    ax[0].plot([0,1.7],[0,1.7],'--',color=GRAY,lw=1)
    ax[0].scatter(x[~covered],y[~covered],color=RED,s=25,alpha=.8,label='Some nominal taxa outside intervals')
    ax[0].scatter(x[covered],y[covered],color=BLUE,s=35,label='All nominal taxa inside intervals')
    ax[0].set(xlabel='Uncorrected L1 error',ylabel='Calibrated L1 error',xlim=(0,1.7),ylim=(0,1.7))
    ax[0].set_title('Held-out cell mixtures');ax[0].legend(frameon=False,fontsize=7,loc='upper left')
    counts=[E['n_all_taxa_cover_nominal'],E['n_present_taxa_cover_nominal'],E['n_samples_with_absent_taxon_reads']]
    bars=ax[1].barh(range(3),counts,color=[BLUE,BLUE,RED],height=.55)
    ax[1].set_yticks(range(3),['All 7 taxa covered','Present taxa covered','Absent taxa have reads'])
    ax[1].set_xlim(0,83);ax[1].invert_yaxis();ax[1].set_xlabel('Samples out of 78')
    ax[1].set_title('Nominal-mixture diagnostics')
    for bar,k in zip(bars,counts):ax[1].text(k+1,bar.get_y()+bar.get_height()/2,str(k),va='center')
    fig.subplots_adjust(left=.07,right=.97,wspace=.8,bottom=.18,top=.86)
    fig.savefig(FIG/'empirical_stress_test.png',dpi=220);plt.close(fig)


fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
for name,file in [('Body','DejaVuSans.ttf'),('Body-Bold','DejaVuSans-Bold.ttf'),('Body-Oblique','DejaVuSans-Oblique.ttf'),('Code','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Oblique',boldItalic='Body-Bold')
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=9.1,leading=13.4,textColor=colors.HexColor(INK),spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Body',fontSize=7.8,leading=11,textColor=colors.HexColor(GRAY),spaceAfter=6),
 'h1':ParagraphStyle('h1',fontName='Body-Bold',fontSize=19,leading=23,textColor=colors.HexColor(INK),spaceAfter=14),
 'h2':ParagraphStyle('h2',fontName='Body-Bold',fontSize=11,leading=15,textColor=colors.HexColor(BLUE),spaceBefore=6,spaceAfter=7),
 'title':ParagraphStyle('title',fontName='Body-Bold',fontSize=28,leading=34,textColor=colors.HexColor(INK),spaceAfter=18),
 'table':ParagraphStyle('table',fontName='Body',fontSize=8,leading=11,textColor=colors.HexColor(INK)),
}
story=[];equations=[]
def p(s,style='body'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'h2')
def page(title):
    if story:story.append(PageBreak())
    p(title,'h1')
def table(headers,rows,widths):
    data=[[Paragraph(str(x),styles['table']) for x in headers]]+[[Paragraph(str(x),styles['table']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5F0F3')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor(BLUE)),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F8')])]))
    story.extend([t,Spacer(1,10)])
def eq(s):
    equations.append(s);path=FIG/f'equation_{len(equations):02}.png'
    fig=plt.figure(figsize=(8,.55));fig.text(.02,.5,'$'+s+'$',fontsize=12,va='center')
    fig.savefig(path,dpi=250,bbox_inches='tight',pad_inches=.08,transparent=True);plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:w,h=im.size
    width=min(514,w*72/250);story.append(Image(str(path),width=width,height=width*h/w,hAlign='LEFT'));story.append(Spacer(1,7))
def pct(x):return f'{100*float(x):.2f}%'


def content():
    p('BIO-300  /  REPRODUCIBLE RESEARCH REPORT','small')
    p('Microbiome bias calibration','title')
    p('Why intervals failed, and what changes when contamination and sample variation are modeled','h1')
    p('17 October 2026 | Version 2 | Prepared by Codex | Scoped methodological contribution','small')
    h('Finding')
    p('The original intervals failed for two reasons consistent with their model assumptions being violated: stray reads at nominally absent taxa and excess variation in present-taxon ratios. Grouped held-out evaluation improves from 4/78 nominal compositions contained to 74/78 with a contamination-plus-variation model; a training-bootstrap sensitivity envelope contains 77/78. These empirical results do not establish general 95% coverage.')
    table(['Evidence','Result'],[
        ['Mathematics','A simultaneous finite-sample confidence construction propagates both calibration and target count noise. A separate sharp bound controls externally bounded multiplicative drift.'],
        ['Model-conforming simulations',f'No coverage failures in {10*NREPS:,} replicates across 10 declared settings. Intervals are conservative; this is not evidence of exact 100% coverage.'],
        ['Calibration bottleneck',f"With M=200 calibration reads and N=50,000 target reads, freezing estimated bias gives {pct(D['depth_M200_N50000']['frozen_calibration_coverage'])} simultaneous coverage; propagated intervals cover every simulated case but are wide."],
        ['Original stress test',f"In 78 held-out mock mixtures, median L1 error falls from {E['median_l1_raw']:.3f} to {E['median_l1_corrected']:.3f}, while only {E['n_all_taxa_cover_nominal']}/78 interval vectors contain all nominal proportions."],
        ['New empirical model','Grouped five-fold validation: contamination alone contains 11/78 nominal compositions; contamination plus Dirichlet-multinomial variation contains 74/78, with median L1 error 0.077.'],
    ],[132,388])
    h('What this report establishes')
    p('The original proofs and 140,000 simulations are retained. The extension adds a conditional known-parameter interval theorem, 30,000 synthetic target draws, 600 complete training refits, 500 bootstrap refits, and 8 further numerical checks. All inputs, fitted parameters, generated counts, and code are supplied.')
    h('Claim boundary')
    p('This does not solve the full BIO-300 target. The construction combines standard concentration and normalization arguments; no claim of research priority or a previously unknown statistical method is made. There is no new wet-lab experiment, proof-assistant verification, or independent human statistical review.')
    p('Original analysis: sections 1-9. New diagnosis, model, verification, and reproducibility: sections 10-15.','small')

    page('1. Scope and prior work')
    p('The selected we-solve entry asks for transferable bias correction, calibration designs, and uncertainty accounting for counts, dropout, and contamination. Its recorded attempt establishes calibration-ratio identification and a noiseless log-error bound, leaving the other components unresolved [S0]. This study takes one narrow next step: a single known, full-support calibration mixture.')
    h('Relevant methods already exist')
    p('McLaren, Willis and Callahan introduced and empirically evaluated the multiplicative-bias model [S1]. The metacal tutorial describes a deterministic normalized correction and its uncertainty limitation [S2]. Count-aware Bayesian calibration is described with fido [S3, S7]. Clausen and Willis develop a broader semiparametric model with detectability, contamination, boundary estimates, and asymptotic inference [S4]. These sources rule out presenting count-aware microbiome calibration as a new research area. No head-to-head performance comparison with those packages is performed.')
    h('Declared model')
    table(['Symbol','Meaning'],[
        ['q','A fixed, prespecified list of taxa.'],['r','Known calibration composition; every r_i > 0 and the entries sum to one.'],
        ['p','Unknown target relative composition on the same taxa; zero target proportions are allowed.'],
        ['b','Positive taxon efficiencies shared by calibration and target, identifiable only up to scale.'],
        ['C, M','Calibration count vector and its positive total read depth.'],['Y, N','Target count vector and its positive total read depth.'],
    ],[68,452])
    eq(r'C\sim\mathrm{Mult}(M,\sigma),\quad Y\sim\mathrm{Mult}(N,\rho),\quad \sigma_i=\frac{b_i r_i}{\sum_j b_jr_j},\quad \rho_i=\frac{b_i p_i}{\sum_j b_jp_j}')
    p('Counts are multinomial conditional on the stated depths. Within a table, counts across taxa are dependent; the proof does not assume independent taxa. Simulations use independent calibration and target draws, although the union-bound guarantee only needs the two stated marginal laws. There is no contamination, extra-multinomial variation, uncertain calibration composition, or unknown bias transfer in this model.')
    eq(r'p_i=\frac{r_i\rho_i/\sigma_i}{\sum_j r_j\rho_j/\sigma_j}')
    p('This identity follows by cancellation of b_i and the two normalizing constants. It is the inversion being bounded, not an unbiasedness claim for a plug-in estimator. Pooling calibration counts is justified here only for the same composition and the same efficiency vector. Different mixtures require a different analysis.')

    page('2. A finite-sample confidence construction')
    h('Step A: bound each count probability')
    p('For a binomial marginal K with depth n, let x=K/n. For q taxa and total failure budget alpha, use kappa=log(4q/alpha). The factor 4 accounts for two tails, two count tables, and q taxa. The confidence interval is the set of probabilities t satisfying:')
    eq(r'd(x\Vert t)=x\log\frac{x}{t}+(1-x)\log\frac{1-x}{1-t},\qquad n\,d(x\Vert t)\leq\kappa')
    p('Use the continuous boundary convention 0 log 0=0. This is a conservative Chernoff interval, not Clopper-Pearson. At K=0 it is [0, 1-exp(-kappa/n)]; at K=n it is [exp(-kappa/n), 1]. Thus zero observed reads do not establish absence. Each tail has probability at most alpha/(4q).')
    h('Step B: invert the simultaneous boxes')
    p('Write [l_i^Y,u_i^Y] and [l_i^C,u_i^C] for the target and calibration probability intervals. Form lower and upper weights, then normalize their worst cases:')
    eq(r'a_i=\frac{r_i l_i^Y}{u_i^C},\quad A_i=\frac{r_i u_i^Y}{l_i^C},\quad L_i=\frac{a_i}{a_i+\sum_{j\ne i}A_j},\quad U_i=\frac{A_i}{A_i+\sum_{j\ne i}a_j}')
    p('If l_i^C=0, set A_i=+infinity. If an upper numerator is infinite, set U_i=1; if a lower denominator contains an infinite competing weight, set L_i=0. Zero lower numerators give zero lower bounds. For q=1, return [1,1]. The implementation explicitly handles these cases instead of silently deleting taxa.')
    h('Theorem 1: simultaneous coverage')
    eq(r'\Pr\{L_i\leq p_i\leq U_i\ \mathrm{for\ all}\ i\}\ \geq\ 1-\alpha')
    p('The guarantee holds for every positive b, full-support r, p in the simplex, and positive M,N under the model on page 2. It concerns all taxa of one target sample. It is not simultaneous across many target specimens; a study-wide guarantee needs a further error-budget allocation.')
    h('Proof of the count step')
    p('For K~Bin(n,t), Markov\'s inequality applied to exp(lambda K) gives P(K/n >= z) <= inf_{lambda>=0} exp(-lambda n z)(1-t+t exp(lambda))^n. For z>t, optimizing at exp(lambda)=z(1-t)/[t(1-z)] yields exp[-n d(z||t)]. The lower tail follows using lambda<=0. The divergence is monotone as an observation moves away from t on either side. Therefore the event that the true t lies beyond either confidence endpoint has probability at most exp(-kappa). The discrete threshold uses the nearest attainable count in the relevant tail; boundary parameters follow directly. A union bound over 4q tails gives failure probability at most alpha. No independence between coordinates is required.')

    page('3. Normalization and an L1 error certificate')
    h('Completion of Theorem 1')
    p('On the simultaneous count event, each true unnormalized weight w_i=r_i rho_i/sigma_i lies in [a_i,A_i]. The function w_i/(w_i+sum_{j!=i}w_j) increases in its own weight and decreases in each competing weight. Its extremal values over the weight box are exactly the displayed L_i and U_i, with the stated infinite-endpoint limits. The true p therefore lies in every returned interval. This completes the coverage proof.')
    p('The weight box is an outer relaxation: it discards the sum-to-one constraints within each original probability table and their dependence structure. Consequently these are not generally shortest or sharpest intervals. This deliberate conservatism is visible in the experiments.')
    h('A descriptive point estimate')
    eq(r'\widehat p_i\ \propto\ r_i\frac{Y_i+1/2}{C_i+1/2},\qquad \sum_i\widehat p_i=1')
    p('The fixed half-count makes a point estimate available when a calibration count is zero. It is not used to construct confidence intervals, and it carries no unbiasedness or coverage guarantee of its own. The certificate below applies to any simplex-valued point estimate, including a data-dependent one.')
    h('Theorem 2: a computable global error radius')
    p('Let B be the interval box [L,U] intersected with the simplex. For every subset S of taxa, its largest feasible probability mass is min(sum_{i in S} U_i, 1-sum_{j not in S} L_j). Hence the exact maximum L1 distance over this outer set is:')
    eq(r'\varepsilon_B=2\max_{S\subseteq\{1,\ldots,q\}}\left[\min\left\{\sum_{i\in S}U_i,1-\sum_{j\notin S}L_j\right\}-\sum_{i\in S}\widehat p_i\right]')
    eq(r'\Pr\{\|\widehat p-p\|_1\leq\varepsilon_B\}\geq1-\alpha')
    p('Proof. Since p and the point estimate both sum to one, their L1 distance equals twice the largest excess mass over any subset. Maximizing this identity over B permits exchanging two finite maxima. For each subset, mass is limited both by its own upper bounds and by the complementary lower bounds. Either limiting value is attainable by distributing the remaining mass within the two boxes, because B is nonempty. This proves the expression and the probability statement follows from Theorem 1.')
    h('Interpretation and numerical limits')
    p('L1 error lies in [0,2]; total variation distance is half of L1. The formula is sharp for B, not for the original nonlinear confidence set. The supplied implementation enumerates 2^q subsets and deliberately rejects q>12. It uses float64 arithmetic, outer bisection endpoints, and a small outward radius allowance. Numerical audits support the implementation but do not constitute a certified floating-point proof.')

    page('4. Transfer: what a drift bound can buy')
    p('Suppose the target efficiencies are b_i times d_i, with the same positive b used for calibration. Correction with the calibration bias then identifies p-star=normalize(p times d), rather than p. With arbitrary unknown positive d, any positive candidate p can be made compatible with p-star by choosing d_i proportional to p-star_i/p_i. Transfer cannot be identified from these two count tables alone.')
    h('Theorem 3: an externally bounded drift')
    p('Assume an external deterministic bound R on the range of log d_i. Only relative drift matters: multiplying all d_i by the same constant changes no composition. Then:')
    eq(r'\max_i\log d_i-\min_i\log d_i\leq R\quad\Longrightarrow\quad\|p^*-p\|_1\leq2\tanh(R/4)')
    p('Proof. Rescale the weights so their minimum is 1 and their maximum is H=exp(R), and let mu=sum_i p_i d_i. Convexity places |d-mu| below the chord connecting its endpoint values on [1,H]. Taking expectation under p gives E|d-mu| <= 2(H-mu)(mu-1)/(H-1). Divide by mu to obtain the L1 distance. Its largest possible value over 1<=mu<=H occurs at mu=sqrt(H), yielding 2(sqrt(H)-1)/(sqrt(H)+1). For R=0 the result follows directly. Equality is attained with two taxa, weights (1,H), and probabilities (sqrt(H)/(sqrt(H)+1), 1/(sqrt(H)+1)). Thus the bound is sharp under only this range restriction.')
    h('Two ways to use the bound')
    p('Theorem 1 still covers p-star under the drifted model. A triangle inequality gives the L1 radius min(2, epsilon_B+2 tanh(R/4)). Alternatively widen each interval by bounding the ratio of d_i to the weighted mean drift among other taxa:')
    eq(r'L_i^{(R)}=\frac{L_i}{L_i+e^R(1-L_i)},\qquad U_i^{(R)}=\frac{e^R U_i}{1-U_i+e^R U_i}')
    p('These widened intervals simultaneously cover p with probability at least 1-alpha if the supplied R is valid. R is not estimated by this implementation. In simulations it is known from the generator; that is an oracle sensitivity exercise, not a data-derived certificate of real-world transfer.')
    h('Relation to the existing deterministic bound')
    p('If individual log-efficiency errors are bounded in absolute value by delta, their range is at most 2 delta. The resulting L1 bound is 2 tanh(delta/2), which is tighter than min(2, exp(2 delta)-1) in the catalogue attempt. This is an elementary normalization bound derived here with an equality example; no originality claim is attached to it.')
    h('What the bound does not cover')
    p('Additive contamination, uncertain r, and overdispersed counts are outside the proved count model. A zero observed calibration count is allowed and widens uncertainty. A taxon with r_i=0 is structurally uncalibrated: adding a pseudocount cannot identify its efficiency. The implementation rejects such an input.')

    page('5. Simulation design and complete results')
    p(f'All scenarios were written into experiment_plan.json before execution; this was not an external preregistration. Each scenario uses {NREPS:,} replicates, alpha=0.05, NumPy PCG64, and seed 20261017 plus its index. Ten settings obey the model; four deliberately violate it. Counts and per-replicate diagnostics are retained.')
    p('Base p=(0.40,0.30,0.15,0.10,0.05), r=(0.20,...,0.20), and b=(0.25,0.5,1,2,4). Rare-target p=(0.70,0.20,0.09,0.009,0.001). The calibration-dropout setting gives the last taxon calibration abundance 0.0002 and efficiency 0.1. Exact parameters for every setting are in the JSON input.')
    rows=[]
    names=['500 / 500','2,000 / 2,000','20,000 / 20,000','200 / 50,000','1,000 / 50,000','50,000 / 200','50,000 / 1,000','Rare target','True target zeros','Rare calibration','1.5x drift','4x drift','5% contamination','Overdispersion']
    for label,row in zip(names,S):
        rows.append([label,pct(row['coverage']),pct(row['frozen_calibration_coverage']),f"{float(row['mean_interval_width']):.3f}",f"{float(row['median_l1_corrected']):.3f}",f"{float(row['median_l1_radius']):.3f}"])
    table(['M / N or setting','Coverage','Frozen','Mean width','Median L1 error','Median radius'],rows,[137,69,69,76,86,83])
    p('Coverage is simultaneous across all five taxa. Width is averaged across taxa and replicates. The radius is the per-replicate bound from Theorem 2, summarized by its median. Neither a median width nor a median radius is itself a universal error guarantee. Frozen is a deliberately incomplete baseline that treats a smoothed estimated bias as known; it is not an evaluation of an existing software package.','small')
    p('For drift, only the first efficiency changes; full-model count assumptions otherwise hold. Contamination sends 5% of target read probability to taxon five while leaving calibration unchanged. Overdispersion draws separate Dirichlet probability vectors, with concentration 200, before multinomial calibration and target sampling. These violate the shared fixed-probability model in different ways.')
    p('For a scenario with 10,000/10,000 covered replicates, a pointwise 95% Wilson interval for its simulated coverage is approximately [99.962%,100%]. This Monte Carlo interval describes the chosen setting only. The mathematical guarantee, not simulation, supports the uniform model-based statement. No cross-scenario multiplicity claim is attached to the Wilson intervals.','small')

    page('6. Reading the simulation evidence')
    story.append(Image(str(FIG/'simulation_coverage.png'),width=520,height=209))
    p('Figure 1. Dashed lines mark 95%. Blue diamonds on the right use the correct externally supplied log-drift bound. No corresponding correction is applied to contamination or overdispersion. All results use the declared simulation settings.','small')
    h('Calibration uncertainty cannot be removed by deeper target sequencing')
    row=D['depth_M200_N50000']
    p(f"At M=200 and N=50,000, the frozen-bias baseline covers only {pct(row['frozen_calibration_coverage'])}. Accounting for both count tables covers every replicate, but mean coordinate width is {float(row['mean_interval_width']):.3f} and median L1 radius is {float(row['median_l1_radius']):.3f}. Deep sequencing of the target does not resolve imprecise efficiency estimates.")
    h('Zero counts honestly expose missing information')
    row=D['rare_sample']
    p(f"For a target taxon with true proportion 0.001, its sample count is zero in {pct(row['sample_last_taxon_zero_rate'])} of runs. The median upper bound is {float(row['median_last_taxon_upper']):.3f}; this is valid but far too loose to precisely estimate that rare abundance. Zero observed counts must not be converted into certainty of absence.")
    row=D['calibration_dropout']
    p(f"When the same difficulty occurs in calibration, at least one calibration count is zero in {pct(row['calibration_any_zero_rate'])} of runs. The last taxon's median upper bound is 1, and the median L1 radius is {float(row['median_l1_radius']):.3f}, nearly the maximal possible error 2. Coverage alone would conceal this lack of information.")
    h('A correct drift budget restores coverage at a cost')
    p('In the 1.5x and 4x drift settings, the unadjusted coverage is 1.23% and 0%, respectively. Widening with the known R=log(1.5) or log(4) restores coverage in every simulated run. For 4x drift the sharp deterministic L1 allowance is 2/3. Such an allowance can dominate sampling uncertainty; additional reads do not remove it.')
    h('Conservatism is a substantive finding')
    p('All ten model-conforming settings had no observed failures, while some intervals and radii were very wide. The construction is a transparent reference certificate, not a recommendation for an efficient production estimator. Tighter joint likelihood methods and explicit measurement-error models are natural next comparisons, subject to their own assumptions.')

    page('7. Public mock-community stress test')
    p('The Brooks et al. cell-mixture experiment has 80 samples across two plates and seven named taxa [S5]. The processed tables are bundled from metacal at a fixed commit [S6]. Before evaluating results, the rule selected the unique full-support mixture on each plate as calibration: s1-23 and s2-14. The remaining 39 mixtures per plate are targets. This uses nominal mixture composition as the reference, not independently remeasured truth.')
    p('Only the Other category is excluded, and depths are conditioned on the retained seven named taxa. Reads assigned to nominally absent named taxa are retained, so model incompatibilities remain visible. No ground-truth support filtering is used to improve the results.')
    story.append(Image(str(FIG/'empirical_stress_test.png'),width=520,height=200))
    p('Figure 2. Points below the diagonal have lower L1 error after calibration. The counts on the right are descriptive diagnostics across dependent samples sharing two calibrators. They are not independent binomial trials.','small')
    table(['Diagnostic','Observed'],[
        ['Median L1 error, raw -> calibrated',f"{E['median_l1_raw']:.3f} -> {E['median_l1_corrected']:.3f}"],
        ['All seven nominal proportions inside intervals',f"{E['n_all_taxa_cover_nominal']} / 78"],
        ['All nominally present proportions inside intervals',f"{E['n_present_taxa_cover_nominal']} / 78"],
        ['Samples with reads from nominally absent taxa',f"{E['n_samples_with_absent_taxon_reads']} / 78"],
        ['Such reads / all retained target reads',f"{E['absent_taxon_reads']:,} / {E['retained_reads']:,}"],
        ['L1 certificate contains nominal composition',f"{E['n_l1_certificates_cover_nominal']} / 78"],
    ],[352,168])
    p('If nominal zero abundances were exact and the no-contamination model held, reads from those taxa would have probability zero. Their presence demonstrates incompatibility with that idealization, though these data do not isolate contamination, misclassification, preparation error, or other mechanisms. Noncoverage among present taxa also shows that nominal zeros are not the only difficulty. A single calibrator per plate and uncertainty in nominal compositions further limit interpretation.')

    page('8. Verification, limitations, and status')
    h('Verification actually performed')
    table(['Check','Evidence'],[
        ['Automated verification',f"{V['test_methods']} test methods passed; no failures or errors."],
        ['Exact binomial sums',f"{V['exact_binomial_parameter_cases']} parameter cases; maximum failure {V['max_exact_binomial_failure_at_nominal_0p02']:.5f} at nominal two-tail budget 0.02."],
        ['Exact joint count enumeration',f"{V['exact_joint_parameter_cases']} parameter cases, {V['outcomes_per_joint_case']} outcomes each. Minimum coverage {V['minimum_exact_joint_coverage_at_nominal_0p80']:.6f} at nominal 0.80."],
        ['Independent geometry checks','100 random normalization boxes checked against every corner; 100 L1 radii checked against an independently enumerated set of simplex-box vertices.'],
        ['Drift checks',f"1,000 random reweightings and seven equality examples. Maximum equality discrepancy {V['drift_sharpness_max_error']:.2g}."],
        ['Boundary and interface checks','Zero target reads, zero calibration reads, one taxon, invalid inputs, batch/scalar agreement, and the exact inversion identity.'],
    ],[150,370])
    p('These tests are stronger than checking a few illustrative point estimates: they enumerate probability mass and compare distinct computational formulations. They still do not replace the written proof or establish correct floating-point behavior for every possible input.')
    h('Material limits')
    p('<b>Statistical:</b> exact calibration composition, complete taxon support, positive common efficiencies, and multinomial counts are assumed. The guarantee is per target sample. Unknown contamination, arbitrary bias drift, preparation uncertainty, and extra-multinomial variation are not absorbed by a wider sample-count interval.')
    p('<b>Computational:</b> the confidence construction is conservative; the exact outer-box L1 calculation is exponential and capped at 12 taxa. Float64 arithmetic is audited but not formally certified. No large-community or runtime-performance claim is made.')
    p('<b>Empirical:</b> one small processed mock-community dataset, two calibrators, nominal references, and shared calibration dependence support a stress test only. No cross-laboratory validation, absolute-abundance inference, raw-read pipeline validation, or biological discovery is claimed.')
    p('<b>Research:</b> the literature check was targeted. Comparable or stronger methods may already exist. Neither the elementary sharper normalization bound nor this confidence-box construction is asserted to be original. The original BIO-300 target remains unresolved.')
    h('Extension in this version')
    p('Sections 10-15 evaluate a contamination and process-variation model on grouped held-out mixtures. Replicated calibration across matrices, independently characterized composition uncertainty, and external coverage validation remain necessary.')

    page('9. Reproduction and source ledger')
    p('The original analysis uses Python 3.12.14 and NumPy 2.3.5; the extension additionally uses SciPy 1.16.3. Figure/PDF rebuilding uses the separately pinned report dependencies. Follow README.txt from the package root:')
    p('<font name="Code">python -m pip install -r requirements.txt<br/>python code/verify.py<br/>python code/experiments.py --reps 10000 --seed 20261017</font>')
    p('The CSV inputs are unchanged from commit f56792d02bd722ab16c1ed215b07c1187829a226. Input hashes and access notes are in data/sources.json; all delivered files are listed in MANIFEST.sha256. The clean-copy comparison is recorded separately in results/reproduction_check.json. Every table and scientific figure in this report is generated by code/build_report.py from those outputs.')
    h('Primary sources and project records')
    for ref in refs:
        p(f"<b>[{ref['id']}] {ref['title']}</b><br/><link href=\"{ref['url']}\" color=\"#176B87\">{ref['url']}</link>",'small')
    h('Attribution and disclosure')
    p('Original experimental work is credited to Brooks et al.; the public processed data and multiplicative-bias framework to McLaren and collaborators. Their source license is included. This report, implementation, derivations, simulations, and self-audit were prepared by Codex, an OpenAI GPT-6-based assistant; the exact serving revision is unknown. The human user selected the problem, prompted the work, and requested publication. No independent human statistical review occurred. Publication state is recorded separately in publication.json. Complete mathematical arguments, executable calculations, and evidence are supplied.','small')


def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#D8E0E5'));c.line(46,39,566,39)
    c.setFont('Body',7);c.setFillColor(colors.HexColor(GRAY))
    c.drawString(46,27,'BIOCAT-68 / BIO-300 | Calibration study v2 | 17 October 2026')
    c.drawRightString(566,27,str(doc.page))


if __name__=='__main__':
    figures();content()
    from report_extension import extension_content
    extension_content(globals())
    doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(612,792),rightMargin=46,leftMargin=46,
        topMargin=43,bottomMargin=52,title='Microbiome bias calibration: a reproducible finite-sample study',author='Codex',
        subject='BIO-300 scoped contribution; assumptions, proofs, simulations and public-data stress test')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (FIG/'equations.json').write_text(json.dumps(equations,indent=2)+'\n')
    print(ROOT/'report.pdf')
