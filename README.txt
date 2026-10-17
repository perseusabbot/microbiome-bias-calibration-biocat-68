MICROBIOME BIAS CALIBRATION - REPRODUCIBILITY PACKAGE
BIOCAT-68 (displayed as BIO-300), version 2, prepared 2026-10-17

Read report.pdf first. This is a scoped methodological study, not a full
solution to BIO-300 and not a claim of a previously unknown method.

Scientific reproduction (Python 3.12.14, NumPy 2.3.5, SciPy 1.16.3):
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python code/verify.py
  python code/experiments.py --reps 10000 --seed 20261017
  python code/verify_extension.py
  python code/extension.py

The calculations are offline: all empirical inputs are bundled at a pinned
source revision. No API keys, raw sequence processing, or external accounts
are needed. Expect seconds to a few minutes, depending on hardware.
The run overwrites the scientific files in results/ with regenerated outputs.
Other seeds or repetition counts are permitted but will change the results.
The report builder intentionally requires the reference seed and 10,000
replicates; restore that run before rebuilding the fixed report narrative.
NumPy RNG reproducibility requires the pinned version and PCG64 specified in
results/run_environment.json. Cross-platform floating-point differences may
prevent byte identity even when numerical results agree.

Optional figure and PDF regeneration:
  python -m pip install -r requirements-report.txt
  python code/build_report.py
The app-specific artifact marker used in the original authoring environment
is not required for reproduction outside the app.

File map:
  report.pdf                    Updated 16-page report: findings, proofs, limits
  github-submission.txt         Complete Markdown issue body for BIOCAT-68
  publication.json              Actual GitHub publication state and URLs
  PUBLISH.txt                   Publication record and submission information
  code/extended_model.py        Contaminated multinomial and DM model
  code/extension.py             Diagnosis, grouped validation, bootstrap, simulations
  code/verify_extension.py       8 additional verification methods
  code/report_extension.py       Source of the 6 new report sections
  data/extension_plan.json       Exploratory extension plan, fixed before fitting
  results/extension/             All extension results, fits, and generated counts
  code/calibration.py           Confidence intervals and L1/drift certificates
  code/verify.py                12 verification methods and exact finite checks
  code/experiments.py           All simulation and empirical analyses
  code/build_report.py          Scientific figures and full PDF source
  data/experiment_plan.json     Fixed scenarios and empirical selection rule
  data/brooks2015-*.csv          Unmodified public empirical inputs
  data/sources.json             URLs, retrieval date, source revision, scope
  data/ATTRIBUTION.txt           Credits and redistribution information
  results/simulation_summary.csv       One row per simulation scenario
  results/simulation_replicates.csv    Every synthetic count pair and metrics
  results/empirical_samples.csv        Held-out sample-level diagnostics
  results/empirical_intervals.csv      Every empirical interval and estimate
  results/verification.json            Exact checks and numerical audit results
  results/reproduction_check.json      Clean-copy reproduction comparison
  MANIFEST.sha256                      Checksums of delivered files

Definitions:
- Coverage means all named taxa's intervals contain their true proportions
  simultaneously, for ONE target sample, at nominal level 95%.
- Empirical 'cover_nominal' compares with nominal prepared mixtures. This is
  a model stress test, not estimated repeated-sampling confidence coverage.
- L1 error is sum(abs(estimate - truth)), between 0 and 2. Total variation
  distance is half the L1 error. Interval widths are in abundance units [0,1].
- The 'frozen' baseline deliberately treats a smoothed estimated bias as known.
  It is not an implementation or evaluation of metacal, fido, or tinyvamp.
- The drift adjustment uses an externally supplied bound on log bias drift.
  In simulations that bound is known from the generator, not inferred.

Using the core functions:
  import sys
  sys.path.insert(0, 'code')
  from calibration import calibration_intervals, point_estimate, l1_radius
  y = [100, 50, 25]
  c = [60, 80, 100]
  r = [1/3, 1/3, 1/3]  # known composition of the calibration community
  lo, hi = calibration_intervals(y, c, r, alpha=0.05)
  point = point_estimate(y, c, r)
  certificate = l1_radius(point, lo, hi)

IMPORTANT CONTRACTS FOR THE ORIGINAL FINITE-SAMPLE MODEL:
All target taxa must be present in the known calibration composition.
A zero observed calibration count is permitted and may make bounds vacuous.
A taxon structurally absent from the calibration mixture is not permitted.
The model assumes multinomial read counts and a shared positive efficiency
vector; it does not cover unknown contamination, overdispersion, arbitrary
matrix-specific drift, uncertain nominal compositions, or absolute biomass.
The pseudocount 0.5 affects only the descriptive point estimate. Confidence
intervals use raw counts. No finite-sample unbiasedness is claimed.
L1 subset enumeration is restricted to at most 12 taxa; confidence intervals
are mathematically defined more generally. No scalability benchmark is made.

Checksums:
  shasum -a 256 -c MANIFEST.sha256
A mismatch after deliberately rerunning with changed settings is expected.

The scientific code was written for this study by Codex. No external human
statistical review, theorem prover, wet-lab experiment, or novel-discovery
certification was performed. All source publications retain their authorship.

VERSION 2 FINDINGS AND CONTRACTS:
The old model missed nominally absent taxa due to stray reads in 71/78
samples; 43/78 also missed nominally present taxa. The original failure
breakdown and dependent pairwise log-ratio diagnostics are fully retained.

Grouped five-fold validation keeps identical nominal support sets together
and includes both full-support calibration controls in each training fold.
Every held-out composition is inferred without using its nominal support.
Containment of all seven nominal proportions: contamination-only 11/78;
contamination plus Dirichlet-multinomial variation 74/78; a parametric
training-bootstrap envelope 77/78. These are exploratory stress-test
results, not independent estimates of repeated-sampling confidence coverage.

The new conditional interval theorem requires KNOWN efficiencies,
contamination fraction/profile, and DM concentration. Plugging in estimates
and taking a finite bootstrap envelope does not inherit that theorem.
Intervals may reject the model via an empty set, represented by NaN
endpoints. Such cases count as failures and are excluded from mean width.
The point estimate uses nonnegative contamination subtraction and
normalization; it is not asserted to be a likelihood MLE or unbiased.

Default extension run: seed 20261018, 100 bootstrap fits per fold,
10,000 known-parameter targets per fixed composition, and 300 complete
training refits per selected composition. There are 30,000 unique target
draws evaluated by two procedures, 500 bootstrap training fits, and
600 repeated-training fits. All resampled counts and fitted parameters are
stored. Do not count the two procedures as independent synthetic samples.

All 14 extension scientific output files reproduced byte-for-byte in a
clean directory on the same pinned environment. The 7-file original
reproduction audit is retained. Use results/extension/reproduction_check.json
and results/reproduction_check.json to inspect the actual hash comparisons.

GitHub repository: https://github.com/perseusabbot/microbiome-bias-calibration-biocat-68
See publication.json for the current issue-submission state.
The issue draft includes user-confirmed submitter declarations,
complete methods/proofs, source references, and truthful contributor/model
disclosure. The user confirmed all declarations and authorized public-repository access.
The intended new-work license is CC BY 4.0 (text) / Apache-2.0 (software),
confirmed by the submitter. Upstream data retain their MIT
notice in data/LICENSE.md. No claim of novel methodology or full solution
to the original transferable calibration problem is made.
