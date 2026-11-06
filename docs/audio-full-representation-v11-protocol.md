# Prospective full-representation audio comparison

6 November 2026. Freeze this method, code/runtime/checkpoint digests and original
source roles publicly before project fitting or new neural forwards. This follows
the rejected v9/v10 results; they are consumed development, not fresh evidence.
This protocol does not authorize serving changes or new overall-confidence input.

## Question and single candidate

Can supervised use of all frozen representation coordinates improve useful
noise/background-speech conclusions at the same error limits, where PCA16 and
two added quality descriptors failed? The projection may discard low-variance
forensic information. This is a hypothesis, not an established failure cause.

Retain the same commercial-permission speech backbone, original1024-vector and
raw response. Every pretrained parameter stays frozen. Remove the component
parallel to the existing predictor head. Compute a weighted center and diagonal
standard deviations from fitting inputs only, with floor1e-6. Keep every1024
coordinate, multiplying normalized residuals by sqrt(16/1024). Their weighted
total variance is at most16, matching the prior PCA16 variance budget without
selecting its directions. Calibration/development never set normalization.

Use the identical v9 weighted smoothed binary-log-loss objective: equal truth
classes, generator families within class, canonical groups within class/family,
originals within group and assessed variants within original. Weights sum1.
Class-specific target smoothing uses only connected fitting-group counts.
Predictors are raw clipped logit,1024 residual coordinates and intercept.
The raw slope is nonnegative; ridge penalties are0.001 on raw slope,0.05 on each
residual coefficient, zero on intercept. Initialize raw slope1 and all other
coefficients0. L-BFGS-B, analytic gradient,ftol1e-12,gtol1e-9,maxiter1000, no
hyperparameter search or early stopping. Nonconvergence fails the attempt.

This is one fixed full-representation candidate. The exact fitted v9 PCA16
object remains the representation control; it is not refitted or retuned from
development. The unchanged current guarded service remains the adoption baseline.
Quality/context/source/generator IDs are not predictors. Metadata is used only
for declared weighting, grouping, source auditing and stratified reporting.

## Unchanged inputs and calibration

Reuse the exact4,248 corrected consumed source/features and their roles:1,926
fitting,873 calibration,1,449 development inputs,214/97/161 originals and
169/82/106 connected groups. All62 unassessed inputs remain unassessed; preserve
any failed input/probe, with an inconclusive verdict and complete denominator.
Verify all existing representation hashes, original raw head, source/PCM clocks,
commercial checkpoints and runtime before fitting or reusing any cache.

The original v9 receipt, projection/head, complete service journal and completed
v10 journal are pinned by SHA-256. The union supplies424 already audited filtered
probes. Reuse only exact per-input raw/representation/filtered-PCM matches. Any
newly required probe must decode the unchanged source, apply the same band view,
run the same frozen neural model and record complete hashes/values. Acoustic
quality values are not used. Do not change waveforms to improve scores.

Fit only fitting-role sources; save the fitted object before separate calibration.
Calibrate with the unchanged worst nonconformity per canonical group and truth
class, retaining missing variants. A paired genuine/cloned group may appear in
both truth-class pools within its one role. At least nine groups/class required.
Keep the original research0.8/0.2 points, alpha0.1 class guards and both original
raw and mapped logit-change limits ln2 at six-decimal probabilities. These are
internal comparison rules, not public confidence/accuracy promises.

## Development, auditing and decision

Compare every1,449 development input with the exact cached current-service result
and separately describe differences from the rejected PCA16 control. Preserve all
failures/abstentions. Report both false-call directions, correct AI/original
conclusions, AI without an AI conclusion, inconclusive rates and source-balanced
Brier/log loss/AUC. Include every generator, noise/music/speech condition and
background stratum. Use unchanged connected-group paired bootstrap, seed and
2,000 replicates; retain conditional uncertainty and original counts. Nine related
variants are not nine independent users or recordings.

Use all sixteen existing v9/v10 acceptance gates unchanged. Better proper scores
or fewer false alarms do not excuse lost useful conclusions. All gates must pass
before considering a separately frozen confirmation. No rule/point/feature/penalty
change after seeing outcomes. If any gate fails, reject without scoring reserved
confirmation. Record a failure honestly instead of seeking a more favorable gate.

Before any confirmation, reproduce the fitting-only coefficients/normalization
and all calibration ranks/decisions/statistical summaries. Re-run the unchanged
99-case native development plan and every newly computed probe, preserving
source/PCM/raw/feature bindings. Supplemental new-probe cases audit provenance,
not fresh accuracy. Publish safe numerical parameters/outcomes and rejection or
admission evidence for outside reproduction, excluding private/customer media,
provider IDs, transcripts, local paths and full pretrained checkpoints.

The1,080 reserved confirmation inputs/known identity roles remain fixed and
unscored unless development and all native/numerical audits pass. Their own
freeze and complete confirmation comparison precede any adoption decision.
Actual paid-flow, source evidence, PDF/JSON filtering and rollback tests precede
local enablement. No short-window result alone admits arbitrary-duration audio
or whole-video confidence. Image/video/writing improvement, real outside human
testing and eventual GCP release remain separate unfinished requirements.
