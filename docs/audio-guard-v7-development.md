# Source-separated speech decision layer and uncertainty guard, v7

Research only; no serving, billing, overall confidence or checkpoint change.
V4/V5/V6 remain rejected under their original gates. This is a new experiment
with separate fitting and class-set calibration, compared against the actual
guarded service rather than a refitted unguarded comparator. Freeze code and
these rules before extracting new representations or fitting parameters.

## Data and independence

Keep all original V3 fitting40 and calibration80 sources, all six conditions:
720 new feature inputs. Keep the complete V4–V6 consumed development panel:
120 source files,720 inputs, including its30 unassessed variants. Do not score
unused fresh-generator material. The three roles have28/75/60 connected source
groups respectively; fitting has20 genuine speakers and eight synthetic voice
groups, calibration40/35 class groups, development60/20 class groups. Class
groups can share an evaluation voice across genuine/generated speech and must
remain connected for uncertainty resampling.

Connect used sources by known voice/cluster, source-group identifier, normalized
known transcript and conservative DFADD prompt index. Refuse a cross-role
connection, duplicate source/input identity, incomplete condition set or changed
source/PCM hash. Background source hashes must be disjoint across roles too.
Preserve all sources; do not silently drop a problematic group or unassessed
input. Existing source/weight permissions apply. No new assets or packages.

Fitting/calibration waveforms are2–4 seconds; development2.5–4 seconds. The
existing short-recording, one-complete-window scope remains required for an
operational development conclusion. Source/channel/duration and generator
confounding remain possible. Exact DFADD transcripts and pretraining overlap
remain unknown. These are consumed-source development results, not fresh or
outside validation, and the720 variants are not independent source recordings.

## Fixed learner and independent class-set calibration

Use the unchanged, pinned commercial-permission speech detector and its1024
dimensional head-input representation. Fit/cal representations are newly run,
requiring original raw-response and linear-head reconstruction parity. Reuse
only the immutable, previously audited V6 development representations, verifying
identities, native PCM, embedding digests, weights and original responses.
Do not use generator, transcript, true SNR, condition, clean counterpart or
source identity as predictors. Original unassessed windows remain unassessed.

Fit the V6 residual projection/decision objective on FIT ONLY: source-balanced
head-orthogonal PCA16, minimum singular scale1e-6, original clipped logit plus
sixteen residual coordinates/intercept, nonnegative original-logit coefficient,
ridge .001/.05, no intercept penalty, source/class weights, source-count target
smoothing and balanced reference prior. No parameter sweep or alternative arm.
The resulting sigmoid is this learner's reference estimate; it never enters
the old detector's probability mapping. Calibration rows do not fit PCA,
normalization or coefficients. Record fit-only coefficients before scoring
calibration. This new role scheme is not a promotion of V6 cross-validation.

On CAL ONLY, obtain the new learner's probabilities and record worst
nonconformity per connected group/class over all sources/conditions; an
unassessed score makes that group maximum infinity (serialized null).
At least nine groups per class are required. Use the existing class-set rank
formula with alpha .1 and inclusive comparisons; independently fitted guards
must select only the corresponding class for an internal tendency. The fixed
research directional points remain .8/.2. They are internal diagnostics,
not a public accuracy or threshold claim. No finite-sample coverage guarantee
is asserted across these source/generator shifts.

An original tendency also requires a same-recording stability probe, with no
label-dependent triggering: fourth-order100–3800Hz Butterworth SOS forward/
backward filtering, float32, exact sample count and original complete window.
Capture the same frozen representation on this view and apply the SAME new
head. Compare clipped logits of new original/probe probabilities rounded to
six decimals, using the fixed ln2 bound. Missing/nonfinite/unstable probes
withhold original; they cannot create an AI verdict or alter the probability
used by proper scores. Calibration uses original probabilities only. The
unmodified waveform is the evidence source; filtering is a sensitivity test,
not recovered speech. No model-based claim is made from background tags.

## Matched baseline, gates and reporting

Before new inference, require exact current-bundle probability/decision replay
for all720 development inputs, including its voice guard and original-only
stability veto. Preserve the bundle/version and every saved baseline outcome.
Baseline neural responses were previously audited, not newly rerun here.

Compare new guarded outcomes with this unchanged serving reference on all720
inputs, separately for six conditions, genuine/generated classes, joint voices,
generator, source dataset and background. Proper scores use common assessed
inputs even where a guard withholds a tendency. Retain both error directions,
inconclusive/unassessed rates, conditional source-group bootstrap intervals,
all failures, new probe costs and source bindings. Use the existing2000
conditional connected-group draws/seed461109; no source counts inflated by
variants or independent-review claims from local scripts.

Advance only if all fixed gates pass: Brier improvement>=.005, log-loss
improvement>=.01, paired conditional Brier interval wholly below0; improved
Brier in noise0, music5 and the joint-voice panel; at most ONE additional
false-AI and ONE false-original call total and per condition; at most two
lost clean correct calls per class; at least five additional correct calls
in EACH noise0 and music5; identical assessed inputs, all planned cases and
zero new extraction/probe failures. Do not relax gates or select a cutoff
after results. A failure preserves the current service and unused confirmation.

A passing development result permits only a separately frozen fresh evaluation
with new generators/voices/background sources, including background speech.
Default/overall-confidence changes require fresh evidence, own calibration
artifacts, source-linked UI/PDF/JSON and complete paid-flow/rollback verification.
Outside human validation remains outstanding. GCP remains last.
