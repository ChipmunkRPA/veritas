# Audio calibration v3 — fresh multi-generator validation

**Status: frozen for offline inference at 2026-11-01 04:10:32 UTC.** The final
cohort, method and preprocessing are bound by the run lock before any fresh
inference. Fitting/calibration must finish and their exact coefficients, maxima,
thresholds and score prefix must be locked before the first v3 test score.
This document extends, rather than relaxes, the admission criteria
in `docs/independent-evaluation-protocol.md`. It is an offline experiment.

## Question and interpretation

Does a prospectively fixed audio method reduce the false-original errors seen
with strong background noise, while preserving useful authentic and AI
decisions and improving calibration against a fairly refitted old detector?
The consumed v2 study had two incorrect original conclusions among twenty AI
recordings at 5 dB noise; that result is motivation for development, not new
confirmatory evidence. The 160 v1 and 104 v2 originals are permanently excluded
from the fresh v3 fitting, calibration and test cohorts. Their derivatives and
identities cannot become fresh evidence by changing their file names or noise.

Development may inspect the old studies. The final v3 test may not select a
method, threshold, family, background, duration, source replacement or subgroup.
A result reported after such a change is exploratory and needs a new holdout.
Fresh relative to these experiments does not prove absence from pretrained
detector data; the candidate's training corpus is not disclosed.

## Cohort and fixed sample size

The fixed target is **420 originals and 2,520 derived condition inputs**:

| Split | Authentic originals | AI originals | Population |
| --- | ---: | ---: | --- |
| Fit | 20 | 20 | Fresh LibriSpeech; FLEURS-HS Google News/Standard voices |
| Calibration | 40 | 40 | Fresh LibriSpeech; FLEURS-HS Azure, Amazon and Google Neural2/WaveNet voices |
| Test | 150 | 150 | 50 fresh VCTK + 100 fresh LibriSpeech; 50 each DFADD PFlow, StyleTTS2 and NaturalSpeech2 |

FLEURS-HS aliases represent eight fit voice groups and 35 calibration voice
groups, not one independent voice per provider/model spelling. Merge Azure
Jenny/JennyMultilingual, Amazon standard/neural variants, and matching Google
Standard/Neural2/WaveNet voice letters where they may share an underlying identity. Retain
the alias table and evidence. DFADD generators are absent from the fit/calibration
cohorts. The 57 common fresh VCTK identities available across the selected
generator archives stay 57 joint voice groups when reused for authentic and
three generated populations, not two hundred independent speakers. The final manifest must state exact
attained counts and all aliases before freeze. No count shortfall can be hidden
by extra passages or derived mixtures.

This exceeds the required 100 authentic and 100 AI **original files**, uses two
authentic collections and three held-out generator families. Extra passages
from one voice remain extra files, not independent voices. Source metadata
availability may restrict the supported scope but cannot turn a smaller pilot
into the requested larger validation.

Fitting and calibration use different source, speaker/voice and passage groups.
The final test also excludes all fitting, calibration and old-study groups.
Generator and speaker IDs must be normalized across archives: generated variants
of a VCTK speaker share that speaker's group, including any authentic recording
of that speaker in the same experiment. Repeated source-text/prompt groups are
retained as groups even when the output comes from a different generator.

Use natural two-to-four-second speech prefixes without repetition, truncating
at four seconds. Match authentic durations to generated durations prospectively
using metadata, never detector output. Apply the same decoding, amplitude and
selection rules. Keep authentic collections represented within each relevant
duration stratum; do not repeat v2's eight-second stratum with only one authentic
test source. Record the actual class/source/channel/language/duration table. Where
possible, include authentic recordings from the same source-speaker/channel
population used to generate AI recordings, alongside an independent authentic
collection, and report the matched and cross-source panels separately. If a
population is confounded with class, state that explicitly and do not describe
the resulting discrimination as source-independent.

The attained test interval is **2.5–4 seconds**, with matching counts in each
class: 2.5 s × 5, 2.75 s × 2, 3 s × 5, 3.25 s × 9, 3.5 s × 8, 3.75 s × 17,
and 4 s × 104. Fitting/calibration still include 2-, 3- and 4-second prefixes.
Only seven test originals per class are shorter than three seconds, so that
stratum is descriptive and has inadequate support for a separate confidence
claim. Do not describe two-second inference as independently validated. A
duration restriction, unlike true SNR, can be observed by the service; any
admitted duration rule must be explicitly tied to adequately supported results.

Exclusions check canonical source IDs, publisher URLs, speaker/voice groups,
normalized passage strings or documented prompt groups, original byte hashes
and decoded PCM hashes against all earlier and new splits. Unknown transcripts
must never be described as exact passage checks. A conservative whole-corpus
text exclusion is acceptable only when primary publisher evidence establishes
that every selected prompt belongs to that corpus; retain the evidence and
label that exclusion as corpus-level. An undocumented index mapping across
generator archives is not proof of distinct texts. Missing identity, rights or
provenance required for the design stops preparation before inference.

The DFADD paper states that its 300 prompts are five-to-ten-word sentences from
LJ Speech. The complete 13,100-record LJ transcript corpus is retained for
exclusion. No complete LJ passage of at least five words matches or is contained
in the 264 consumed-study transcripts; nine earlier passages nevertheless share
generic five-word strings, and a three-word passage appears in one earlier text.
Those flags remain recorded. New fitting/calibration transcripts reject all LJ
five-word overlaps. The selected DFADD filenames have distinct prompt indices,
but their exact text mapping is unpublished: this is **not** a guarantee of
exact prior prompt independence or cross-generator text uniqueness. Report the
shared DFADD text pool as a dependence/provenance limitation. It must remain
unresolved in any release record unless verified mapping or a separately
adequate fresh-text confirmation supplies the missing evidence.

Retain the primary DFADD paper/revised archive declarations, FLEURS-HS publisher
metadata and CC BY 4.0 notice, VCTK attribution to Junichi Yamagishi, Christophe
Veaux and Kirsten MacDonald/Edinburgh CSTR, and LibriSpeech's CC BY 4.0 notice.
LJ Speech source texts are public domain. Dataset permission does not verify
pretrained-detector independence, and selected source audio is not bundled into
customer reports or redistributed as part of the application.

Select and replace sources deterministically from metadata and decoding checks
only. Log every rejection and replacement before scoring. No label correction
or replacement after observing detector scores may remain in confirmatory
results without separately preserving the original evaluation.

## Background challenge and preprocessing

The fixed challenge is clean speech, realistic non-speech noise at 0, 5 and
15 dB, and music at 5 and 15 dB: **six conditions per original**. This prioritizes
the observed noise failure and adds 0 dB noise while keeping inference bounded;
0 dB music is outside the evaluated scope.
Use actual licensed background recordings, including changing/nonstationary
sound, rather than treating white noise as a complete background evaluation.
All noise and music recordings must be new relative to the old experiments and
disjoint across v3 fitting, calibration and test. No alternative excerpt of the
same background work counts as an independent background group.

The fixed target test pool is eight distinct noise recordings and four music
works; fitting and calibration each use separate pools of six noise recordings
and two music works. Noise sources are individual MUSAN FreeSound public-domain
records; music sources are individual MUSAN RFM works under CC BY 3.0, retaining
Kevin MacLeod/Incompetech attribution and the publisher's per-file declaration.
Before freeze specify
exact attained background groups and an assignment matrix balanced
across authentic/AI classes and generator/source panels. Each background must
occur with both classes. Carry the same assigned background through that
original's three noise levels. Report both SNR and individual background
results; a single pooled noise metric can hide a failed background. Background
group reuse is dependence, not increased independent sample size.

Use a fixed speech interval without speech looping or padding, loudness matching
and a common clipping-safe attenuation across conditions. Record full-window
RMS SNR, background offset, sample bounds and hashes; SNR is a controlled mixing
parameter, not a service-observable feature. The service's production decoder
must produce the final signed-16-bit-derived mono 16 kHz PCM actually scored.
Pin the full-profile adapters, speech gates, window planner, aggregation,
weights and receipts. Record accepted and rejected windows with their original
sample intervals. A failed/no-speech input remains in operational denominators.

Both detectors run locally on CPU with four threads, using their already
installed weights. The bound is 5,040 original detector-input attempts across
2,520 inputs, plus at most 1,800 conditional filtered-view attempts on test
inputs; speech-gate abstentions reduce actual network calls. No external
inference API or customer billing submission is used. The final run lock pins
runtime/library/FFmpeg versions in addition to implementation and weight bytes.

## Development, fitting and calibration boundary

Method development uses consumed v1/v2 data only. Any proposed filter, quality
feature or guard must be computable from the uploaded waveform and pinned
production outputs, without true SNR, known class, source collection, generator
identity, filename or transcript. Filtering does not establish that removed
audio was fake; disagreement may justify an inconclusive result only.

Before fresh scoring, specify one candidate, its unguarded ablation and a fairly
refitted old-model baseline. Select neither the most confident view nor the
best test result. Preserve a single explicit probability definition for proper
scoring. If a method changes only abstention, its Brier and log loss are those
of its unchanged underlying probability; it cannot claim improved calibration
by deleting hard cases. Score additional transformed views only as predeclared
ablations or development diagnostics.

The current development proposal is an **original-clearance sensitivity veto**.
Apply a fourth-order zero-phase Butterworth bandpass at 100–3800 Hz to the entire
production-decoded float32 clip. Use precisely the original-view speech gate and
window sample intervals, with no second gate, normalization or requantization.
Aggregate the same candidate's window scores by the production arithmetic mean
and round both view aggregates to six decimal places. Clip raw probabilities
to [1e-6, 1-1e-6] before logits. If the absolute logit difference exceeds ln(2),
or the filtered score is unavailable, veto an otherwise-original decision.
AI decisions are not changed by this veto; neither direction receives a changed
probability. This addresses the observed false-original failure direction and
must report its cost to genuine original coverage. The fixed-threshold evaluation
on all 520 **consumed v2** inputs accepted this exact rule for fresh assessment:
false-original calls fell from two to zero, false-AI calls remained one, and
decisions fell from 476 to 453. At 5 dB noise, authentic decision coverage fell
from 92.31% to 80.77%, while correct AI calls stayed at 39. These are development
diagnostics, not independent performance estimates. No threshold sweep was used.
The method source hash is
`9f47f54d2386b637c025a9a1ef1acca447b406695e7f1efb7e585f11225e3b24`;
retain `development/method-selection.json` and its diagnostic hashes in the
run lock before fresh inference. The filtered view is a sensitivity probe, not
evidence that any frequency band was synthetic or a proven noise-removal method.

The scoring wrapper may skip the filtered view where it cannot affect a result:
fitting/calibration use only original scores, and test clips require a filtered
view only after the frozen mapped probability and class set would otherwise
return original. AI and inconclusive paths explicitly record that the probe is
not required, rather than reporting a probe failure. This execution optimization
does not change the frozen rule, probability or any decision. Pin its separate
source file and verify behavioral tests plus actual consumed-record CPU
equivalence before freeze; no fresh result may be used to validate the shortcut.

Fit the same nonnegative-slope Platt mapping used in v2 on the **fit split only**
for each detector, with class-balanced per-original weights and smoothing based
on distinct original files. Use the unrounded production internal mean for
probability fitting and mapping, as the v2 evaluator did; only the sensitivity
comparison uses six-decimal aggregate scores. Retain both representations so
this distinction is reproducible. Use the separate calibration split for identity-level
rejection maxima and operating thresholds. The v2 rank rule is retained: take
the worst class nonconformity across all an identity's source files and six
conditions (p(AI) for authentic; 1-p(AI) for generated). An unavailable condition
makes that identity's maximum infinite, not absent. For proposed class c use
`(1 + count(calibration maxima >= new nonconformity))/(identity count + 1)`;
include c only above .10, inclusive ties in the count. Fewer than nine calibration
groups in either class disables definitive decisions. A leaning decision needs
both p(AI) >= .80 or <= .20 and a singleton rank set agreeing with that class,
plus the predeclared candidate quality gate if selected. Missing probability
is inconclusive. The implementation permits a fixed 1e-12 arithmetic tolerance
at the original boundary `1-.80`; this is not a tunable decision margin.
The .10 set threshold is not the separate 5% FPR operating
point. Do not treat rank p-values as posterior confidence. The probability
interpretation is a 50:50 reference prior, not measured customer prevalence.

Freeze an explicit nominal 5% false-AI operating threshold from authentic
calibration identities only. For each identity take its maximum mapped p(AI)
across all conditions; any missing score makes that maximum infinite. With n
identities use the ascending order statistic at rank `ceil((n+1)*.95)`, and set
the calling threshold to the next representable float above that value. Thus
ties at the order statistic do not become AI calls. If the rank exceeds n,
the order statistic is infinite or is >= 1, record an unavailable threshold.
If the available identities cannot establish it, record
`not_established`; do not substitute a test-derived threshold. Its AI recall
evaluation is separate from the 80% leaning rule. Missing scores abstain but
remain in the operational rate denominators.

## Lock and test release

The protocol, final cohort/assignment manifest, exact selected method,
preprocessing code and weight receipts are hashed before any fresh inference.
Development/fitting and calibration records are append-only and identify their
stage. Before the first test result, freeze fitted coefficients, guard thresholds,
calibration identity maxima, threshold calibration source IDs and the exact
development score prefix. A test-start record includes every hash and prefix
length. Validate those locks again before every resumed test batch.

No test file may be scored through an exploratory script before its test-start
lock. Once read, its result is consumed. Retries are permitted only for logged
operational failures under identical code, weights and input; retain failed
attempts. A deterministic first-success rule prevents cherry-picking. Do not
expand the cohort after looking at results to obtain a narrower interval.

## Metrics and uncertainty

Report the baseline, unguarded candidate and guarded candidate side by side:

- Original, speaker/voice, text-group and background-group counts; assessed,
  no-speech, failed, timed-out and inconclusive input counts.
- Raw ranking, probability-only and guarded decision counts at p(AI) >= .80
  or <= .20, overall and by each condition, authentic collection, AI family,
  duration and predefined matched/cross-source panel.
- False AI / authentic inputs, false original / AI inputs, correct AI recall,
  correct-original recall, AI-call and original-call error risks, and authentic
  and AI decision coverage separately. Report both count numerator and denominator.
- Original/class/source-balanced Brier and log loss on each system's assessed
  population and on the same paired common-assessed population. State population
  weights. Guards must not remove cases from the proper-score comparison.
- Calibration-selected 5% operating-point FPR and AI recall with uncertainty.
  Thresholds remain frozen; test EER is descriptive only.
- Each individual background's errors in both directions and class coverage,
  with the small original/identity counts visible. Unknown metrics are null,
  not zero; omit ambiguous inherited fields such as guarded `*_80` summaries.

Use 2,000 paired stratified speaker/voice cluster bootstrap replicates with
seed **3101985**, carrying all that group's sources and conditions together.
Strata are joint VCTK identities and LibriSpeech identities. A VCTK draw carries
its authentic and all generated recordings together; do not independently
resample class or generator copies of the same speaker. Within each reported
slice assign equal original-source weight, shared across that original's rows,
then normalize the two class totals equally. Keep each original's weight fixed
before applying bootstrap multiplicities, then renormalize class totals.
Record and report undefined replicates.
Explicitly account for reused background groups with a second, paired crossed
speaker/background bootstrap (or an equally prespecified conservative cluster
analysis); retain both the conditional-on-background and crossed results. The
prespecified crossed procedure independently resamples noise and music groups
with replacement within their types, uses product speaker/background draw
multiplicities on each row, and gives clean rows a background multiplicity of
one. Apply precisely the same weights to all compared systems. Normalize within
class after weighting. This is a crossed-cluster sensitivity analysis, not
proof that the small background pool represents all acoustic environments.
Eight noise and four music works provide limited background-population support.
Even a crossed bootstrap does not justify arbitrary-environment performance;
claims are conditional on the fixed tested pool unless additional independent
background evidence establishes a broader scope.
Generator families are fixed tested populations, not a random sample of every
possible generator. Repeated text groups need a documented dependence analysis.

Add exact one-sided 95% upper bounds for independently grouped error events:
count a speaker/voice once if any relevant source/condition makes an incorrect
call. Identify when such groups are not independent because of shared prompts
or backgrounds. A zero-error bootstrap interval [0, 0] does not establish zero
risk; report its nonzero finite-sample upper bound and the effective group
count. All intervals condition on frozen fitting/calibration unless additional
uncertainty is explicitly calculated. They are not a population guarantee.

For the paired noninferiority check, also count harmful and beneficial discordant
speaker/voice identities separately: a harmful identity has a new false call
where the old system did not, and a beneficial identity has the reverse. If
neither kind occurs, a collapsed bootstrap interval cannot qualify the two-point
margin unless the exact one-sided 95% upper bound for zero harmful events among
the independent identity count is <= .02. This is a supplemental precision gate,
not a substituted confidence interval for the original-file rate difference.
Shared backgrounds/text assumptions still limit independence. With roughly
57 AI voices, zero discordance leaves this comparison unestablished; reporting
that limit accurately is part of validation rather than a reason to add test
records after results are seen.

## Prospective admission and coverage safeguards

Apply the existing criteria to a **predefined supported scope**, not a subgroup
discovered because it happened to pass. There is no automatic prohibition on a
successful well-supported method, and no automatic promotion after this run.
SNR and generator labels are experiment metadata, unavailable during service
use; they cannot silently route deployment eligibility. All six evaluated
conditions must pass for this method's presently proposed short-speech scope.
A future restriction based on observable waveform features must itself be
frozen and independently validated, rather than inferred from a favorable SNR
subset after this test.

1. Verify traceable labels/rights, disjoint cohorts, replayable hashes and
   production scoring parity; meet the original/group/source/family support
   requirements above. Insufficient support is unestablished, not a pass.
2. On the shared assessed population, both source-balanced Brier and log loss
   must improve over the fairly recalibrated baseline with paired 95% intervals
   supporting improvement. Report the unguarded ablation so an abstention-only
   change is not described as a probability-calibration improvement.
3. At the fixed 80% rule, the upper paired 95% bound on candidate-minus-baseline
   false-AI rate must be <= two percentage points in each supported condition.
   Define the formerly qualitative false-original noninferiority requirement
   symmetrically: its upper bound must also be <= two percentage points. Report
   exact grouped error bounds and decided-case risk as well; collapsed bootstrap
   intervals alone cannot establish safety.
4. A guard must not claim success merely by withholding all original calls.
   Require at least 75% decision coverage **within each class** for clean,
   noise-15 and music-15 inputs, and at least 50% within each class for noise-0,
   noise-5 and music-5. Relative to the same candidate with its rank guard but
   without a new quality veto, the additional absolute class-coverage loss must
   be <= ten percentage points in clean/mild conditions and <= 25 points in
   strong conditions. These are prospective usefulness floors, not statistical
   error guarantees. Report their paired intervals and correct-class recall
   alongside the point estimates. Every proposed admitted condition must meet
   them; a large loss is not disguised by reporting overall coverage alone.
5. At the separately calibrated nominal 5% FPR threshold, the authentic-file
   95% upper FPR bound must be <= 7.5% within the proposed scope, with paired
   evidence of AI recall improvement. An unsupported threshold or inadequate
   precision fails to establish eligibility.
6. Report each of the six predefined condition panels, including 0 dB noise,
   without selectively hiding failed conditions. These panels locate failures and cannot be used
   as serving eligibility rules requiring unavailable true SNR. The proposed
   scope is assessed on English 2.5-to-4-second speech
   prefixes from the tested populations, conditional on the fixed background
   pools, predominantly four-second intervals; the sparse shorter-duration
   stratum cannot inherit a confidence claim. Declare channels, generator and background populations. The production
   candidate adapter currently supports CPU only, so CPU is the matching
   candidate device; verify representative raw-score/gate/window parity against
   that actual adapter. A later device change needs separate parity checks.
   Do not extrapolate to arbitrary long recordings,
   other languages, partial edits, telephone channels or whole-video soundtracks.
7. Save an explicit versioned admission record with evidence. Never write a trial
   into `backend/data/fusion/` or `backend/app/engine/fusion/release/`, whose loader
   can auto-select it. Do not invent a human reviewer. Until an eligible scoped
   artefact is deliberately integrated and locally verified, candidate fusion
   weight remains zero.

Old-detector removal requires the separately predeclared candidate-versus-combined
comparison: rule out > .01 Brier increase and > .02 recall loss, no error increase
or important coverage loss. A study without that combination comparison cannot
justify removal. Preserve historical weights and calibration for replay.

## Independent audit and frozen preparation evidence

An independent pre-score audit checks rights, group separation, all exclusions,
counts and background balance; byte/PCM hashes; representative production decode
replays; and code/weight/runtime locks. Independent post-score calculations must
reconstruct probability maps, calibration maxima, decisions, operating thresholds,
point metrics and confidence intervals from raw records without importing the
evaluation harness's metric helpers. Audit exact test-start order and all retries.
Persist audits, failures and limitations under
`backend/data/independent-evaluation/audio-v3/audits/`.

Preparation completed without fresh inference. The independent audit verified
2,968 current byte hashes, all 420 source PCM hashes, separately decoded all
264 previous and 420 current full originals/prefixes, directly decoded all
2,520 final inputs through FFmpeg, and replayed 18 through the production
mapping helper. It verified class-duration matching, every background's class
balance, source/voice exclusions, all original/condition counts, and the entire
LJ text corpus against the stated conservative exclusion rules. Documented
unknown prompt mapping and limited background support remain limitations.

Pinned preparation evidence (SHA-256):

- Manifest logical digest:
  `6e5959c3a5dd58a6c5deb2c190756c54e4d1ec6f4f9e3b9e7f33a4194b6144b6`.
- Independent preparation audit:
  `ff6c16ff2a556c8df62f5368f999bb538ad547b02e962850871730dac7064be5`.
- Fixed method selection:
  `cc58124dc34e9feb61b46da410ef20432f382d6fd48f12cd655ca16223099c48`.
- Original production-adapter parity:
  `9c6bf9fa97f36f9e2ed8e867a046fad777345b65e807cabee8736b7c01b744cd`.
- Lazy/full execution equivalence:
  `ef9494e4bd399ff332b157742531825ded7078b13e8b5d57569dc4a0244643b9`.

The manifest's `preparation_protocol_sha256` refers to the earlier drafting
snapshot used during input generation. The run lock pins this final protocol
as the authoritative inference specification, all implementation/runtime/model
bytes and the evidence above. Recheck those pins on every resume. The later
calibration and test-start artefacts must be frozen before final-test access;
this preparation freeze cannot substitute for them.
