# Audio v8: broader fitting sources, independent calibration and fresh reservation

Research protocol, 6 November 2026. No serving or overall-confidence change.
This protocol precedes new v8 representation extraction and project-data fitting.
The previous v7 candidate failed fresh confirmation despite better average
scores: false AI35→38 and false original39→52 on1080 variants. Its result and
immutable code remain published. V8 is a new hypothesis, not a rescue threshold
or a declaration that source diversity will fix that failure.

## Sources and roles

Account for all540 previously consumed source originals:420 in the complete v3
manifest, plus120 from the failed v7 confirmation. Exclude61 shorter than four
seconds and three conservative provider-name collisions with reserved profiles.
The remaining476 are all consumed development material, including old test rows.
Keep known voices, source groups, exact/eight-word-opening text, original bytes,
full production PCM, actual four-second prefixes and conservative DFADD prompt
connections together. Assign roles before new extraction; never split a genuine
and cloned known voice between roles merely to balance counts.

| Role | Foreground originals | Known connected groups | Genuine / synthetic | Nine-condition inputs |
|---|---:|---:|---:|---:|
| Fit |219|174|124 /95|1971|
| Independent calibration |99|84|60 /39|891|
| Consumed development |158|106|76 /82|1422|
| Reserved fresh confirmation |120|120|60 /60|1080|

Eight synthetic families occur in fitting/calibration/development: Google,
Amazon, Azure, PflowTTS, StyleTTS2, NaturalSpeech2, ElevenLabs and Chatterbox.
Some family calibration strata are small; this does not support a family-specific
error guarantee. Identity groups are known metadata connections, not biometric
proof or evidence of absence from the pretrained backbone's training corpus.

Reserve confirmation before fitting:60 previously unused LibriSpeech train100
speakers,24 new provider profiles from LAION's ElevenLabs-v4 recordings,12
Kokoro voices, and24 unused Chatterbox prompt/dialogue IDs. Scripts and full/
prefix decoded audio have no known connection to the consumed source pool.
Publisher-labelled generation metadata supplies class labels; learned annotation
scores never supply labels, predictors or case-selection criteria. LAION text
is intended provider text with bracket delivery controls removed, not verified
spoken words or alignment.

Source declarations: LibriSpeech and LAION generated recordings CC-BY4.0;
selected Dictation Speech recordings/text CC-BY-SA4.0; Chatterbox publisher PLN
scenario MIT. Dataset permissions do not relicense generator weights or prove
speaker consent. No new pretrained generator/detector weights are needed here.
Pinned revisions, source hashes, extraction coordinates and notices are in the
public source capsule. Preserve applicable attribution/share-alike terms for
upstream/derived data; the project software license does not override them.

## Backgrounds and exact audio

Use separate original recordings in each of four roles:8 public-domain MUSAN
FreeSound noise files and4 CC-BY3.0 named MUSAN music works per role. Genuine
background speakers are4 each for fit/calibration/development and6 for fresh
confirmation, separate from all foreground speakers and known scripts.
Noise contributor/session identities are unpublished. Music works are distinct
but share a composer across roles; do not claim independent creators.

The initial cached noise inventory had only18 unused recordings at least eight
seconds long. Preserve that preparation failure and its exact code. Acquire the
publisher-checksummed original noise Parquet and extend the inventory by its
fixed row group1, restricted to the same public-domain subset. This happens
before model inference, without changing duration/energy eligibility or choosing
by outcomes. The additional100 originals include52 at least eight seconds long.

Each foreground uses its first64000 production-decoded samples (four seconds,
16kHz mono). Use clean; noise0/5/15dB; music5/15dB; genuine background speech
0/5/15dB SNR. Normalize foreground RMS to0.1, add the selected original background
interval, and apply one common peak gain across a source's nine conditions.
Within each role and truth class, hash-order sources and round-robin the fixed
background pools. Offset is a fixed source/kind hash modulo available samples.
An invalid/silent chosen interval stops preparation; do not search offsets.
Bind source interval, background interval, intended/measured SNR, common gain,
float payload, WAV bytes and production-decoded PCM hashes. These correlated
variants are not independent recordings.

## Fixed learner and service comparison

- Keep the current commercially permitted Pella V2 speech backbone entirely
  frozen. Use the same original-input speech activity/window policy as the full
  service. Unassessed originals remain unassessed; a missing check is not original.
- Extract the original1024-dimensional frozen representation on CPU float32.
  Reconstruct the original sigmoid response from its head as an arithmetic check.
- Fit only the219 assigned originals and their assessed variants. Equalize truth
  classes, then generator families within class, connected groups within family/
  class, sources within group/family/class, and assessed variants within source.
- Remove the fixed original-head direction, fit weighted PCA16 on fitting rows
  only, and floor PCA coordinate scales at1e-6. Predictors are the original clipped
  logit,16 residual coordinates and intercept; no corpus/voice/label metadata.
- Fit one smoothed logistic objective: ridge0.001 on the original slope,0.05 on
  residuals, zero on intercept, original slope nonnegative. Targets use connected
  truth-class fitting-group counts. L-BFGS-B fixed convergence and1000-iteration
  maximum; no sweep, outcome-based feature choice or neural finetuning.
- Freeze fitted parameters before reading independent calibration responses for
  guard fitting. Use worst nonconformity per connected group **and truth class**;
  a known genuine/cloned voice may appear in both class pools within its one role.
  Missing calibration responses remain conservatively unbounded. Minimum9
  calibration groups/class; internal class-set alpha0.1.
- Keep internal0.8/0.2 directional diagnostics. These are research decision points,
  not a public confidence/accuracy claim. For an original proposal, both the
  candidate calibrated-logit change and the original raw-logit change under the
  fixed300–3400Hz view must be≤ln2 using six-decimal probabilities. Missing or
  failed probes withhold original. This prevents a flatter map alone passing the
  original raw-response stability check.
- Comparator is the actual current `local-speech-v3.1` service calibration and
  its original-input stability guard, not a refitted or unguarded substitute.
  Share a computed probe when either arm requires it; keep both decisions.

## Frozen advancement and fresh-confirmation requirements

Evaluate every1422 assigned development input and, only after a pass and separate
head/input/code lock, every1080 reserved confirmation input. Use the same16 gates
at both stages. No development pass enables serving by itself.

1. Balanced, source-weighted Brier improvement at least0.005 and log-loss
   improvement at least0.01; paired95% Brier interval entirely below zero.
2. Lower observed Brier separately for noise0, music5 and speech0/5/15.
3. At most one additional false AI and at most one additional false original,
   both overall and separately in every condition. Each represented generator
   permits at most one additional false-original call.
4. At most two fewer correct calls in either clean truth class and at most two
   fewer correct AI calls for each represented generator.
5. At least five more correct conclusions for noise0, five for music5, and five
   across background-speech conditions. In each speech condition, at most two
   fewer correct calls in either truth class. Extra abstention alone cannot pass.
6. Identical assessed input identities, all planned source/condition outcomes,
   zero extraction failures and zero required-probe failures.

Report false AI/genuine denominator, false original/synthetic denominator, AI
without an AI conclusion (including incorrect-original and inconclusive), each
class's correct calls, unassessed and execution failures, and inconclusive rate.
Include per-condition, generator, dataset and background-source breakdowns.
Proper scores use common assessed responses even when a guard abstains.
Bootstrap2000 connected-group resamples, stratified by represented truth/family
connections, fixed seed461109. Intervals condition on fixed fit/calibration and
selected backgrounds, excluding their population uncertainty. These are not
per-upload error probabilities. A failed gate rejects advancement; no post-hoc
threshold relaxation, dropped family or replacement source.

## Reproduction and release boundaries

Run the source-planning/acquisition/materialization helpers in `backend/scripts`
with fixed originals and the public capsule's hashes. Private source manifests
contain upstream text and local paths; publish only an explicit safe metadata
allowlist. Customer uploads are never evaluation sources.

The source capsule is `docs/evaluation-panels/audio-v8-source-plan.json.gz`.
Its source, voice and text connection identifiers are hashed; upstream audio must
be acquired separately under its own terms. A reviewer can check every published
role, connection, condition, background interval and implementation pin without
installing a model or signing in:

```sh
python3 backend/scripts/review_audio_multisource_plan_v8.py \
  docs/evaluation-panels/audio-v8-source-plan.json.gz /tmp/audio-v8-plan-review.json \
  --project .
```

This command checks metadata and source-code consistency only. It does not replay
source audio, neural responses or the eventual fitting experiment.

The development runner stages are `freeze`, `extract`, `train`, `compare`, and
`review`. Its development-only input validation refuses fresh-confirmation rows.
Freeze/publish the protocol, source/role references, exact input hashes and code
before extraction/fitting. Native decoder/input/probe replay and independent
numerical reconstruction remain required alongside a numerical pass. Only after
fresh validation and the paid analysis/evidence/export/rollback integration checks
could a scoped local adoption be considered. GCP, payment, authentication, existing
model defaults and overall confidence remain unchanged in this experiment.

Published methods and a reproducible capsule enable outside review. They do not
constitute participation by an outside reviewer; no external reviewer has yet
validated this experiment, and no outreach is sent without authorization.
