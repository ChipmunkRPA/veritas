# Audio-aware mouth evidence: sensitivity gain, replacement rejected

The audio-aware candidate located **31/32 known mouth edits**, versus **26/32**
for the visual-only v12 control, but ordinary flags doubled from **12/56 to
24/56**. It fails the unchanged total/per-condition false-flag rules and the
additional comparison against that control. **Do not adopt it, change customer
confidence or describe its margins as AI probabilities.**

All 272 planned audio tracks and 6,373 windows completed. A separate full audit
re-decodes every audio track and reproduces all new feature vectors, source
positions and diagnostic outcomes. These are correlated, consumed development
clips from 28 actor roles, not fresh people, customer accuracy or outside review.
The original v13 partial attempt and input failure remain retained separately.

## Complete matched comparison

Both newer methods preserve retained-v8 absolute evidence first, then use their
own highest-margin window. All 88 diagnostic clips remain assessed: 32 authored
AI edits from two generator families and 56 ordinary controls on eight consumed
actors. A useful location retains the original temporal IoU >= .2 definition.
Spatial overlap with the known mask is reported separately and includes misses.

| Method | Located edits /32 | AI clips without a flag /32 | Ordinary flags /56 | Mean time IoU | Mean spatial IoU |
|---|---:|---:|---:|---:|---:|
| Retained v8 research reference |14|18|0|.302009|.151177|
| Visual-only v12 control |26|6|12|.532484|.239908|
| Audio-aware candidate |31|1|24|.647610|.266661|

The candidate improves first-family locations from 10/16 to 15/16. Both methods
locate all 16 second-family edits. No old useful location is lost, and the
candidate passes the other original location/coverage requirements. Those gains
do not override its three failed specificity requirements.

| Ordinary condition | Clips | Visual-only flags | Audio-aware flags |
|---|---:|---:|---:|
| Unedited original |8|3|5|
| Ordinary constant audio delay |8|3|6|
| Local blur, sigma 2 |8|0|1|
| Local blur, sigma 4 |8|0|0|
| Local lighting |8|1|3|
| Portrait resampling |8|2|4|
| Speech-renderer matched resampling |8|3|5|

Sound/mouth correspondence is therefore not manipulation-specific here. In
particular, an ordinary timing offset or an original recording can trigger it.
This comparison does not measure achieved lip-sync error, authenticate AI origin
or establish that the flagged pixels caused the response. No successful family
or point is selected after observing these outcomes. No fresh reserve was used.
The [complete aggregate](evaluation-panels/mouth-audio-context-v14-result.json)
retains both standalone and preserve-v8 outcomes, every family/condition, gates,
source commitments and resource profiles.

## Fixed measurements, learning and source separation

The [v14 correction protocol](mouth-audio-context-v14-protocol.md) and
[input/code freeze](evaluation-panels/mouth-audio-context-v14-freeze.json) were
published and verified as
[fddd024](https://github.com/ChipmunkRPA/veritas/commit/fddd0241b724985b20d750de2c654d9f94f45500)
before new extraction. The underlying [v13 protocol](mouth-audio-context-v13-protocol.md)
defines the unchanged mathematical experiment.

Twenty new features compare actual source log-RMS audio with measured lip
aperture at -8/-4/0/4/8 video-frame offsets. Middle and whole-support groups are
dependent. Every lag uses the same valid anchors, with no wrap or padding.
Flat traces retain an explicit unusable-correlation fraction. The candidate
adds these features to 49 previously audited visual measurements. No source
identity, condition, true mask, interval or paired original enters inference.

Keep sixteen fitting, four ordinary-point and eight diagnostic actors disjoint.
Both small tree learners use identical fixed v12 hyperparameters and weights.
The two complete heads/points were saved before diagnostic audio processing.
Their internal raw-margin points are not public thresholds or calibrated
authorship confidence. The visual-only refit, point, all diagnostic margins and
outcomes match its published v12 result exactly.

Complete native-estimator/export correspondence passes on all 6,373 windows
for each arm, maximum difference zero. These native estimator reconstructions
fit only the original fitting role and must exactly reproduce the previously
sealed heads; they do not refit from diagnostic responses. Four small head fits
include those two reconstructions. No new pretrained weight was acquired or
visual neural forward executed. Existing CREMA-D ODbL/DbCL attribution and
separate commercial component permissions remain applicable.

## Full audit and retained failed attempt

The separate process verified:

- All 272 original file hashes, complete signed16 audio decodes and
  **10,841,600 PCM samples** against the original input commitments.
- **16,940 native frame clocks/dimensions**, complete inherited geometry,
  independent scalar log energies and aperture calculations.
- All **6,373 new vectors /127,460 values**, lag anchors, raw tree responses,
  genuine-only points, evidence selections, source-mask/time overlaps, outcomes,
  conditions and advancement gates.

The visual pixels, geometry models and prior visual neural responses were
audited previously and are inherited, not newly re-inferred. Polygon rasterization
uses the same OpenCV primitive, with separately reconstructed mask intersections.
The audit trains no head and performs no neural forward. It is local verification,
not an outside person's review or independent truth certification.

The original v13 attempt completed 160 fitting and 18 point records, then
rejected an intact 50-frame control because its envelope guard incorrectly
required 51 frames. No head or diagnostic response had been produced. Its failed
profile, journals and code remain unchanged. The separate correction accepts
complete shorter sample blocks, rejects out-of-source centers and distinguishes
extraction/training commands and completion files. All original **178 persisted
records reproduce exactly** before training. Earlier broad source-preflight and
dispatch problems also remain explicit; no old failed guard is claimed passed.

| Corrected phase | Seconds | Sampled RSS lower bound, bytes | Sampling errors | Exit / guard |
|---|---:|---:|---:|---|
| Fitting-source audio |73.357|1,078,214,656|1|0 / none|
| Ordinary-point audio |16.544|1,077,772,288|0|0 / none|
| Both head/point fits |9.752|1,065,500,672|0|0 / none|
| Diagnostic audio |46.993|1,076,903,936|2|0 / none|
| Complete native/export scoring |10.762|1,097,564,160|1|0 / none|
| Full separate audio/outcome audit |101.737|1,084,784,640|2|0 / none|

All phases ran sequentially under 900-second/6-GiB guards and minimum available
RAM/disk checks. Sampling errors remain recorded; RSS is a sampled lower bound.
The earlier failed attempt is additional work, not included in the corrected
phase durations. No failure or incomplete case was silently dropped or restarted.

## Public review and next work

All **20 authored local tests pass**, including lag sign, exact sample blocks,
short controls, degenerate traces, no wrap, complete support, point/head binding,
fit-role isolation and invalid-tree refusal. The public NumPy-only review needs
no media, model weights, Torch or scikit-learn:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_mouth_audio_public_v14 /tmp/new-mouth-math-review.json
```

It checks five known signed lags, fifty source-style sample blocks, flat traces
and no wrap against the separately written scalar equations. The local maximum
difference is 2.220446049250313e-16 within the frozen 1e-12 review tolerance.
The [actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42488459238)
also passes, with the same maximum difference. Its
[decoded-log-bound receipt](evaluation-panels/mouth-audio-context-v14-linux-math-review.json)
records the actual source/workflow revisions and job. It is authored arithmetic,
not a replay of private audio, learned-head fitting or detection quality.

Raw recordings, new audio traces, private fitted heads and per-case journals
remain private. Outside human reviewers remain zero. An authoritative read of
open GitHub issues and pull requests found no incoming review to address; that
does not establish absence of feedback through other channels. No outreach was
sent or independent participation claimed.

Further work needs manipulation-specific evidence with stronger ordinary
controls and source diversity, not another post-hoc scalar point. Clean studio
speech does not qualify noisy mixtures, music or overlapping speakers. Fresh
actors/new sentences, independent human evaluation and full customer evidence,
PDF/JSON/prepaid-flow tests remain required before any replacement. Serving
models, voice calibration, billing, reports and GCP resources are unchanged.
