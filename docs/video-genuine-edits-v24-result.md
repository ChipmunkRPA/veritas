# Ordinary video edits expose an authorship limit

The face appearance cue located changes in 22 of 40 ordinary edited clips.
All 22 flagged spans overlapped the known insertion, with best interval IoU
at least 0.3. That is useful evidence of an appearance transition. These
clips introduced no new AI generation: interpreting those same alerts as AI
authorship would be unsupported. Keep this cue outside customer AI confidence.

## What was tested

The [frozen protocol](video-genuine-edits-v24-protocol.md) used all 40
publisher-original sources from the already consumed confirmation role. They
were paired into 20 connected components before detector execution. In both
directions, donor frames 48–95 replaced the target's same frames, producing
40 new clips with an ordinary insertion from 2 to 4 seconds. Forty original
baselines were retained. No response selected sources, replacements, positions,
parameters or evaluation points.

The new clips used lossless RGB encoding at the verified native 24 fps and
640 × 480 geometry. All 5,760 decoded frames matched the expected source pixels
exactly. The 120 private PNG anchors at frames 36, 72 and 108 matched the source
RGB commitments and timestamps. They represent before, during and after at
1.5, 3 and 4.5 seconds. Their existence does not establish AI authorship.

Source authenticity is a publisher label, not independently verified truth.
The construction adds no AI generation, but does not establish the upstream
history of the source material. Different sources may show the same person.
This is a new edit construction on consumed sources, not fresh people,
generators, a calibration set, or a full application test.

## Fixed reference results

Every denominator remains 40, including inconclusive cases. The five v23
research artifacts and internal evaluation points were frozen before this
run. None was fitted or selected using these ordinary-edit outcomes.

| Reference method | Original proxy flags /40 | Original inconclusive /40 | Ordinary-edit proxy flags /40 | Ordinary-edit inconclusive /40 |
|---|---:|---:|---:|---:|
| Direct appearance calibration | 0 | 1 | 22 | 18 |
| Positive appearance calibration | 0 | 1 | 22 | 18 |
| Family maximum | 0 | 2 | 7 | 19 |
| Same maximum repeated with matched penalty | 0 | 2 | 7 | 19 |
| Preserved per-view maxima | 1 | 2 | 21 | 19 |
| Appearance calibration at mapped raw inspection point | 0 | 1 | 22 | 18 |

These are fixed diagnostic AI-proxy flags, **not customer confidence or service
false-positive rates**. All methods flag ordinary edits in this unvalidated
context. Lower flag counts alongside high inconclusive counts do not establish
a better detector. The direct, positive and mapped appearance references each
flagged 11 of 20 ordinary-edit components; the per-view combination also flagged
11, and both family maximum controls flagged 6.

The raw inspection cues were also retained separately:

| Cue | Original alerts /40 | Original inconclusive /40 | Ordinary-edit alerts /40 | Ordinary-edit inconclusive /40 |
|---|---:|---:|---:|---:|
| Face crop, any sampled response above existing raw zero point | 35 | 2 | 36 | 3 |
| Face appearance transition | 0 | 1 | 22 | 18 |
| Face feature transition | 0 | 1 | 0 | 18 |

These raw points are inspection references, not calibrated file probabilities.
In particular, “no feature-transition alert” cannot be turned into evidence
that the file is original. Missing geometric or face context stays explicit.
The 22 appearance spans correspond to ordinary source substitutions, not
correctly localized deepfakes or identified edited pixels.

## Local verification and consequences

The native run completed all 40 planned clips and two predeclared original
parity clips. The latter matched their cached responses and availability
exactly, with maximum raw difference 0. Three first eligible new model inputs
passed the frozen repeat checks. Recorded inputs totaled 1,428 face detection,
491 crop classification, 1,393 appearance and 1,393 patch feature inputs, plus
three repeat inputs. These are input counts, not counts of batched forward calls.

An additional read-only audit checked source pairing commitments, all decoded
RGB commitments, every saved anchor, every measured input, units, availability,
original parity, frozen reference scores, internal flags and aggregate counts.
It performed no model inference or fitting. The source audit took about 184
seconds and native scoring about 276 seconds. Both exited successfully within
their declared guards. Sampled process-tree RSS lower bounds peaked at about
1.32 and 2.42 GiB; these are not complete accelerator-memory measurements.
Transient process sampling errors remain recorded in the aggregate receipt.

The selected 105 local tests passed, including construction, adverse source
and counter inputs, appearance evidence binding, inspection-only fusion, and
shared finding interpretation. This verifies the selected code paths; it is
not a new end-to-end customer analysis or PDF rendering test.

The app already keeps appearance transitions inspection-only, with zero fusion
weight and no AI probability. This result supports retaining that separation.
No serving model, calibration, billing rule, public threshold or customer
verdict was changed. A future AI-authorship calibration needs ordinary edits
in its development controls and independent confirmation with both genuine
edits and AI manipulations. These 20 consumed components must not become a
rescue holdout or justify fitting a new default.

Across modalities, a check must retain its actual target: an appearance change,
background audio discontinuity, unusual image region or writing-style pattern
does not by itself establish AI authorship. Separate calibration helps only
when its labels, input conditions and intended claim match. The combined
result must be evaluated separately and preserve unsupported or missing checks.

The public package contains [aggregate results](evaluation-panels/video-genuine-edits-v24-result.json),
the [read-only aggregate audit](evaluation-panels/video-genuine-edits-v24-audit.json),
and reviewed code. Raw media, private source pairings, face vectors and
per-frame geometry/hash journals remain private. This package does not provide
a public raw-media or numerical replay. It reports a local diagnostic; no
outside human review or independent detection validation has occurred.
