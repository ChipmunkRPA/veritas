# Independent testing and collaboration

We welcome independent evaluations of images, video, audio, and writing. The
current detection-improvement program is in [this plan](detection-improvement-plan.md).
An implementation by our team is not an independent review; an external test
must identify who ran it, how inputs were selected, and what was independently
checked. We do not currently claim recruited reviewers or completed community pilots.

Repository-managed GitHub Actions is disabled at the project owner's request.
Run verification locally and push reviewed updates directly. Historical Linux
workflow receipts below describe past runs; they are not instructions to start
or rerun a workflow.

## Start with a bounded check

The [four-modality receipt example](#four-modality-receipt-example) runs with
the Python standard library and fictional inputs. It shows how to reproduce a
comparison and detect a changed operating point.

The [standalone measurement reviewer](#recompute-selected-image-and-video-measurements)
replays a PNG seam or video frame schedule directly from a supplied original
and its signed evidence record. Start there to check those measurements on a
local export.

The [large-image copy comparison](image-affine-native-scale-2026-11-11.md#reproduce)
provides a licensed roster, exact source commitments, and a separate reader
for the PNG and JPEG localization and ordinary-control results.

The [earlier-image orientation comparison](image-reference-orientation-transfer-2026-11-12.md)
reconstructs right-angle rotations and reflections. Its Pillow-only reader
checks the original sources, native crops, and submitted difference mask.

The [earlier-video comparison](video-reference-comparison-2026-11-11.md#replay-without-the-app)
re-decodes two supplied clips and checks the JSON, native picture timestamps,
before/during/after previews, and exact soundtrack excerpts. On supported
pairs it also replays the short local picture range and its native crops.
An additional Python and FFmpeg command independently checks the exported local
interval and all six native crops without installing the app or model weights.

The [earlier-audio comparison](audio-reference-comparison-2026-11-11.md#replay-without-the-app)
replays changed sample ranges, difference waveforms, and playable excerpts
from two continuous, clock-matched recordings.

The [two-check writing replay](writing-checkpoint-agreement-2026-11-11.md#local-replay)
checks whether both detectors responded to the same source ranges and reproduces
the agreement and false-flag counts on the retained publisher panels.
The [editorial transfer](writing-agreement-ordinary-edit-2026-11-11.md#replay)
adds dated before-and-after revisions and separate scientific publisher controls.

The [audio supplement example](#review-an-audio-supplement-by-acoustic-condition)
uses authored scores to show how an acoustic routing error can add a genuine
false flag.

The [mono speech-interference study](audio-mono-dereverb-transfer-2026-11-11.md)
reconstructs matched noisy inputs and checks which true and false voice flags
change when reducing predictable late sound.
The [competing-voices disagreement study](audio-disagreement-transfer-2026-11-12.md)
reconstructs exact mixtures across calibration and two held meeting rooms.
It compares four voice policies and records when the second opinion cannot act.
The [stereo removed-speech comparison](audio-removed-source-transfer-2026-11-12.md)
rebuilds the sound taken out by interference suppression. The replay measures
target-speech leakage separately for centered and displaced recordings.
The [short edits in longer voice recordings](audio-overlap-window-transfer-2026-11-12.md)
replays four window rules on exact six-second waveforms, with known edit times
and matched loud competing speech.

The [video tracking evidence fixture](video-face-tracking-context.md) exercises
the same recorded values in a saved packet and report without downloading
detector weights. The [matched image cue](image-bg-real-detail-deficit-v265-result.md)
and [passage context replay](text-context-shift-v266-result.md) give reviewers
two recent examples of what a failed localization candidate looks like, with
frozen methods, exact aggregate results, and limitations. Full-media replay
requires the separately obtained source inputs and retained local bindings.

The [strong-background speech test](audio-noise-speech-gate-strong-noise-v269-result.md)
documents how to reproduce a comparison between the current speech admission
rule and a fixed fallback. Its aggregate distinguishes genuine and synthetic
coverage at two noise levels. The [downstream comparison](audio-noise-speech-rescue-downstream-v270-result.md)
scores the rescued clips and the already admitted comparison group against
their own clean versions.

The [science-explainer intake](text-elife-source-intake-v272-result.md) gives
reviewers a new five-source writing cohort with work-level attribution, an
immutable publisher revision and exact replacement targets. The receipt records
all nine dispositions and contains no excerpts. The [rewrite result](text-elife-rewrites-v273-result.md)
records all four unavailable attempts. The [detector comparison](text-elife-passage-score-v275-result.md)
reports localization, coverage and raw position ranking separately.
The [fresh transfer roster](text-elife-transfer-intake-v286-result.md) adds 26
groups with separately checked work licenses and fixed rewrite targets. The
[transfer rewrite construction](text-elife-transfer-rewrites-v289-result.md)
keeps the source roles and all 52 planned attempts in the record.
The [relative paragraph comparison](text-elife-relative-paragraph-v292-result.md)
reports recovered edit locations and false highlights on separate confirmation articles.
The [editorial-control intake](text-elife-editorial-controls-v301-result.md)
reports exact source and offset checks for a new set of ordinary edits, including
the coverage shortfall that stopped the study before scoring.
The [writing and measured-room transfer checks](writing-and-room-transfer-2026-11-11.md)
record the false-flag and recall trade-off from an archived writing replay
and the separate acoustic geometry preflight.
The [fresh editorial and mixed-passage test](text-editorial-transfer-2026-11-11.md)
reports newly measured false highlights and source-localization results from two generator families.
The [paragraph context transfer](writing-paragraph-context-transfer-2026-11-11.md)
compares isolated and neighboring paragraph responses on 11 new source articles,
with separate counts for originals, ordinary edits, and two generators.
The [fiction transfer](writing-fiction-paragraph-transfer-2026-11-11.md) applies
the same frozen rule to a new genre and counts ordinary human replacements.
The [practical DOCX comparison](writing-practical-notice-transfer-2026-11-11.md)
binds both human and generated replacements to exact extracted paragraph positions.
The [correspondence holdout](writing-correspondence-transfer-2026-11-11.md)
uses separate archival sources and synthetic multipart email containers. It
counts both useful paragraph locations and ordinary replacements flagged.
The [modern bulletin holdout](writing-modern-bulletin-transfer-2026-11-11.md)
includes exact quote and template variants across six separate publishers.
The [article context holdout](writing-article-context-transfer-2026-11-11.md)
compares a paragraph-only read with the usual surrounding input and a fixed
paragraph-omission experiment on six source-disjoint web features.
The [transcribed speech transfer](writing-speech-paragraph-transfer-2026-11-11.md)
checks whether an isolated paragraph preserves known rewrite locations while
reducing false highlights on oral histories and prepared public addresses.

The [image edit construction check](image-flux-reference-edit-v271-result.md)
uses public, authored RGB fixtures to verify exactly which pixels change when a
generator output is composed into a source region. A separate verifier checks
the pinned bundle without running the model or accessing project photos. The
[licensed photo pilot](image-flux-photo-preparation-v276-result.md) carries the
same geometry checks into two exact edits with six matched ordinary controls.
The [paired detector comparison](image-flux-detector-comparison-v280-result.md)
reports the same edit footprints under PNG and JPEG75, including false highlights.
The [six-photo expansion](image-flux-photo-expansion-v285-result.md) extends
that design to new licensed source groups. Its [format-specific comparison](image-flux-product-comparison-v288-result.md)
uses the actual PNG and JPEG routes and checks the resulting masks against exact truth.
The [copy corroboration fixture](image-copy-corroboration-v295-method.md)
checks whether a proposed region has a matching donor elsewhere in the image
while retaining the original highlight.
The [licensed photo comparison](image-copy-transfer-v298-result.md) measures
donor recovery on fresh ordinary edits and unchanged controls in two formats.
The [native copy follow-up](source-evidence-native-coverage-2026-11-11.md#images)
compares the supplied-region and automatic proposal paths on the same images,
with separate counts for translated and reflected copies.
The [regional patch response comparison](image-relative-patch-transfer-2026-11-11.md)
reports a compact paired-feature method's localization and false highlights
when the edit generator and source groups change.
The [small-insertion transfer](image-bg-real-insertions-2026-11-11.md)
compares source-localized results with unedited originals and matched
processing controls. Its exact precompression footprint is kept for both formats.
The [two-editor background patch panel](image-background-patch-transfer-2026-11-11.md)
tests PNG and JPEG75 with region-matched blur and processing controls.
The [native copy comparison](image-translation-patch-transfer-2026-11-11.md)
tests two crop sizes on 16 held-out source groups and reports hard controls
with natural repetitions in both PNG and JPEG90.
The [boundary blend comparison](image-boundary-blend-transfer-2026-11-11.md)
reconstructs the exact masks for sharp and feathered edits and verifies the
format-specific current checks against matched controls.
The [noise-guided image comparison](image-noise-region-candidate-2026-11-11.md)
matches the same input digests and evaluates a separate checkpoint at its
publisher point and two points fitted on disjoint controls.
The [fine-detail comparison](image-wavelet-region-transfer-2026-11-11.md)
adds 12 independently qualified photo groups and compares two sealed regional
checks on exact sharp and feathered edits after PNG and JPEG delivery.
The [residual color comparison](image-residual-color-transfer-2026-11-11.md)
reports exact regional overlap and control flags for three spatial-band arms
on two panels of sharp and feathered image edits.
The [semantic context comparison](image-semantic-context-transfer-2026-11-11.md)
compares a forensic patch response with its response relative to distant,
semantically similar patches on two licensed panels.
The [three-layer image comparison](image-multidepth-transfer-2026-11-11.md)
publishes exact region counts and a standard-library replay of all 384 held
decisions without the source media. A second verifier rebuilds native masks
when the evaluator has the original intake.
The [learned residual comparison](image-residual-patch-transfer-2026-11-11.md)
rebuilds exact edited-pixel masks for 496 deliveries, including seven separately
qualified SDXL and Kandinsky sources. The [remote crop transfer](image-remote-crop-editor-transfer-2026-11-11.md)
replays the same inputs and verifies whether suppressing broad processing
responses changes useful localizations.
The [paired PNG and JPEG crop comparison](image-crop-delivery-consistency-2026-11-12.md)
checks whether converting the same PNG to JPEG corroborates the native crop,
with separate counts for lost edit locations and added ordinary flags.
The [flexible single-image boundary comparison](image-boundary-extent-transfer-2026-11-12.md)
has a standalone public count verifier and native source replays. It records
small- and large-region tradeoffs and the separate follow-up on generated edits.
The [image flip comparison](image-regional-flip-transfer-2026-11-11.md) tests
whether the sensitive detector reproduces a region under a reversible transform.
It separately reports useful regions, processing controls and untouched photos.
The [stereo speech coverage comparison](audio-channel-speech-gap-transfer-2026-11-11.md)
checks exact channel samples and annotated speech in intervals left unscored
by the mono voice check. The [blind writing probe](writing-blind-rewrite-probe-2026-11-11.md)
checks source-character positions and human false highlights on new rewrites.
The [dated journal body comparison](writing-journal-body-transfer-2026-11-11.md)
replays exact earlier and later text from 21 distinct work groups and checks
localization on 32 separately generated one-sentence edits.
The [archived PEP writing pilot](writing-pep-editorial-intake-2026-11-11.md)
replays five dated work histories and compares exact sentence locations on
generated rewrites and ordinary editorial revisions.
The [archived technical prose transfer](writing-archived-passage-transfer-2026-11-12.md)
replays 125 held passages, including one historical revision, and compares a
fixed sentence ranker and its combination with the current check.
The [multi-paragraph writing transfer](writing-multiparagraph-localization-transfer-2026-11-12.md)
replays native article passages and constructed email equivalents, including
unrelated paragraphs and quoted text.
The [public-archive writing comparison](writing-archive-source-evidence-2026-11-12.md)
replays exact archive matching and source offsets, including a publisher
correction, missing-source controls, and quoted text that must be ignored.

The [offline video interval verifier](video-native-interval-replay-2026-11-11.md)
checks the source clock and exact before, during and after pixels with Python
and FFmpeg. Its local receipt distinguishes native source verification from
retrospective scoring.
The [short face-return transfer](video-short-return-margin-transfer-2026-11-11.md)
checks a fixed margin at three placement phases, then counts distinct secondary
ranges in a separate retrospective analysis. Its reviewer checks native anchors
and control outcomes independently of the scorer.
The [single-file mouth texture trial](video-mouth-native-texture-transfer-2026-11-11.md)
counts separately localized and missed redraws, ordinary retouching flags,
and source-compatible visible motion. The [paired H.264 transfer](video-reference-h264-earlier-transfer-2026-11-11.md)
verifies two independently encoded video copies, including tandem re-encodes
and unrelated clips with the same picture clock. The [second held H.264 group](video-reference-h264-grid-reserve-2026-11-11.md)
checks native crops and clocks on separate 360 × 288, 25 fps sources with
ordinary brightness changes, blur, and multiple encoding controls.
The [scene-referenced face motion comparison](video-webcam-motion-transfer-2026-11-11.md)
checks whether genuine head movement can explain flagged ranges and tracks
which clips lack enough stable background for that measurement.
The [lip-sync coverage comparison](video-lipsync-mouth-coverage-2026-11-11.md)
checks held mouth frames on real moving footage, matched to original and
recompressed controls, with a separate authored false-collapse regression.
The [local audiovisual timing comparison](video-lipsync-local-contrast-transfer-2026-11-12.md)
reconstructs 160 exact soundtrack deliveries and checks signed timing estimates,
ordinary controls, native mouth crops, and PCM slices.
The [mouth-pose comparison](video-mouth-pose-recurrence-2026-11-12.md)
tests short neural redraws and independently reopens the seven pairs with
source-compatible mouth motion, including their native pictures and PCM.
The [six-picture face-swap comparison](video-face-six-picture-transfer-2026-11-12.md)
re-creates the authored H.264 swaps and hard cuts and compares denser sampling
with the additional consecutive support rule.
The [timing and codec follow-up](video-timing-coverage-and-codec-holds-2026-11-11.md)
records when a single timing window leaves change unassessed and compares
exact and approximate picture holds on the same lossy and lossless deliveries.
The [longer timing replay](video-longer-timing-transfer-2026-11-11.md)
tests native windows with actual timing-change opportunity and separately reports
measured windows, redraw overlaps, ordinary controls, and unassessed inputs.
The [speaker-attribution comparison](video-speaker-attribution-transfer-2026-11-11.md)
checks visual speech separation, lip geometry and existing timing on known
voiceovers, timing interventions and continuous camera controls.

The [stereo audio preflight](audio-native-stereo-preflight-v274-result.md)
checks exact channel recovery and source clocks on five authored WAVs. The
same tool can also verify the four publisher background archives and the
fixed development and holdout windows. The [paired stereo benchmark](audio-native-stereo-pairs-v277-result.md)
records construction and exact readback of all 385 planned files. The
[development speech-gate comparison](audio-native-stereo-gate-v279-result.md)
compares mono admission and a fixed acoustic channel choice without using
authenticity scores for selection. The [downstream comparison](audio-native-stereo-authenticity-v282-result.md)
reports correct, wrong and inconclusive directions on those recovered clips
and on the comparison group. The [AI-only policy replay](audio-ai-only-rescue-v283-result.md)
counts the directions that a fixed mono-plus-channel rule would change. The
[200-case holdout](audio-native-stereo-holdout-v284-result.md) compares that rule
with mono on separate sources and reports the effects by noise condition.
The [AMI intake](audio-ami-native-stereo-v291-result.md) adds 24 naturally
recorded two-microphone windows for a separate genuine-audio test.
Its [stereo transfer comparison](audio-ami-stereo-transfer-v293-result.md)
reports identical decisions and unchanged coverage across both rules.
The [room-transfer preflight](audio-room-transfer-preflight-v296-method.md)
verifies a renderer for asymmetric background noise and confirms the archive
inventory for a future paired comparison with two voice generators.
The [measured room intake](audio-room-assets-v299-result.md) qualifies the
documented Surrey source positions and records the open second-room geometry.
The [BBC holdout intake](audio-bbc-room-assets-v303-result.md) verifies a
separate room with documented front and lateral positions.
The [matched Surrey and BBC comparison](audio-measured-surrey-activity-transfer-2026-11-11.md)
reports speech-window admission on the same foregrounds and strong background noise.
The [stereo suppression test](audio-stereo-reference-suppression-2026-11-11.md)
adds 36 matched noise-only controls and measures whether channel-difference
filtering recovers synthetic or genuine speech windows.
The [waveform fallback transfer](audio-waveform-fallback-transfer-2026-11-11.md)
tests repeat localization across three held meeting groups and twelve separate
speakers, with a standalone reviewer that checks source clocks and paired spans.
The [second voice representation](audio-wavlm-strong-traffic-2026-11-11.md)
keeps the admission decisions fixed and compares clean and −10 dB traffic
responses from two voice representations, with retained misses and exclusions.
The [two-background transfer](audio-wavlm-two-background-transfer-2026-11-11.md)
tests the same rule with distinct background recordings and records how each
condition changes flags and unassessed coverage.
The [independent voice veto comparison](audio-independent-noise-veto-2026-11-11.md)
replays 212 held audio views to quantify both removed false alerts and lost
correct detections under strong noise, Opus, and audio restoration.
The [paired noise correction](audio-paired-nuisance-transfer-2026-11-11.md)
checks separate fit, calibration, selection and held roles and replays 77 exact
room deliveries under the same fixed correction.
The [temporal corroboration trial](audio-temporal-corroboration-2026-11-11.md)
uses three native windows per track and holds the speech gate fixed when
comparing the outcomes under two additional background recordings.
The [two-microphone noise trial](audio-natural-stereo-coherence-2026-11-11.md)
compares score directions after a fixed filter with two recorded meeting
backgrounds, six genuine readers and three synthetic generator families.
The [stereo component comparison](audio-stereo-component-transfer-2026-11-11.md)
uses two other recorded meetings and explicitly counts admitted genuine false
alarms after two fixed decision rules.

The [video brightness readout](video-luma-context-v278-result.md) uses the
original selected appearance flags, known edit controls and exact native
frame bindings to document where a luma change extended beyond the face.
The [fresh camera qualification](video-vcd2-unseen-controls-v287-result.md)
adds eight unused camera types with independently verified native clocks.
The [matched ordinary controls](video-vcd2-matched-controls-v290-result.md)
verify 24 unchanged, chroma-converted and recompressed views on those clocks.
The [first current-app triplet](video-vcd2-current-app-v294-first-result.md)
records measured appearance coverage and the separate frame-check budget skip.
The [later reference snapshot](video-vcd2-current-app-v294-reference-snapshot.md)
preserves all 20 completed controls and four cases pending resource admission.
The [aligned face detail and motion comparison](video-aligned-face-detail-probe.md)
reuses committed landmarks to evaluate two new measurements on separate
development and confirmation groups. It includes the local replay command and
reports the false alarms that kept both measurements experimental.
The [whole-clip face-swap comparison](video-full-clip-face-swap-transfer.md)
uses pinned crop classifiers on exact common frame inputs, with genuine paired
and camera controls and a separate publisher swap.
The [whole-clip pooling transfer](video-whole-clip-pooling-transfer-2026-11-11.md)
compares four frozen rules across two source-disjoint delivery cohorts. Its
anonymized scalar replay reproduces every operating point and paired count
using only Python. The [facial color-step comparison](video-face-chroma-transfer-2026-11-11.md)
tests a separate native-region measurement on the same verified source inputs.
The [same-source delivery comparison](video-face-delivery-transfer-2026-11-11.md)
holds the source pictures and times fixed across two deliveries and tests the
frozen rule that selected a different check for each delivery.
The [paired face-swap transfer](video-face-pe-transfer-2026-11-11.md) includes a
standard-library score replay on 108 clips. The [prospective confirmation](video-face-pe-confirmation-2026-11-11.md)
applies the frozen rule to separate source groups using identical native faces in both arms.
The [outside publisher and camera transfer](video-face-pe-publisher-2026-11-11.md)
checks known full and partial swaps and ordinary delivery changes. It reports
the four partial variants as one source and camera outcomes by recording group.
The [recorded-noise repetition comparison](audio-repeat-robustness-transfer-2026-11-11.md)
checks exact and similar repeated audio across recorded backgrounds and four
codecs, with matching ordinary controls and anonymized per-group results.
The [frequency-domain separation comparison](audio-frequency-separation-transfer-2026-11-11.md)
keeps the production voice scores and calibrations fixed while measuring one
stereo projection under two kinds of competing speech.
The [voice timestamp replay](audio-clock-voice-window-replay-2026-11-11.md)
holds decoded samples fixed while introducing timestamp breaks and checks the
resulting voice coverage through the signed app report.

The [active PNG transfer](image-active-png-transfer-2026-11-11.md) uses exact
source-aligned masks to check which gains survive on small edited patches with
untouched backgrounds. The [color-residual comparison](image-color-residual-transfer-2026-11-11.md)
shows how many misses come from region selection before a review point is applied.
The [integer image reference comparison](image-reference-comparison-2026-11-11.md#integer-crops-and-translations)
preserves both original coordinate spaces for a corroborated crop or translation.
The [short face texture comparison](video-face-texture-return-2026-11-11.md)
checks appearance and texture on native video clocks with ordinary camera edits.
The [denser timing comparison](video-face-native-boundary-2026-11-11.md) measures
boundary error on a different set of recordings while preserving the existing flags.
The [fixed one-second rule](video-face-short-boundary-confirmation-2026-11-11.md)
tests the resulting timing rule prospectively on separate moving faces and camera controls.
The [short email contrast study](writing-short-email-contrast-transfer-2026-11-11.md)
compares individual sentences and matched context with exact MIME positions.
The [contextual token study](writing-context-token-transfer-2026-11-11.md)
checks whether exact sentence features improve selection on source-separated emails.

The [portable PACC-P replay](audio-pacc-portable-replay.md) runs the 98 genuine
speech controls using publisher files, separately obtained model assets and
caller-owned directories. It compares aggregate decisions and clean-to-noisy
transitions against the published result. Qualification runs without model
weights, and the aggregate comparison can be shared without sharing recordings
or individual scores.

The [fresh GRID transfer preparation](native-mouth-transfer-progress-v134.md)
separates publisher metadata, complete archive checksums, native video/audio
clocks and neural qualification. Fifteen new authored transport/selection checks
use synthetic bytes/ZIPs and mocked requests in the project dev environment;
they need no media, model weights, credentials or actual network requests:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_grid_acquisition_continuation_v135.py backend/tests/test_grid_clip_plan_v136.py
```

These check missing-range boundaries, unchanged-prefix assembly, rejected short/
ignored responses, bounded ZIP-directory reads and complete source/utterance
selection. They do not measure detection quality or establish the complete fresh
cohort's input eligibility. Read the [explicit acquisition continuation](grid-acquisition-continuation-v135-method.md)
and [full clip planning method](grid-clip-plan-v136-method.md) before interpreting
their receipts. Raw source identities/media and individual fingerprints remain
private. No outside reviewer is claimed.

The [expression-driving prototype](portrait-expression-drive-v138-design.md)
also has nine NumPy/pytest checks for the independent displacement equation,
unchanged source expression, invalid inputs and model-free imports:

```sh
PYTHONPATH=backend python -m pytest -q -o addopts='' backend/tests/test_portrait_expression_drive_v138.py
```

Those checks do not run pretrained networks or qualify generated motion. The
CPU bridge is unqualified and is not part of serving confidence. Its model
execution, native composite checks and achieved-motion comparison remain open.

The [complete native mouth construction/replay](native-mouth-clips-v120-complete.md)
now includes all 283 ready clips and 4,700 source-bound difference masks. Those
private-media checks are internal lineage validation, not an outside review.
For the next small cross-modal feature core, 13 authored checks need no media,
weights or credentials in the project dev environment:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_mouth_articulation_core_v122.py
```

They cover exact native-time context, speech-layer order, gap/padding rejection,
weighted projection/reconstruction and protection against learning the reference
from calibration/confirmation or edited rows. They establish arithmetic/role
contracts, not detection quality or a calibrated probability.

For exact source/output mask correspondence, the seven authored checks in the
[native clip replay](native-mouth-clip-review-v121-method.md) need no private
media, model weights, credentials or network in the project dev environment:

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_review_native_mouth_clips_v121.py
```

They reject wrong mask positions, nonbinary masks, extra changed pixels and
incorrect pixel-edge transforms. The full real-source audit needs separately
retained clips/source plans; its passing authored tests are not evidence of
outside human review, achieved synchronization or detector accuracy.

For native mouth-edit support and counterfactual response arithmetic, the
[new input bank](native-mouth-inputs-v116-result.md) and
[renderer contract](native-mouth-renderer-v117-method.md) offer 24 authored
checks in the project development environment. They require no source media,
weights, credentials or network. These checks cover native context preservation,
exact-time conditioning shifts, repeated-image equality and nonresponse failures;
they do not establish neural accuracy. Three separate installed-runtime checks
require the pinned v7 generator dependency overlay. The actual v118 native
[renderer response experiment completed](native-mouth-renderer-v118-result.md);
the earlier admission denial remains historical. That success qualifies response
and containment, not achieved phonetic alignment or detector accuracy. The
separate [CPU second-generator qualification](portrait-native-qualifier-v133-method.md)
later started after RAM admission, but hit its sampled RSS limit; see the
[actual stop](evaluation-panels/portrait-native-resource-stop-v133.json). Its
earlier [readiness receipt](evaluation-panels/portrait-native-readiness-v133.json)
remains historical. The [bounded-convolution context](portrait-tiled-runtime-v139-method.md)
preserves that failed attempt and tests an explicit memory implementation change
at the same weights, precision, resolution and resource limits.
The new [actual bounded CPU run passed](portrait-tiled-runtime-v139-result.md),
including exact own repeats and byte-identical first-crop comparison with the
preserved partial original. This is construction-runtime qualification, not fresh
detector accuracy or achieved phonetic alignment.

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_native_mouth_inputs_v116.py backend/tests/test_native_mouth_renderer_v117.py
```

For the newest native speech-state/source routing contract, run the 21 authored
checks in the [source encoder result](av-source-encoder-v115-result.md). They use
NumPy, pytest and the project code, with no media, model download or credentials.
The complete real-source state review requires separately retained private
inputs. It verifies saved outputs and routing, not detector accuracy. The
[native video selection result](video-native-appearance-v112-result.md) likewise
separates zero false flags from two inconclusive controls and reports incomplete
time support; it is not reserved confirmation or full production qualification.

The [photo attribution/pixel-lineage review](image-source-lineage-v20-result.md)
adds 49 authored local checks requiring no model, real photo, credentials or
network request. Its actual private-panel run covers all 523 attribution records
and 465 source-pixel comparisons, with separate full reconstruction and dual
decoders. These verify source correspondence, not detection accuracy or training
permission. Additional useful contributions are permitted original/edit pairs
with photographer attribution, exact edit positions and source-group separation;
do not post customer media or assume a dataset-wide license clears every image.

For a model-free audit of photo licenses and source grouping, the
[image training-data review](image-source-rights-v19-result.md) provides a
public-only local command path. It downloads about253MB of official/pinned
metadata and no photographs. All2,242 metadata source records and per-role
counts reproduce without the private pairings; attribution/pixel lineage and
training permission remain unestablished. Its52 authored local tests also cover
used-source exclusion and inconsistent role files. GitHub Actions stays disabled.

For the new localizers' source-coverage and pixel-projection checks, use the32
authored tests below (NumPy, Pillow, Torch and pytest; no weights or photos):

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_image_imdet_v17.py backend/tests/test_relayformer_v18.py
```

Their [native execution record](new-localizers-and-editing-methods-2026-11-08.md)
is separate:24 actual forwards on authored patterns, not a real-edit accuracy
evaluation. Neither candidate is admitted to serving confidence.

For an image-coordinate review, use the public authored crop/interpolation
check. It needs NumPy, SciPy, Pillow and OpenCV, with no photographs, pretrained
weights or account credentials:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_image_iml_zoom_public_v16 /tmp/new-image-crop-review.json
```

For a smaller audio/video timing review, the signed-lag and sample-block check
needs only NumPy:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.review_mouth_audio_public_v14 /tmp/new-mouth-timing-review.json
```

Use a fresh receipt filename; these commands refuse overwrite. Include your
commit, environment and generated receipt in a proposed evaluation issue.
These checks validate arithmetic. To evaluate actual detection accuracy, use
the source-separated manifest contract below with separately permitted real
and manipulated inputs, retaining false alarms, misses and unassessed cases.
Do not attach private source media or claim that an authored check measures
real-world detector accuracy.
The image arithmetic check passes while the [real-source exact-mask audit](image-iml-zoom-v16-result.md)
fails at rounding boundaries. Its result and supplementary sensitivity remain
explicit; passing continuous math does not establish discrete mask parity.

## Verify image crops from a signed export

From `backend`, with the project environment active and the instance key obtained
separately, run the verifier against the evidence ZIP and your original image.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-image original.png
```

This optional check runs locally. It compares the original SHA-256 and
then checks every unannotated inspection crop and both sides of a same-image
match at their recorded EXIF-oriented coordinates. Pillow is enough for
the crop comparison. For a geometric copy match in `image-evidence-13`, the
project Python environment also recomputes both masks with the recorded
alignment and starting box. This replay does not rerun sparse proposal
generation. [Geometric copy evaluation](image-affine-copy-transfer-2026-11-11.md)

When the export includes an aligned repeat-pass mask, the project environment
also recomputes the overlap of every inspected component with that mask, even
when the check withheld all crops. It uses the recorded response and quality
masks without running a detector. [Regional repeat pass](image-regional-flip-transfer-2026-11-11.md)

New image coverage records can include how much of the original contains
clipped or very dark pixels, low contrast, low-detail JPEG blocks, and patterns
consistent with resizing. Those categories can overlap. The recorded
reliability mask shows where the check applied exclusions on its measurement
grid. With the original supplied, the same command independently recomputes
those native shares and the exact grid mask, including analyses with no
selected regions. This check uses the submitted pixels and JPEG structure
without running detector weights.

The ordinary archive
verification remains available with only the ZIP and a key. The pixel check
provides source evidence for a separate review of the detector’s interpretation.
For a source-free setup check, run the project's authored fixture tests. They
cover mirrored and translated pairs, EXIF rotation, an incorrect source file,
and a signed crop whose declared position was shifted.

```sh
python -m pytest tests/test_public_package_integration.py -k '(image and multimedia_zip) or native_source_verifier'
```

When an earlier aligned image is available, the separate [reference comparison](image-reference-comparison-2026-11-11.md)
can produce a full-size mask and unaltered crops from both files. It runs on the
reviewer's machine and commits to both source hashes.

## Verify audio clips and timestamps from a signed export

Run the optional comparison with the local project environment, FFmpeg and
FFprobe installed. Supply the exact uploaded file. For a video export, give
the original video containing the soundtrack.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-audio original.mkv
```

The verifier checks the original SHA-256, re-decodes the first audio stream
to the committed mono 16 kHz samples, and compares its source timeline and
all published WAV intervals. This includes voice and lip-sync windows,
background sound, acoustic events, and both sides of repeated-audio evidence
when present. Timeline gaps stay visible in the comparison. The output counts
each playback interval separately, even when two intervals share one WAV asset.
For current measured stereo context, it also recomputes both physical channel
levels, their correlation and the mono level drop in each bounded source
window. It replays the bounded waveform similarity result when the signed
package contains that measurement.
The ordinary verifier also checks a reported audio clock discontinuity against
the soundtrack timeline committed in a signed video export.
The same file decoded with a different FFmpeg build can produce different samples.
This check verifies source correspondence for independent listening. It does
not rerun the detectors, recreate charts, or validate an AI authorship claim.

```sh
python -m pytest tests/test_audio_repetition_context.py tests/test_audio_similarity_context.py tests/test_video_nested_evidence.py
```

## Verify video frames and crops from a signed export

With the local project environment, FFmpeg and FFprobe installed, supply the
exact original video and the separately obtained instance key.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-video original.mkv
```

This checks the source hash and native presentation timestamps, then compares
unannotated frame PNGs and crops to the decoded pixels. When present, it also
checks the measured mouth patches, the committed aligned face inputs and the
source clock of saved context clips. It reports how many frames and inputs
matched. An inferred timestamp cannot verify as a measured frame. Native
replay accepts frames up to 16 megapixels. Different FFmpeg builds can produce
different results, particularly for older exports.

Review the report's interpretation separately. The verifier does not rerun the
detectors or recreate annotations and context-clip pixels. When the package has
frozen soundtrack evidence, run both checks together against the same original.
Each channel reports its own result.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem \
  --source-video original.mkv --source-audio original.mkv
```

### Recompute selected image and video measurements

After the trusted-key verification above, extract the signed `evidence.json`
entry and run the standalone reviewer from `backend`. Use separate work files
for different exports.

```sh
review_dir=$(mktemp -d)
unzip -p image-evidence.zip evidence.json > "$review_dir/image.json"
python -I scripts/review_evidence_measurements.py image original.png "$review_dir/image.json"

unzip -p video-evidence.zip evidence.json > "$review_dir/video.json"
python -I scripts/review_evidence_measurements.py video original.mkv "$review_dir/video.json"
```

The reviewer imports no application code and reads the original locally. The
image path needs Pillow. It recomputes the four sharp seam responses from
native color steps and checks their combined response. The output gives the
largest absolute difference from the packet. The tolerance is 0.00005 for
float32 accumulation and packet rounding. This reproduces the published
rectangle's measurement. Whole-image candidate ranking is a separate test.

The video path needs FFprobe and a complete native clock of at most 108,000
frames. It reconstructs the 12 Hz schedule from the source presentation
timestamps, checks the recorded input indices and ticks, and derives the two
transition timestamps and the before, during, and after frame references from
the recorded supports. Use the signed-source verifier above to verify the
actual frame pixels and aligned crops. Face comparison scores and candidate
ranking require detector re-execution.

The output includes source and evidence digests, packet counts, and either
`matched` or `no_measurement_recorded`. A mismatch exits with status 1. A
source without a published seam or timing refinement yields
`no_measurement_recorded`. Keep the JSON and the tool versions with a review.
Share the output only if disclosure of those source commitments is permitted.

Local replay of the existing paid-flow test fixtures reproduced all five seam
responses in the pasted PNG and all 21 scheduled positions, 21 measured
inputs, and three displayed frame roles in the known short swap. The original
PNG, JPEG, and original video had no recorded measurement in this replay.
On the 120-view [second PNG panel](image-boundary-prominence-holdout-2026-11-11.md),
the reviewer checked all combined responses and the four side responses for
25 highlighted rectangles. The largest absolute difference was 0.00000160.
The authored regression tests reject altered values, schedule positions,
source ticks, and displayed frame references.

```sh
python -m pytest tests/test_evidence_measurement_review.py
```

### Verify an earlier-copy image comparison

Download the comparison JSON in the local app and obtain both original files
from the reviewer who supplied them. From the backend directory, use the
project's local Python environment to replay the comparison.

```sh
.venv/bin/python -m scripts.review_image_reference_export \
  chipmunk-verify-image-comparison.json earlier.png submitted.jpg
```

The first two comparison schemas also support the standalone Python and Pillow
reviewer.

```sh
python -I scripts/review_image_reference_export.py \
  chipmunk-verify-image-comparison.json earlier.png submitted.jpg
```

The reviewer checks both file SHA-256 commitments and native dimensions after
orientation. It checks the overlap against the recorded integer offset,
compares both decoded crops with the corresponding source pixels, and checks
the embedded PNG hashes and binary mask bounds and pixel totals. Where the export
includes an amber difference overlay, the reviewer checks every pixel against the
full mask sliced at the submitted crop's coordinates. For the resize
schemas, the reviewer reruns the resize comparison using the two source files
and checks the recomputed framing, regions, and entire mask. It independently
maps the submitted region back to the exact earlier-copy crop using rational
coordinates. The crop-and-resize schemas verify the earlier area's exact offset
and size. The scoped PNG fallback also verifies the reproduced resize, pixel
match count, and exact support tiles. The reviewer reports empty comparable
and unassessed results separately. A mismatch exits with status 1. [Resize evaluation](image-reference-resize-transfer-2026-11-11.md) · [JPEG crop and resize](image-reference-crop-resize-transfer-2026-11-11.md) · [Exact PNG crop and resize](image-png-exact-crop-resize-2026-11-11.md)

The comparison JSON is a standalone, unsigned export. This replay checks its
geometry and source binding. The resize replay also checks the algorithm's
selected pixels. Establishing when or how either source was created requires
separate evidence. On the isolated browser fixture, the reviewer matched both crops and
counted 7,693 pixels within the reported mask bounds. Authored tests reject
shifted coordinates, missing alignment support and rehashed altered crops.
The reviewer also accepted all 180 genuine exports from the earlier integer
alignment panel. It checked 106 source crops and counted 1,036,494 retained
mask pixels across 99 comparable views. The other 81 views retained their
reported unassessed reasons. [Source and transforms](image-reference-comparison-2026-11-11.md#integer-crops-and-translations)

### Verify an earlier-draft comparison

Download the standalone comparison JSON and keep the original earlier and
later files. From the backend directory, replay both source hashes, the
decoded text positions, and the exact changed-token ranges locally.

```sh
.venv/bin/python -m scripts.review_document_revision_export \
  chipmunk-verify-draft-comparison.json earlier.eml later.eml
```

Use matching `.eml`, `.txt` or `.docx` files. The output gives the verified range
counts by role. Word replay also checks the top-level body paragraph numbers,
XML body-child positions and paragraph-local character ranges. The comparison
locates differences and leaves revision order and authorship to other evidence.
When a plain-text export includes the experimental paired check, replay also
verifies the exact source spans, both input hashes and the arithmetic of the
saved responses. This offline replay verifies saved values against the stated
rule. Reproducing the responses themselves requires a separate inference run.

### Replay document evidence against the original

New document evidence records distinguish source ranges with a local response,
text read only as context, and text with no exact local record. The analysis
view, PDF and JSON use the same extracted Unicode positions. Counts include
every mapped character excluding whitespace. The analysis view and JSON retain
up to 256 ranges, and the PDF shows up to 12. Source replay verifies the
retained positions and character counts against the supplied original. The
server also replays the map against the frozen local responses.

An isolated local app run used a short multipart email containing a controlled
rewrite, a genuine quotation and a signature. With the current classifier
available, all 307 mapped characters had local responses. A second run of the
identical file with the classifier skipped showed all 307 characters as having
no exact local record. The evidence API, ordinary JSON export and rendered PDF
agreed on the three exact source ranges in both runs. Both runs used simulated
payment and a separate local database.

```sh
python -m app.cli verify evidence.zip --public-key instance-key.pem --source-document original.pdf
```

With the project dependencies installed, this compares the original's SHA-256,
re-extracted text and source mapping with the signed record. It checks the exact
excerpts, positions, supporting input ranges, repeated passages and any native PDF
page images in the package. It supports plain text, DOCX, plain-body EML and
native-text PDF. OCR-dependent files can still use signature and package integrity
verification. The replay does not score authorship.

## Choose a review to reproduce

These are bounded entry points for contributors. Automated Linux reproduction
is separate from an outside person's review. A reproduced calculation does not
establish source truth, neural input correspondence or detection quality.
The numerical paths and synthetic colour experiment use this public repository;
the writing source audit and real-video renderer supplement additionally require
separately acquired study data.

| Package | What can be reproduced | Current evidence and limitation |
| --- | --- | --- |
| [Photo attribution and pixel lineage](image-source-lineage-v20-result.md) | Public authored identity/crop/decoder checks; complete real-panel replay requires separately retained permitted photos and private source plans | All 523 metadata responses reconstructed; 465 accepted source identities compared. 397 exact RGB matches, 68 differences; all 465 geometry replays and 930 dual-decoder checks match. No model inference, training admission, camera-authenticity proof or outside human review. Current generator-output terms remain separately scoped. |
| [Image source rights/grouping](image-source-rights-v19-result.md) | Standard-library public metadata/count reviewer after pinned metadata acquisition; no images, model weights or private pairings required | Full local private reconstruction and public-only replay pass for2,242 photos;597 pass a preliminary license filter.523 queued for attribution/lineage review, none admitted to training. Misnamed FLUX original files excluded from role assignment; actual generation splits disjoint. Not detector accuracy, final rights clearance or outside human participation. |
| [Full-source localizer v18](image-relayformer-v18-result.md) | Frozen native/source reviewer, independent OpenCV/IEEE ranking and contingency counts; exact panel additionally requires private sources and permitted cached weights | All384 inputs/798 project forwards complete. Full384-map repeat and both genuine-point refits pass. Useful regions0/48 per format vs9/48 native and6/48 JPEG current; rejected. Raw photographs/maps remain private; no fresh accuracy, serving admission or outside human review. |
| [Controlled writing-span learning](writing-spans-v7-result.md) | Public source construction, native loaders, two small learners, calibration/points, and full local source/refit reviewer; exact panel requires private source plan and separately acquired licensed CSV | All992 planned documents and full source/native/refit replay passed. All864 diagnostic arm masks recounted; both learned heads and three calibrators/points refit exactly. Interval found38/128 useful mixed locations vs45/128 current and lost27 previously useful documents; rejected as replacement. Consumed source groups, no serving confidence, fresh-author or outside human review claim. |
| [Magnified image crops](image-iml-zoom-v16-result.md) | Public authored grid/point arithmetic and complete aggregate results; native source/grid replay requires the retained private panel | All384 candidate variants and96 ordinary-reference runs completed. Candidate found0/48 useful edits per format versus9/48 native and6/48 JPEG current; rejected. Strict real-source independent-mask audit remains failed. Full post-result canonical/sensitivity audit retains25/768 ungated metric discrepancies and one changed point; rejection unchanged. Actual Linux continuous arithmetic passes, without outside human review or fresh accuracy. |
| [Supervised image-region head](image-patch-head-v14-result.md) | Frozen native/source replay, separate NumPy dense-head and coordinate arithmetic, and [actual Linux synthetic checks](evaluation-panels/image-patch-head-v14-linux-synthetic-review.json) | All1,344 development cases completed; candidate rejected for fewer useful regions. Original strict mean audit failed and remains failed. A separately frozen complete diagnostic replay reproduces all native features/maps and original outcomes; no mean-sensitivity mask changes. Raw sources/vectors/head remain private; no fresh accuracy or outside human review. |
| [Fresh JPEG calibration confirmation](image-fresh-v12-result.md) | Public frozen scoring/audit code and complete aggregate results; source/map replay requires the separately retained private panel and cached permitted weights | All 96 fresh inputs completed; 192 independent probability calculations and 384 source-coordinate means match. Candidate rejected: 12/48 edited flags vs 10/48 current, but 5/48 genuine false flags vs 2/48. Current masks meet IoU >= .3 on 3/48 edits. No serving promotion, public photos/maps or outside human review. |
| [Four-media specialist calibration](specialist-calibration-design.md) | Public authored128-file/512-observation version2 example, distinct context heads and calibrated combination; source-role/missing-evidence/duplicate-vote/source-geometry checks |63 local selected tests and [actual Linux integration](evaluation-panels/specialist-calibration-v2-linux-review.json) pass. Unrelated genuine regions no longer subtract from positive evidence elsewhere. This is symbolic-score integration only, not real-media calibration or detection quality. |
| [Region-aligned image calibration](image-region-calibration-v9-result.md) | Standard-library312 source/plateau-certificate bindings, two genuine point refits,120 display audits,384 metric sets and14 deliberate alterations | Complete private-grid/source replay passed. Area-aligned points still located0/48 edits per format vs12/48 current; rejected. Cached consumed responses, no new neural inference or outside human review. |
| [Regional image patch drift](image-patch-drift-v8-development-evaluation.md) | Standard-library312-case source/role binding, genuine-only point refits,576 pixel-metric sets, family counts and24 advancement checks; thirteen deliberate alterations | Separate local native/vector/pixel audit and [actual Linux scalar review](evaluation-panels/image-v8-patch-drift-linux-review.json) passed. Both readouts located0/48 edits per format vs12/48 current and were rejected. No public photos/features or fresh accuracy claim. |
| [Face-transition development review](video-v16-sface-development-evaluation.md) | Standard-library source/range/overlap/count review and ten deliberate packet alterations; no images or face embeddings | Local and actual Linux arithmetic passed. Located 55/60 edits vs 19/60 with 21/60 genuine review flags for both. Consumed-source development only. |
| [Fresh face-transition confirmation](video-v17-sface-confirmation-evaluation.md) | Standard-library source/donor exclusions, all120 ranges, paired/fallback counts and twelve deliberate packet alterations | Combined method located60/60 vs25/60, same26/60 genuine review flags. New cue supplied58 edit ranges; two use declared fallback. All ten requirements and [actual Linux replay](evaluation-panels/video-v17-sface-linux-review.json) passed. [Local integration and activation](local-face-appearance-v17-rollout.md) are complete; repository default remains legacy. |
| [Whole-image generation cue](image-smaug-v7-development-evaluation.md) | Standard-library all224 scalar responses, directional counts and ten original gates; optional numerical intervals | Actual Linux replay passed. Whole-generation indications improved, but local-edit detection was 0/24 per format and the original mixed-image advancement gate failed. No serving promotion. |
| [Overlapping-speech separation review](audio-separation-v14-development-evaluation.md) | Complete outcomes, numerical/refit and native-audit evidence, with public-source feature references | Separation arms rejected under the original useful-detection requirements. Better loss or fewer calls alone did not justify replacing the serving calibration. |
| [Mouth-window review](mouth-public-portable-review-v12.md) | Standard-library arithmetic, or seven-package exact fitting-only tree refit; no media or neural weights | Both Linux modes passed. Candidate still rejected for excess ordinary-clip flags. |
| [General-video review](video-demamba-v13-result.md) | NumPy-only all192 outcomes per method, losses, grouped intervals and11 requirements from nine lossless members | Linux arithmetic passed. Candidate rejected; four of16 native RGB checks remain failed. |
| [Video colour-path audit](video-v14-colour-rendering-audit.md) | Self-created synthetic colour chart; optional real-source renderer/crop/tensor replay without neural weights | Local32-frame supplement passed. Linux synthetic run completed but did not reproduce the Mac-specific hypothesis; different bundled codecs and separate encoding remain explicit. Original native failures remain. |
| [Fixed native-YUV comparison](video-v15-native-yuv-protocol.md) | Self-authored planar samples with exact full native recovery before renderer comparison | Mac/Linux native samples and clocks exact; explicit renderer RGB hashes still differ across hosts. Original RGB hypothesis/native gate remain failed. |
| [Audio review](audio-public-portable-review-v13.md) | All4,248 feature records,1,449 development outcomes and optional exact fitting-only numerical refit | Linux arithmetic passed; strict exact cross-platform refit failed. Diagnostics changed no directional calls; exact check was not relaxed. |
| [Writing-control source audit](writing-v4-human-controls.md) | Original human-only history replay and overlap audit, using separately obtained study data | Local source audit only;45 supported/eight unresolved controls, no classifier scores. Source text and edit logs are not in the public package. |

For a useful review, identify your commit and environment, choose which claims
you independently inspect, retain failures and return the generated receipt
with any proposed issue or pull request. Exact commands and data/dependency
boundaries are linked above. No account credentials, customer uploads or model
downloads are needed for the three numerical paths or synthetic colour experiment.

## Useful contributions

- Reproduce a published failure on a permitted input, including its source hash,
  exact check coverage, app revision, hardware, and relevant source positions.
- Compare a proposed detector or preprocessing change against the current check
  on identical, source-separated inputs. Include ordinary editing, compression,
  background sound, and other plausible non-AI explanations.
- Audit a calibration, sample split, mask or timestamp measurement, license
  declaration, or claimed numerical parity. A code audit and a detection-quality
  evaluation answer different questions; label them separately.
- Test whether ordinary users can follow findings to evidence and understand
  inconclusive outcomes. Document the task and feedback without private uploads.

Use the repository's **Independent evaluation or detection failure** issue form.
Issues and pull requests are reviewable proposals; they do not automatically
change a serving model, calibration, or stored result. Do not submit media that
you do not have permission to share. Public references, hashes, methods, and
aggregate outcomes are sufficient when source media cannot be redistributed.

The [public evaluation source catalogues](evaluation-panels/README.md) identify
the exact new image/video panels with public references, hashes, and edit
positions. Their dataset metadata retains its separate attribution and terms.

## Portable evaluation contract

`backend/app/bench/independent.py` is standard-library-only. Its command-line
wrapper runs without downloaded weights and does not start the app:

```sh
python3 backend/scripts/independent_review.py audit manifest.json
python3 backend/scripts/independent_review.py freeze manifest.json development.json thresholds.json frozen.json
python3 backend/scripts/independent_review.py summarize manifest.json confirmation.json frozen.json summary.json
python3 backend/scripts/independent_review.py compare manifest.json baseline.json baseline-frozen.json candidate.json candidate-frozen.json comparison.json
```

Outputs refuse overwrite. Keep the original manifests, frozen decisions, and
receipts. Store private evaluation media outside Git. The summary deliberately
omits individual source identifiers and media; review it before public sharing.

For a new evaluation, add `--print-digest` to the `freeze` command to obtain the
canonical digest. Put it in the confirmation receipt as `freeze_sha256`.
`summarize` and `compare` check a digest when present. To require it, pass
`--require-freeze-commitment`. The digest binds the exact threshold and other
frozen fields. A dated, separately retained record establishes when the freeze
was made. Historical receipts can still run without the new flag.

The manifest has `schema: "chipmunk.independent-evaluation/1"` and a `cases` array.
Each case records:

| Field | Meaning |
| --- | --- |
| `id` | Unique input variant |
| `modality` | `image`, `video`, `audio`, or `text` |
| `split` | `development` or `confirmation` |
| `group` | Shared parent-source group, consistent across variants and modalities |
| `label` | Integer 0 for declared genuine control, 1 for known generation/manipulation |
| `family`, `condition` | Generator/control family and transformation or recording condition |
| `source_sha256`, `input_sha256` | Parent source and exact scored-file digests |
| `lineage` | Known linked original, speaker, synthetic voice, topic, transcript, or source identifiers |
| `license` | `id`, explicit `evaluation_permitted`, and explicit `redistribution_permitted` |

Known lineage, group identities, and source/input hashes cannot cross splits.
These are declared identities: the checker cannot discover undisclosed training
overlap, authenticate a publisher's labels, grant rights, or infer subject consent.
Check near duplicates and connected source lineage before freezing the selection.
Keep source families and transformed copies together. Explain any unknown overlap.

Each system receipt records `manifest_sha256` (canonical JSON digest from the
audit), `split`, `system`, `implementation_sha256`, a nonempty `weight_sha256`
array, `score_kind` (`raw` or `calibrated`), `response_units`, and `predictions`.
Calibrated receipts also require `calibration_sha256`.

For checks based on direct measurements, set `method_kind: "model_free"` and
`weight_sha256: []`. The freeze records the method and implementation digest.
Existing weighted receipts use their pinned weight digests.

Every planned case appears exactly once with its `id`, `input_sha256`, and
`status`.

- `ok`: a finite `response`, oriented towards the declared positive class. Raw
  measurements retain their actual units, including signed responses. Calibrated
  estimates must lie between 0 and 1.
- `inconclusive`, `not_applicable`, or `failed`: `response: null` and a specific `reason`.

Raw responses are not confidence probabilities. Missing responses must not be
written as zero. Candidate calibrations have their own identities; changing a
system, checkpoint, implementation, or calibration after the freeze is refused.
The `thresholds.json` file maps each evaluated modality to a fixed **internal
benchmark flag point**. This is not a public screening cutoff or a rule for the
customer headline. Operating points use the declared response units; signed raw
measurements are not clipped or mislabeled as probabilities. Select operating points only on development data and explain
the choice; the wrapper intentionally does not optimize them on confirmation.

All planned inputs remain in alert/miss and coverage denominators, including
errors and abstentions. AUC covers only scored inputs. Brier is reported only for
declared calibrated responses and scored cases. AI without an alert includes
unassessed inputs; it must not be described as a genuine-content verdict.
Paired comparisons resample declared source groups and report each media type and
condition separately. A small bootstrap interval describes this panel, not future
customers or unknown generators. Report all original/edit counts alongside rates.

### Four-modality receipt example

From the repository root, run with Python 3.11 or later. The example contains
32 authored input records and no media, app dependencies, model weights, or
accounts. It demonstrates counts and localization for images, video, audio, and
writing. The values are fictional and measure receipt arithmetic.

```sh
scratch=$(mktemp -d)
example="$scratch/four-media"
review=backend/scripts/independent_review.py
python3 -IS "$review" demo "$example"
python3 -IS "$review" compare "$example/manifest.json" \
  "$example/baseline-confirmation.json" "$example/baseline-freeze.json" \
  "$example/candidate-confirmation.json" "$example/candidate-freeze.json" \
  "$scratch/replayed.json" --require-freeze-commitment
cmp "$example/comparison.json" "$scratch/replayed.json"
```

Each method retains 24 confirmation cases and four unavailable outcomes.
The comparison records false flags, recovered edits and source overlap.
Changing a frozen operating point while using the original confirmation receipt
causes verification to stop before writing an output. Try the same review
commands on your own receipts after replacing the fictional source roster.

The local clean-source test used Python 3.11.9 with site packages disabled.
The example comparison reproduced byte for byte and a changed video point was
rejected. A separate clean-source run of the published mouth review reproduced
100 recorded trees, 492 point windows, 2,321 development windows and 88 case
outcomes. The audio identity reviewer also reproduced 5,328 planned inputs
from public source metadata. Those two checks used public files from commit
`b0aa41f`. [Mouth review instructions](mouth-public-portable-review-v12.md) ·
[Audio metadata commands](#model-free-checks)

## Review an audio supplement by acoustic condition

The [separated-stream presence comparison](audio-stream-presence-transfer-2026-11-11.md)
adds a natural meeting panel selected from public word annotations before
inference. Its separate verifier reconstructs exact input clocks, confirms
the development and confirmation roles and checks the published arithmetic.
The local recordings and per-interval outputs remain with the evaluator.

The [hard-overlap source panel](audio-hard-overlap-transfer-2026-11-11.md)
is reconstructible from pinned public VoxPopuli test partitions and the
commercially reviewed model assets. It includes a separate standard-library
verifier for the saved local audio hashes, interval grids, count distributions
and route totals. The method and aggregates are public. Keep native source audio
and per-recording responses in the evaluator's local workspace.

The [independent speaker-count comparison](audio-independent-overlap-count-2026-11-11.md)
includes a fixed acoustic routing replay with recorded input hashes, the author
checkpoint and scaler commitments, and separate controlled and natural-room
counts. An evaluator can replay the original five-second count and the two
level-normalized variants with the published method and locally held source
permissions. The decision review below works from a receipt without these weights.

The standard-library reviewer replays the effect of adding a second audio
response only where the current check was inconclusive. The authored example
uses fictional source groups, score values and routes. It reproduces the
counting logic without model weights or audio files.

```sh
review_dir=$(mktemp -d)
python3 -I backend/scripts/review_audio_supplement.py demo "$review_dir/example"
cat "$review_dir/example/summary.json"
```

The example recovers one synthetic input. Two extra genuine flags in the
mostly-single-interferer stratum belong to one source group. A third genuine
flag comes from a source annotated as overlapping speech that the router
classified as mostly one interfering voice. The summary counts source groups,
individual inputs, routing, and assessment coverage separately.

For your own evaluation, use the [portable manifest](#portable-evaluation-contract)
with audio cases only. Use `clean`, `mostly_single_interferer`,
`overlapping_interferers`, or `unknown` for the reference `condition`. Put
known foreground speaker and background recording or meeting links in
`lineage` so that they stay in one split. Use the foreground identity for
the parent `source_sha256` when a mix has a separate background recording.
Labeling and source permission remain the contributor's responsibility.

Use portable development and confirmation receipts with two additional
top-level fields. `incumbent_sha256` and `router_sha256` identify the incumbent
and routing implementations and their pinned configuration. Add
`incumbent_direction` and `observed_condition` to every prediction. Use
`ai`, `original`, `inconclusive`, or `unassessed` for the incumbent. Use the
four acoustic condition values for the measured route. Obtain that route
from the candidate's measurement and record an unavailable route as `unknown`.

Choose the internal operating point for each enabled noisy route on
development data. Use a JSON object such as the generated example's
`{"mostly_single_interferer": 0.6}`. Freeze it before running confirmation.

```sh
freeze_sha256=$(python3 -I backend/scripts/review_audio_supplement.py freeze \
  manifest.json development.json points.json frozen.json)
```

Put the printed digest in the confirmation receipt's top-level `freeze_sha256`.
Keep the freeze and its recorded time with the study. The replay validates
matching content, and the study log establishes when the freeze was made.

```sh
python3 -I backend/scripts/review_audio_supplement.py compare \
  manifest.json confirmation.json frozen.json summary.json
```

A scored response strictly above the frozen point can promote an incumbent
inconclusive result to an AI flag when the measured route is enabled. The
summary evaluates the result by reference condition and includes all planned
cases. For a genuine stratum, compare the `ai` count with `planned` and
`incumbent_assessed` to see false flags and coverage together. The aggregate
contains counts and whole-receipt digests. Review the aggregate before sharing
it, especially small groups, and keep media, per-case scores and private
manifests local.

## Evidence localization

The portable receipt can also count localization outcomes without carrying source
media, coordinates, or private text. To include them, add `localization_truth`
to each planned case with `unit`, `source_units`, and `target_units`. Use
`pixels` for images, `frames` for videos, `samples` for audio, and
`unicode_codepoints` for writing. Use zero target units for controls and the
exact union of affected units for positive inputs. In the corresponding
prediction, add a `localization` object with its own `status`, plus measured
`reported_units` and `overlap_units` for `ok`, or a reason for an unassessed
outcome. An assessed input with no region has zero reported and overlapping
units. The summary reports per-condition counts, control highlights, and
positive IoU, F1, and recall, including misses and unassessed cases in the
positive means. File response and localization may have different coverage.
The comparison command also reports paired localization differences by media
type, condition and truth label. It averages related variants within each
declared source group and resamples source groups for its intervals. It reports
individual case transition counts alongside the group-weighted difference.
Read the coverage difference with overlap and false regions so a missing
source mapping cannot be mistaken for an improvement.
For an image with 1,000 source pixels and a 100-pixel truth mask, a report
covering 80 pixels with 60 pixels inside the mask contributes an IoU of
60 / 120. Keep the exact region geometry in your private evaluation records.

The receipt checks arithmetic bounds. A separate audit should verify the
underlying masks, clocks, and text positions:

- Images: original dimensions and orientation, reference-mask hash, exact scoring
  grid and projection, displayed-region policy, IoU/pixel F1 including misses,
  and authentic inputs receiving highlights. Bounding rectangles are not masks.
- Video: known manipulated frame ranges, exact presentation timestamps, track
  coverage, interval overlap including misses, and before/during/after frames.
- Audio: original and decoded hashes, sample rate, sample bounds, independent
  source groups, measured mixture SNR, and whether the background was separated
  or remains part of a mixed recording.
- Writing: immutable extracted text, Unicode offset convention, scored context
  windows, source passage positions, and whether word-level labels are actually
  available. A passage response is not proof of individual AI-written words.

## Model-free checks

For a first independent contribution, the corrected audio source graph and
execution freeze can be reviewed with the standard library alone:

```sh
python3 backend/scripts/review_audio_identity_plan_v9.py \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz /tmp/new-source-review.json
python3 backend/scripts/review_audio_identity_execution_v9.py review \
  docs/evaluation-panels/audio-v9-execution-freeze.json \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz \
  /tmp/new-execution-review.json --project .
```

Use new output paths. Report the commit, your environment, receipt hashes and
which checks you independently inspected. A successful metadata/code review
does not establish native inference parity, predictive accuracy or customer
usability. The [writing source preflight](writing-source-preflight-v2.md) similarly
distinguishes source-position verification from detector validation and records
unresolved source terms before reuse.

The [complete corrected audio result](audio-identity-v9-development-evaluation.md)
also supplies all consumed representations, fitted parameters and outcomes for
a numerical review with optional CPU refit. That command needs Python numerical
dependencies but no source audio or pretrained-model downloads. It reproduces
the original feature/fit artifacts and all conclusions/gates; it does not
re-execute native model forwards or establish fresh accuracy. The native replay
receipt is separate. The candidate failed development and remains disabled.

The [conditional audio result](audio-conditional-v10-development-evaluation.md)
adds a separate nonlinear/quality comparison with all outcomes, small fitted
numerical trees and 121 native replay receipts. Its public-only numerical refit
uses the already published vectors/quality packet, without private manifests or
pretrained downloads. Both eligible arms failed; reproducing that rejection is
a useful contribution. Please distinguish arithmetic reproduction, source/audio
re-execution, predictive evaluation and usability feedback in any review.

The [full-vector audio comparison](audio-full-representation-v11-development-evaluation.md)
also remains disabled after four failed requirements. Its small public packet
reuses previously published vectors/cached probes. The supplied command refits
only fitting sources and reproduces all conclusions/gates without private
manifests or model downloads. Native source re-execution covers 125 fixed/new-probe
cases and is a separate recorded check.

For a model-free starting point with a clean installation, use the
[portable v13 numerical review](audio-public-portable-review-v13.md). It has a
separate seven-package requirements file and blocks app/neural imports. Local
verification reconstructed every feature record, both complete fitting-only
objects and all 1,449 outcomes in a new environment. The result remains rejected;
this is a reproducibility check, not fresh accuracy or outside participation.

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q backend/tests/test_independent_review.py
```

Those tests use explicitly synthetic response fixtures. They verify the audit
contract and denominators, not the predictive accuracy of any detector.

The [image context/region-point comparison](image-region-point-v15-result.md)
preserves all four arms and a rejected primary, with complete native/source
replay and an actual 12-test Linux arithmetic receipt. Its frozen data were
already consumed; no new outside human reviewer or fresh-quality claim follows.
The [broader provider-signal inventory](provider-signal-broad-search-2026-11-08.md)
identifies additional qualification routes and explicit documentation gaps;
provider policy review is separate from authentic file/response testing.

The [background-context video experiment](video-background-context-v25-result.md)
adds 18 authored spatial/temporal and source-role tests, with an actual isolated
Linux receipt. The native280-file study and separate600-outcome audit are
complete, but its primary fails replacement rules. Source footage/geometry/
descriptors stay private; the public tests reproduce arithmetic and safety
contracts, not native face inference or fresh accuracy.

The [public LTX audio marking contract](ltx-marking-audio-v1-result.md) adds
80 authored native/file cases and a complete separate sample/codec audit. The
[actual Linux receipt](evaluation-panels/ltx-marking-audio-v1-linux-array-review.json)
passes 20 arithmetic/contract tests without media or neural inference. No
provider-authentication, natural-speech accuracy or outside human review follows.
The [provider follow-up](provider-signal-followup-2026-11-08.md) records 82
families and explicit gaps, separately from operational verification.

The [audio-aware mouth study](mouth-audio-context-v14-result.md) provides another
model-free starting point: `python -m scripts.review_mouth_audio_public_v14`
with a new receipt path and `PYTHONPATH=backend`. Only NumPy is required; it
checks authored signed lags, exact audio blocks and no-wrap/degenerate cases
against separately written scalar equations. It does not replay private audio
or train the small head. The [actual Linux math receipt](evaluation-panels/mouth-audio-context-v14-linux-math-review.json)
also passes. The complete local 272-track/6,373-window study rejects
the candidate for excess ordinary flags. Outside human review remains pending.
