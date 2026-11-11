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
array, `score_kind` (`raw` or `calibrated`), `response_units`, and `predictions`. Calibrated receipts
also require `calibration_sha256`. Every planned case appears exactly once with
its `id`, `input_sha256`, and `status`:

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

## Evidence localization

The portable receipt assesses input-level responses. It does not certify masks,
lip-sync offsets, or source positions. Include a separate localization audit:

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
