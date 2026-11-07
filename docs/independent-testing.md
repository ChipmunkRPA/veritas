# Independent testing and collaboration

We welcome independent evaluations of images, video, audio, and writing. The
current detection-improvement program is in [this plan](detection-improvement-plan.md).
An implementation by our team is not an independent review; an external test
must identify who ran it, how inputs were selected, and what was independently
checked. We do not currently claim recruited reviewers or completed community pilots.

## Choose a review to reproduce

These are bounded entry points for contributors. Automated Linux reproduction
is separate from an outside person's review. A reproduced calculation does not
establish source truth, neural input correspondence or detection quality.
The numerical paths and synthetic colour experiment use this public repository;
the writing source audit and real-video renderer supplement additionally require
separately acquired study data.

| Package | What can be reproduced | Current evidence and limitation |
| --- | --- | --- |
| [Face-transition development review](video-v16-sface-development-evaluation.md) | Standard-library source/range/overlap/count review and ten deliberate packet alterations; no images or face embeddings | Local and actual Linux arithmetic passed. Located 55/60 edits vs 19/60 with 21/60 genuine review flags for both. Consumed-source development only. |
| [Fresh face-transition confirmation](video-v17-sface-confirmation-evaluation.md) | Standard-library source/donor exclusions, all120 ranges, paired/fallback counts and twelve deliberate packet alterations | Combined method located60/60 vs25/60, same26/60 genuine review flags. New cue supplied58 edit ranges; two use declared fallback. All ten requirements passed; local integration remains required. |
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
