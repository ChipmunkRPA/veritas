# Public evaluation source references

These are public dataset references, selection metadata, file hashes, and known
edit positions for the new local comparisons. **No source media, customer uploads,
or upstream model checkpoints are included.** The v6 fitting capsule includes
derived numerical features and fitted linear parameters, with separate data
terms. These resources let independent reviewers identify the
same inputs and audit source separation, rather than relying on aggregate numbers
or a generic dataset link.

- `iml-vit-v4-plan.json` and `iml-vit-v4-manifest.json`: the 48 original/edit
  pairs in the [IML-ViT comparison](../iml-vit-localization-evaluation.md).
- `tiled-splice-v5-plan.json` and `tiled-splice-v5-manifest.json`: the next 48
  pairs in the [tile comparison](../tiled-splice-evaluation.md), retaining one
  explicitly unavailable secondary author mask.
- `video-v6-sources.json`: the 90 downloaded original/two-generator videos,
  their public release locations, file hashes, and matched-encoding recipe.
- `video-v6-intervals.json`: all 120 derived original, swap-insert, and ordinary
  controls, including source groups and known inserted frame/time intervals.
- `video-v6-exclusions.json`: prior source/donor identifiers excluded before
  selection. It preserves the exclusion identities, not local setup paths.
- `audio-v4-v6-development.json`: 120 already-consumed original audio files,
  12 noise/music recordings, all 720 input recipes, exact file/decoded/sample
  payload hashes, original WAV container timestamps, and five connected-source
  folds. It includes frozen baseline responses and guarded reference decisions.
  These are development inputs, never a new confirmation set.
- `audio-v4-v6-reconstruction.json`: local offline reconstruction receipts for
  all 720 inputs. Independently recomputed sample payloads, complete WAV files
  and production-decoded PCM matched their declared hashes. This is local
  reproducibility evidence, not outside review or a fresh accuracy study.
- `mouth-v4-development.json`: 48 AI mouth-edit/non-AI control cases, eight
  consumed CREMA-D sources, separate raw/normalized hashes, exact edit intervals,
  mask references, source-frame hashes and frozen comparison.
- `mouth-v4-source-audit.json`: all 48 cases passed exact native pixel, mask,
  clock, surrounding-frame and soundtrack replay checks. This is local source
  verification, not detection accuracy or outside review.
- `mouth-v4-renderer-assets.json` and `mouth-v4-renderer-LICENSE.txt`: permitted
  human generator source/checkpoint pins and terms. The restricted InsightFace
  detector is excluded. These create benchmark inputs, not customer detections.

Image selection plans and manifests are exact copies of the frozen metadata.
The video source/exclusion catalogue is deliberately limited to reproducibility
fields; original ffprobe tags and local setup details are omitted. The derived
video interval manifest is exact. Native source checksums and referenced source
revisions identify the inputs. Derived encoded bytes can depend on ffmpeg and
codec versions; verify exact frames as the preparation scripts do and record any
differences. Do not silently replace an unavailable source or relabel a mismatch.

Source metadata originates from [TGIF/TGIF2 (IDLabMedia)](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
and the [RTFS-10K publisher release](https://huggingface.co/datasets/godmodes/rtfs-10k/tree/bca865e675852666a426368b2f5614abf1c1926a).
These metadata extracts retain the publishers' [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) terms and
attribution; the project's MIT code license does not relicense the datasets.
Follow the linked cards for original source credits and media terms. Project
methods, curation, derived fields, unavailable-mask handling, and outcome
limitations are described in the linked experiment reports. Publisher labels
and declared rights are not independent authentication of every source.

## Audio review and offline reconstruction

The audio catalogue has its own per-source licence declarations and attribution:
DFADD's revised release, underlying VCTK and LJ Speech, LibriSpeech, and selected
MUSAN FreeSound/RFM recordings. The TGIF/RTFS terms above do not apply to audio.
Full transcripts, private paths, audio payloads and model weights are omitted.

To reproduce inputs, acquire the separately permitted original/background bytes
at the pinned references. For archive members retain the declared member; for
Parquet sources use the source identity and recorded background row coordinates.
Verify the source-byte hashes and place each extracted recording in a separate
cache as `<sha256>.audio`. Then, from the repository root:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/rebuild_audio_development_panel.py \
  docs/evaluation-panels/audio-v4-v6-development.json \
  /path/to/separately-acquired-source-cache /path/to/new-output-directory \
  --catalogue-sha256 c0efaa0c9924f0a57fa42bf3d613e351c122070d43dce47a500ac5969eecbabd
```

The helper is offline: it downloads nothing, loads no detector and submits no
paid analysis. It recomputes the speech normalization, background intervals,
SNR mixtures and shared attenuation, verifies the float sample payload before
restoring the recorded WAV header timestamp, and checks both complete file and
production-decoded PCM hashes. Only reconstructed benchmark copies receive the
recorded timestamp. Original acquired recordings remain unchanged. Every planned
case receives a receipt, including failures; an existing output is refused.
Decoder/library differences must be reported, not silently relabelled.

The local source-cache reconstruction passed all 720 cases. Its exact receipts
are in `audio-v4-v6-reconstruction.json`. An earlier attempt reproduced the
samples but failed exact file hashes because libsndfile writes a current PEAK
timestamp; that failed attempt is retained in ignored local research storage.
The revised catalogue records that header field, and the helper checks the
independently recomputed sample hash before restoring it. The combined audio
development, export/reconstruction and portable review suite passed 83 tests.
This verifies the stated local workflow, not outside reviewer participation.

Rebuilding consumed data checks reproducibility, not fresh detector accuracy.
The [v4 enhancement](../audio-noise-view-v4-evaluation.md),
[v5 quality calibration](../audio-quality-v5-evaluation.md), and
[v6 decision layer](../audio-residual-v6-evaluation.md) were rejected under their
frozen advancement criteria. A new method or outside confirmation needs a
separately frozen source selection and
decision. No outside reviewer has independently verified these audio results.

## AI mouth-edit replication

The [mouth comparison](../mouth-sequence-v4-evaluation.md) includes true neural
redraws, ordinary local edits and timing controls, with exact before/during/after
positions. It is consumed development. CREMA-D source/derived data retains
ODbL/DbCL terms and publisher attribution; MIT code does not relicense it.
Media and mask PNGs are not bundled. The catalogue distinguishes raw FLV hashes
from normalized and generated files; do not compare a downloaded FLV with a
derived MKV digest.

The standard-library helper downloads only reviewed human animation modules,
checkpoints, the permitted landmark bundle and publisher licence:

```sh
python3 backend/scripts/fetch_mouth_benchmark_assets.py \
  docs/evaluation-panels/mouth-v4-renderer-assets.json /path/to/benchmark-assets
```

Use `--verify-only` for an existing cache. Generation uses the isolated MediaPipe
0.10.32 runtime described in `video-forensic-followup.md`, preserving the app's
OpenCV dependency. `smoke_mouth_renderer.py` supports CPU/MPS; the tested full
generation driver in `prepare_mouth_edit_v4.py` uses MPS on the local Mac. The
catalogue links source references and the original normalization recipe.
Container bytes and neural outputs can vary by library/device; record new
hashes and masks rather than silently substituting them in frozen results.

`audit_mouth_edit_v4.py` verifies input pixels, masks, clocks and audio.
`compare_mouth_sequence_v4.py` freezes, scores and reviews the paired production
adapter inputs. Its internal point has feature-distance units and is not an AI
confidence percentage. The candidate failed its gates and is not enabled.
No outside reviewer has validated this panel.

## Model-free review of learned mouth-edit responses

The [v5 comparison](../mouth-artifact-v5-evaluation.md) publishes
`mouth-v5-public-responses.json` with all retained source-clock/model-input
hashes and raw logits. `mouth-v5-source-audit.json` verifies 2,016 input bindings;
`mouth-v5-invalid-first-attempt.json` preserves the invalid run's notice. The
candidate was rejected. These are consumed development and local audits, not
outside validation.

An outside reviewer can check arithmetic and temporal denominators without
installing the app or downloading weights:

```sh
python3 backend/scripts/review_mouth_artifact.py \
  docs/evaluation-panels/mouth-v4-development.json \
  docs/evaluation-panels/mouth-v5-public-responses.json /path/to/new-review.json \
  --catalogue-sha256 82d90c2c1e0c05d3a8ecb42370f51c2fd62a139a1b8972e31fb6724d203b5b65 \
  --receipt-sha256 a54e474404ccc8f136306bafeb4844e51dd86ad31b62df5a97f29146430c6339
```

The helper refuses changed input digests and existing outputs. It recomputes
each reported window from retained logits, checks its recorded source timing,
recomputes the non-AI-only point, and includes every planned case. It cannot
verify actual pixels/model execution, missing eligible windows, face-track
membership, spatial masks or independent accuracy. Those require separately
permitted media and full source/weight replay.

## Refit the experimental mouth head without vision weights

The [v6 result](../mouth-trained-v6-evaluation.md) includes a compact
`mouth-v6-fitting-capsule.json.gz`: 800 generic mouth-feature vectors, actor roles,
source/PNG identities, fitted coefficients and normalization. Its underlying
CREMA-D-derived data retains ODbL/DbCL attribution and terms, recorded inside the
capsule. It contains no source images, private paths or pretrained vision weights.

In an environment with NumPy, OpenCV and scikit-learn, run:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/mouth_probe_capsule.py replay \
  docs/evaluation-panels/mouth-v6-fitting-capsule.json.gz /path/to/new-refit.json \
  --sha256 11f6c6dcdeb755aa25aee8b01498f3c59619176abd886bd50fc533584831cfb4
```

The helper loads no pretrained model. It validates vector commitments and actor
separation, refits the same actor/class-balanced head and point, and compares
coefficients/normalization to the stored result. It refuses a different capsule
digest or an existing output. The local CPU replay had zero differences. Library
differences must be reported; the helper's numerical tolerance is explicit in code.

The training and 48-video source audits are `mouth-v6-training-audit.json` and
`mouth-v6-video-audit.json`; the numerical replay is `mouth-v6-fitting-replay.json`.
These are local correspondence/refitting checks. No outside reviewer has
independently verified them. Refit success does not validate the neural feature
extraction, source rights, fresh accuracy or the failed detector. This head is
not enabled and its raw margin is not AI confidence.

## Second-family controlled mouth edits

The [v7 extension](../mouth-multifamily-v7-evaluation.md) adds forty clips from
the same eight consumed actors. `mouth-v7-development-inputs.json` retains every
new input, native interval, mask reference, crop mapping and render proof hash.
`mouth-v7-source-neural-audit.json` verifies all 2,740 native frames/800 masks and
320 neural input bindings, plus a fixed sample of 32 neural output replays.
Raw current-check windows, frozen code/point, and the 88-clip two-family summary
are separate JSON files in this directory. This is development characterization,
not fresh accuracy or independent review.

Only seven permitted generator assets are downloaded; no publisher demo media,
face-parsing, DWPose/S3FD or optional SyncNet weights are acquired:

```sh
python3 backend/scripts/fetch_mouth_generator_v7.py \
  docs/evaluation-panels/mouth-v7-generator-assets.json /path/to/benchmark-assets \
  --sha256 bc823fb8f21e21c8c1cd722d225c927b0102b724b76942457f166ef047caf9fc
```

`--verify-only` checks an existing complete cache without downloading. Publisher
LFS/Git identities and byte bounds are mandatory; an unexpected existing file
is refused. The source notice and separate CREMA-D terms remain applicable.
No publisher Python/shell file is executed by this helper.

The tested research runtime adds four pinned wheels to the project's existing
backend environment through a separate path:

```sh
backend/.venv/bin/python -m pip install --require-hashes --no-deps --only-binary=:all: \
  --target /path/to/new-research-runtime -r docs/evaluation-panels/mouth-v7-runtime.lock
```

Select that path only for research processes, for example
`PYTHONPATH=backend:/path/to/new-research-runtime`. This does not upgrade the app's
environment. Package/version identities are in `mouth-v7-runtime.json`.
The GPU smoke and generation/audit drivers are tested on this Mac with explicit
MPS/float32. The smoke exposes a CPU option, but this port has not been forward
tested on CPU. Full generation is not validated for CUDA or GCP.
Source reconstruction needs the separately acquired v4 original cache;
its raw, normalized and lossless-parent digests have distinct meanings.
Re-encoding/device/library differences require new receipts, not substitution in
the frozen results. Model/source names here document research rights and methods;
they do not add identifiers to customer analysis exports or reports.
# Two-family learned mouth experiment (v8)

The [v8 result](../mouth-learning-v8-evaluation.md) contains a real learned
localization gain, but it failed the frozen both-family advancement requirement.
The candidate is research only. Public `mouth-v8-*` metadata includes source
roles, every native training/view commitment, learning/point receipt, all 88
outcomes and source/numerical audit receipts. No customer data, media or model
binary is included. These were consumed-source local experiments, not outside
validation. Component and source-data terms remain separate.

Anyone with Python 3 can check retained-window arithmetic and complete outcome
counts without ML packages or checkpoint downloads:

```sh
python3 backend/scripts/review_mouth_learning_v8.py \
  docs/evaluation-panels/mouth-v8-review-responses.json.gz \
  docs/mouth-learning-v8-evaluation.json \
  docs/evaluation-panels/mouth-v8-head.json \
  /tmp/mouth-v8-public-review.json
```

The output must be a new file. This checks 1,056 retained candidate windows and
176 outcomes. It does not certify model inference, omitted windows, native
masks, spatial overlap, statistical generalization or outside participation.

For a new local MPS experiment, first reconstruct the existing v6/v7 sources and
permitted assets using their sections below. Preserve their original files.
Set `PYTHONPATH=backend`, `VERITAS_MODELS_DIR` and `VERITAS_HF_HOME` to the
reviewed local caches and enable offline mode. The new generation stages need
the isolated hash-locked v7 renderer library path appended to `PYTHONPATH`;
learning/comparison use the app's existing runtime. Then run:

1. `prepare_mouth_learning_v8.py plan <v6-training-plan> <new-v8-plan> --parent-manifest <v6-still-manifest>`.
2. `prepare_mouth_learning_v8.py build <new-v8-plan> <new-v8-inputs> --asset-manifest docs/evaluation-panels/mouth-v7-generator-assets.json --assets <v7-assets>`.
3. With `lipsync-v4`, `lipsync-v6`, `lipsync-v7` and the new `lipsync-v8` under one data root, run
   `run_mouth_learning_v8.py freeze <data-root> <new-v8-experiment>`, then its
   `cache <lock> <new-cache>` stage.
4. Run `audit_mouth_learning_v8.py <new-v8-plan> <lock> <cache> <new-audit> --asset-manifest <asset-manifest> --assets <v7-assets>` and require all planned cases.
5. Run `run_mouth_learning_v8.py train <lock> <new-trained-dir> --cache <cache>`,
   then `score <lock> <new-score-jsonl> --head <trained-dir/head.json>`, followed
   by `review <lock> <new-summary> --receipt <score-jsonl>`.
6. Run `audit_mouth_video_v8.py <lock> <score-jsonl> <new-video-audit>` and
   `replay_mouth_learning_v8.py <lock> <cache> <head> <new-forward-audit>`.

These script names are under `backend/scripts/`, run with the backend Python
environment. Each output directory/file is new; failed receipts remain
separate. Physical root paths are part of private manifest bytes, so a new
location produces new container receipt hashes. Public source/pixel/tensor
commitments identify the original run; a reconstruction is a new run, not a
claim of original receipt-byte identity. Cross-device training parity remains
unverified. No customer or cloud service is started by these research stages.
# Source-matched mouth experiment (v9)

The [v9 comparison](../mouth-paired-v9-evaluation.md) failed its frozen gates
and lost the earlier candidate's gain. It is not enabled. Its public metadata
retains the complete 88-case, three-arm comparison, 768 source-matched fitting
pairs, 4,096 teacher fitting targets, and source/full-forward audits.

Standard Python, without ML packages or weights, can replay retained-window
arithmetic, all pair/source-role correspondences, and teacher payload hashes:

```sh
python3 backend/scripts/review_mouth_paired_v9.py docs /tmp/mouth-v9-review.json
python3 backend/scripts/diagnose_mouth_ranking_v9.py \
  docs/evaluation-panels/mouth-v9-review-responses.json.gz \
  /tmp/mouth-v9-ranking-diagnostic.json
```

Outputs must be new files. The second command is an explicitly post-hoc
consumed-data bound, not an adopted point or fresh false-alarm claim. These
tools do not replay neural inference, omitted windows, masks/spatial IoU or
generalization, and local reproduction is not outside review.

For a new local experiment using the existing backend environment, offline
reviewed model caches and immutable V8 inputs, run the following stages of
`backend/scripts/run_mouth_paired_v9.py`: `freeze <v8-experiment> <new-v9-dir>`,
`train <v9-lock> <new-trained-dir>`, `score <v9-lock> <new-score-jsonl> --head
<v9-head>`, and `review <v9-lock> <new-summary> --receipt <v9-scores>`.
The lock is `<new-v9-dir>/lock.json`; MPS is explicit, with no silent fallback.
Run `audit_mouth_pairs_v9.py <v9-lock> <v9-head> <new-fit-replay>` and
`audit_mouth_paired_v9.py forward <v9-lock> <v9-head> <new-forward-replay>`;
its `video <v9-lock> <v9-scores> <new-video-audit>` stage replays native inputs.
The unchanged `/8` source/forward audit formats bind the V9 lock/head; they do
not indicate use of a V8 candidate at inference. Public source data retain
their separate ODbL/DbCL terms. Media and learned weights stay outside Git.
No customer or cloud service starts through these research stages.
# Surrounding-footage mouth evidence (v10)

The [V10 comparison](../mouth-context-v10-evaluation.md) recovers more edit
locations but fails both frozen ordinary-flag gates. Neither arm is enabled.
It preserves24 ordinary calibration inputs, complete112-case/two-head raw
inputs/context schedules and every eligible window, not just the top12.

Anyone with standard Python can reconstruct every eligible context window,
point selection, old raw-response correspondence and combined arm decisions:

```sh
python3 backend/scripts/review_mouth_context_v10.py docs /tmp/mouth-v10-review.json
```

The output must be new. Expected:24 calibration/88 development cases,5,626
windows, primary21 locations/12 ordinary flags, secondary26/24. This arithmetic
review does not prove neural inference, original pixels/masks/spatial IoU,
generalization or outside participation. Existing metadata retain their
separate CREMA-D ODbL/DbCL terms. Media and checkpoint binaries remain outside Git.

To reconstruct a new local research run, preserve V6–V9 inputs/receipts and the
permitted face task. Use the backend Python, offline model/cache settings and
`PYTHONPATH=backend`; ordinary construction/source auditing additionally needs
the existing isolated `video-v3/av/mesh-runtime` on that path (MediaPipe0.10.32).
It is not installed into the app. Source picture/audio offsets are explicit;
never loop/pad missing context or assume picture index equals audio time.

Run `prepare_mouth_context_v10.py plan <v6-source-plan> <new-v10-plan> --task
<verified-face-task>`, then `build <new-plan> <new-calibration-inputs>`. Require
all24 ready and run `audit_mouth_context_v10.py <plan> <manifest> <new-source-audit>`.
The frozen experiment layout expects these as `lipsync-v10/calibration-plan.json`,
`calibration-inputs-v2/manifest.json` and `calibration-source-audit.json` under
the common independent-evaluation root. The initial `calibration-inputs` failure
is preserved separately; a new reconstruction must use new output directories.

Run stages of `backend/scripts/run_mouth_context_v10.py`: `freeze <common-root>
<new-experiment-dir>`, `calibration <lock> <new-calibration-scores>`, `calibrate
<lock> <new-points> --receipt <calibration-scores>`, `development <lock>
<new-development-scores>`, and `review <lock> <new-summary> --receipt <development-scores>
--points <points>`. Learned scoring requires explicit local MPS; there is no
silent fallback or new head training. Run `audit_mouth_context_inputs_v10.py
<lock> <calibration-scores> <development-scores> <new-native-audit>` to replay all
source bindings. Physical root paths bind private manifest bytes, so a new run
has new receipt hashes. No customer or cloud service starts through these stages.

# Complete-video temporal learner (v11, rejected)

The [completed experiment](../mouth-temporal-v11-evaluation.md) describes
all160 audited fitting clips from sixteen actors. The compressed
`mouth-v11-preparation.json.gz` includes the exact source links/roles/common
clocks, masks' hashes and geometry/conditioning bindings, runtime declarations,
frozen method hashes and160 complete audit outcomes. It contains no source
pictures, audio or checkpoints. Known construction masks are evaluation truth,
not a detector's pixel attribution. The learner located32/32 known edits but
flagged22/56 ordinary clips, versus14/32 and0/56 for retained V8. It remains
disabled under the frozen false-flag requirements; no fresh confirmation was
scored. The complete fitting/result capsule is published too.

Review its metadata consistency without application dependencies or models:

```sh
python3 backend/scripts/review_mouth_temporal_preparation_v11.py docs/evaluation-panels/mouth-v11-preparation.json.gz
```

The [local review receipt](mouth-v11-preparation-review.json) checks all160
source-condition/audit records, original and encoded clocks, exact known edit
positions, sample offsets, source-bound seeds and construction input bindings.
It does not decode source bytes, execute the generators or prove label truth.
Six focused tests include corrupted offset, timestamp, role, condition and seed
cases. Local reproduction is not evidence of an outside reviewer participating.

To reconstruct, provision the existing V4 permitted portrait/mesh assets, V7
speech-rendering assets and isolated runtime, and the unchanged V6 source-role
plan. With the application environment and relevant isolated runtime paths,
run `prepare_mouth_temporal_v11.py plan <v6-plan> <new-plan>
--asset-manifest <v7-assets-json> --portrait-root <v4-assets-root>` then
`build <new-plan> <new-input-dir> --speech-assets <v7-asset-dir>`.
Run `audit_mouth_temporal_source_v11.py <plan> <manifest> <v7-asset-dir>
<new-source-audit>`. New outputs are exclusive; do not overwrite a failed run.
The comparison expects `lipsync-v11/training-plan.json`,
`training-inputs/manifest.json` and `training-source-audit.json` under the common
independent-evaluation root. Native MPS generation/replay is explicit.

`run_mouth_temporal_v11.py freeze <common-root> <new-experiment>` binds the
method before new responses. The `fit`, `calibration` and `development` stages
take `<lock> <new-feature-receipt>`; fit uses the sixteen fitting actors,
calibration uses the24 ordinary V10 point clips, and development reuses the88
consumed clips. The latter two reuse immutable audited neural responses and
freshly replay native pixels/chips and measure geometry. Run `train <lock>
<new-head> --fit <fit-features> --calibration <point-features>` only after the
source audit, then `review <lock> <new-summary> --head <head> --receipt
<development-features>`. All160 fitting,24 point and88 development cases have
completed; each new reconstruction must retain every planned source/condition.

`export_mouth_temporal_v11.py <v11-root> <destination> --preparation-only`
exports the preparation metadata; the default full export includes the fitting
capsule. Review all6373 eligible windows, feature/normalizer/decision arithmetic
and optionally refit the49-variable classifier on CPU:

```sh
PYTHONPATH=backend python3 -m scripts.review_mouth_temporal_v11 docs/evaluation-panels/mouth-v11-fitting-review.json.gz
PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_mouth_temporal_v11 docs/evaluation-panels/mouth-v11-fitting-review.json.gz --refit
```

The first command needs only the standard library; refitting additionally needs
NumPy/scikit-learn. The local refit matched all coefficients exactly. These
reviews do not rerun the vision networks, physically inspect source masks or
establish outside participation. Raw responses are not authenticity probabilities.

`diagnose_mouth_temporal_ranking_v11.py <capsule> <new-output>` reconstructs the
explicitly post-hoc scalar-ranking bound. Preserving V8 first and allowing<=1
ordinary flag retains at most16/32 locations, including only two first-family
edits. This is not a new calibrated review point or a way to rescue the failed
experiment. The [record](mouth-v11-ranking-bound.json) remains separate from the
frozen [full outcome](../mouth-temporal-v11-evaluation.json).

## Reproduce source-separated audio v7 fitting and decisions

The [v7 development result](../audio-guard-v7-evaluation.md) passed its frozen
development gates against the actual guarded service. The candidate remains
disabled pending a separately frozen fresh comparison. This is not a rescue
of V4–V6 or an outside validation claim.

`audio-v7-fitting-review.json.gz` contains all1,440 fit/calibration/development
feature rows, source metadata and hashed known text connections, fitted
projection/decision parameters, class-set calibration groups, all720 development
outcomes and297 stability-probe representations. Waveforms, full transcripts,
private paths and pretrained neural checkpoints are excluded. Source attribution
and separate dataset rights still apply.

With the declared NumPy/SciPy/scikit-learn runtime, reconstruct all source roles,
independent guard arithmetic, exact serving comparator, conclusions, complete
operational counts and gates, and optionally refit the small decision head:

```sh
PYTHONHASHSEED=0 PYTHONPATH=backend backend/.venv/bin/python -m scripts.review_audio_guard_v7 docs/evaluation-panels/audio-v7-fitting-review.json.gz /tmp/audio-v7-review.json --refit
```

The output must be a new file. Omit `--refit` to skip fitting/bootstrap replay.
`PYTHONHASHSEED=0` is part of the recorded statistical runtime: the inherited
bootstrap helper's set-based strata order is otherwise process-dependent.
The capsule records the exact package versions and source implementation pins.
Its SHA-256 is `37d775c639b6105ca7b451fbe9e2b6aa117fabace7a7646e55c6c58aeb814bee`.

The local CPU receipt is `audio-v7-fitting-replay.json`; all coefficients matched
exactly. This shares the declared fitter/projection/bootstrap, with separately
reconstructed guard/decision arithmetic. It does not decode audio or rerun
neural inference. `audio-v7-source-audit.json` records the separate audit of720
source/PCM bindings,297 regenerated probe views and six sampled neural forwards.
`audio-v7-tamper-checks.json` records four targeted integrity rejection checks.
None of these local scripts establishes outside participation or generalization.

To rebuild from separately provisioned licensed inputs, use
`prepare_audio_guard_v7.py` to create/check a frozen manifest,
`run_audio_guard_v7.py extract` for representations, then its `train` and
`evaluate` stages, keeping fitting and calibration roles disjoint. Consult each
CLI's help and the [fixed protocol](../audio-guard-v7-development.md); use new
output paths and preserve every original failed/unassessed case. Source replay
uses `audit_audio_guard_v7.py` with the original manifest/representation/head/
score bindings. Existing consumed inputs are not fresh test sources.
