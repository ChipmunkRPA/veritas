# Public evaluation source references

These are public dataset references, selection metadata, file hashes, and known
edit positions for the new local comparisons. **No source media, customer uploads,
or model checkpoints are included.** They let independent reviewers identify the
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
