# Temporal texture on whole-clip face swaps

This comparison tests whether rapid local changes around the cheeks and
face boundary add a useful check when a swap occupies the entire clip.
The serving comparison is Chipmunk Verify's existing paired face review
on six verified interior pictures. Recent work on pixel-wise temporal
frequency motivated testing temporal texture, while a November
compression study motivated encoding each source at two delivery levels.
The experiments use a separate implementation and the app's pinned
commercially licensed face locator. [Temporal-frequency paper](https://arxiv.org/abs/2507.02398)
· [November H.264 study](https://arxiv.org/html/2608.28685v2)
· [Face locator license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Sources and frozen comparison

Select 36 fresh RTFS targets with both publisher-rendered swap variants
by a fixed hash order. The roster excludes 258 previously recorded target
groups and their known donor keys. It binds each original and swap to
its publisher SHA-256 and byte count. Fourteen distinct target groups
supply development and 22 supply the holdout. The publisher supplies
the swap labels and reports success. Published source and donor keys
do not independently establish biological identity or training-set
separation. [Pinned source card and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
· [Earlier source qualification](video-common-source-preparation-v89.md)

Starting at 0.40 seconds, select the nearest verified native picture for
each of 100 output times, scaled and padded to 640 × 360. Encode H.264
at CRF 23 and re-encode the result at CRF 36. All 72 target-to-swap
comparisons retained the same selected-frame repetition pattern and
matched relative source times within 4 ms. The two encodes reuse the same
subjects and are paired measurements.

The candidate examines every second delivered picture. It aligns a
tracked face by its eye positions and compares rapid luminance and
chroma changes in two inner and two outer cheek regions. The first
view uses that alignment directly. The second uses optical flow to
reduce neighboring-picture motion within the aligned face. A third
combines both views. Each fit requires at least two fully measured
windows and uses source-matched development contrasts between the
original and the corresponding swap. Ordinary source-group maxima
from leave-one-group-out scoring supply both comparison points. The
protocol also freezes fits on each generator separately for transfer
to the other. The current paired face check uses its saved operating point.

## Held results

| Holdout delivery | Paired face check, swaps flagged | Paired face check, ordinary flags | Temporal swaps flagged | Temporal unassessed |
| --- | ---: | ---: | ---: | ---: |
| Direct CRF 23 | 35/44 | 5/22 | 0/44 | 3/66 |
| Tandem CRF 36 | 22/44 | 6/22 | 0/44 | 3/66 |

All nine temporal fits found zero swaps at both frozen points. Every
optical-flow fit added the same single ordinary alert at the wider
point on a tandem delivery. The other arms added zero ordinary flags.
All six unassessed deliveries came from the same source group. This
leaves the temporal measurements in evaluation.

The paired face check assessed all 132 held deliveries. Of the 44 swaps,
18 flagged in both encodes, 17 in the direct encode only and four in
the tandem encode only. Five remained unflagged in both. One ordinary
source flagged in both encodes, four in the direct encode only and
five in the tandem encode only. The paired losses identify strong
recompression as a practical gap in the current sampled-face check.
The data covers two publisher engines and one fixed four-second
delivery recipe. The study measures detection on whole-clip labels.

The independent pixel audit checked 21,600 delivery-frame timestamps
and 3,150 face-window RGB hashes across all 216 deliveries. It
compared source cadence separately. The compact receipt contains
the measured response values, frame indexes, hashes and aggregate
counts, with no raw videos, crops or face boxes.
[Numeric results and frozen protocol](evaluation-panels/video-whole-face-temporal-transfer-2026-11-16.json.gz)

## Replay

The portable comparison refits both development operating points and
recounts the paired decisions on Python 3.12 and 3.14.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-whole-face-temporal-transfer-2026-11-16.json.gz
python3 -B -S -m scripts.video_face_temporal_fit verify "$receipt"
```

For native replay, use the pinned official publisher file index and
the media SHA-256 roster to restore exact source paths. Then execute
the preparation and scoring commands in a fresh private output
directory. The matching code uses local FFmpeg and OpenCV. The incumbent
replay uses the verified installed weights on CPU or Mac GPU.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runner=backend/.venv/bin/python
repository=https://huggingface.co/api/datasets/godmodes/rtfs-10k
revision=bca865e675852666a426368b2f5614abf1c1926a
curl -fsSL "$repository/revision/$revision?blobs=true" -o "$run/publisher.json"
python3 -B -S -m scripts.audit_video_face_temporal restore "$receipt" "$run/publisher.json" "$run/roster.json"
"$runner" -B -m scripts.video_face_temporal_panel acquire "$run/roster.json" "$run/source"
"$runner" -B -m scripts.video_face_temporal_delivery "$run/roster.json" "$run/source" "$run/delivered"
"$runner" -B -m scripts.video_face_temporal_readout "$run/delivered" "$run/dev.json" development
python3 -B -S -m scripts.video_face_temporal_fit freeze "$run/dev.json" "$run/freeze.json"
"$runner" -B -m scripts.video_face_temporal_readout "$run/delivered" "$run/held.json" held
"$runner" -B -m scripts.video_face_temporal_reference "$run/delivered" "$run/reference.jsonl" held
python3 -B -S -m scripts.audit_video_face_temporal pixels "$run/delivered" "$run/dev.json" "$run/held.json" "$run/audit.json"
python3 -B -S -m scripts.video_face_temporal_fit summarize "$run/dev.json" "$run/held.json" "$run/freeze.json" "$run/reference.jsonl" "$run/result.json.gz"
```
