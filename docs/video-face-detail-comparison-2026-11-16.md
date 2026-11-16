# Facial detail after video recompression

Strong recompression hides many of the face swaps that the local app can
flag on a first encode. The earlier fallback recovered six of the 42
heavily recompressed swaps the original face checks missed. This comparison
asks whether changes in facial detail supply a separate signal alongside
those checks. [Current face review and earlier comparison](video-face-quantization-calibration-2026-11-16.md)

## Method

Use 16 scheduled pictures on the native presentation clock. Keep the
longest uninterrupted run in which adjacent sampled face boxes overlap
and two detected eye positions give enough support for the same crop geometry. A
run needs at least eight usable crops. Sample spacing is 240 ms. The
whole-clip response cannot determine when a face swap started or ended.

Use two intermediate layers of a frozen patch encoder trained on images.
Take measurements inside the sampled face, through a surrounding ring,
and in a narrow band of wider context. Compare patch appearance,
differences between spatial smoothing scales, and differences between
temporal smoothing scales and adjacent samples. These measurements
evaluate facial detail and how it changes across space and time. The
approach draws on a recent study of learned features and multiscale
video residuals. The extractor, crop support, feature projections,
and fitted heads are local implementations. [MARS paper](https://arxiv.org/html/2611.12508v1)
· [Pinned patch encoder and licence](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
· [Face detector licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The fit uses 60 existing source groups, two swap families and matched
original footage. Simple regularized linear heads learn from whole-clip
measurements and from differences within matched original and edited
clips at identical delivery settings. The 30 separate sources for
choosing operating points come from the earlier compression comparison.
The comparison includes the new measures alone, the two existing readings
alone, and combinations. The wider fallback can add at most two ordinary
source groups during point selection. The stricter point can add zero.
Selection ranks added swaps under strong recompression first. This
selects the combined appearance, spatial and temporal measurements as
the new signal before measuring the fresh holdout.

## Held sources and verification

The fresh panel contains 24 source groups, each with one ordinary video
and publisher-created swaps from two generators. The fixed order excludes
earlier project source identifiers, known donor keys, source links and
file hashes. Donor keys come from recorded filenames. Separation of
people's identities or of the pretrained encoders' original training
data remains unknown. Each item is delivered at 640 × 360 and 25 fps,
first at H.264 CRF 23 and then from that encode at CRF 36. The eight
separate camera recordings are ordinary negative controls with prior
project exposure and zero fitting or point-selection observations in
this comparison. [Swap dataset card and licence](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
· [Camera source qualification](video-vcd2-matched-controls-v290-result.md)

A separate FFprobe and FFmpeg replay checked 72 acquired asset lengths
and hashes, 7,200 source picture mappings, 16,000 delivered picture
timestamps, and the original RGB hashes of all 2,560 inspected frames.
Both variants for every target had matching source cadence. Each pair
of re-encodes shares its source history, so report counts by encode
and track independent gains by source group.

## Local results

The current check and the selected facial detail check assessed all 144
fresh deliveries and all 16 camera deliveries. Ordinary alerts are false
alarms on the labeled footage. Each fresh column has 48 swaps and 24
ordinary videos. The camera columns each have eight ordinary videos.

| Comparison | Direct swaps | Strongly recompressed swaps | Direct ordinary alerts | Strongly recompressed ordinary alerts | Camera alerts, direct / recompressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current app check | 33 | 23 | 6 | 6 | 4 / 3 |
| Facial detail alone at the matched selection point | 26 | 28 | 5 | 7 | 3 / 4 |
| Current plus facial detail, stricter setting | 33 | 23 | 6 | 6 | 4 / 3 |
| Current plus facial detail, wider setting | 35 | 28 | 9 | 8 | 5 / 4 |

At the wider setting the five added alerts on strongly recompressed
swaps span four targets and both generators. Twenty strongly recompressed
swaps still go unflagged. Across both delivery settings, the addition
flags seven more swap deliveries, five more ordinary deliveries from
three fresh targets, and one separate camera recording in both encodes.
That false-alarm increase matters alongside the two added swap alerts
and three added ordinary alerts on the direct deliveries. The stricter
point produces zero added alerts. The app keeps its current face checks.

The appearance-only arm warrants separate testing for strongly
recompressed input. On those deliveries, its standalone head flagged
29 swaps and seven originals. Direct deliveries fell to 20 swap alerts.
Its stricter fallback added four swaps and one ordinary alert on strongly
recompressed delivery, plus one camera alert. These are secondary held
observations. A quality-specific rule would need its own separate panel
with more ordinary videos and different renderers.
[Per-clip comparison and native audit](evaluation-panels/video-face-detail-comparison-2026-11-16.json.gz)

## Replay

The lightweight reader rebuilds the selection points, reconstructs the
current face decisions from both raw readings, and recounts the held
and camera panels. The receipt contains scalar measurements, fitted
head parameters, source hashes, and native sample positions. Local
face images, coordinates, and intermediate features stay in the run
directory.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-detail-comparison-2026-11-16.json.gz
python3 -B -S -m scripts.video_face_residual_fit verify "$receipt"
```

Use the [earlier preparation commands](video-face-quantization-calibration-2026-11-16.md#replay)
to restore the two fitting banks, operating-point deliveries and
separate camera controls. Restore the public held roster using the
pinned publisher file index with `scripts.audit_video_face_temporal
restore`. Put that roster at `$study/fresh-roster.json`, the acquired
sources in `$study/fresh-sources`, and the rendered deliveries in
`$study/fresh`. Recreating exactly the private roster digest also
requires the pinned metadata and its retained donor history, using
`scripts.video_face_residual_sources`. Keep those private inputs outside
the published receipt.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
runner=backend/.venv/bin/python
previous=docs/evaluation-panels/video-face-quantization-calibration-2026-11-16.json.gz
"$runner" -B -m scripts.video_face_residual_readout "$first" "$study/fit-a.jsonl"
"$runner" -B -m scripts.video_face_residual_readout "$second" "$study/fit-b.jsonl"
"$runner" -B -m scripts.video_face_residual_readout "$selection" "$study/selection.jsonl"
"$runner" -B -m scripts.video_face_residual_fit freeze "$study" \
  --first "$first" --second "$second" --selection "$selection" --previous "$previous"
"$runner" -B -m scripts.video_face_residual_readout "$study/fresh" "$study/held.jsonl"
"$runner" -B -m scripts.video_face_residual_readout "$camera" "$study/camera.jsonl"
VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW=false \
  "$runner" -B -m scripts.video_face_temporal_reference "$study/fresh" "$study/fresh-reference.jsonl" held
"$runner" -B -m scripts.video_face_residual_fit assess "$study" --previous "$previous" --camera "$camera"
python3 -B -S -m scripts.audit_video_face_residuals native "$study" "$camera" "$study/native-audit.json"
```
