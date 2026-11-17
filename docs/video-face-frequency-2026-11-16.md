# Facial detail and appearance after strong video recompression

The follow-up combines matched facial detail and local appearance with
the app's two existing face readings. It detected ten additional swaps
after a strong second encode, with one additional matched original
flagged and zero added alerts on twelve separate camera recordings.

## Method and calibration

The comparison schedules 16 source pictures across four seconds. It
keeps the longest continuous track with at least eight usable aligned
face crops. A three-level Haar transform measures luminance and color
detail inside the face, around its boundary and in nearby context.
The relative version measures differences among these regions within
each frame. The appearance version compares patch features across the
same regions. Both views use the same track. No new pretrained weights
were introduced. [Frequency-domain research](https://openaccess.thecvf.com/content/CVPR2026W/PPMisDet/html/Ghosh_WGN_Wavelet-Guided_Network_for_Efficient_and_Generalised_Deepfake_Detection_CVPRW_2026_paper.html)
· [Recent appearance and residual research](https://arxiv.org/html/2611.12508v1)
· [Pinned Apache-2.0 feature encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

Sixty earlier source groups supplied matched training pairs. The fit
weights both classes and gives each source equal influence within its
class. It also fits the score difference between original and swapped
variants of the same source and encode. Fifty-four separate paired
source groups and eight ordinary camera groups selected the fallback
and set its points. The study compared five feature combinations with
and without qualification by the observed median inter-picture luma
quantization. It froze the favored combination before fresh scores
were measured. The selected method combines the earlier two scalar
readings, relative appearance, and both detail measures. It uses a
shared calibration across the two coding qualities. [Earlier face-detail study](video-face-detail-comparison-2026-11-16.md)
· [Native quantization and reference](video-face-quantization-calibration-2026-11-16.md)

## Fresh local results

Thirty-two fresh publisher targets supplied one original and two
swaps each. The fixed source selection excludes recorded target and
donor keys, source links, and hashes from earlier panels. Identities
or recordings seen during pretrained model development are unknown.
Each input was delivered as upright 640 × 360 H.264 MP4 at 25 fps with
100 verified native pictures. The first encode uses CRF 23 and the
second uses CRF 36 on those bytes. Twelve separate ordinary camera
recordings add 24 controls. These cameras were used previously for the
retiming study and supplied no measurements to this face fit or calibration.
[Publisher swap dataset and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
· [Camera recording terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

| Review | Direct swaps /64 | Strongly recompressed swaps /64 | Direct matched originals /32 | Strongly recompressed originals /32 | Camera flags, direct / recompressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current paired face review | 53 | 27 | 5 | 5 | 3 / 2 |
| Current plus selected check, stricter | 53 | 29 | 6 | 5 | 3 / 2 |
| Current plus selected check, wider | 54 | 37 | 6 | 6 | 3 / 2 |
| Replace the current review when the selected check can assess | 53 | 37 | 4 | 6 | 2 / 2 |

Both renderers benefit under strong recompression. The gain is six of
32 for the first renderer and four of 32 for the second. Twenty-seven
strongly recompressed swaps still go unflagged at the wider point. The
group-resampled difference on this panel is +15.6 percentage points,
with a 7.8 to 25.0 point interval. The matched-original difference is
+3.1 points, with a 0 to 9.4 point interval. The intervals describe this
panel and its two related encodes. The comparison measures whole-clip
swaps and lacks labeled starts and stops for partial swaps.
[Frozen outcomes and exact frame commitments](evaluation-panels/video-face-frequency-2026-11-16.json.gz)

An independent native readback verified 96 publisher asset lengths and
digests, 9,600 source-picture mappings, 21,600 delivered timestamps
and slice quantization readings, and 3,343 decoded RGB anchor hashes.
All 64 original-to-swap source pairs had comparable picture cadence.
The released receipt carries derived scores, calibration, timing and
hashes. Face crops and intermediate embeddings stay with the local run.

## Local application

`VERITAS_VIDEO_WHOLE_FACE_DETAIL_REVIEW=true` enables the fallback in
development alongside `VERITAS_VIDEO_WHOLE_FACE_REVIEW` and
`VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW`. Both earlier checks must
finish without a flag. The finding gives the measured native timestamps
and three original frames with face crops. The check contributes zero
weight to the combined AI assessment. The earlier checks can serve on
their own by turning off the fallback setting.

Five simulated USD 9.99 purchases exercised the normal worker with the
face checks selected. No analysis queued before simulated payment. The
two recovered swaps both displayed 16 source positions from **0.200 to
3.800 s**, with original frames and face crops at **5, 53 and 95**.
The browser sought to **2.120 s**. The PDF and JSON matched the API, and
the signed verifier reproduced each aligned input from the original
video. A matched original produced a completed face-detail check with
no flag. A previously flagged swap kept its earlier finding. The
on/off replay on the same file agreed on the overall assessment.
[Local checkout and export receipt](evaluation-panels/video-face-frequency-app-2026-11-16.json)

The app shares its crop and feature extraction with the benchmark.
Replaying that extractor reproduced all 216 held and camera readings.
The source inspection sorts frames by native index before joining
the face track, including when some frames are already cached.

## Independent replay

The standard-library reader recalculates the decisions and grouped
comparisons. The export creates a manifest, both frozen decision
receipts and a source-group comparison for `scripts.independent_review`.
A reviewer can use the same manifest to submit an additional method
or identify coverage disagreements. These are labeled benchmark inputs.
An independent team should use a separate holdout when choosing methods.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-frequency-2026-11-16.json.gz
python3 -B -S -m scripts.video_face_frequency_fit verify "$receipt"
python3 -B -S -m scripts.video_face_frequency_review export "$receipt" "$run/independent"
review="$run/independent"
python3 -B -S -m scripts.independent_review compare "$review/manifest.json" \
  "$review/baseline-confirmation.json" "$review/baseline-freeze.json" \
  "$review/candidate-confirmation.json" "$review/candidate-freeze.json" \
  "$run/comparison.json" --require-freeze-commitment
```

For a native replay, restore `source_roster` from the receipt with
`scripts.audit_video_face_temporal.restore_roster` and the pinned
publisher file index used in the earlier study. Use the restored roster
with `scripts.video_face_temporal_panel acquire` and
`scripts.video_face_temporal_delivery`. Add each local camera input's
`path` to `camera_roster` and build the camera panel with
`scripts.video_face_frequency_readout cameras`. Use the model registry's
pinned snapshots to rerun measurements. The reported run used MPS for
the learned feature extractor and CUDA-free source decoding on Apple
Silicon. The native audit also runs without any model weights.
[Pinned source index and earlier readout commands](video-face-detail-comparison-2026-11-16.md#replay)
· [Frozen research implementation](https://github.com/ChipmunkRPA/veritas/commit/dcce309707e41083a96742d0ede90364f5ecf5fa)

```sh
runner=backend/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
for panel in held camera; do
  "$runner" -B -m scripts.video_face_frequency_readout measure "$run/$panel" "$run/$panel.jsonl" --device mps
  VERITAS_VIDEO_WHOLE_FACE_SAME_FRAME_REVIEW=false "$runner" -B -m scripts.video_face_temporal_reference \
    "$run/$panel" "$run/$panel-reference.jsonl" held --device mps
done
python3 -B -S -m scripts.video_face_frequency_review native "$run" "$run/native-audit.json"
npm --prefix frontend run build
"$runner" -B -m scripts.video_face_frequency_app "$run/held" "$receipt" "$local/paid" --browser
```
