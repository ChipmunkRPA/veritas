# Short face changes within a continuous shot

The app already compares pooled face appearances before, during and after a candidate interval. This experiment asks whether many individual face comparisons support the same temporary change in a continuous shot. It focuses on 0.375-, 0.50- and 0.75-second swaps. Earlier tests of jaw texture and color transitions produced little supporting evidence. [Current appearance method](video-v16-sface-development-protocol.md) · [Previous boundary color comparison](video-chroma-boundary-transfer-2026-11-11.md) · [Identity-aware video research](https://arxiv.org/abs/2012.02512) · [Face consistency research](https://arxiv.org/abs/2203.01318)

## Frozen comparison

The current arm uses the published appearance window and point with quarter-second centers and their immediate native neighbors. The 8 Hz centroid arm uses the same pooled appearance formula on the new short windows. The rank arm uses the same dense samples and short windows as the centroid arm. It compares each core face with individual faces in each flank, then uses lower-quartile distances to require persistent separation. It subtracts the largest upper-quartile dispersion from within each period and between the two flanks. This controls for noisy individual crops and temporary differences without a return to the earlier appearance. The local study uses the existing commercially licensed face detection and aligned representation and loads no new model. [Face detection terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE) · [Weight-inclusive face representation terms](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/README.md)

The guarded rank arm compares textured scene tiles outside the detected face on each side of both candidate boundaries. It removes each tile’s average brightness and requires at least eight comparable tiles. A low median correlation removes that candidate window from consideration. A window with insufficient scene support is unavailable for the guarded arm. The unguarded rank arm shares the same point and shows the effect of applying the scene check. The study compares temporal face observations and does not establish a person’s identity. The exact region recorded in the private evidence is the detected face box, since the reference labels only the edited time.

Both new scores seal the largest measured response from 45 older genuine views before the held run. The current point stays fixed. All arms use source-verified frame clocks, and the short-window arms use half-open intervals ending at the first native sample after the core. The current comparator retains its existing reported endpoints. The evaluator counts a flag as localized when its intersection-over-union with the known edit is at least 0.50. This comparator exercises the current appearance calculation with a single observed face and omits the full app tracker and fallback. [Frozen protocol](evaluation-panels/video-face-return-transfer-freeze-2026-11-11.json)

## Sources

Six held RTFS groups have distinct known source links and donor keys and fall outside five prior localization rosters. They supply 36 swaps from two publisher families and 60 matched unchanged and ordinary light, blur and time-step controls. Two separate camera sources supply six unchanged, chroma-converted and recompressed deliveries. Six additional control views briefly switch between two genuine videos of different people and return to the first. Each cut pair has exact matching native clocks and dimensions, and the cut views share the held RTFS sources. The held footage has prior project-level exposure outside this method. Publisher rights and donor-linkage limitations are documented in the source review. [Native edits](video-native-inserts-v95-method.md) · [RTFS source review](video-common-source-preparation-v89.md) · [Camera data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Results

| Arm | Correct 0.375s intervals | Correct 0.50s intervals | Correct 0.75s intervals | Ordinary false flags | Source-cut false flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current appearance | 0/12 | 10/12 | 10/12 | 0/66 | 4/6 |
| 8 Hz centroid | 4/12 | 0/12 | 0/12 | 0/66 | 0/6 |
| 8 Hz rank with scene check | 10/12 | 4/12 | 0/12 | 42/66 | 3/6 |
| 8 Hz rank without scene check | 10/12 | 4/12 | 0/12 | 42/66 | 3/6 |

All four arms had six unassessed swaps. The current method had 13 unassessed ordinary controls and each new arm had ten. All arms had one unassessed cut. The guarded rank arm selected eight wrong intervals in swap views. The scene check discarded 32 candidate windows and final outcomes matched the unguarded rank arm. The new arm missed the frozen advancement criteria. [Exact results and sealed points](evaluation-panels/video-face-return-transfer-2026-11-11.json)

The ten correct shortest intervals came from five source groups. Each had 0.80 intersection-over-union, with absolute start and end errors of 0.04 seconds. Three private before, during and after face-crop exhibits were independently verified against their original frame hashes and timestamps. The frozen rank point was below zero. In a post-hoc diagnostic, requiring a positive margin preserved all ten of the shortest hits and three half-second hits, and removed all rank false flags. This diagnostic motivates independent testing of the shortest-range cue. The frozen comparison remains the result for these sources.

The two local CPU phases verified 3,715 native pictures and clocks in 672 seconds, with 332 MiB peak worker memory. A separate standard-library audit reproduced the 45 development and 108 held outcomes, 361 selected intervals and 819 exact frame-anchor checks. Six focused tests cover face-pair separation, the half-second timing case, scene cuts, missing scene support, brightness shifts and classification outcomes.

## Reproduce

Use the retained archive and private directories containing the earlier frozen rosters. The local journal retains frame indices and clocks. The public receipts contain aggregate counts and hashes.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_return_transfer freeze "$run_dir" \
  --prior "$texture_run" --prior "$boundary_run" --prior "$short_boundary_run" \
  --prior "$landmark_run" --prior "$chroma_run"
for stage in development seal held readout; do
  backend/.venv/bin/python -B -m scripts.video_face_return_transfer "$stage" "$run_dir" || break
done
```
