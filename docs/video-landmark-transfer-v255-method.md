# Landmark support transfer on retained selection packets

Check whether eye separation and facial geometry changes occur in existing flags on constructed swaps as well as in ordinary control footage. The 23 source groups are the consumed v112 selection split. Each contributes an unchanged clip, two publisher-labelled swap variants, and lighting, blur, and time-step controls. There are 138 clips and five fixed windows per clip. The separate webcam discovery clips and reserved confirmation groups are outside this readout. [Selection report](video-native-appearance-v112-result.md)

Before measuring geometry, verify the exact v112 public execution and readout receipts, the eight successful extraction profiles and journals, the per-batch source bindings, and the derived plan. Replay the historical metadata and scalar verification to bind the recorded picture times, sampled-frame hashes, face eligibility and strict current comparison point. The original weighting and score are used only to identify existing flags. The new readout uses the saved box and landmarks with zero new inference and no media decode.

Measure the same six before, nine during and six after sampled frames as the frozen appearance check. Preserve unassessed windows and failed geometry measurements as missing. Compute median eye separation divided by face-box width for each phase, the minimum during ratio, and the during median minus the mean of the two flank medians. Center, rotate and scale the landmark shapes using their eye points. Take the coordinate-wise median shape per phase and report root-mean-square displacement between the before and during shapes, after and during shapes, and the two flanks. The normalized distances may be inflated when the eye points are unusually close. Summarize these continuous measurements by condition across source-group medians. Report coverage for all windows and for windows flagged by the current check.

For each already-flagged swap window, compare geometry with the same source group and exactly scheduled window in each control. Report paired counts, missing endpoints and signed differences for each measurement. Repeat comparisons from the same source group share footage and are dependent. This comparison estimates neither false-flag reduction nor production coverage. The fixed v112 checks had no false flags on these controls, and the archived selection path has a single-visible-face contract and fixed sparse windows. No threshold, model, head, flag or serving decision changes.

The historical records classified the model weights as commercially usable. The source panel still has portrait attribution gaps and unknown pretraining overlap. Only aggregate counts and whole-artifact commitments are suitable for publication. Keep media, per-clip measurements, landmarks, source identifiers, individual timestamps and fingerprints private. [Source scope](video-appearance-native-v108-method.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_landmark_transfer_v255 preflight
backend/.venv/bin/python -B -m scripts.video_landmark_transfer_v255 freeze
backend/.venv/bin/python -B -m scripts.video_landmark_transfer_v255 readout
```
