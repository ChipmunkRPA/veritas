# Motion-compensated exposure around video face flags

The earlier brightness diagnostic compared corresponding coordinates in unregistered pictures. Some flagged webcam clips had large brightness changes outside the face, and the changes varied by recording. This replay tests whether a temporary exposure shift survives camera-motion registration and also appears in the aligned face. [Earlier brightness diagnostic](video-luma-context-v278-result.md)

## Fixed comparison

The prior face-appearance outputs supply 110 views across six moving source groups and two genuine camera groups. The replay checks all 50 existing flags using their recorded before, during and after anchors and the immediate native neighbors of each. Exact native presentation times and RGB hashes are verified against the archived source receipts. The 36 known swap views include two swap families and three durations. The two genuine recordings supply 14 unchanged and ordinarily processed views. One camera group also appeared in the earlier brightness diagnostic. The other did not. All views already had baseline scores in the short-boundary study. This is a prospectively specified comparison on saved flags, with no fitted point and no newly acquired source footage. [Archived source qualification](video-common-source-preparation-v89.md) · [Prior appearance results](video-face-short-boundary-confirmation-2026-11-11.md) · [Genuine camera data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

The replay uses the existing face detector solely for box and landmark geometry. It excludes expanded face boxes, registers 320-pixel-wide grayscale pictures to the middle anchor using forward and backward optical flow and a robust partial affine fit, then measures registered scene tiles separately from the aligned face interior. A temporary shift must point in the same direction relative to both flanks. Its support must extend over at least 60% of measured scene tiles in at least three quadrants and at least 60% of face sample sites. The signed face and scene responses must agree and their magnitudes must have a ratio between 0.5 and 2. A failed registration or insufficient common coverage keeps the baseline flag. The runner keeps the original appearance point and performs no appearance classification forwards. [Optical-flow documentation](https://docs.opencv.org/4.5.3/dc/d6b/group__video__track.html) · [Affine-fit documentation](https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html) · [Face-localization weights](https://huggingface.co/opencv/face_detection_yunet)

Advancement requires withholding at least four baseline false flags spanning both genuine camera groups, retaining every swap range with at least 0.50 temporal intersection over union, and measuring at least 90% of selected views in both cohorts. The candidate can only mark an existing range as ambiguous because of broad scene change. It has no originality verdict and no confidence contribution. No points were adjusted after scoring.

## Results

| Existing appearance outcome | Baseline flags | Context measured | Flags withheld |
| --- | ---: | ---: | ---: |
| Known swaps at 0.50 or greater overlap | 36/36 | 12/36 | 0 |
| Genuine camera views | 14/14 | 7/14 | 0 |
| Matched original and ordinary-edit controls | 0/60 | — | 0 |

Registration was unavailable for 31 of 50 flagged views. The seven measured genuine camera views came from one recording. Their aligned face log shifts ranged from about −0.360 to 0.045, and their registered scene median shifts were zero. The measured scene tiles did not establish a coherent temporary exposure change under the fixed rule. The candidate retained every correct swap range and cleared no genuine flag. Both the improvement and coverage criteria failed. The current appearance check remains in place. [Exact counts](evaluation-panels/video-registered-light-context-2026-11-11.json) · [Frozen settings](evaluation-panels/video-registered-light-context-freeze-2026-11-11.json)

The CPU replay verified 226 native pictures in 149.5 seconds with 217 MB peak worker memory. It made 226 face geometry calls and zero appearance classifier calls. A separate standard-library readout recomputed the baseline labels, overlap, coverage, context decisions and aggregate commitments for all 110 views. Two authored regression tests verified small camera translation, broad temporary exposure, face-only changes, opposite directions and insufficient scene texture. The prior landmark evidence points to another specific failure mode. A future comparison can evaluate abstention on the selected range when the face alignment collapses, while preserving the other observed ranges for review. [Landmark evidence](video-saved-landmark-support-v254-result.md)

## Reproduce

Use the exact private archive and the prior short-boundary output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_registered_light_context freeze "$run_dir" --previous "$short_boundary_run"
backend/.venv/bin/python -B -m scripts.video_registered_light_context run "$run_dir" --previous "$short_boundary_run" --archive "$archive"
backend/.venv/bin/python -B -m scripts.video_registered_light_context readout "$run_dir" --previous "$short_boundary_run"
```
