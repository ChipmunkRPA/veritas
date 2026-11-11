# Landmark support in saved face-appearance windows

The local audit measured all 144 retained windows. It covered six webcam uploads from two genuine recordings and six controlled renders from one source with one known swap. Four renders contain that swap and two are matched controls. The audit separately counted the 12 windows shown in customer evidence and the 132 other retained windows. [Exact aggregate](evaluation-panels/video-saved-landmark-support-v254-complete.json)

| Source and selected window | Renders | During eye separation minus mean of before and after, relative to box width | Before-to-during landmark distance, in eye-separation units |
| --- | ---: | ---: | ---: |
| Genuine webcam false flag | 6 | −0.217 to −0.198 | 0.408 to 5.539 |
| Known face swap | 4 | +0.011 to +0.014 | 0.032 to 0.037 |
| Matched no-swap false flag | 2 | +0.024 to +0.024 | 0.048 to 0.053 |

In the webcam flags the eyes moved substantially closer relative to the face box during the selected window. That can amplify the landmark distance because it is normalized by the measured eye separation. This pattern is consistent with unreliable landmark support. The matched no-swap flags kept stable eye spacing, so the pattern explains only some false flags. The other retained windows include overlapping measures, and the four known-swap renders represent one intervention. These results support evaluating landmark quality on another panel with both swaps and controls.

The runner verifies the frozen v190 input and database hashes and every v253 webcam archive against its published commitment. It verifies the saved raw detector object and evidence packet, source clock, and native frame hashes before measuring. Each before, during and after group uses the saved consecutive frames around the measured centers. The audit pools eye-separation ratios by role and compares role medians of the normalized five-point configurations. It records unavailable coverage, and counts repeated frame observations across different windows where applicable. It reads the saved geometry without decoding frames or invoking models. [Known-swap comparison](video-faceswapai-full-app-v190-result.md) · [Webcam controls](video-vcd2-remaining-webcams-v253-result.md)

The app retained at most 12 windows per upload. This audit examines those retained windows and gives no estimate of how a production quality filter would affect all candidate windows or the published verdict. The runner writes the aggregate to standard output.

```sh
PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.video_saved_landmark_support_v254
```
