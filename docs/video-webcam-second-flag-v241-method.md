# Repeat the native webcam measurements on a second false flag

The frozen raw webcam comparison found a second genuine recording with a face appearance flag. All three versions shared the exact 32-frame interval. Bind the published six-recording archive set, then verify the three matching archives and use the same frame, geometry and in-memory database checks as the first source replay. [Six-recording comparison](video-vcd2-full-app-v202-two-triplets.md) · [First source replay](video-webcam-source-geometry-v231-method.md)

Measure face and remainder luma, and normalized landmark motion for the saved before, during and after frames. Compare the ordinary variants with the unchanged recording only when the saved role indices match. These descriptive measurements test whether the second false flag has the same observable pattern as the first. They are neither causal claims nor detector decisions. The local replay makes zero model calls and publishes aggregate measurements without exact frame indices or per-source fingerprints.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.video_webcam_second_flag_v241 freeze
backend/.venv/bin/python -B -m scripts.video_webcam_second_flag_v241 readout
```
