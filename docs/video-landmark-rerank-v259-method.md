# Exploratory landmark-quality rerank on saved appearance windows

This replay checks the practical effect of withholding a face-appearance interval whose landmark geometry has weak support. A window qualifies if its before, during and after geometry can be measured and the during median inter-eye distance relative to face-box width is at least 60% of the mean of the two flank medians. The 60% cutoff is an exploratory choice informed by the webcam results in v254. It is frozen before this rerank and has no independent calibration. [Observed landmark geometry](video-saved-landmark-support-v254-result.md)

Re-verify the twelve full-app uploads and reproduce the v254 geometry aggregate from the exact saved evidence. Preserve the original order by descending exact appearance response, then ascending native start and track ID. Verify that the original flag was the first saved window above the existing review point. Within the retained list choose the first qualifying window strictly above that same point. Report retained and replacement ranks, cases with no qualifying saved flag, and unavailable geometry separately. Compare selected ranges and overlap with the verified edited frames for the four renditions of the known swap. Count render variants and source groups separately.

Apply the same fixed geometry check to the existing appearance flags from the separate 23-group constructed selection panel. Preserve that panel's own historical appearance point and unavailable windows. This panel can show how many existing swap flags would be withheld. It has no prior control flags for measuring a false-flag reduction. [Constructed transfer panel](video-landmark-transfer-v255-result.md)

The production archive retained only the top 12 appearance windows per upload. The replay has no access to lower-ranked discarded windows. It evaluates a research hypothesis and changes no customer result, serving route or overall confidence.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_landmark_rerank_v259 freeze
backend/.venv/bin/python -B -m scripts.video_landmark_rerank_v259 readout
```
