# Video appearance flags and broad luma changes

This diagnostic measures whether temporary brightness changes around the originally selected appearance flags extend across the image outside the tracked face. Use all selected flags from the completed v254 cohort, including four renders of one known swap, two matched renders without a swap and six genuine webcam renders from two other source groups. Renders of a group are dependent. The alternative windows from the landmark reranking are outside this cohort. This is a retrospective diagnostic and does not set a detector threshold. [Saved window audit](video-saved-landmark-support-v254-result.md) · [Prior reranking result](video-landmark-rerank-v259-result.md)

Reopen only the exact archived source and analysis records, verify the saved native PTS and RGB commitments, then sample luma at matching native coordinates on a 16-pixel grid. Require at least four sites in each face box, 128 common sites outside every expanded face box and three frames per role. The three role medians are computed independently at each common site. Use the smaller luma shift from the during role to the two flanks if both shifts agree in sign. Report the fraction of common sites with a temporary shift of at least 12 RGB-code-value units, and the fraction moving in the same direction as a face shift of at least 12 units. These describe spatial support for a change. Camera or subject motion, cuts and relighting can all affect the result. The grid is fixed to the image, with no camera-motion registration.

Keep source frames, per-window coordinates and individual measurements private. Report coverage and ranges by cohort, with absent support visible. The comparison calls no detector and preserves the current serving route.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_luma_context_v278 freeze
backend/.venv/bin/python -B -m scripts.video_luma_context_v278 readout
```
