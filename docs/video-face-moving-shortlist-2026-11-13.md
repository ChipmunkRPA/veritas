# Finding brief face changes when the subject moves

The four-second H.264 check used a single median face box to shortlist
possible boundaries in the entire clip. In a previously missed source,
measuring the correct local position recovered strong transitions at the
known edit boundaries. The existing final measurement already uses the
face box from the picture before each transition. The new candidate
changes where the initial scan looks. It interpolates the position and
size of the face box between verified sampled pictures and uses the
nearest available box at the ends. The local motion tests, face
comparisons and response point remain the same. This is a separate use
of frame-to-frame boundary evidence of the kind explored in recent
research. [Short-interval boundary research](https://arxiv.org/abs/2606.07090)

The comparison keeps earlier results and routes a clip only when it
has no existing brief range. It tests wider duration alone, moving
boxes alone, and the two together. The base search is five to seven
frames and the wider search is five to nine. Each shortlist can retain
16 edges and schedule up to 84 face inputs within the same 120-picture
clock. Candidate readings share decoded source pictures during the
offline run. No new model or weights are used. [Previous comparison](video-face-distinct-avc-2026-11-13.md)
· [Face assets and terms](video-appearance-plan-v107-method.md)

## Previously examined sources

The development replay opened all 181 roster entries. Two ordinary
files had identical delivered bytes in both banks and count once in
the results. A useful location covers at least half the temporal union
with a previously missed labeled range.

| First-range fallback | Edited intervals located | Exact additions | Ordinary clips with any flagged range |
| --- | ---: | ---: | ---: |
| Existing check | 87/110 | — | 15/101 |
| Wider search with the fixed box | 89/110 | 2 | 15/101 |
| Moving box at the existing duration | 87/110 | 0 | 15/101 |
| Moving box and wider search | 91/110 | 4 | 15/101 |

The combined change recovered four exact intervals from three source
groups. Two encodes of the same eight-picture edit were the additions
unique to the moving box. Neither individual change recovered the
missed source by itself. Ordinary flags here include existing coarse
appearance observations. The fallback added no ordinary or misplaced
ranges. [Development replay receipt](evaluation-panels/video-face-moving-shortlist-development-2026-11-13.json)

## Source-separated confirmation

The held panel contains 56 newly authored deliveries from eight
archived publisher captures. Selection excludes earlier source slots
with recorded archive indices and their known links, capture components
and donor keys. Broader project exposure and incomplete publisher
donor attribution limit the independence claim. Five
captures were recorded at 30 fps and three at 29.97 fps. The matching
controls include original clips, same-recording jumps, genuine cuts to
another speaker, and one-swap files with ordinary exposure or timing
changes. The predeclared gates require at least three added locations
across two source groups, with at most one new ordinary or misplaced
range and zero new genuine cut findings.
[Source archive and terms](video-common-source-preparation-v89.md)

On the 48 labeled intervals, the existing review located **38**.
All three candidates kept those results and added zero locations.
The existing check and all candidates added zero ranges to the
24 ordinary clips. The four edited deliveries eligible for the
fallback came from one capture. Its first encode had a second
detected face in 56 of 120 pictures, including samples at both
authored swaps. Interpolating the face box cannot recover the
missing single-face measurements. The confirmation did not pass the
advancement criteria, so the route remains off by default.
[Source-separated receipt](evaluation-panels/video-face-moving-shortlist-confirmation-2026-11-13.json)

## Local customer flow

The combined search is available as a local opt-in for Full and Deep
analyses of upright 720p H.264 MP4 with exactly 120 verified source
pictures at 30 fps. Enable `VERITAS_VIDEO_FACE_MOVING_AVC_REVIEW=true`
alongside `VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true` in appearance mode.
It reuses the decoded motion pictures and overlapping face measurements
and runs within the existing 12-second budget. If it finds a first
range, a second range is outside this fallback's scope. Previously
reported ranges remain available. Its evidence carries zero weight in
the overall AI estimate.

When proposed boundaries lack two complete sets of comparable
face views at supported distances, the fallback records insufficient
face context. A face that appears alongside a second face can leave
too few usable inputs even when the source frames decode correctly.

With the option on, the isolated Full worker recovered an exact
**3.100–3.367 s** change with original pictures **92, 97 and 101**.
The API, JSON, Standard PDF and signed original replay matched all
six original images and crops. A genuine two-cut upload added no range.
An edited control with two existing ranges kept both. The overall
assessment was unchanged across all six on/off runs. Simulated checkout
queued zero unpaid jobs. [Application receipt](evaluation-panels/video-face-moving-shortlist-app-2026-11-13.json)

## Replay

The Python standard-library reader reproduces the decisions, source
times and measured input counts. The separate FFmpeg reader regenerates
the native clock and the original selected frame digests from local
media. Public receipts contain no media or appearance vectors.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.video_face_moving_shortlist review "$receipt"
python3 -B -S -m scripts.review_video_face_moving_shortlist_pixels "$receipt" "$media_bank"
```

Provide all three local bank directories to the pixel reader for the
development replay. The 181 delivery records reproduced 21,720 source
timestamps and 12 original selected frames. Two duplicate source
deliveries remain in the ledger for provenance and are counted once.

To prepare and score the confirmation again, provide the retained
source archive, the earlier private two-range confirmation directory
as `$prior`, and a new private run directory. The runner freezes the
routing, source commitments, input budgets and local app code before
preparing the test deliveries. The public replay includes a content-hash
check when the local media is available.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist freeze "$run" --prior "$prior" --panel confirmation
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist prepare "$run" --prior "$prior"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist run "$run" --prior "$prior"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app "$prior" "$app_run"
```
