# Brief face changes in four-second WebM clips

The optional brief face-appearance check can now find a short change and
return in a qualifying four-second VP9 WebM without a previous face finding
or an earlier copy. The browser, PDF and JSON show the same end-exclusive
source time, with original frames and face crops before, during and after
the change. Ordinary cuts, expressions or lighting changes can also affect
appearance. The check carries zero weight in overall AI confidence.

## Frozen comparison

The paired boundary idea is supported by recent work on complementary
appearance and residual measurements and on locating brief intervals in
otherwise genuine footage. This test transfers the existing production
shortlist. Its measurement first ranks local near-head motion, then compares
three facial views on each side of both edges. It uses the original five- to
seven-picture range, 84-measurement cap and frozen response point. Only a
distinct range can supplement the existing Full face check. The tests use
the already pinned MIT and Apache-2.0 face detection and appearance weights.
[Multiscale appearance and residual research](https://arxiv.org/abs/2611.12508)
· [Short-interval localization research](https://arxiv.org/abs/2606.07090)
· [Original production shortlist](video-face-unrouted-short-transfer-2026-11-12.md)
· [Asset licenses](video-appearance-plan-v107-method.md)

Each carrier supplies 56 retained source cases. Sixteen positive deliveries
have one exact six-picture swap and one eight-picture swap from two publisher
families, including eight tandem-encoded positives. Forty ordinary controls
include untouched, same-recording cuts, whole-picture cuts and fixed-layout
speaker changes. VP9 files were newly encoded from the same H.264 inputs and
run through the actual Full face stages with the fallback disabled. The H.264
comparator uses the recorded Full face outcomes on those same archived bytes.
Both banks were previously seen in the project. Their paired carrier variants
are not independent source observations. A useful location needs at least half
temporal overlap by intersection over union. Advancement requires four useful
additions from at least three source groups, including two exact, at most one
ordinary addition, and zero new ordinary speaker or cross-recording cuts.
[Source permission and exact construction](video-face-two-returns-hd-2026-11-12.md)

## Held results

| Carrier | Full six-picture locations | With shortlist | Exact additions | Groups with additions | New ordinary flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| VP9 WebM | 0/16 | 15/16 | 15 | 8 | 0/40 |
| H.264 MP4 | 5/16 | 9/16 | 4 | 2 | 0/40 |

Both carriers had sufficient geometry to measure all 56 files. VP9 found
all four tested placements. Its only miss came from the tandem encode of a
known edit recovered on its primary delivery. Neither carrier added a
mistimed edited range or flagged any ordinary speaker or cross-recording
cut. H.264 adds four exact locations on two distinct source recordings at
the same time placement. It misses the frozen three-group requirement and
stays experimental. Independent FFmpeg replay checks 6,720 timestamps in
each carrier and all 57 selected original RGB frames. The public ledger
contains no videos, images or identifying paths. [Held ledger](evaluation-panels/video-brief-face-120-carrier-2026-11-13.json)
For this panel, all 9,888 possible proposal comparisons with recorded
prior ranges produced the same overlap decision on frame indices and native
WebM timestamps. All 48 comparisons between recorded and authored ranges
also agreed.

## Local product scope

The new path requires a four-second, upright 1280 × 720, 8-bit 4:2:0 VP9
WebM with exactly 120 native timestamps matching a rounded 30 fps clock.
Use Full or Deep, appearance review, and the existing
`VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW=true` option. The option remains off
by default. The five-to-seven-picture check keeps the original HEVC path
and the distinct eight-second carrier check. This comparison does not measure
mouth synchronization, longer videos, or videos with multiple clear faces.

An isolated simulated paid worker tested the two ordinary cases, the known
swap and the same swap with the option off. With the option on, the swap
produced a range at **2.067–2.267 seconds** with source pictures **61, 65 and
68**, using native timestamps **2033, 2167 and 2267** at time base **1/1000**.
The three original frames and three exact face crops matched the downloaded
PDF, evidence JSON and API. Signed replay recomputed the image shortlist and
both near-head motion boundaries from the supplied source and rejected changed
source bytes. Chrome loaded all six images and showed all three source clocks.
Neither ordinary case added a range. Turning off the option removed the new
range without changing the overall assessment. [Local application receipt](evaluation-panels/video-brief-face-120-webm-app-2026-11-13.json)

## Replay

The independent scalar reader needs Python only. The original comparison
requires the retained archives and commercially licensed assets above. The
scorer verifies the source file, its complete native clock, and all archived
RGB commitments for the H.264 carrier. VP9 output stays local.

```sh
python3 -B -IS backend/scripts/review_video_brief_face_120_carrier.py \
  docs/evaluation-panels/video-brief-face-120-carrier-2026-11-13.json
```

For a native rerun, start from application source revision `ff9ac04` with
the scorer from this study. Use the retained archive directories named in
the scorer. The release adds a carrier to the portable proof binding, so
the original study freezes the earlier production file hash.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze avc vp9; do
  backend/.venv/bin/python -B -m scripts.video_brief_face_120_carrier "$phase" "$run" || exit 1
done
```
