# Proposing six-picture face-change ranges in 720p30 HEVC video

The opt-in timing refinement starts from a range already selected by the Full face check. Four six-picture swaps in the preceding HEVC comparison supplied no such range. This study tests whether brief changes can supply their own candidate range. The research on localizing sparse manipulated intervals motivates the separation between candidate search and evidence confirmation. This implementation uses native pictures and the existing qualified face assets. [Earlier HEVC evaluation](video-face-paired-edge-transfer-2026-11-12.md) · [Boundary-sensitive proposal research](https://arxiv.org/abs/2606.02402) · [Face asset licenses](video-appearance-plan-v107-method.md)

## Frozen comparison

The scope is upright 1280 × 720 HEVC `hvc1` MP4 with 8-bit 4:2:0 color, exactly 120 verified pictures, and a 30 fps native clock. Each arm searches for intervals of five through seven pictures. Each endpoint requires three consecutive observations on each side, with compatible detected face boxes, a stable difference at both edges, and a return toward the surrounding face appearance. The existing near-head motion calculation checks that a broad change outside the center does not accompany either boundary. The selected ranges carry zero AI-confidence weight.

The dense comparison uses every native picture. The bounded arm estimates a common face box from 12 evenly spaced pictures, keeps up to 16 locally strong near-head motion edges, and uses no more than 84 pictures for face measurements. Both arms rank the supported intervals by their weakest response and retain at most two nonduplicate intervals. The development rule uses the stricter of the existing 0.10 short-face response point or the largest selected ordinary response. The score is an uncalibrated comparison of visual measurements.

The 61 development deliveries use the preceding HEVC panel. The 56 held deliveries comprise eight different RTFS source groups and eight different fixed-layout speaker pairs, each with a tandem encode. The eight RTFS groups supply 16 edited deliveries. Each has one six-picture and one eight-picture edit. The 40 ordinary deliveries include originals, same-recording jumps, cross-recording whole-picture cuts, and 16 fixed-layout speaker cutaways. The media had prior project exposure. Their HEVC deliveries and the comparison against this method were prepared for this study. The test preserves the source frame clock and known placements. [RTFS inputs and edit boundaries](video-face-two-returns-hd-2026-11-12.md) · [Genuine cutaway controls](video-face-hd-retry-scene-challenge-2026-11-12.md)

Full runs on the same HEVC bytes. A new location needs at least 0.50 temporal intersection over union with a six-picture edit that Full missed. The gate requires four additional locations in three different source groups, two exact ranges, and at least 75% of the dense comparison's gains. It allows one ordinary flag and one wrong range on edited footage. It allows zero fixed-layout speaker-cut flags.

## Held results

| Outcome | Current Full | Dense native search | Bounded shortlist |
| --- | ---: | ---: | ---: |
| Six-picture ranges correctly localized | 0/16 | 16/16 | 16/16 |
| Exact six-picture ranges | 0/16 | 16/16 | 16/16 |
| New ordinary ranges | — | 0/40 | 0/40 |
| New fixed-layout speaker cut ranges | — | 0/16 | 0/16 |
| Wrong ranges on edited deliveries | — | 0/16 | 0/16 |

Both arms passed the held comparison using the frozen 0.10 point. They recovered all eight source groups and their tandem deliveries. The four measured intervals were **2.067–2.267 s**, **2.133–2.333 s**, **3.033–3.233 s**, and **3.133–3.333 s**. The bounded arm requested 12–71 face measurements per file with a median of 26.5. The earlier Full check flagged 15 fixed-layout speaker cutaways. These checks did not suppress those ranges. The separate development bank recovered three of four six-picture swaps with zero added flags among 53 ordinary clips. Independent replay reproduced all 48 original frame hashes, 48 aligned face crops, and 32 surrounding-motion measurements. [Public case ledger](evaluation-panels/video-face-unrouted-short-transfer-2026-11-12.json)

## Local replay

`$development` is the retained HEVC confirmation directory. The held runner uses the archived 720p30 dual-edit and genuine cutaway banks and creates a new private `$run` directory. It retains the experimental method independently of the serving code. Original media stays local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in prepare development held incumbent; do
  backend/.venv/bin/python -B -m scripts.video_face_unrouted_short_transfer "$phase" "$run" "$development" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_unrouted_short_transfer \
  "$run/receipt.json" --assemble "$run" "$development" --native "$run"
```
