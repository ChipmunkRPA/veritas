# Correcting zoom and brightness around video face changes

The combined correction removed six ordinary zoom or brightness annotations, then also removed four genuine speaker cutaways. It missed the frozen preservation requirement. Scale correction alone removed all five zoom annotations, preserved all ten genuine cutaway deliveries, and left one brightness annotation. The scale result was an ablation on this bank and stays experimental. [Exact counts and source audit](evaluation-panels/video-face-context-correction-2026-11-12.json)

## Method and source separation

The baseline measures central face change and four sectors around the head at both boundaries of a brief face range. It subtracts ordinary adjacent-picture motion measured outside the boundary neighborhood. A broad annotation needs a central transition plus supporting change in at least two surrounding sectors at each edge. A positive observation would describe the extent of the visible change and would carry zero AI-confidence weight. [Earlier native test](video-face-near-head-transfer-2026-11-12.md)

The correction variants measure bounded scale and translation, a bounded affine brightness adjustment, and the two adjustments together. Each fit uses alternating four-pixel blocks from the near-head sectors. Acceptance requires improvement on the other blocks and at least two gray levels of improvement in the center, which was excluded from the fit. The geometry search covers a scale of 0.8–1.24 and offsets up to eight pixels on the 96-pixel analysis grid. Brightness correction allows a gain of 0.75–1.30 and an offset of up to 24 levels. Insufficient pixel coverage leaves the original measurement in place.

Four GRID source pairs supplied eight genuine cutaway deliveries and sixteen ordinary controls. Four temporary brightness controls were built from the existing unchanged deliveries. All six prespecified correction acceptance points kept the eight cutaways and removed the eight raw ordinary annotations. The fixed tie-break chose three levels and 45% relative improvement on the off-fit blocks. That point was sealed before measuring the held banks. The other four GRID pairs supplied only the saved application timing replay. The CREMA-D and RTFS sources are separate from calibration. These recordings had earlier project exposure, and the CREMA-D footage uses engineered fixed-layout controls. [GRID source and license](https://zenodo.org/records/3625687) · [CREMA-D terms and exact transfer bank](video-face-near-head-transfer-2026-11-12.md)

## Held results at authored boundaries

| Condition | Intervals | Raw broad | Scale only | Brightness only | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| CREMA-D speaker cutaways and tandem copies | 10 | 10 | 10 | 10 | 6 |
| CREMA-D digital punch-ins | 5 | 5 | 0 | 5 | 0 |
| CREMA-D local gain controls | 5 | 1 | 1 | 0 | 0 |
| CREMA-D unchanged and motion-jump controls | 10 | 0 | 0 | 0 | 0 |
| RTFS native swap intervals and tandem copies | 32 | 0 | 0 | 0 | 0 |

All 62 intervals had sufficient spatial support. Combined correction retained three of the five genuine speaker pairs, below the required four. The two losses each reproduced in both encodes. Neither single correction lost a genuine cutaway in this panel. A separate scale-only confirmation would need a newly declared operating point and additional source groups, including real framing changes and app-selected ranges. The present results cannot establish that the combined correction can safely explain an ordinary cut.

## Replay from the app's selected ranges

On the other four GRID pairs, the worker had selected ranges on eight of eight cutaway deliveries and on none of the twelve unchanged, jump or reframing controls. The replay searched for the largest central change within eight native frames of each recorded endpoint before measuring correction. All four arms gave broad observations for all eight cutaways, with selected edges within two pictures of the exact authored edges. The app had a usable primary range on twelve of sixteen RTFS swap deliveries. Each arm gave zero broad annotations across those twelve ranges. This route replay measures only the ranges that the existing app actually selected. Ordinary processing controls without an app range provide no conditional false-annotation estimate.

Across both phases, the study verified 2,840 native frame reads and checked all 212 selected adjacent-picture pairs against source PTS and RGB commitments. A separate scalar audit reproduced every correction and broad annotation decision from the saved measurements. The held pass took 26.4 seconds on one CPU worker. Encoded additions occupied under 1 MiB. No detector, confidence, or report behavior changed.

## Local reproduction

Use the retained source runs from the near-head transfer and the earlier Full video tests. Keep the exact clips, per-source geometry and measurement journals local. Run calibration separately before opening the held phase.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze calibrate held report; do
  backend/.venv/bin/python -B -m scripts.video_face_context_correction "$phase" "$run" || exit 1
done
```
