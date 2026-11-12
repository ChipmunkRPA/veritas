# Native mouth-shape dynamics on timed retalking edits

A local measure of high-order lip motion correctly located five of 32 held redraws, including two of seven with previously verified source-time mouth motion. The previous direct picture comparison found one redraw and none in that seven-clip subset. The motion measure also flagged 16 of 56 genuine or conventionally edited controls. It failed the frozen advancement gate and remains experimental. [Aggregate and native replay](evaluation-panels/video-mouth-dynamics-transfer-2026-11-12.json)

## Frozen comparison

The October 2026 revision of LADDER reports that the relative variation across successive lip-landmark derivatives transferred more consistently than raw movement amplitude across its tested generators. This experiment adapts that observation to short local intervals using four retained geometry measurements. It uses neither the paper's trained classifier nor its dataset. [LADDER v5](https://arxiv.org/html/2604.16808v5)

The experiment reuses the exact native GRID recordings and component-separated roles from the earlier reconstruction test. Eight recordings from four components set the points and eight from four other components supply the held results. Each recording supplies four local neural redraws and seven controls covering original pictures, local brightness and resampling edits, genuine frame retiming, and an audio-only delay. All sources had earlier project exposure. The unrendered reserve has no comparable known edited counterpart. [GRID publisher and CC BY 4.0 terms](https://zenodo.org/records/3625687) · [Earlier native comparison and roles](video-mouth-reconstruction-transfer-2026-11-11.md)

Previously verified geometry measures three inner-lip gaps and inner-mouth width, normalized by eye separation. The original observer recorded each variant at native pictures 25 through 48 and its original counterpart at pictures 15 through 58. The new measurement reuses an existing observation only when its native picture hash and time match the tested file. This supplies verified flanks without borrowing reference geometry from different pixels. The historical observer profile binds the exact licensed geometry assets, inputs, artifacts and successful completion. This replay uses saved geometry and makes zero new neural calls. [Native observer protocol](grid-mouth-observer-v157-method.md) · [Isolated runtime and reviewed geometry assets](grid-portrait-geometry-v144-method.md)

At exact 25 fps the comparison scans 12-, 16- and 20-picture intervals with ten consecutive pictures on each side. For each lip measure it computes how standard deviation changes between position, first, second and third temporal differences. It averages the log ratios relative to position using a fixed floor of .002. The candidate requires a higher response inside the interval than on either flank. A second arm subtracts a simultaneous increase in the same measurement of eye scale, eye roll and nose position. The amplitude comparison uses position variation alone. Each arm uses strict exceedance of the third-highest ordinary development maximum. The previous direct-picture outcome retains its prior point and native intervals.

A correct range needs at least 0.50 temporal intersection over union with the authored interval. The gate requires at least six correct redraws across three components, at least two more correct locations than the direct picture comparator, two correct source-time-compatible redraws, at least 75% coverage, and at most two additional control flags. Source-time compatibility means the existing measurement found compatible visible mouth motion and the output retained its original PCM. Phoneme-level and listener-perceived synchrony remain unverified. [Prior source-time motion qualification](grid-mouth-motion-v158-result.md)

## Held outcomes

| Check | Measured views | Correct redraw ranges | Misplaced redraw ranges | Correct source-time-compatible ranges | Ordinary flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct picture | 88/88 | 1/32 | 0/32 | 0/7 | 0/56 |
| Mouth motion amplitude | 88/88 | 2/32 | 0/32 | 0/7 | 6/56 |
| Relative derivative variation | 88/88 | 5/32 | 1/32 | 2/7 | 16/56 |
| With head motion adjustment | 88/88 | 4/32 | 1/32 | 2/7 | 14/56 |

Both derivative arms selected **1.080–1.880 s** exactly in one source-time-compatible redraw. The other selected **1.320–1.800 s** within the same authored span, for 0.60 IoU. The first range uses native mouth pictures at **1.040**, **1.480** and **1.880 s**. Correct derivative ranges span two held components. The ordinary flags include two untouched recordings and their two visually identical audio-only variants. The unadjusted arm also flagged six brightness changes, four local resamples and two genuine frame retimes. The head adjustment removed two brightness flags and one correct redraw range. The prior direct picture comparison produced no flags on the ordinary controls.

The cached held calculation took 3.3 seconds. The independent reviewer recomputed 264 decisions and reopened 147 selected mouth crops at their source timestamps. For the seven source-time-compatible clips it also verified identical original PCM, decoded 1,050 native source and output pictures, and rebuilt all 140 authored changed-pixel masks.

## Reproduce

Use the retained GRID bank and prior observer receipts. The independent reviewer writes exact native examples to the private run folder.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development calibrate held summary; do
  backend/.venv/bin/python -B -m scripts.video_mouth_dynamics_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_dynamics_transfer "$run"
```
