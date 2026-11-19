# Cross-cut timing on a separate 29.97 fps camera corpus

Reapplying a stricter point of .21 to the original 25 fps results retained four correct signed changes, one wrong estimate and no alerts on the synchronized or equal-offset controls. That comparison used the same files that suggested the new point. This separate trial froze both .10 and .21 before scoring 30 new files authored from 12 different CREMA-D camera actors in six pairs. The recordings and speakers are separate from GRID. The project previously used these recordings to evaluate eye evidence. [CREMA-D publisher and licence](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

Each AVC file has two 60-picture shots on a 30000/1001 clock and continuous mono 48 kHz AAC audio. The cut is at frame 60, 2.002 s, PTS 180,180 on a 1/90,000 clock. The renderer accounts for each camera source's verified 27 ms picture-to-audio origin difference. It preserves one native picture per output frame. Before creating any tests, the clock intake tolerance widened from 1.0 to 1.5 ms because millisecond timestamps in all available sources exceeded the initial limit. The largest admitted source deviation was 1.234 ms. The exact delivered PTS and RGB frames stayed fixed. A correct result requires the exact cut and a correctly signed lag change within 60 ms of the planted change.

| Held case | Planned | Cut measured | Original .10 | Stricter .21 |
| --- | ---: | ---: | --- | --- |
| Second-shot sound advanced | 6 | 4 | 2 correct, 2 wrong | 0 correct, 2 wrong |
| Second-shot sound delayed | 6 | 4 | 2 correct, 1 wrong | 1 correct, 0 wrong |
| Synchronized cut | 6 | 4 | 1 alert | 1 alert |
| Aligned speaking pause | 6 | 4 | 2 alerts | 0 alerts |
| Same signed lag in both shots | 6 | 4 | 1 alert | 0 alerts |

The two groups with missed cuts remain in every planned denominator. The existing near-silence route had measurements available for all 30 files and produced no alerts. The stricter point reduced ordinary and equal-lag alerts while losing three of four correct changes. It missed the frozen minimum of four correct changes across three groups and exceeded the maximum of one wrong estimate. Both points failed the overall gate.

In the original GRID data, the inferred offset of an identical first shot changed by 20–340 ms when the second shot alone changed, with spreads of at least 100 ms in five of six pairs. Before opening transfer scores, we froze a separate test that estimated each shot independently using common visible and voiced rows and required both halves to support a distinct peak. It abstained for unstable peaks on all 20 files with observed cuts and for shot coverage on the remaining 10. The current representation did not offer stable timing in this cohort. Neither method moved into the app. These staged timing changes test localization and signed direction, with AI attribution outside the study.

Independent replay checked 3,600 delivered picture timestamps and RGB frames, 1,921,920 presented audio samples at the analysis rate, 720 camera picture ordinals and 12 source clock origins. It verified the before, at and after mouth crops for all 11 selections at .10, including the four retained at .21. The anonymous receipt replays the numeric adjudication and reports the locally verified coverage. It contains neither media nor individual source identifiers.

```sh
python3 -I -S backend/scripts/review_shot_lag_point21.py \
  docs/evaluation-panels/video-shot-lag-point21-transfer-2026-11-19.json
```
