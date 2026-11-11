# Timing change-rule coverage on the GRID pilot

The dense common-input diagnostic recorded a lag window on 166 of the 176
pilot cases. All 166 had insufficient support to detect a change. The
remaining 10 had no attributed speaking face. No case reached the support
needed for either a lag step or a correlation collapse. Change-detection
sensitivity and synchrony remain unassessed. [Original pilot result](grid-native-mouth-scores-v155-result.md)

| Saved diagnostic opportunity | Count |
| --- | ---: |
| Pilot cases | 176 |
| Audio permits exactly one complete lag-search window | 176 |
| Measured cases and measured windows | 166 |
| Measured windows at or above the 0.4 trusted-peak floor | 141 |
| Measured cases with insufficient change-rule support | 166 |
| Unassessed after speaker attribution | 10 |
| Cases with enough support for either change rule | 0 |
| Change flags | 0 |

The detector compares a 25-tick window at 25 Hz over lags from −15 to +15
ticks. Its window starts are `floor(12.5 × j)`. The 47,648 samples at 16 kHz
give 74 complete ticks. Only window 2, from 1.00 to 2.00 s, has both sides of
the lag search available. The next window would need 77 ticks. Four
full-search windows first fit at 102 ticks (4.08 s), before the detector
applies the picture, speech, and speaker gates. [Frozen window and lag rules](../backend/app/detectors/video/xcorr.py)

A lag step needs two stable runs of at least two trusted, non-collapsed
windows. Each run admits offsets within one tick of its running median.
The longest retained run sets the baseline, with the earliest used on a
tie. A second run must differ from the baseline by at least three ticks
(120 ms). Stable offset runs can span gaps in the window indices. A
correlation collapse needs at least four measured
windows to establish its threshold at half the track median, then at least
two consecutive grid windows strictly below it. The relative threshold is
the only collapse threshold for this route. [Frozen change rules](../backend/app/detectors/video/lipsync.py)

All 16 ordinary delayed-audio controls retained the original video, sample
count, and clock. The constructor records copying samples `[12160, 24960)`
into `[17280, 30080)`, moving waveform values 5,120 samples (320 ms) later
within the 1.08–1.88 s interval. It retained the waveform values outside
that interval. Fifteen delayed cases produced a measurement. Fourteen
original/control pairs had a measurement at the same time, and 12 of those
pairs met the peak floor on both sides. The winning lag shifted by exactly
+8 ticks in two of 14 pairs, including one of the 12 above the floor on both
sides. The edited interval covers 0.80 s of the only 1 s measurement. A
winning lag in that mixed window measures envelope correlation. Identifying
a lag step requires a comparison across runs. [Construction method](grid-native-clips-v149-method.md)

Run the replay from the repository root.

```sh
cd backend
.venv/bin/python -B -m scripts.review_grid_timing_opportunity
```

The replay binds the published v155 collection, the retained v151 clip 0,
the 15 completed continuation stages, their v149 case journals, and the
v126 timing code. It checked the winning lags against 5,146 saved lag
coefficients and replayed the frozen change rules on those saved windows.
The script reads metadata and reports aggregates for the common-input
pilot. [Completion receipt](evaluation-panels/grid-native-mouth-scores-complete-v155.json)
