# Acoustic timing support for short voice edits

A source-only timing cue found no extra useful edit locations when combined
with the existing dense voice response. This test distinguishes a temporary
acoustic change from AI authorship. It reuses the exact native mixtures and
locked voice scores from the earlier isolation trial and adds no model calls.
[Isolation trial and source terms](audio-source-isolation-diagnostic-2026-11-13.md)

## Fixed comparison

Compare the middle 840 ms of a one-second candidate with 320 ms of audio
on each side, leaving an 80 ms gap from the reported interval to each
flank. Candidate starts advance in half-second steps and retain the
existing speech gate. An interval near the start or end may lack a
complete flank. Compute median relative spectral power in 24 logarithmic
bands between 250 and 6,400 Hz from overlapping 512-sample Hann frames.
Use Jensen–Shannon divergence to compare each interior/flank pair and the
two flanks. The basic cue uses the weaker interior/flank divergence.
The acoustic return subtracts the difference between the two flanks.
Ordinary words, speaker changes and background sounds can produce this
shape. No feature reads an earlier copy, isolated voice or known edit time.
[Earlier paired boundaries](audio-original-pcm-boundary-transfer-2026-11-12.md)
· [Environmental signature research](https://arxiv.org/abs/1411.7084)

For the combined readout, scale the dense voice logit and timing cue by
the median and interquartile range of their ordinary development windows.
Add half the nonnegative timing value, capped at four scaled units,
to the voice score. Without timing, retain the voice term and its coverage.
Each new point is the maximum ordinary response on 40 development
deliveries, which include 28 ordinary controls. Seal those points before
measuring the 52 held mixtures. Development and held roles use separate
known speakers and rooms. The sources have earlier project exposure. A
useful location needs at least 0.30 temporal IoU and covers at least half
the known edit. Each ordinary group counts once if any of its controls
flags. [Held source and construction audit](audio-remote-neighbor-replication-2026-11-12.md)

## Results

| Readout | Useful at fixed point / 12 | Ordinary groups flagged at fixed point / 16 | Useful before the point / 12 |
| --- | ---: | ---: | ---: |
| Current two-second voice check | 1 | 3 | 5 |
| Dense voice check | 0 | 0 | 3 |
| Weaker spectral edge | 0 | 0 | 2 |
| Acoustic return | 0 | 0 | 2 |
| Dense voice plus acoustic return | 0 | 0 | 3 |

Every file had at least one assessed interval in every readout. Eleven
edits had a potentially useful interval with both flanks, compared with
12 for the dense voice readout. At the descriptive allowance of one
held false group, the acoustic return located one stock-speech edit the
current check missed. The same descriptive cut flagged three genuine
deliveries from one group and nominated an incorrect time on one other
edit. At two or three false groups, the timing-only readout and both
voice comparators each located one edit. The combined readout never
improved useful counts on the dense voice check at the same allowance.
No candidate has evidence for an app or confidence change.
[Scalar measurements and decisions](evaluation-panels/audio-acoustic-return-timing-2026-11-13.json)

## Replay

The standard-library reader checks all window slots, time bounds,
published development factors, selected points and held outcomes from
the pinned original score receipt and the new acoustic measurements.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_acoustic_return_timing review \
  docs/evaluation-panels/audio-acoustic-return-timing-2026-11-13.json
```

For native replay, use the same retained source folders as the isolation
study and a new run directory. The script checks original PCM and source
commitments. It performs signal measurement and reuses the committed
voice scores, so no model assets are required for this replay.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development assessment; do
  backend/.venv/bin/python -B -m scripts.audio_acoustic_return_timing "$phase" "$run" || exit 1
done
```
