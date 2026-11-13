# Local spectrotemporal modulation under competing speech

The modulation check localized zero of 22 held edits under strong interference. Adding it to local Mel cepstra also localized zero. The existing check localized one of 12 edits with recorded room speech at its frozen point. [Score ledger and replay receipt](evaluation-panels/audio-modulation-partial-transfer-2026-11-12.json)

## Method

A two-dimensional cosine transform measures how short-term acoustic energy changes across both time and frequency. Compute 32 Mel-band log energies on 25 ms frames with a 10 ms hop. In each 200 ms cell, remove the time-average from each band, transform across time and bands, and pool squared coefficients into four temporal by five spectral bands. Fit one small regularized head to local energy, relative energy, and neighboring-cell contrast. Fit a second head to the same features plus the prior Mel cepstra. No external code, model weights, or source separation enter the method. [Gao et al., joint modulation for spoof detection](https://www.isca-archive.org/interspeech_2021/gao21c_interspeech.html) · [Zaman and Unoki, segmental modulation](https://arxiv.org/html/2604.23241v1) · [Prior cepstral comparison](audio-cepstral-partial-transfer-2026-11-12.md)

Fit on the prior eight matched source groups and set each point using the same 46 genuine examples held apart from fitting. Compare with the saved incumbent readings on both previously exposed held panels. The room panel uses 12 distinct speech groups from four AMI meetings, with three generator families, ordinary same-reader and other-reader splices, unchanged clips, and four isolated backgrounds. It supplies 430–960 ms of annotated competing talker overlap per edit. The acoustic candidate ranks 0.6-second windows on a 200 ms grid; the room panel incumbent ranks its fixed two-second windows. Useful localization needs at least 0.30 intersection over union and coverage of at least half of the edit. [Fit and rights](audio-periodic-excitation-transfer-2026-11-12.md) · [Room source qualification and baseline](audio-remote-neighbor-replication-2026-11-12.md)

| Held condition | Edits | Existing useful / false groups | Modulation useful / false groups | Modulation with cepstra useful / false groups |
| --- | ---: | ---: | ---: | ---: |
| Matched dry | 10 | 2 / 0 | 0 / 0 | 0 / 1 |
| Matched competing speech, −5 dB | 10 | 0 / 0 | 0 / 0 | 0 / 3 |
| Recorded room speech, −5 dB | 12 | 1 / 3 | 0 / 0 | 0 / 0 |

With up to two retrospectively matched false groups, both candidates still localized zero noisy edits. The incumbent found two in the matched competing-speech panel and one in the room panel. Before applying a cutoff, the strongest modulation window overlapped usefully on three of 12 room edits, or one of 12 when combined with cepstra. All held files were measurable. This narrows the failure to score ranking and interval selection under the tested conditions. The candidate remains offline.

## Replay

```sh
python3 -B -IS backend/scripts/review_audio_modulation_partial_transfer.py \
  docs/evaluation-panels/audio-modulation-partial-transfer-2026-11-12.json

export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_modulation_partial_transfer run "$run"
backend/.venv/bin/python -B -m scripts.audio_modulation_partial_transfer review "$run"
```

The public reader recomputes both points, all 126 held overlap decisions, and all false-group counts. The native replay verifies exact source PCM and repeats feature extraction, fit and outcome generation. It uses the retained inputs described in the fit and room-source receipts. No audio or fitted parameters are published.
