# Waveform-residual test on noisy speech

A lightweight residual check found four useful edit locations missed by the
current two-second MP3 check. It also added 21 ordinary source-group flags
across the two panels. The residual check stays experimental. [Case-level
scores and comparison](evaluation-panels/audio-mp3-residual-transfer-2026-11-12.json)

## Comparison

The 2026 study by Pizarro and colleagues uses the statistical properties of
waveform residuals for synthetic-speech detection and generator attribution.
This trial independently tests a simpler question. It measures how far a
two-second window departs from ordinary noisy speech in the same MP3 profile.
It uses NumPy and FFmpeg with no paper code, weights, or training corpus.
[Residual fingerprint research](https://www.isca-archive.org/interspeech_2026/pizarro26_interspeech.html)

For each of the five two-second windows, subtract a five-tap binomial
low-pass view of the original decoded samples. In 20 ms frames at a 10 ms
hop, measure residual-to-source power and normalized autocorrelation at
lags of 1, 2, 4, 8, 16, and 32 samples. Retain the quartiles and median of
each measurement. The existing detector's speech admission governs the
windows that can be measured. Score each vector's squared distance from an
equal-group-weighted ordinary reference, using a covariance matrix shrunk
halfway to its diagonal.

Divide the 16 ordinary calibration groups by a deterministic digest order.
Fit the reference on eight groups and choose the second-highest group maximum
from the other eight. A response must strictly exceed this value. Freeze the
reference and point before either held run. The comparison retains the
strongest eligible window with the earlier window winning a tie. A useful
location covers at least half of the edit and reaches 0.30 temporal IoU. For
further consideration, the supplement needs at least two added useful
locations, at least one per panel, and at most two added ordinary groups.

The retained panels are the same 50 and 52 six-second, 16 kHz mono MP3
recordings used in the previous codec work. Each has 12 known edits from
three generator families, with ordinary splices, unchanged voices, and
room-only controls. The two panels are disjoint in exact audio and host
speaker. Two generated voice profiles appeared in prior work with separate
clips. These are new residual measurements on previously used recordings.
[First panel and terms](audio-short-codec-transfer-2026-11-12.md) ·
[Confirmation sources and terms](audio-short-mp3-confirmation-2026-11-12.md)

## Results

| Panel | Current useful edits | Residual useful edits | Additional useful edits | Current ordinary groups flagged | Residual ordinary groups flagged | Additional ordinary groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transfer | 2/12 | 4/12 | 3 | 1/14 | 12/14 | 12 |
| Confirmation | 1/12 | 1/12 | 1 | 1/16 | 9/16 | 9 |

All 102 held files had a measurable window. The residual arm flagged 17
generated files and localized five usefully. The other 12 flags pointed to
the wrong interval. Four room-only files flagged, compared with zero for the
current check. The result is consistent with sensitivity to background and
recording conditions. A residual anomaly by itself does not identify a
synthetic voice or a synthetic background. Future residual tests need
matched nuisance controls and a source-separated assessment of what caused
each false alert.

## Reproduce

The public result contains anonymous per-window scalar distances, case
roles, and measurement commitments. It includes no recordings, model
parameters, or residual feature vectors. Review the operating point, window
admission, and every aggregate with the standard library.

```sh
python3 -B -S backend/scripts/audio_mp3_residual_transfer.py review \
  docs/evaluation-panels/audio-mp3-residual-transfer-2026-11-12.json
```

For native replay, use the retained MP3 directories from the preceding
studies. Set `codec_dir` to the directory with the 90 calibration and
transfer source records, `confirmation_dir` to the 52-file confirmation
directory, and `replay_dir` to a new working directory. NumPy and FFmpeg
are required for this step.

```sh
python3 backend/scripts/audio_mp3_residual_transfer.py freeze "$replay_dir/fit" \
  --codec "$codec_dir" --confirmation "$confirmation_dir"
python3 backend/scripts/audio_mp3_residual_transfer.py measure "$replay_dir/result.json" \
  --run "$replay_dir/fit" --codec "$codec_dir" --confirmation "$confirmation_dir"
```

Native replay verifies the committed bytes and exact decoded PCM for all
142 recordings before deriving the window features. The frozen fit and
complete outcome receipt reproduced byte-for-byte on a second local run.
