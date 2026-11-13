# Short speech edits under overlapping voices

A single-upload pitch check did not find additional synthetic replacements
at its frozen operating point. It recovered an exact quieter-pitch change
in a controlled two-tone recording. Under competing recorded speech, the
same check frequently highlighted unrelated portions of the recording.
The app continues to use its existing voice check.
[Portable evaluation receipt](evaluation-panels/audio-overlap-pitch-transfer-2026-11-13.json)

## Method and sources

The previous periodic-excitation test followed the dominant period of a
mixture. This check estimates two harmonic series from the submitted
mono waveform. It looks for the second series to change on both sides
of an 800 ms range while the strongest series stays comparatively
continuous. A 64 ms Hann transform advances every 20 ms. A frequency
whitening step reduces the influence of broad spectral shape. The
second pitch needs at least five qualifying partials away from the
dominant comb. Local comparisons skip the nearest 20 ms at each
boundary. Neither series identifies a speaker. The method was written
locally with NumPy and SciPy. It uses no model, external implementation,
earlier copy, clean stems, estimated or known noise level, or edit
coordinates to score the submitted WAV.
[Prior periodic-excitation test](audio-periodic-excitation-transfer-2026-11-12.md)
· [Research on tracking overlapping pitches](https://ieeexplore.ieee.org/document/9381673)
· [Research on speech transitions at partial edit boundaries](https://arxiv.org/abs/2604.01083)

The primary bank has 108 six-second mono WAVs from four readers. It
contains dry submissions and +5, 0 and −5 dB mixtures. The extra transfer
bank uses 72 submitted WAVs from three other held readers and different
recorded AMI background sessions. It adds 0, −5 and −10 dB conditions
and an ordinary replaced passage. Neither bank is fresh to the project.
They were locked before inference for this method, and prior methods
had already measured their source recordings. Synthetic sources use
two synthesis mechanisms from one publisher. The assessment also
uses untouched, gain, ordinary resampling, and where available
equalization and isolated background controls. No media or reader
identifiers are in the scalar receipt.
[Primary source and rights](audio-phoneme-short-edit-intake-2026-11-13.md)
· [Transfer source and rights](audio-clean-reference-transfer-2026-11-13.md)

The runner binds all 180 waveform hashes, the preceding receipts and its
own code before scoring. It seals the candidate point at **0.44448318**,
the strict maximum of 30 assessed ordinary development recordings out
of 38. Native incumbent scores come from the earlier frozen comparison.
The reader checks all 756 recorded Full and short native window hashes
against the identical original samples. No new neural inference was
needed. A useful interval covers half of the actual edit and has at
least 0.30 temporal intersection over union. The promotion rule
requires at least two more useful primary localizations than the
incumbent at −5 dB with at most one extra false flag, and at least
two localizations at both −5 and −10 dB on transfer with at most
one false flag at each level.
[Earlier pinned score receipt](audio-phoneme-short-edit-score-2026-11-13.md)

## Held results

| Source and condition | Useful candidate / synthetic | Candidate false / ordinary | Useful existing short check / synthetic | Candidate unassessed synthetic / ordinary |
| --- | ---: | ---: | ---: | ---: |
| Primary dry | 0/4 | 0/8 | 4/4 | 3 / 8 |
| Primary +5 dB | 0/4 | 0/10 | 3/4 | 2 / 8 |
| Primary 0 dB | 0/4 | 0/10 | 2/4 | 4 / 4 |
| Primary −5 dB | 0/4 | 0/10 | 0/4 | 1 / 0 |
| Transfer 0 dB | 0/6 | 0/18 | Unmeasured | 2 / 5 |
| Transfer −5 dB | 0/6 | 0/18 | Unmeasured | 2 / 6 |
| Transfer −10 dB | 0/6 | 0/18 | Unmeasured | 0 / 0 |

Before applying the point, the candidate found a useful strongest
interval on one of four primary edits at −5 dB. The existing short
check found three useful strongest intervals there. On transfer
the candidate found one of six at −5 dB and two of six at −10 dB.
At −10 dB, three pairs of synthetic mechanisms produced exactly
the same selected interval and score. Two of those intervals missed
the edits in both versions. At −5 and −10 dB, the candidate never ranked
an assessed synthetic edit above its paired ordinary resampling control
on transfer. These observations are consistent with ordinary competing
speech determining the selected intervals. They do not identify the
source of individual pitch estimates.

Digital silence and a single harmonic series produced no eligible range.
Two constant harmonic series produced zero response. An 800 ms change in
the weaker series yielded the exact intended sample interval
`[33792, 46592)` with a 0.97926931 score. The idealized controls exercise
the intended measurement. The held speech comparison found no evidence
for using it as a standalone synthetic edit detector.

## Replay

Rebuild the six-second submitted PCM files from the source-cache steps in
the linked primary and transfer studies. Use the dictionaries returned by
`scripts.audio_phoneme_short_edit_score.file_inputs` and
`scripts.audio_clean_reference_transfer.inputs`. For each primary row
write the original waveform as `{case}_{level}_{kind}.wav`, and for each
transfer row write the submitted waveform as `{case}_{level}_{kind}_submitted.wav`.
Use `None` in dry filenames, and signed 16-bit mono WAV at 16 kHz. The
runner rejects a changed waveform or sample clock.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_overlap_pitch_transfer "$phase" \
    --run "$run" --primary-wavs "$primary_wavs" --transfer-wavs "$transfer_wavs" || exit 1
done
python3 -B -S backend/scripts/review_audio_overlap_pitch_transfer.py \
  docs/evaluation-panels/audio-overlap-pitch-transfer-2026-11-13.json
```

The native rerun reproduced all 180 measurements byte for byte.
The separate standard-library reader checked the fitted point,
localization, coverage, false-alarm and matched-ranking arithmetic.
