# Locate speech changes under competing voices with an earlier clean copy

The six-second earlier-copy option gained eight correctly located
synthetic replacements on each noise level in the held comparison.
At −5 dB it reduced unrelated processing highlights from five to one
while keeping all six correctly positioned baseline findings. The
option measures a change relative to a clean earlier recording and
carries zero weight in the AI estimate.
[Frozen study plan](evaluation-panels/audio-dense-clean-reference-plan-2026-11-13.json)

## How it measures the change

The check first requires broadband waveform agreement across the
same-clock pair. In three speech-frequency bands, it estimates how
much of the earlier waveform is retained in the submitted recording.
It uses 120 ms frames every 40 ms. A 960 ms interval is eligible
when the available flanks, up to 640 ms on each side with 80 ms
guards, support a comparison in at least two bands. Within each band
the check measures the signed gain of the earlier waveform and
compares the interval with the weaker flank. A negative gain can
produce a drop statistic above one. The statistic describes waveform
agreement. Ordinary speech splices and some level changes yield the
same observation. This uses the existing reference-match and band
logic with a denser grid and its own operating point. It loads no
additional model.
[Existing 720 ms method](audio-clean-reference-transfer-2026-11-13.md)
· [Research on separating speech and environmental evidence](https://arxiv.org/abs/2606.10791)
· [Research on mixed audio and temporal localization](https://arxiv.org/abs/2609.03620)

## Local results

The development bank used eight hosts with matched untouched, ordinary
splice and synthetic splice deliveries. It selected the geometry before
the assessment and used the maximum development control score as its
operating point. The held placement bank used eight other English
audiobook readers and two licensed stock-voice generators. It includes
retained generated and ordinary splices in 960 ms ranges, plus separately
constructed gain, low-pass, masker-only and wrong-reference controls.
Three held AMI meeting groups provide overlapping background voices.
The bank had previously been used to test standalone classification.
These are new earlier-copy measurements, not new raw source recordings.
[Development sources and terms](audio-stock-voice-transfer-2026-11-13.md)
· [Held placements and terms](audio-random-clock-stock-2026-11-13.md)

| Outcome | Existing at 0 dB | Denser at 0 dB | Existing at −5 dB | Denser at −5 dB |
| --- | ---: | ---: | ---: | ---: |
| Synthetic replacements usefully located | 6/16 | 14/16 | 6/16 | 14/16 |
| Ordinary replacement splices usefully located | 6/16 | 14/16 | 6/16 | 14/16 |
| Wrong interval on a synthetic replacement | 0/16 | 0/16 | 0/16 | 0/16 |
| Untouched, gain or low-pass processing controls highlighted | 5/24 | 0/24 | 5/24 | 1/24 |
| Unrelated-reference or masker-only pairs admitted | 0/16 | 0/16 | 0/16 | 0/16 |

The new locations span four additional hosts and both stock-voice
generators. Both noise levels reuse the same eight hosts and 16
synthetic edits. The one control flag at −5 dB came from a local drop
in volume. The recovered ranges cover at least half of the true
edit with at least 0.30 temporal intersection-over-union. The work
does not measure standalone detection, natural room remixes, codec
changes, clock drift or short edits without supporting flanks.
[Scalar decision ledger](evaluation-panels/audio-dense-clean-reference-2026-11-13.json)

## Use on the local instance

Enable `VERITAS_AUDIO_REFERENCE_DENSE_CLEAN_SPEECH=true` alongside
the existing comparison and clean-speech options. The repository keeps
the new switch off by default. For the six-second method, use two
continuous 16 kHz mono PCM16 WAV or FLAC files with identical timing.
Open a completed audio analysis, choose **Compare with an earlier
audio recording**, and select the clean-copy checkbox. The existing
twelve-second option continues to use its original method.

The browser, comparison PDF and JSON show the selected interval and
both waveforms. The browser and JSON also carry playable excerpts.
The measurements give before, within and after gain for each band.
An unrelated or insufficiently matched reference yields a coverage
reason. Exports save the comparison method so the old and new
versions can replay against the same source files.

In two simulated prepaid analyses, no job ran before confirmation.
The new method recovered **0.560–1.520 s** on a held edit, compared with
the planted **0.622–1.582 s** interval. The two exported waveforms and
playable excerpts matched both source files at ticks **[8960, 24320)**.
An untouched recording produced no local finding. An unrelated reference
was unassessed. All three comparisons preserved the original assessment.
[Local application verification](evaluation-panels/audio-dense-clean-reference-app-2026-11-13.json)

## Replay

The first command recounts the public ledger using only standard-library
Python. The second rechecks all 352 pairs on the retained native banks.
The public files contain waveform hashes and measurements. The source
audio and reader identities stay local.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.audio_dense_clean_reference_transfer review \
  docs/evaluation-panels/audio-dense-clean-reference-2026-11-13.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.audio_dense_clean_reference_transfer score "$new_receipt" \
  --development "$stock_voice_bank" --assessment "$random_placement_bank"
```

Use the independently decoded source waveform and full report replay for
a downloaded comparison JSON:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.verify_audio_clean_speech_export \
  earlier.wav submitted.wav comparison.json
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.compare_audio_references \
  earlier.wav submitted.wav comparison.json --verify
```
