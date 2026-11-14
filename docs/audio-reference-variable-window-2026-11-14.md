# Frequency changes in an earlier-copy speech comparison

The local trial checks whether a broad shift in frequency balance can explain
the difference between a submitted voice clip and an earlier clean copy.
When it can, the app reports the measured frequency changes and leaves the
local interval unhighlighted. The browser, PDF and JSON use the same five
bands and three two-second sections. Supported local changes still show the
source times, both waveforms and both excerpts. The comparison contributes
zero weight to the overall AI estimate.

Both files need six seconds of continuous 16 kHz mono PCM16 WAV or FLAC
on the same sample clock. The recordings must first pass the existing
matching-sound rule. The check divides each band's matching-sound gain by
the mean gain in the two lowest bands. It applies the
[existing consistency conditions](audio-clean-reference-duration-2026-11-13.md)
over three shorter sections. The first and last sections must provide matching
sound, and the middle section also contributes when it has enough. The reported
pattern is consistent with broad processing such as equalization. A local
edit and broad processing may coexist.

## Local comparison

The 824 pairs reuse recorded source banks with eight calibration hosts,
eight distinct held hosts, and 24 previously evaluated SINE pairs across
20 readers. The held hosts supply two synthetic replacements and two human
splices each at preassigned edit times. Six-second copies have no added
noise, recorded rain, or recorded train noise at −5 dB relative to the
original speech. All variants in a group use the same background samples
and peak-safe audio scaling. Ordinary controls include intact speech,
gain changes, whole-clip filtering, background alone and unrelated earlier
copies. Human splices are listed separately because their locations really
changed. The different mixes reuse the same edits.
[Development stock voices and terms](audio-stock-voice-transfer-2026-11-13.md)
· [Held stock voices and edit clocks](audio-random-clock-stock-2026-11-13.md)
· [SINE source selection](audio-sine-resynthesis-transfer-2026-11-14.md)
· [Background recordings and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

The comparison also tested 320, 480 and 640 ms windows as a fallback to
the existing 960 ms check. Each shorter window used the largest ordinary
calibration response as its operating point. Useful overlap requires half
the known edit and temporal intersection over union of at least 0.30.

| Held measurement | Existing | With shorter fallback | With frequency balance |
| --- | ---: | ---: | ---: |
| Synthetic replacements located, clean | 14/16 | 14/16 | 14/16 |
| Synthetic replacements located, rain | 14/16 | 14/16 | 14/16 |
| Synthetic replacements located, train | 14/16 | 14/16 | 14/16 |
| Human splices located, each condition | 14/16 | 14/16 | 14/16 |
| Ordinary pairs flagged locally, held stock | 1/112 | 3/112 | 0/112 |
| Ordinary pairs flagged locally, SINE | 2/336 | 3/336 | 1/336 |
| SINE cut-and-paste edits located, each condition | 0/12 | 0/12 | 0/12 |

Eight of the 12 SINE cut-and-paste pairs passed the matching-sound rule in
each condition. None of the 12 resynthesis pairs passed it. The existing
960 ms window remains in the app. The trial reclassified two filter-induced
locations, one under train noise and one under rain, while preserving all
useful held locations. The study reused earlier source rosters and two
backgrounds. Results apply to these matched recording pairs.
[Measurement ledger](evaluation-panels/audio-reference-variable-window-2026-11-14.json)

## Local use and replay

Enable `VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_EQ_GUARD=true` alongside the
dense clean-speech comparison. In a completed retained audio analysis, open
**Compare with an earlier audio recording** and choose **The earlier file is
a clean copy of the same recording**. The source setting defaults to off.
Three simulated prepaid app runs exercised the filter case, a synthetic
replacement under train noise, and the intact control. The filter report
showed all three measured time sections. The replacement kept its original
**2.800–3.760 s** interval and both source excerpts. The saved overall
assessments matched, and zero jobs ran before payment. The exact-source
replays matched the exported measurements and verified both excerpts.
[Local app receipt](evaluation-panels/audio-reference-frequency-balance-app-2026-11-14.json)

The scalar review needs only standard-library Python. Add the three source
banks linked above and a folder with the pinned `rain.ogg` and `train.ogg`
for a native replay of all 824 pairs. The ledgers contain scores, positions
and decoded sample hashes. Source recordings stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/audio-reference-variable-window-2026-11-14.json
python3 -B -S -m scripts.audio_reference_variable_window review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_reference_variable_window review "$receipt" \
  --development "$development" --assessment "$assessment" --sine "$sine" --sounds "$sounds"
backend/.venv/bin/python -B -m scripts.audio_reference_frequency_balance_app "$app_run" \
  --bank "$assessment" --sounds "$sounds"
backend/.venv/bin/python -B -m scripts.verify_audio_clean_speech_export \
  earlier.wav submitted.wav comparison.json
```
