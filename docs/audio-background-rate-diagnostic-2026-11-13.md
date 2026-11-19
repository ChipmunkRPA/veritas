# False speech alerts under traffic noise

The same five genuine speakers triggered the optional background-reduced
speech review at 48 kHz and at the 16 kHz rate used by the app. The primary
speech detector issued no AI verdicts. The app now explains the risk beside
the flagged interval in the evidence view and the PDF. The optional review
carries zero weight in the overall assessment.

## Matched replay

Use the 12 genuine VCTK speakers, same three-second intervals, and same Vienna
roadside recording from the previous native-rate preflight. The clean and
−5 dB traffic files reproduce the earlier SHA-256 values, analyzed audio, and
decisions. Preserve a single foreground gain per speaker and vary traffic
relative to speech. The −5 dB setting puts speech 5 dB below traffic by RMS.
The 16 kHz repack stores the decoder's exact output in a new PCM16 WAV. The
direct 16 kHz mix resamples the two original tracks separately before mixing
them and changes the resulting samples by at most two 16-bit PCM steps. The
background-only control keeps the exact loud traffic level and excerpt.
[Source roster, assets, and licenses](audio-background-lossless-rate-transfer-2026-11-13.md#native-rate-controls)

| Condition | Primary AI directions | Usable background-reduced readings | False flags from the background-reduced review |
| --- | ---: | ---: | ---: |
| Clean 48 kHz | 0/12 | 1/12 | 0/12 |
| −5 dB traffic, native 48 kHz | 0/12 | 10/12 | 5/12 |
| −5 dB traffic, exact 16 kHz repack | 0/12 | 10/12 | 5/12 |
| −5 dB traffic, direct 16 kHz mix | 0/12 | 10/12 | 5/12 |
| 0 dB traffic | 0/12 | 9/12 | 3/12 |
| +5 dB traffic | 0/12 | 6/12 | 3/12 |
| Loud traffic alone | 0/12 | 0/12 | 0/12 |

The same three speakers remained flagged at both quieter levels. Traffic
alone never passed the speech gate. At the strongest setting the primary
returned eight inconclusive and two unassessed voice results, so the optional
review ran on ten clips. The 16 kHz rate restriction cannot prevent the five
reproduced errors. The matched experiment associates them with traffic mixed
with these voices under the current review. It does not isolate which behavior
of enhancement, speech admission, or classification caused the errors. These
are constructed three-second mixtures with one noise recording and twelve
speakers. They give no new estimate of synthetic detection or false alarms
on other recordings. Of the 19 noisy synthetic clips recovered by the same
review in the earlier 16 kHz panels, 10 scored at or below the highest false
flag in this group. Raising the operating point enough to suppress all five
of these errors would also suppress those 10 earlier detections. The point
and overall weighting stay fixed. A change needs fresh
genuine and synthetic speech, multiple new traffic recordings, and separate
evaluation on naturally noisy speech. [Frozen inputs and measured results](evaluation-panels/audio-background-rate-diagnostic-2026-11-13.json)

One flagged 16 kHz replay also passed the local upload and paid-analysis flow
with a payment stub. The primary assessment stayed inconclusive with or
without the optional review. The standard PDF showed the warning, original
waveform and source interval. The export and signed package retained the
matching evidence and verified against the uploaded samples.

## Paired waveforms in the local app

When this optional review flags a three-second interval, the evidence view
shows the original playback and the background-reduced copy with separate
waveforms. Both plots use the same source samples from 0.000 to 3.000 seconds
and the same fixed digital amplitude scale. The standard PDF places both
waveforms with their source timestamp. The JSON export identifies each
image and its corresponding audio clip. Signed original-file replay
recreates the original waveform from the uploaded source and the processed
waveform from the saved processed playback.

The local worker replayed one of the flagged genuine speech-and-traffic
fixtures and tried a separate silent control after simulated checkout. The
flagged report reproduced both images and their source positions. The silent
file was unassessed and had no paired playback or plots. The optional
response carries zero weight in the overall assessment.

## Run the review locally

The portable check recounts the branches and decisions without downloading
media or weights. To rerun model inference, use the already pinned local
assets and populate `$vctk_cache` with the source FLAC members listed in
`native_freeze.rows` of the previous receipt, named `00-source.flac` through
`11-source.flac` by row ID. The runner checks original member and decoded
PCM hashes, traffic recording hash, exact lengths, source clocks, and
identical prior WAVs before using the fixed operating point. It reuses the
score for the exact repack after verifying identical analyzed samples. The
direct mix is scored on its own samples.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/audio-background-rate-diagnostic-2026-11-13.json
python3 -B -S -m scripts.audio_background_rate_diagnostic review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_background_rate_diagnostic \
  freeze "$new_run" --cache "$vctk_cache" --background "$vienna_wav"
backend/.venv/bin/python -B -m scripts.audio_background_rate_diagnostic \
  score "$new_run" --cache "$vctk_cache" --background "$vienna_wav"
```
