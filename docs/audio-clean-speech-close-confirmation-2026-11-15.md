# Find more short audio changes with both stereo channels

Chipmunk Verify can now find additional short speech changes when a
supported 12-second, 48 kHz stereo AAC upload is compared with a clean
earlier recording. The existing comparison uses two six-second checks
with starts 1.2 seconds apart. When it yields no interval, the new
fallback tries starts 0.8 seconds apart. It requires the mono mix and
both physical channels to support a common span of at least 360 ms.
The matching criteria and frequency-balance exclusions remain in use.
[Earlier dense stereo comparison](audio-reference-fresh-room-validation-2026-11-15.md)

## Native comparison

Rebuild 528 published comparison pairs from their committed source files
and decoded channels. They include two synthesis methods, genuine
splices, resampled and untouched foreground speech, and reference
substitutions. Recorded room speech is 5 dB louder than the foreground.
The first two panels use four AMI meeting recordings. The third uses
three other meetings and a different microphone pair. All test inputs
had prior project exposure. The last panel reuses the same foreground
groups as the first two, so its gains measure room and microphone
transfer. [Native room and source commitments](audio-stereo-room-transfer-2026-11-15.md)
· [AMI corpus licence](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Compare the serving schedule with three fallback rules. The first allows
an additional pair of checks 0.8 seconds apart with the earlier stereo
corroboration. The second requires three consecutive starts with a common
half-range. The third uses the closer pair and requires a common interval
across all three audio views. Each retains the serving interval whenever
there is one. The useful-overlap rule requires at least half the staged
edit and 30% temporal intersection over union.

| Rule | Correct synthetic ranges | Correct genuine splice ranges | Ordinary flags | Alerts with room speech as reference |
| --- | ---: | ---: | ---: | ---: |
| Serving schedule | 56/96 | 28/48 | 0/144 | 2/48 |
| Closer pair | 60/96 | 30/48 | 0/144 | 3/48 |
| Three consecutive starts | 60/96 | 30/48 | 0/144 | 3/48 |
| Closer pair with both channels | 60/96 | 30/48 | 0/144 | 2/48 |

The selected fallback added three correct ranges in the first panel and
three in the room and microphone transfer. The second panel's results
were identical across rules. The fallback preserved every previous
location and added no misplaced ranges or added alerts on unrelated,
half-matching, or room-only controls. The two earlier alerts when room
speech served as the reference remain possible with the serving schedule.
The finding measures shared sound across the two files. It contributes
zero weight to AI confidence. [Numeric comparison](evaluation-panels/audio-clean-speech-close-confirmation-2026-11-15.json.gz)

## Local app verification

Set `VERITAS_AUDIO_REFERENCE_STEREO_CLOSE_CONSENSUS=true` with the earlier
stereo speech comparison enabled. The new option defaults off in source
configuration. The result records the comparison window starts for both
channels and the mono mix and displays them beside the corresponding time
range in the app and PDF. The JSON verifier recomputes both native channel
ranges and their starts from the two files. Earlier schema versions replay
under their original schedules.

The local fixture used two simulated USD 9.99 payments and three
reference comparisons. It recovered **9.520–10.240 s** in the edited
recording. The mono mix and both channels agreed using checks starting
at **5.200 s** and **6.000 s**. The untouched and room-reference
comparisons had no interval. Both waveform excerpts verified against the
source recordings and the saved AI assessments retained their hashes.
No analysis job queued before payment. [Local app receipt](evaluation-panels/audio-clean-speech-close-confirmation-app-2026-11-15.json)

## Replay

The portable reviewer compares the earlier receipts, checks window
geometry, and recalculates all outcomes. Its results matched on Python
3.12 and 3.14. The public data records sample positions and measurements.
The audio remains in the licensed source archives or on the test machine.

```sh
export PYTHONPATH=backend
module=scripts.audio_clean_speech_close_confirmation
receipt=docs/evaluation-panels/audio-clean-speech-close-confirmation-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Use the ICSI and AMI inputs from the
[original stereo native replay](audio-reference-spatial-schedule-2026-11-15.md#replay).
The transfer command re-fetches bounded public AMI excerpts and verifies
each microphone's bytes against the previous room transfer before scoring.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" capture "$icsi" "$first_rooms" "$held_rooms" "$run/retained.json.gz" --panel retained
"$runner" -B -m "$module" capture "$icsi" "$first_rooms" "$held_rooms" "$run/fresh.json.gz" --panel fresh
"$runner" -B -m "$module" merge "$run/retained.json.gz" "$run/fresh.json.gz" "$run/merged.json.gz"
cmp "$receipt" "$run/merged.json.gz"
"$runner" -B -m scripts.audio_clean_speech_aac_app "$run/app" \
  --receipt docs/evaluation-panels/audio-clean-speech-schedule-fresh-rooms-2026-11-15.json.gz \
  --icsi "$icsi" --long --carrier aac48_stereo --close-consensus
```
