# Earlier-copy waveform evidence for Apple AAC exports

Some Apple AAC exports signal how many leading and trailing samples should
be omitted during playback. An export can contain six seconds of usable
audio while its packet clock covers a longer interval. Chipmunk Verify
now accepts the tested mono AAC-LC M4A profile when its gapless table,
packet clock, and decoded clock agree. The comparison requires an earlier
six-second 16 kHz mono WAV or FLAC and a submitted 48 kHz stream with a
measured bitrate of at least 60 kbps. It contributes zero weight to AI
confidence. [Apple packet table documentation](https://developer.apple.com/documentation/audiotoolbox/audiofilepackettableinfo)

For each accepted file, the check requires contiguous 1,024-sample packets
and a declared 2,112-sample encoder delay confirmed by the decoder. The
reported 288,000 usable native samples must fit exactly between that delay
and the declared remainder. The browser, PDF and JSON show elapsed time
from the start of playback. The JSON also preserves the actual native
sample positions, including the offset beyond the encoder delay.

## Paired check against lossless audio

The [frozen plan](evaluation-panels/audio-gapless-aac-freeze-2026-11-18.json)
reused the same 234 source pairs in the [earlier codec study](audio-reference-waveform-aac-transfer-2026-11-18.md).
The submitted recordings used the same bus, office and competing-speech
backgrounds. Both Apple encodes used the same PCM, source truth,
and waveform rule. These measurements test format transfer on known pairs.

| Nominal Apple encoder setting | Observed stream bitrate | Clock verified | Files below bitrate floor | Lossless locations preserved after eligibility | Unchanged files flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 kbps | 76,979–80,356 bps | 234/234 | 0 | 40/40 | 0/117 |
| 64 kbps | 58,037–61,591 bps | 234/234 | 185 | 5/40 | 0/117 |

Both encodes located 40 of 81 staged cut-and-paste edits when scored before
file eligibility. No same-word resynthesis case produced a range. The
higher-quality export passed the frozen approval rule. The lower setting
had too many files below the existing minimum to pass as a group. The
evaluation covers this six-second AudioToolbox profile. Arbitrary iPhone
recordings and other sample rates need their own clock and accuracy checks.

[96 kbps receipt](evaluation-panels/audio-gapless-aac-96-2026-11-18.json.gz) ·
[64 kbps receipt](evaluation-panels/audio-gapless-aac-64-2026-11-18.json.gz)

Two simulated USD 9.99 checkouts verified the completed local application.
The edited clip reported 2.740–3.809 seconds from the start of playback,
corresponding to native AAC ticks `[133632, 184959)`. The earlier FLAC
showed 2.740–3.940 seconds. The unchanged comparison returned no range.
The browser downloads agreed with the API and the JSON replay verified
both excerpts against the original files. [Local checkout receipt](evaluation-panels/audio-gapless-aac-app-2026-11-18.json)

## Reproduce

The results can be recounted without media or model weights. Recreating
the Apple AAC files uses macOS AudioToolbox. The file hash records the
exact scored export, which also includes a creation timestamp. Replay
compares the decoded samples, clock, and range as well as the source and
fixture hashes.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -B -S -m scripts.audio_gapless_aac_transfer review \
  docs/evaluation-panels/audio-gapless-aac-96-2026-11-18.json.gz
backend/.venv/bin/python -B -m scripts.audio_gapless_aac_transfer run \
  "$new_receipt" "$dcase_archive" "$full_sine_bank" "$short_sine_bank" --kbps 96
backend/.venv/bin/python -B -m scripts.audio_waveform_aac_app \
  "$new_app_output" "$aac_fixtures" --gapless --browser
```
