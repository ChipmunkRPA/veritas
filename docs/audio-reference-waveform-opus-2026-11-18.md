# Earlier-copy speech evidence for Ogg Opus

The earlier-copy waveform check can locate changes in the tested six-second
mono 48 kHz Ogg Opus recordings at a constant 64 kbps. It uses an earlier
16 kHz mono PCM16 WAV or FLAC of the same recording. Locations show where
matching speech changes timing and the waveform match falls, with exact
playable excerpts. Ordinary edits can produce the same evidence. The check
contributes zero weight to AI confidence.

## Paired recording trial

The [frozen procedure](evaluation-panels/audio-reference-waveform-opus-freeze-2026-11-18.json)
checked each of 96 reference pairs with the existing lossless workflow and
with 64 kbps constant-bitrate and 24 kbps variable-bitrate Opus. The source
was 16 SINE readers mixed with four AMI room excerpts. The same unchanged
foreground, same-recording splice and publisher-annotated cut-and-paste
version went through each codec at both background levels. The earlier
recording was always the clean six-second crop. Foreground speech was
nominally 5 dB quieter or 3 dB louder than the room sound.
[Exact source reservation](evaluation-panels/audio-standalone-dual-view-sources-2026-11-18.json)

| Submitted carrier | Located at −5 dB | Located at +3 dB | Lossless locations preserved | Unchanged flags | Ordinary splice flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lossless PCM16 | 7/16 | 9/16 | 16/16 | 0/32 | 0/32 |
| Ogg Opus at 64 kbps constant bitrate | 6/16 | 9/16 | 15/16 | 0/32 | 0/32 |
| Ogg Opus at 24 kbps variable bitrate | 7/16 | 7/16 | 13/16 | 0/32 | 0/32 |

All comparisons passed the exact clock and all 96 cases per format were
assessed. The comparison required at least half of the annotated samples
and 30% of the temporal union for a useful location. The 64 kbps profile
passed the frozen 85% paired-retention rule. The 24 kbps profile gained
one location and lost three and stayed below the retention rule. It remains
outside the app’s waveform evidence. This is transport qualification on a
controlled panel. The same readers appeared in a separate standalone
evaluation. The earlier copy supplies information missing from a single
recording, and these results give no estimate for standalone performance or
arbitrary Opus voice notes. [Case measurements](evaluation-panels/audio-reference-waveform-opus-2026-11-18.json.gz)

The SINE publisher release carries Apache 2.0 terms. AMI recordings carry
CC BY 4.0 attribution terms. The receipts retain scalar readings, time
ranges, source hashes and publisher identifiers. The source recordings are
distributed separately by their publishers.
[SINE release](https://huggingface.co/datasets/PeacefulData/SINE#license) ·
[AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Playback positions

The app checks Ogg page checksums, the stream serial sequence, and each
page's granule position. On the qualified profile, the 301 packets occupy
20 ms each. The header and demuxer agree on the 312 leading samples and
648 trailing samples to omit. The 48 kHz playback clock starts at zero
and extends over 288,000 samples. The comparison decoder must independently
produce exactly 96,000 contiguous 16 kHz samples beginning at zero.
The browser, report and JSON use elapsed playback time and the JSON
includes the corresponding 48 kHz tick range. Alternate Opus settings,
incomplete pages and streams with conflicting timing remain unavailable
for this comparison. [Opus framing and playback specification](https://www.rfc-editor.org/rfc/rfc7845.html#section-4.3)

The comparison is available with the existing clean-speech and matching
waveform settings and the compressed waveform option
`VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_AAC=true`. It runs when the user
provides an earlier clean copy of the same recording.

Three simulated USD 9.99 checkouts exercised the local worker, browser,
PDF, JSON and independent original-file replay. The edited 64 kbps clip
showed 4.020–4.516 seconds in the submission and 4.020–4.380 seconds in
the earlier FLAC. Both playable excerpts matched the re-decoded samples.
The unchanged 64 kbps clip returned no range. The 24 kbps export was
unassessed for matching waveform evidence. The three original assessments
were preserved and no real payment was processed.
[Local checkout receipt](evaluation-panels/audio-reference-waveform-opus-app-2026-11-18.json)

## Local replay

The recount uses Python’s standard library and the retained commitments.
Full replay uses FFmpeg 7.0.2 with libopus, the existing Python dependencies
and the exact SINE and AMI inputs. The output file needs an unused path.
The [source-replay guide](audio-standalone-dual-view-2026-11-18.md#replay)
shows how to restore these recordings from publisher archives.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=scripts.audio_reference_waveform_opus_transfer
python3 -B -S -m "$module" review \
  docs/evaluation-panels/audio-reference-waveform-opus-2026-11-18.json.gz
backend/.venv/bin/python -B -m "$module" run \
  "$new_receipt" "$run_sources" "$ami_backgrounds"
backend/.venv/bin/python -B -m "$module" fixtures \
  "$new_fixtures" "$run_sources" "$ami_backgrounds"
backend/.venv/bin/python -B -m scripts.audio_waveform_aac_app \
  "$new_app_output" "$new_fixtures" --opus --browser
```
