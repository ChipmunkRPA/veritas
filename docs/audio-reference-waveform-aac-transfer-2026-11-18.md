# Earlier-copy waveform comparison across WAV, FLAC and AAC

The frozen [earlier-copy waveform check](audio-reference-waveform-2026-11-14.md)
located the same 40 of 81 staged speech edits in six-second WAV, FLAC and
FFmpeg AAC-LC deliveries at nominal 64, 96 and 128 kb/s. Every tested rate
flagged zero of 117 matched unchanged clips. The 32 kb/s delivery lost seven
locations found in the lossless versions and gained three. It fell below
the frozen paired retention criterion. These are earlier-copy timing and
continuity measurements with zero contribution to AI confidence.

## What was compared

Both retained SINE panels supplied 27 cut-and-paste clips and 12 same-word
resynthesis clips. Each of the 39 clips supplied an edited or resynthesized
version and a matched unchanged version. Each pair was delivered under
three backgrounds, for 234 pairs per encoding setting. The early copy is
the dry recording. The submitted clip contains competing speech from AMI
or the left channel of a DCASE bus or office recording. The unchanged and
edited variants use the same room samples at a nominal level 5 dB above
foreground speech. The original 44.1 kHz DCASE recordings contributed
six-second windows starting at 2, 8, 14 and 20 seconds.

- SINE uses recordings from LibriVox via LibriLight. The publisher marks
  its processed release Apache 2.0. The two prepared panels retain the
  source filenames, checksums and crop recipes. [SINE release](https://huggingface.co/datasets/PeacefulData/SINE) ·
  [Full panel](audio-sine-resynthesis-transfer-2026-11-14.md) ·
  [Second panel](audio-aligned-speech-app-2026-11-14.md#held-comparison)
- AMI released the competing-speaker recordings under CC BY 4.0. [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- D. Giannoulis, E. Benetos, D. Stowell and M. D. Plumbley released the public
  2013 DCASE scene archive under CC BY 2.0 UK. The exact stereo archive SHA-1
  was checked against the publisher metadata. [Publisher metadata](https://archive.org/metadata/dcase2013_scene_classification) ·
  [DCASE data description](https://dcase.community/challenge2013/task-acoustic-scene-classification)

The [frozen plan](evaluation-panels/audio-reference-waveform-aac-freeze-2026-11-18.json)
requires at least half of the labeled range and 0.30 temporal intersection
over union for a useful location. It requires AAC to preserve 85% of the
lossless matches and permits no more than three net losses in any
background. It permits up to two additional ordinary alerts from one reader.
WAV and FLAC were native 16 kHz mono PCM16 with matching decoded samples.
Every FFmpeg M4A was native 48 kHz mono AAC-LC with an exact zero origin and
288,000 presented samples. Only verified decoder padding beyond the
presentation extent was excluded.

## Measured results

The rates below are nominal encoder settings. Each bitrate range comes from
the audio stream in the same scored files. Each setting tested all 234 clips.

| Delivery | Observed stream bitrate (bps) | Bus edits | Competing-speaker edits | Office edits | WAV/FLAC matches preserved | Ordinary flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WAV or FLAC | Lossless | 12/27 | 13/27 | 15/27 | 40/40 | 0/117 |
| [AAC 32 kb/s](evaluation-panels/audio-reference-waveform-aac-32-2026-11-18.json.gz) | 31,034–32,465 | 11/27 | 10/27 | 15/27 | 33/40 | 0/117 |
| [AAC 64 kb/s](evaluation-panels/audio-reference-waveform-aac-64-2026-11-18.json.gz) | 64,149–65,755 | 12/27 | 13/27 | 15/27 | 40/40 | 0/117 |
| [AAC 96 kb/s](evaluation-panels/audio-reference-waveform-aac-96-2026-11-18.json.gz) | 96,188–100,058 | 12/27 | 13/27 | 15/27 | 40/40 | 0/117 |
| [AAC 128 kb/s](evaluation-panels/audio-reference-waveform-aac-128-2026-11-18.json.gz) | 121,585–132,104 | 12/27 | 13/27 | 15/27 | 40/40 | 0/117 |

There were 43 proposed ranges on edited clips in lossless and 64 kb/s AAC.
Three missed the overlap requirement. In 39 of 43 proposals both endpoints
matched exactly between those encodings. The largest endpoint change was
40 ms. All edit and unchanged pairs were assessable at each setting. None
of the 36 resynthesis pairs showed a flagged range in any setting. The
receipts retain their separate abstention counts.

The [Apple clock probe](evaluation-panels/audio-reference-waveform-aac-apple-clock-2026-11-18.json)
encoded a dry clip, an edit and an unchanged noisy clip with macOS
`afconvert` at nominal 64 kb/s. All three returned a presentation start
of 2112 ticks on the native 48 kHz clock. The exact-clock check requires
a zero start and abstained on all three. The bitrates and encoders above describe
this finite panel. The evaluation reused known edited source clips and
room excerpts to measure transport on matched files. Other edit generators,
arbitrary recordings and unverified AAC clocks are outside the test scope.
The receipts contain measurements and hashes.

## Evidence in the local app

The earlier-copy waveform check accepts six-second mono 48 kHz AAC-LC in
M4A when its presentation begins at zero, its measured audio bitrate is
at least 60 kbps, and the earlier 16 kHz mono PCM16 WAV or FLAC spans the
same six seconds. This uses the existing clean-copy and waveform settings
together with `VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_AAC=true`.

The comparison view and PDF show the matching waveform on both copies,
including the native AAC timestamps. The JSON commits both clocks and
the exact playable excerpts. AAC processing can add samples before or
after the intended audio. The check binds the zero-origin presentation
extent and excludes trailing decoded samples outside it.
[Apple's AAC timing documentation](https://developer.apple.com/documentation/quicktime-file-format/appendix_g_audio_priming_handling_encoder_delay_in_aac)

Two simulated USD 9.99 checkouts tested the browser and both downloads.
The edited sample reported 2.740–3.809 seconds on the AAC file and
2.740–3.940 seconds on the earlier FLAC. Both excerpts independently
matched the decoded sources. The unchanged clip returned no range.
[Checkout and browser receipt](evaluation-panels/audio-reference-waveform-aac-app-2026-11-18.json)

## Replay

Use Python 3.11 or later to recount any receipt with no model weights or
audio. For a full readback prepare the pinned SINE folders and DCASE archive.
The local readback used FFmpeg 7.0.2. Temporary audio is removed as each
source group finishes.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
base=docs/evaluation-panels/audio-reference-waveform-aac
python3 -B -S -m scripts.audio_reference_waveform_aac_transfer review "$base-64-2026-11-18.json.gz"

backend/.venv/bin/python -B -m scripts.audio_reference_waveform_aac_transfer run \
  "$new_receipt" "$dcase_archive" "$full_sine_bank" "$short_sine_bank" --kbps 64
```

To replay the app on a matched pair, supply `earlier.flac`, `positive.m4a`
and `ordinary.m4a` in one local folder. The app smoke requires the local
analysis weights and a built frontend. It uses a test payment provider.

```sh
backend/.venv/bin/python -B -m scripts.audio_waveform_aac_app \
  "$new_app_output" "$aac_fixtures" --browser
```
