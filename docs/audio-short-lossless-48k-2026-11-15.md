# Short speech review for six-second 48 kHz WAV and FLAC

The Full audio analysis can run the existing two-second speech check
on supported 48 kHz WAV and FLAC uploads. The source must contain one
mono 16-bit stream with exactly 288,000 original samples and a continuous
zero-based presentation. The app checks the original decoded frame
timestamps and sample counts against the container. It must also recover 96,000 samples on the
16 kHz analysis clock. The admission check uses both lengths because
resampling can round a source with one extra sample to the same length.

The browser and Standard PDF identify the decoded input and show the
flagged window, waveform, and available nearby readings. JSON includes
the original time base and duration alongside the analysis positions.
Signed replay verifies these fields against the original file and
recreates the waveform and listening clip. The short review carries zero
weight in the overall assessment. It uses the existing voice and speech
activity weights. [Commercial asset review](commercial-model-audit.md)

## Evaluation protocol

Replay the 302 frozen six-second PCM recordings from the earlier audio
study. Cast each 16 kHz source to 48 kHz without dither and store the
same rendered native samples as PCM16 WAV and FLAC. Verify independent
lossless decoding at 48 kHz, then score the 16 kHz samples returned by
the actual app decoder. The short-window schedule, speech activity rule,
and operating point carry over from the 16 kHz lossless path. No fit is
performed on these inputs. Matched rows retain their original edit clocks,
source grouping, and labels for genuine speech, edits, and isolated
backgrounds. The source waveforms and model assets are verified by digest.
[Source groups and permissions](audio-short-edit-input-views-2026-11-15.md#sources)

Count the results separately by panel and carrier. A useful result covers
at least half the known edit and at least 30% of the temporal union.
With fewer than three prior useful locations, retain them all. Otherwise
retain at least 80%. Allow up to two newly flagged genuine files per panel
and no newly flagged isolated backgrounds. The experiment measures the
effect of delivering retained recordings at 48 kHz. Performance on clips
originally captured at 48 kHz remains a separate question.

## Results

Both formats retained all four useful locations and produced the same
counts as the original 16 kHz inputs. One flagged edited file moved to
another window and still missed the known interval. Across the full panel,
33/128 edited files flagged, with four useful ranges and 29 misplaced
ranges. The 27/128 genuine flags are existing false alarms in these
difficult controls. Five of the 46 isolated backgrounds had no usable
speech windows. Neither delivery added or removed an ordinary flag.

| Held inputs | Correct edit ranges at 16 kHz and in each 48 kHz format | Genuine files flagged | Isolated backgrounds flagged |
| --- | ---: | ---: | ---: |
| Earlier noisy reader comparison | 1/32 | 10/32 | 0/6 |
| PhonemeDF mixtures | 2/48 | 16/48 | 0/24 |
| Stock voice mixtures | 0/16 | 0/16 | 0/8 |
| Separate SINE readers | 1/32 | 1/32 | 0/8 |

A second media pass verified all 604 original-rate frame clocks and
reproduced the encoded inputs and analyzed PCM digests exactly. An
independent numeric recount agreed with the published control and
placement counts. [Measurement receipt](evaluation-panels/audio-short-lossless-48k-transfer-2026-11-15.json.gz)

## Local audio analysis

Enable `VERITAS_AUDIO_SHORT_LOSSLESS_48K_REVIEW=true` in development with
the candidate audio configuration. Six simulated USD 9.99 checkouts ran
the local worker. Both 48 kHz edit deliveries selected **4.000–6.000 s** and
produced identical playback and waveform assets. The staged replacement
itself occupies **4.780–5.220 s**. The range in the report is the window
the app inspected. A matched genuine FLAC produced no short alert. A WAV
with one extra sample and a FLAC with a conflicting duration header were
unassessed for this check even though both yielded six decoded seconds.
The PDF, JSON, and signed package were checked against the uploads. The
same WAV analyzed with the option off kept its overall assessment. No jobs
queued before simulated payment. [Local worker receipt](evaluation-panels/audio-short-lossless-48k-app-2026-11-15.json)

## Reproduce

The numeric reviewer uses only the standard library. It recomputes counts,
selected ranges, and the format-admission decision from the scalar
responses and the earlier 16 kHz receipt. The recordings stay local.

```sh
export PYTHONPATH=backend
module=scripts.audio_short_lossless_48k_transfer
receipt=docs/evaluation-panels/audio-short-lossless-48k-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Use the pinned model weights and reconstructed media from the earlier
[nearby-window experiment](audio-short-edit-input-views-2026-11-15.md#reproduce).
The three input directories contain `pcm/` files named by each raw digest.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" capture "$fresh" "$diagnostic" "$room" models/weights/audio-upgrade "$run/transfer.json.gz"
"$runner" -B -m scripts.audio_short_lossless_48k_app "$fresh" "$run/paid"
```
