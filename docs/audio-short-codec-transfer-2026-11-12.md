# Short speech-window review across audio codecs

All 90 FLAC deliveries preserved the exact WAV samples. At the primary point, shorter windows located two of 12 MP3 edits and one of 12 Opus edits. Each lossy codec added alerts on two of 14 genuine groups. The existing four-second windows located no edits and flagged no genuine groups in either codec. [Audited scores and replay](evaluation-panels/audio-short-codec-transfer-2026-11-12.json)

## Comparison

The previous six-second study identified two useful noisy speech ranges with two-second windows, compared with zero using the four-second analysis windows. This transfer asks whether the same kind of location survives a change in file delivery. Research on partial speech edits separates codec processing from modified content, and recent audio benchmarks test recording and compression shifts separately. This comparison uses the existing local voice and speech-activity assets under their audited terms. [WAV holdout](audio-edge-window-grouped-2026-11-12.md) · [PartialEdit](https://arxiv.org/abs/2506.02958) · [AT-ADD](https://arxiv.org/abs/2608.23437) · [Model terms](commercial-model-audit.md)

The source roster uses 40 ordinary calibration recordings across 16 groups and 50 held recordings across 14 groups. The latter include 12 generated inserts from three generation families, 36 matched genuine views, and two isolated room recordings. Calibration groups are separate from the held groups. The WAV recordings were already measured. All FLAC, MP3, and Opus deliveries were newly encoded from those exact sealed recordings, so the comparison isolates the codec change. The variants share sources and are paired observations. [Source roles and limits](audio-edge-window-grouped-2026-11-12.md#held-recordings)

All deliveries decode to exactly 96,000 mono samples in the app. The FLAC carries signed 16-bit samples at 16 kHz. The MP3 uses a 48 kb/s 16 kHz encode and maps sample zero to source tick 1,105 on the 16 kHz evidence clock after priming. The 24 kb/s Opus uses a 48 kHz codec clock in an Ogg carrier. The app converts the decoded PCM and source timeline to 16 kHz, with sample zero mapped to tick zero. Each file and decoded clock is sealed before scoring. The study does not include AAC, stereo, longer recordings, other bitrates, or other processing chains.

The short and Full arms use the same classifier and half-speech admission. The Full arm covers 0–4 and 2–6 seconds. The shorter arm covers five two-second windows with a one-second hop. Both choose the highest admitted response with earlier-start tie breaking. Each lossy codec and arm uses its own strictly exceeded group-ranked operating point. The primary point allows two flagged calibration groups, with a separately frozen one-group comparison. A useful location covers at least half of the authored edit and has temporal IoU of at least 0.30. The held rule for an additional evidence check is at least two newly useful groups and at most one new genuine group compared with Full on that same codec.

## Held results

| Delivery and point | Four-second edit flags /12 | Two-second edit flags /12 | Useful locations, four → two seconds /12 | Genuine groups flagged, four → two seconds /14 |
| --- | ---: | ---: | ---: | ---: |
| WAV, primary | 1 | 2 | 0 → 2 | 0 → 0 |
| MP3, primary | 1 | 3 | 0 → 2 | 0 → 2 |
| Opus, primary | 0 | 2 | 0 → 1 | 0 → 2 |
| WAV, stricter | 0 | 1 | 0 → 1 | 0 → 0 |
| MP3, stricter | 1 | 3 | 0 → 2 | 0 → 1 |
| Opus, stricter | 0 | 2 | 0 → 1 | 0 → 2 |

All 50 held files were assessed in both windows and each lossy delivery. At the primary point, the shorter MP3 arm flagged four genuine files across two groups. Opus flagged two files across two groups. The stricter MP3 point retained both useful locations and dropped two genuine alerts from one group. Both lossy profiles failed the primary addition rule. The later [MP3 confirmation](audio-short-mp3-confirmation-2026-11-12.md) uses the stricter point unchanged and explains the narrow release scope. The two-second check continues to abstain on Opus.

## Local FLAC support

All 90 native FLAC deliveries reproduced the exact analyzed samples from their WAV counterparts. The Full worker reproduced the two known flagged ranges, 3–5 and 4–6 seconds, and the two known ordinary outcomes. An otherwise equivalent 24-bit file was skipped by the short review. The native 16-bit FLAC scope carries the existing evidence-only calibration point for the analyzed samples.

Chrome loaded the correct two-second excerpt and marked waveform in the plain view. The Standard PDF embedded the same waveform pixels as the JSON asset. Signed source replay matched the exact FLAC source clock, extracted samples, and waveform. Simulated checkout admitted each test analysis only after confirmed payment.

## Replay

The public receipt contains anonymous per-window scalars and known edit intervals. The standard-library review checks source roles, matches the held rows to the published WAV panel, refits the codec-specific points, and reproduces coverage, useful locations, misses, and ordinary-group errors.

```sh
python3 -B -IS backend/scripts/review_audio_short_codec_transfer.py
```

To recreate the deliveries, use the retained local WAV panel, pinned installed checkpoints, and local FFmpeg. Use a new private directory for the sealed run. Source recordings and decoded WAVs stay local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_short_codec_transfer prepare "$run"
backend/.venv/bin/python -B -m scripts.audio_short_codec_transfer score "$run"
backend/.venv/bin/python -B -m scripts.review_audio_short_codec_transfer "$receipt" --native "$run" "$held"
```
