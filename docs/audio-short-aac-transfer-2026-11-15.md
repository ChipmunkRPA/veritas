# Short speech-window review on 48 kHz AAC

This comparison measures how well the existing short speech check can
locate a brief edit in a six-second M4A upload. It compares the existing
two-second and four-second window schedules with the same local voice
model and speech-activity check. Each schedule uses separately derived
research points. The measurements apply to the complete flagged window.
[Earlier MP3 and Opus transfer](audio-short-codec-transfer-2026-11-12.md) ·
[Commercial model terms](commercial-model-audit.md)

## Exact presentation clock

Each retained 16 kHz mono file was encoded to one 48 kHz mono AAC-LC
track at a nominal 64 kb/s. The preparation pinned the encoded file
and decoded PCM hashes. It accepted exactly 288,000 presentation
ticks at 48 kHz starting at zero, with continuous timing after resampling
to 16 kHz. The 225 carriers declared average bit rates of 64,100 to
66,245 bits/s. All decoded to 96,256 samples without a length limit.
The first 96,000 matched the independently decoded presentation prefix
byte for byte. The last 256 samples fall after the six-second source
presentation and were excluded from the window comparison.
[FFmpeg container timing](https://ffmpeg.org/ffmpeg-formats.html#mov_002fmp4_002f3gp)

## Test design

The 40 ordinary calibration files cover 16 disjoint source groups from
the existing short-codec study. Each schedule picks its strongest
admitted interval, then fits one point allowing one calibration group
to flag and a second allowing two. These points were fixed before the
held runs. Both schedules require at least half the window to contain
detected speech. A useful location covers at least half the actual
edit and reaches 0.30 temporal intersection over union.

The first held bank contains 50 files and 12 staged edits. A separate
52-file confirmation bank has 12 edits and disjoint genuine hosts.
The random-clock bank adds 83 retained files, including 32 staged
stock-voice replacements, 48 matched genuine controls, and three
isolated backgrounds. It preserves the separate 0 dB and −5 dB mix
conditions. These are new AAC deliveries of retained PCM with no
across-role duplicate PCM. The sets do not measure every capture device,
encoder or generator. Their sources and prior reuse are documented in
the earlier studies.
[Grouped source method](audio-edge-window-grouped-2026-11-12.md) ·
[Confirmation source method](audio-short-mp3-confirmation-2026-11-12.md) ·
[Random-clock source method](audio-random-clock-stock-2026-11-13.md)

## Held results

The tighter point gives the following results. Useful ranges count
edited deliveries. Ordinary counts include isolated backgrounds, with
flagged source groups in parentheses. All 185 held inputs had an
assessment in both schedules.

| Bank | Useful ranges, four seconds → two seconds | Ordinary flags, four seconds → two seconds |
| --- | ---: | ---: |
| Grouped held files | 0 → 1 of 12 | 0 (0) → 0 (0) of 38 |
| Separate confirmation | 0 → 0 of 12 | 4 (2) → 3 (1) of 40 |
| Fixed roster, randomized time, competing voice 5 dB louder | 0 → 1 of 16 | 1 (1) → 1 (1) of 24 |
| Fixed roster, randomized time, even levels | 0 → 2 of 16 | 1 (1) → 2 (2) of 24 |
| Additional isolated backgrounds from the random-clock bank | — | 0 (0) → 0 (0) of 3 |

The grouped check located **3.000–5.000 s** for a staged edit at
**3.120–4.080 s**. All three useful random-clock deliveries share a
host and edit time. The review located **2.000–4.000 s** for the
**2.861–3.821 s** edit in two synthesis families at even levels and one
family with a louder background. Keeping both schedules adds four
useful deliveries across two source groups and one ordinary file from
one additional group. Neither schedule flags isolated background.
The shorter schedule found the same cases at the wider point.

The confirmation bank contributes a wrong-window example. One edited
delivery and three matched genuine files all flag **2.000–4.000 s**.
The true edit occurs at **5.120–5.760 s**. With these brief edits a
four-second interval is too broad to satisfy the overlap criterion
even when it covers the edit. The report retains wrong locations and
half-edit coverage separately from useful ranges.

## Local trial

The optional Full review uses the tighter AAC point. In development
with the candidate voice configuration, enable
`VERITAS_AUDIO_SHORT_AAC_REVIEW=true`. It admits a single 48 kHz mono
AAC-LC M4A track with a reported bit rate from 64 to 67 kb/s and a
verified six-second presentation. The speech input is the exact
16 kHz presentation prefix. The marked waveform ends at six seconds
and the exports keep the original presentation length alongside the
analysis clock. The evidence carries zero weight in overall confidence.

Three simulated USD 9.99 worker checkouts exercised the option off,
the staged edit with the option on, and an unchanged control. With
the option enabled, the API, PDF and JSON agreed on **3.000–5.000 s**.
Signed replay independently regenerated the audio excerpt, waveform
and five coverage windows against the uploaded M4A. The staged file's
automated assessment agreed with the option off and on. No job was
queued before payment confirmation.
[Local worker receipt](evaluation-panels/audio-short-aac-transfer-app-2026-11-15.json)

## Replay

The public receipt contains scalar scores and source and window
hashes. The standard-library reviewer recomputes cutoffs and held
outcomes. The media stays local.
[Evaluation receipt](evaluation-panels/audio-short-aac-transfer-2026-11-15.json.gz)

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.audio_short_aac_transfer review \
  docs/evaluation-panels/audio-short-aac-transfer-2026-11-15.json.gz
```

With the retained source folders from the linked studies and the pinned
local weights, create a new private run directory. This run used FFmpeg
7.0.2. Preparation verifies the source rows and native presentation
before classification.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_short_aac_transfer prepare "$run" "$random_media"
backend/.venv/bin/python -B -m scripts.audio_short_aac_transfer score "$run" models/weights/audio-upgrade
backend/.venv/bin/python -B -m scripts.audio_short_aac_app "$run" "$app_run"
```
