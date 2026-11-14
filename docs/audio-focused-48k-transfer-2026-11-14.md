# Short speech edits in 48 kHz WAV recordings

The optional overlapping speech review can now assess continuous
48 kHz mono PCM16 WAV files longer than six seconds and up to eleven
seconds. The app decodes the file to its existing 16 kHz analysis
format and applies the same three-second windows, speech activity
check, voice model and calibration. When two neighboring windows lean AI,
their overlap gives a shorter range to review. The report links that range
to both windows and shows a waveform and playable excerpt. It identifies
the submitted sample rate and labels the decoded sample positions.
This review carries zero weight in the overall AI assessment.

## Evaluation

The comparison re-delivered the 98 files from the earlier rain and
train study as mono PCM16 WAV at 48 kHz. It covers 18 publisher-labeled
partial edits from six generators, 12 genuine recordings, and eight
isolated background excerpts. Each speech source has a gain-only view
and matched mixes with recorded rain and train sound. The original
16 kHz files were upsampled before entering the app's usual decoder.
This tests the delivery format and clock. The sources and background
recordings were previously evaluated, and native 48 kHz capture needs its
own confirmation. Like the [RADAR challenge](https://arxiv.org/abs/2605.09568),
the comparison records delivery transformations separately from
the underlying source. It uses [LlamaPartialSpoof](https://zenodo.org/records/14214149)
and the [backgrounds and rights listed in the earlier study](audio-focused-window-transfer-2026-11-12.md#frozen-comparison).

| Delivered condition | Exact 16 kHz source, useful locations | 48 kHz WAV, useful locations | 48 kHz WAV, ordinary files flagged |
| --- | ---: | ---: | ---: |
| Gain only | 13/18 | 13/18 | 0/12 |
| Rain, −5 dB | 3/18 | 3/18 | 0/12 |
| Train, −5 dB | 0/18 | 0/18 | 0/12 |
| Isolated backgrounds | — | — | 0/8 |

The selected ranges matched on all 98 files. Window directions matched
on 96 files. Two formerly unassessed train windows became inconclusive
without changing their selected ranges. At least one window was scored
on 15 of 18 train-masked edits and eight of 12 train-masked genuine
recordings. The isolated backgrounds remained unassessed. Useful
overlap means covering at least half of a known edit and reaching 0.30
temporal intersection over union. The experiment supports a limited
format expansion. Loud train noise still prevented useful localization.
[Comparison receipt](evaluation-panels/audio-focused-48k-transfer-2026-11-14.json)

## Local app verification

Six simulated prepaid runs compared one known rain-masked edit, one
genuine recording mixed with rain, and one rain-only control with the
trial on and off. Zero jobs ran before payment. All three overall
decisions matched across settings. The trial reported **3.000–4.500 s**
for the edited example, or **[48000, 72000)** at the decoded 16 kHz
rate, and produced no ranges for either control. The API, Standard PDF,
JSON, and signed original replay agreed on the interval and exhibits.
The native comparison replay checked 511 window commitments across all
98 submitted files.
[Local app receipt](evaluation-panels/audio-focused-48k-app-2026-11-14.json)

## Reproduce locally

Reconstruct the parent `pcm/` and `roster.json` with the
[original preparation command](audio-focused-window-transfer-2026-11-12.md#replay).
The local comparison uses FFmpeg 7.0.2. The optional native replay
reopens the WAVs and checks their decoded clocks and sample hashes.
The public receipt contains scalar decisions and hashes. Audio and
model assets stay local. To try the app route, enable
`VERITAS_AUDIO_FOCUSED_LONG_48K_REVIEW=true` alongside the existing
focused review in candidate audio mode. The 48 kHz setting defaults to off.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-focused-48k-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_focused_48k_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_focused_48k_transfer prepare "$bank" --native "$parent"
backend/.venv/bin/python -B -m scripts.audio_focused_48k_transfer score "$bank"
backend/.venv/bin/python -B -m scripts.audio_focused_48k_transfer review "$bank/receipt.json" --native "$bank"
backend/.venv/bin/python -B -m scripts.audio_focused_window_app paid-48k "$bank" "$app_run"
```
