# Shorter speech measurements in longer noisy recordings

The two-second timing audit found that 55 of 155 labeled edits have no
matching scheduled two-second position. This comparison measures shorter
intervals directly using the existing 0.5-, 0.75- and 1-second focused
speech detector. It reuses the 240 documented deliveries for an exploratory
comparison with the newly enabled two-window rule. [Two-second timing and inputs](audio-two-second-long-review-2026-11-16.md#timing-limits)
· [Focused detector and weight pins](audio-focused-edit-local-2026-11-16.md)

The existing focused detector reads six seconds at a time. We ran it on
the app's scheduled six-second excerpts at the same calibration used for
isolated six-second uploads. It returns the strongest eligible span in each
excerpt. One comparison adds every qualifying short span to the current
long and two-second reports. Another adds a short span when a second
excerpt that starts at least one second away also flags an overlapping
span. The overlap must cover at least half of the shorter span. These
are checks for longer mixed recordings using the established extractor,
speech admission, and weights. [Original fitting and split](audio-joint-generator-transfer-2026-11-16.md)

## Results

Every delivery produced at least one measured six-second excerpt. Each
condition has 155 labeled edit intervals and 24 ordinary files. At each
noise level the evaluation also includes 48 isolated-background inputs.
The same original voices and labels are reused across conditions. The
event comparison requires at least half of a known interval recovered
and 0.30 intersection over union.

| Condition | Located edits out of 155, current / add short / repeated short | Ordinary flags out of 24, current / add short / repeated short | Background flags out of 48, current / add short / repeated short |
| --- | --- | --- | --- |
| Clean | 81 / 81 / 81 | 0 / 0 / 0 | — |
| −5 dB | 47 / 48 / 47 | 5 / 9 / 5 | 1 / 1 / 1 |
| −10 dB | 22 / 22 / 22 | 2 / 4 / 3 | 1 / 1 / 1 |

The additional −5 dB match selected **6.700–7.200 s** around the known
**6.757–7.097 s** edit. The four extra ordinary alerts at that level
appeared on four recordings. At −10 dB the shorter measurement added
ordinary alerts and zero correct locations. The shorter check stays in
evaluation for this recording length. These results point to a need for
calibration on long, noisy recordings with exact subsecond labels. The
next confirmation needs a separate set of edited and ordinary sources.
[Numeric measurements](evaluation-panels/audio-focused-long-transfer-2026-11-16.json.gz)

## Replay

The portable reader recounts the selections and source groups. The export
uses the common manifest and truth schema to independently count events,
ordinary alerts, and genuine samples covered by reported ranges. Native
audio and checkpoint assets stay local. Set `$media` to the native input
directory from the [two-second comparison](audio-two-second-long-review-2026-11-16.md#reproduce).

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
baseline="$panels/audio-two-second-long-fresh-2026-11-16.json.gz"
measured="$panels/audio-focused-long-transfer-2026-11-16.json.gz"
module=scripts.audio_focused_long_transfer
python3 -B -S -m "$module" review "$baseline" "$measured"
python3 -B -S -m "$module" export "$baseline" "$measured" "$run/independent"
backend/.venv/bin/python -B -m "$module" score "$baseline" "$media" "$run/focused.json.gz"
```
