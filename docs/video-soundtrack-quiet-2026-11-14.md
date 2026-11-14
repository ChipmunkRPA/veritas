# Earlier-copy timing in quieter soundtracks

The previous four-second 720p comparison missed seven of 28 known
200 ms shifts per delivery set. Two missed edits were outside the picture
quality limits. In several other cases, the waveform search found a
local shift but lacked enough same-time audio to bound it on both sides.
The tests here keep the picture limits and isolate two waveform changes.
[720p cadence panel](video-soundtrack-cadence-2026-11-14.md)
· [Local audio matching research](https://arxiv.org/abs/1304.0793)

The first change compares 100 ms windows every 20 ms. The second admits
quieter waveform windows, lowering the level floor from 0.003 to 0.00075
of full scale. The third combines the two. Each uses the same
correlation, alternative-peak margin, time displacement, and requirement
for distinct matching excerpts before and after. Each is evaluated as
a second pass, preserving an existing location.

The first panel re-decodes the 24 fps test audio and requires an exact
PCM hash match for every role in both PCM and AAC. It uses the previously
verified picture qualification, including the actual synchronized video
controls.

The transfer uses new non-overlapping excerpts from the same 14 speakers.
Each excerpt starts five seconds after the earlier one, or five seconds
before it if the source ends too soon. The runner decodes 4.25 seconds
and uses the first 64,000 available samples to avoid padding trimmed
source packets. Both copies are delivered 12 dB quieter. Controls
include ordinary gain and bandwidth changes, a constant offset, other
speech, a mute, a repeated quiet bed, and a deeper local gain change.
This transfer measures the waveform search on same-picture inputs by
construction. The recording families remain shared with the earlier
panel.

## Results and local use

| Second pass | Previous panel located /28 | Quiet transfer located /28 | Control flags in each panel /112 |
| --- | ---: | ---: | ---: |
| Current check alone | 21 | 12 | 0 |
| Short windows | 23 | 12 | 0 |
| Quieter windows | 25 | 18 | 0 |
| Shorter and quieter windows | 25 | 19 | 0 |

PCM and AAC gave the same counts. The quieter 160 ms pass is enabled for
qualified four-second 720p earlier-copy pairs at 24 or 25 fps. The shorter
100 ms pass added one more location on the reduced-level panel. That
addition comes from one source recording, so it remains in evaluation.
Any first-pass finding is retained. The comparison keeps the same two
picture grids, source timestamp requirements, and zero weight in the
overall AI assessment.

In two simulated prepaid analyses of an earlier missed public example,
the quieter pass measured the analyzed excerpt at 1.520–2.400 s and the
earlier excerpt at 1.320–2.200 s, a +200 ms movement. The PDF, browser data,
and JSON use the same before, during and after pictures and both
waveforms. The JSON records which pass was used, so the verifier can
reproduce existing exports under their original rules.

## Replay

The receipt records the decoded-audio commitments, support windows and
measured shifts. Reviewers can use Python's standard library to recount
matches or reproduce the full decoded comparison from the parent
panel's pinned media.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
panel=docs/evaluation-panels/video-soundtrack-quiet-2026-11-14.json
parent=docs/evaluation-panels/video-soundtrack-cadence-2026-11-14.json
python3 -B -S -m scripts.review_video_soundtrack_quiet "$panel" "$parent"
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.video_soundtrack_quiet "$phase" \
    "$new_run" "$conversation_cache" "$other_public_cache" || exit 1
done
cmp "$panel" "$new_run/receipt.json"
```
