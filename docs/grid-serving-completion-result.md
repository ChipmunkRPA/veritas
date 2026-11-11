# Complete local GRID serving pilot

The frozen app completed all **176 planned analyses** across 16 pilot
recordings. All 176 background-sound checks completed. The frame
authenticity check skipped on the app's activation budget in all 176 cases.
Every file fell outside the saved validation envelope, so the app returned
no overall AI or original tendency for any case.
[Freeze](evaluation-panels/grid-serving-baseline-freeze-v156.json) · [Serving method](grid-serving-baseline-v156-method.md) · [Actual completion receipt](evaluation-panels/grid-serving-completion-v156.json)

The replay verified sampled picture input, decoded PCM, sample count, and
audio origin in all 176 cases. The face appearance route measured all cases
and returned no flags. Its coverage included 64 clips with observable
neural mouth redraws. The audio/picture route bound one timing window in
174 cases and found no eligible measurement in two. Each soundtrack permits
only one full-search window per face track before the picture, speech, and
speaker-attribution gates. Either timing-change rule needs at least four
windows. Zero timing-change flags leave timing-change sensitivity
unassessed on this cohort.
[Window coverage audit](grid-timing-opportunity.md)

All 16 guarded workers finished. Maximum sampled descendant RSS was
5,662,949,376 bytes, and summed worker time was 2,324.548 seconds.

## Ordinary edits and the separate soundtrack estimate

| Genuine speech condition | Leaning AI | Leaning original | Inconclusive | Unavailable |
| --- | ---: | ---: | ---: | ---: |
| Untouched original | 0 | 9 | 3 | 4 |
| Local 320 ms copy within the original audio | 12 | 0 | 2 | 2 |

The copied waveform uses genuine samples from the same recording and leaves
the picture, sample count and origin intact. It creates ordinary edit
boundaries and can affect speech-window eligibility. All 12 false AI
tendencies came from the selected speech check with its local calibration.
They displayed **93.0%–95.1%** confidence. The background-sound check reports
context separately. The serving run did not execute the old speech checks,
so it does not establish their relative accuracy.

All 144 other matched comparisons with the same PCM, sample count and
audio origin produced identical soundtrack assessments. Both assessments
were present in 108 of those comparisons. This confirms the saved
soundtrack projection was invariant to the picture conditions on the
tested waveforms. The local-copy arm establishes an ordinary-edit failure
for these short clips. The 16 source recordings come from eight connected
components, so the 12/16 count is specific to this paired pilot.

A follow-up comparison should freeze fresh, source-disjoint genuine and
AI speech with unchanged audio, a timestamp-only offset, hard and crossfaded
copies, ordinary speech edits, and background noise or music. Compare the
existing calibration with one exposed to those ordinary edits and a
separate stability check for AI tendencies. Report false AI tendencies,
missed AI speech, and unassessed results by source and intervention.

Run the complete metadata replay from the repository root. It checks the
actual guarded stage receipts and matched construction records. It does
not rerun inference, start the service, or invoke billing.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_serving_completion
```
