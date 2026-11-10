# Native video appearance comparison: selection result

All 138 planned clips from 23 source groups completed the fixed comparison,
using the same scheduled native frames for the current raw cue and sealed
file-calibrated candidate. No model, point, source or missingness rule changed
after inspection. These are constructed inserts from publisher-labelled swap
streams and matched controls, not a general real-world accuracy estimate.

| Outcome | Current raw cue | Calibrated candidate |
| --- | ---: | ---: |
| Flagged constructed swap clips | 46/46 | 46/46 |
| Missed swap clips, including inconclusive | 0/46 | 0/46 |
| False AI flags on controls | 0/92 | 0/92 |
| Inconclusive controls | 2/92 | 2/92 |
| Assessed files | 136/138 | 136/138 |
| Assessed windows | 681/690 | 681/690 |

Each of the two swap conditions supplied 23 clips. Original, ordinary lighting,
blur and time-step controls each supplied 23 clips. The blur and time-step
conditions each had one inconclusive file because required face/alignment or
geometric support was unavailable. Nine scheduled windows were unassessed.
These cases did not become original-file verdicts. All original and lighting
controls were assessed without an AI flag.

## Timestamp support

Both checks selected the same 46 windows and 414 sampled picture positions.
All 414 positions fell within the intended changed-source ranges, with no
flagged control positions. However, 552 AI-range positions were scheduled and
2,404 additional changed-source pictures were never sampled. The flags cover
only part of each edit: mean continuous interval-union IoU was **0.551**.
Bounding ranges had zero seconds outside the known ranges on this panel,
but may contain unexamined gaps. These results do not establish complete
frame coverage, pixel-level manipulation, identity truth or precise edit edges.
Candidate window support is a single-window replay into a file head, not a
separately calibrated local confidence score.

## Decision and limits

Keep the current serving check and retain the calibrated candidate for further
testing. They tie on both directional calls and local support here; this panel
supplies no evidence that replacing the current check finds additional edits.
This does not erase the candidate's earlier gains on the consumed v106 panel.
Its selection-panel source/class-balanced Brier is 0.00824, but the raw cue has
no comparable calibrated probability. This single loss is not evidence of
general calibration or a customer confidence guarantee.

The panel uses five predetermined windows and a single-visible-face condition.
It is not the complete production tracker/fallback path, lip-sync testing,
audio evaluation or fully generated video testing. Artificial insertion
boundaries may be easier to detect than subtle continuous manipulations.
Publisher labels and exact construction/decoder checks do not prove AI-pixel
or identity truth. The 24 reserved source groups remain unused by this
selection comparison. No threshold tuning or fresh confirmation claim follows.

## Execution and preservation

Eight successful corrected batches made 9,244 native calls in 359.73 seconds,
with zero guards or sampling errors. Maximum sampled descendant RSS was
731,512,832 bytes; minimum available RAM was 6,656,884,736 bytes. Samples are
lower bounds on peaks. The readout made no neural calls and separately checked
all saved metadata, scalar calculations and commitments.

The original v108 source stage failed after unsaved neural work because it
compared a float with the saved cutoff record. Its complete response count is
zero and its partial neural count remains unknown. It was preserved unchanged.
The [isolated v112 correction](video-appearance-corrected-v112-method.md) changed
only that record/scalar handoff and paths; it explicitly repeated the unsaved
first case. The corrected result is not an unexposed confirmation experiment.

See the [full aggregate readout](evaluation-panels/video-appearance-corrected-v112/video-appearance-selection-result-v111.json)
and [execution commitments](evaluation-panels/video-appearance-corrected-v112/execution-summary-v112.json).
The retained readout schema is v111; the new frozen context is v112. Private
source IDs, media, face features and learned parameters remain private. Serving
confidence and rollback assets are unchanged. No Actions workflow was run.
