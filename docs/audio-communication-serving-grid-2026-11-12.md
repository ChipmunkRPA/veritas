# Voice stability on the Full analysis grid

On the confirmation cohort, requiring agreement between original and simulated call views reduced generated flags from two to one and lost the only original flag that covered the true edit. Both arms flagged the same two genuine groups. The discovery cohort also showed no reduction in false groups. These results keep the pairing in offline evaluation.

## Local comparison

| Result at the frozen research point | Discovery original | Discovery call | Discovery both | Confirmation original | Confirmation call | Confirmation both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Assessed files of 52 | 52 | 51 | 51 | 52 | 51 | 51 |
| Speech-admitted windows of 104 | 102 | 101 | 101 | 103 | 101 | 101 |
| Generated flags of 12 | 3 | 3 | 3 | 2 | 3 | 1 |
| Flags covering at least half the known edit | 3 | 3 | 3 | 1 | 3 | 0 |
| Flags containing the entire known edit | 2 | 2 | 2 | 1 | 3 | 0 |
| Useful localizations of 12 at 0.30 IoU | 0 | 0 | 0 | 0 | 0 | 0 |
| False genuine files of 40 | 4 | 6 | 4 | 4 | 8 | 4 |
| False genuine groups of 16 | 2 | 3 | 2 | 2 | 3 | 2 |

The lone paired flag on generated material in confirmation was at 0–4 seconds, entirely outside the 5.12–5.76 second edit. In both cohorts all 12 edits lay wholly inside at least one speech-admitted window in each arm. The two unassessed call views came from room-only controls. No room-only control was flagged. Each cohort's earlier two-second research test retained its useful locations and reduced false genuine groups from three to two. The pairing did not reproduce that trade-off on the Full grid.

## Input and score boundaries

The discovery and confirmation cohorts each contain 52 sealed original and simulated call deliveries. Each has 12 generated speech inserts, 36 genuine controls matched by foreground group, and four room-only controls. The inserts are 0.64, 0.80 or 0.96 seconds long, with background speech at a fixed −5 dB whole-clip foreground-to-background ratio. Four of the 12 edits sit at the start or end of the six-second recording. Discovery and confirmation used separate exact source bytes and declared foreground speakers and room participants. The original studies document archive and voice-profile reuse and source rights. This run reuses both cohorts. [Discovery panel](audio-communication-transfer-2026-11-12.md) · [Confirmation panel](audio-communication-replication-2026-11-12.md)

The run passed the retained six-second, 16 kHz mono PCM through the production candidate voice adapter with the explicit Full profile. It used the same local CPU voice checkpoint, learned speech-activity gate and production calibration bundle. The adapter plans 0–4 and 2–6 seconds on a continuous six-second clock and admits each window only when at least half contains detected speech. The call view uses the previously frozen bandpass, compressor and 24 kb/s Opus voice chain. The app decoder verified all 104 call deliveries against their retained PCM and continuous zero-start source clock. [Commercial model audit](commercial-model-audit.md)

The research comparison takes the strongest admitted response in each view. For the paired result, take the lower score at every shared admitted interval and then select the strongest paired interval. All three arms use strict excess over the original two-second research point, with no refit. A generated flag is a file whose selected interval exceeds that point. A false group is one foreground group with one or more flagged genuine controls, or one flagged room-only control. Record whether the selected interval overlaps the known edit at all, covers half or more, or fully contains it. A useful location also needs at least 0.30 temporal intersection over union. The receipt tracks these research counts and the adapter's actual calibration and assessment fields. [Anonymous scores and replay](evaluation-panels/audio-communication-serving-grid-2026-11-12.json)

## What the four-second windows can establish

A 0.64–0.96 second edit can occupy at most 16–24% of a four-second window. That limits intersection over union to 0.24 even when the window contains the complete edit. The edge edits are reachable on the Full grid, but the broad window cannot identify their boundaries at the frozen 0.30 localization criterion. The two-second research grid supports a higher potential overlap, and its 1–3 and 3–5 second intervals omit the edge edits. These differences make the grid change a comparison of both score behavior and temporal reach.

The installed whole-recording calibration covers 2.5–4 seconds of continuous audio that fits one speech-admitted window at time zero. Every admitted six-second view in this study returned an inconclusive voice direction and a null calibrated probability. The short-window cutoff was used solely to compare responses offline. The study tested the detector adapter and source clock on native input, with the existing local customer-assessment tests for standalone audio and video soundtracks. [Local model scope](audio-measured-spatial-transfer-2026-11-12.md#voice-comparison)

A separate localization experiment could scan overlapping two-second speech windows from the beginning and through the end of the file, then test if a repeated flag in the processed copy helps distinguish an edit from competing voices or call artifacts. It would need its own range-level calibration and fresh controls, including ordinary splices, multiple room and call chains, and edge edits. Both deliveries share the same voice and checkpoint. The paired observation could support a specific timestamp if independently validated. It carries no additional statistical weight as a second detector.

## Replay

Public replay reconstructs all metrics from the anonymous scores and known edit intervals, and checks the comparison against the two earlier receipts. It needs no source media or model files.

```sh
python3 -B -S backend/scripts/review_audio_communication_serving_grid.py
```

With the sealed local inputs and pinned weights, prepare and score the actual Full adapter once, then check the native score ledger against the original and decoded call samples. Use a new run directory if the original freeze already exists. The original run made 407 voice classifier forwards. The native replay checks all 208 complete recording bindings and all 416 planned window bindings, including windows that did not pass the speech gate. It re-decodes the 104 call deliveries and verifies their exact sample clocks. The reviewer uses saved classifier responses.

```sh
run=backend/data/independent-evaluation/audio-communication-serving-grid-2026-11-12
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export VERITAS_AUDIO_DETECTOR_MODE=candidate
backend/.venv/bin/python -B -m scripts.audio_communication_serving_grid prepare --run "$run"
backend/.venv/bin/python -B -m scripts.audio_communication_serving_grid score --run "$run"
backend/.venv/bin/python -B -m scripts.review_audio_communication_serving_grid --native "$run"
```
