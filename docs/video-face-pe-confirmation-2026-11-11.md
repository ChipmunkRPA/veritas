# Prospective native face-swap confirmation

The earlier transfer found a substantial gain from adding the Perception Encoder arm to the reference. This prospective run held both solo operating points and the normalized two-arm maximum fixed. It measured new scores on a separate set of source groups. [Earlier transfer and portable score replay](video-face-pe-transfer-2026-11-11.md)

## Fixed inputs

All 24 v95 confirmation source groups qualified. Each supplied an original native video and two whole-clip swaps, for 72 clips. The source groups and source-byte digests were disjoint from the development and two transfer panels. The v91 acquisition hashes and v92 qualified frame clocks and picture hashes matched the native v95 inputs. Seven groups used 720p and 17 used 1080p. [Native source methodology](video-native-inserts-v95-method.md)

The test used the frames nearest six interior times at successive sevenths of the full native time span. Each frame retained its original source timestamp. The candidate and reference used identical decoded RGB, indices, timestamps, and largest-face boxes from the same local face detector. Each arm needed four finite frame responses and took their median. Both runs checked the archived picture bytes and timestamp for every sample before scoring. The paired rule took the larger of the two normalized medians at its pre-existing operating point. The ten genuine development controls supplied all scales and thresholds. The confirmation responses did not change them.

## Results

| Rule | Inswapper swaps detected | Uniface swaps detected | Total detected | Original clips flagged |
| --- | ---: | ---: | ---: | ---: |
| Reference | 15/24 | 11/24 | 26/48 | 0/24 |
| PE | 24/24 | 22/24 | 46/48 | 1/24 |
| Reference + PE | 23/24 | 22/24 | 45/48 | 0/24 |

The paired rule retained every reference flag and added 19 swap flags. It detected 13 of 14 swaps at 720p and 32 of 34 at 1080p. Every clip had all six responses in both arms. There were no partially scored or unassessed clips. The two runs finished in 147.746 and 157.656 seconds on local Metal at batch size one. They used approximately 1.35 GB and 1.66 GB of peak sampled process-tree memory. An independent readback recomputed every score reduction, group count and paired transition, and verified all 432 native RGB, clock and face-box records per arm. [Aggregate and independent coverage](evaluation-panels/video-face-pe-confirmation-2026-11-11.json)

This test measures whether a rule flags sampled faces on whole-clip swaps from these two generator families. It provides evidence for a sampled face review and leaves partial-edit boundaries, unrelated generators, broader ordinary-edit false alarms, and overall AI confidence unmeasured.

## Local application setup

Set `VERITAS_VIDEO_WHOLE_FACE_REVIEW=true` for the backend process to enable the supplemental check. It uses pinned weights in the local model cache. From the backend directory, `python -m app.cli models fetch gend.pe_large` fetches the additional checkpoint when online fetching is enabled. The six-frame check supports verified, unrotated native frame clocks with at least 31 frames, a short side of at least 360 px, and a long side of at most 1,920 px. Both checks require at least four usable positions.

The analysis links the finding to three representative source frames and their original face crops. The JSON includes all six native frame indices and timestamps. The PDF shows the planned coverage and the three examples. The signed evidence export independently checks that its projected check and packet commit to the same measurement. The cue carries zero overall confidence weight.

A local offline upload test ran the normal Quick analysis on one 1,080p original and its whole-clip swap. The original stayed below the review point and the swap was flagged. All six source positions and decoded RGB inputs matched the frozen comparison. The paired scores matched within CPU numerical tolerance. The API and JSON export agreed. The three PDF face crops for each input matched the exact source pixels, and both signed packages verified against the instance public key. The supplemental detector took 48.2 and 40.5 seconds on the local CPU.
