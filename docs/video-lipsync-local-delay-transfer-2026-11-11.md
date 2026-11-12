# Short lip-sync timing comparisons

Both short timing checks measured 31 of 66 held views. The current one-second comparison measured nine. None correctly localized a signed local delay among 22 known edits. The short aperture check returned one false range on a control and reversed the direction on one overlapping edit. Both checks remain experimental.

## Source and fixed comparison

The study uses 15 verified CREMA-D originals, one actor code per recording, from the official pinned release under its database and content license grants. The actor codes were absent from the earlier archived acquisition roster and the available historical media manifests. All recordings contain the same spoken sentence. This separates known actors within the study while providing limited phrase coverage. Four actors set separate comparison points. The other 11 supplied only held data. Every clip matches the SHA-256 and length of its upstream large-file pointer. A system restart removed the earlier temporary study files. The published results come from a new roster frozen before download and scoring, with the rerun receipts retained in persistent local storage. [Publisher source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Prior source census](video-common-source-preparation-v89.md)

The study reconstructs continuous 16 kHz mono audio by mapping each sample to its source tick and keeping the earlier decoded sample when two occupy the same tick. The normalizer permits up to four overlapping samples at a boundary and requires complete coverage. Each video uses its exact native presentation table. The serving crop geometry and speech-activity calculation are shared across arms. The controlled comparison uses a directly observed single face on native pictures. The separate app replay uses production track selection.

Each actor supplies an unchanged delivery, an ordinary local gain change, two constant global sound delays, and two temporary local sound delays. The central local edit lasts 16 ticks at 25 Hz, or 640 ms. The sound within it is moved 200 ms earlier or later and mixed at each edge over five milliseconds. Both local variants preserve every source sample outside the half-open authored interval and leave the original pictures intact. The test measures timing mismatch from controlled audio edits. Synchronized retalking, generated mouth motion, and AI authorship remain outside this trial's scope.

The current comparison uses its existing one-second windows, full ±600 ms lag search, visible mouth requirements, speech attribution, and offset-step and collapse rules. The first new arm compares three contiguous 480–800 ms windows on the same observed face with a ±240 ms search. It requires agreement on the flank delay, evidence of a different delay in the middle, and a directional change of at least 120 ms. The second arm additionally compares motion within the mouth against changes in the speech spectrum. Both arms select only their strongest range. The separate points were set before held scoring from the maximum ordinary development response plus 0.025, with a floor of 0.10. A useful range needs at least 0.40 temporal intersection-over-union and the correct direction when the check estimates one.

Recent research tests mouth-to-head coordination and frame-level alignment between speech features and lip articulation. Both cues are relevant to retalking that preserves coarse synchronization. This trial tests untrained timing calculations using existing app inputs. [Lip and pose research](https://proceedings.mlr.press/v306/she26a.html) · [Speech-content and lip consistency research](https://arxiv.org/abs/2610.30872)

## Held results

| Method | Measured held views | Correct local delays /22 | Ordinary false ranges /44 |
| --- | ---: | ---: | ---: |
| Current timing comparison | 9/66 | 0 | 0 |
| Short aperture and energy | 31/66 | 0 | 1 |
| Short joint motion and spectral change | 31/66 | 0 | 0 |

The short aperture check flagged 1.12–1.60 s inside the authored 1.04–1.68 s edit, for an intersection-over-union of 0.75. It estimated sound 280 ms earlier. The edit moved sound 200 ms later. The same check also produced a false local range on a clip with a constant global offset. Both arms fell short of the predeclared improvement. Unassessed views account for 57 of 66 on the current check and 35 of 66 for each short check. [Aggregate and verification receipt](evaluation-panels/video-lipsync-local-delay-transfer-2026-11-11.json)

The held CPU replay committed 597 native source pictures. A separate implementation reconstructed all 66 PCM inputs, checked 102 lag-specific sample slices, and re-decoded 234 distinct original frames and their exact mouth patches. Two playable local MOVs preserve all 83 original pictures, their exact presentation timestamps, and all 45,132 samples of the corresponding genuine or edited audio. Raw recordings and exhibits stay local.

Both MOVs went through the local serving pipeline with the current timing, face, and tracking adapters enabled in the full profile. The service measured one window on each, produced no timing finding, and published both results. The isolated adapter set used fallback fusion and checked the production timing path on one matched pair. The 66-view comparison supplies the coverage and error counts above.

The reproduction entry points are `scripts.prepare_video_lipsync_local_delay_transfer` for the intake freeze and publisher-byte verification, `scripts.video_lipsync_local_delay_transfer` for the comparison, and `scripts.review_video_lipsync_local_delay_transfer` for the independent replay. Each command takes a working folder. Intake also takes the prior archive with the publisher tree and historical manifests. Run `freeze` and `acquire` in the intake module. Run `freeze`, `development`, `calibrate`, `held`, and `readout` in that order in the comparison module, then run the review module on the same folder.
