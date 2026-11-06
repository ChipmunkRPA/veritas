# Fresh consecutive-face sequence comparison

Completed locally on 5 November 2026. The primary raw consecutive-crop method
passes the predeclared comparison for **local evidence integration**. It has subsequently
passed isolated production-adapter, source-binding, PDF/JSON, name-filtering and
prepaid-flow verification. The [local release record](local-face-sequence-rollout.md)
describes opt-in and rollback; the repository default remains legacy. Whole-file
confidence is unchanged.

## Fixed comparison

Development used all 30 previously consumed v5 source groups: 120 clips, with
60 swaps and 60 genuine/ordinary-edit controls. Their former test split is now
explicitly development. Raw and brightness-normalized crops used the same face
detections, source schedule, alignment, feature weights, and window statistic.
Each arm selected its own operating point using development only. A disabled
candidate was included in selection. Raw crops localized 17 development edits
versus 12 for normalized crops and 10 for the current check. Raw was therefore
selected as primary **before any fresh candidate inference**. Normalization
remained secondary; test outcomes cannot choose a different default.

The raw inspection operating point is `0.007326911389827726`; the secondary
normalized point is `0.03725649546831847`. These are feature-statistic units,
not AI confidence percentages or public screening-threshold claims.

Both use the existing commercially permitted, digest-verified YuNet and DINOv2
Small weights. No research-only checkpoint was added. The sequence statistic
compares inner-face features to both surrounding time groups, discounts outer
face changes, and discounts differences between the flanks. Each sample uses
three actual consecutive aligned face crops. The secondary method adds robust
brightness/contrast normalization and a declared low-contrast exclusion.

Fresh RTFS source acquisition excluded all 102 prior source-video IDs and prior
donor filenames. Thirty new groups supplied 30 originals, 30 InSwapper clips,
30 UniFace clips, and 30 ordinary-edit controls: seven lighting changes, thirteen
recompressions, and ten jump cuts. All 120 planned clips were available. Swap
insert boundaries were fixed from source hashes and verified by decoded frame
digests. Exact face-pixel masks are not established. Known source/donor metadata
is disjoint; shared subjects or pretraining overlap remain unknown.

## Held-out results

Correct localization requires the first displayed review range to reach
intersection-over-union of at least 0.30 with the known inserted range. Missing
or inapplicable windows remain in the denominator. The current range is the
fallback when an additional check supplies no eligible interval.

| Method | Localized swaps (60) | Missed localizations | Genuine/control clips flagged (60) | Ordinary-edit controls flagged (30) | Mean edited IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current point/interval check | 7 | 53 | 21 | 12 | 0.09249 |
| Primary raw consecutive crops | 19 | 41 | 21 | 12 | 0.25089 |
| Secondary normalized crops | 7 | 53 | 21 | 12 | 0.09249 |

The primary method recovered twelve additional locations and lost none of the
seven current successes. No genuine/control clip gained or lost a review flag.
It recovered six additional InSwapper and six UniFace locations. Mean paired
IoU improved by 0.15840; the 95% source-group bootstrap interval is
0.06888–0.26180. Resampling keeps each source's two swap variants together.
All 120 baseline and candidate executions completed. Sequence windows were
available on 108 clips; twelve clips had insufficient eligible temporal support
and retained the current check. [Aggregate results](video-sequence-v6-evaluation.json)
include family-specific counts, coverage, fixed operating points, and hashes.

These are **review-cue** outcomes, not whole-file AI false-positive rates or
calibrated authenticity probabilities. A 21/60 genuine review rate remains high,
and 41/60 swaps were not correctly localized. The method is neither a general
deepfake solution nor evidence of lip-sync or speech performance. It does not
enter overall confidence on the strength of this localization study.

## Reproduction and integrity

The isolated trial is in ignored local storage at
`backend/data/independent-evaluation/video-v6/`. Scripts are
`prepare_face_swap_upgrade_eval.py`, `prepare_face_interval_eval.py`,
`score_face_interval_eval.py`, `score_photometric_face_sequence.py`, and
`review_photometric_face_sequence.py` under `backend/scripts/`.

Candidate frames are fixed by the current sample plan; no extra dense frame
classifier scores enter this experiment. Shared feature/window functions retain
identical ASTs to the released sequence experiment. A package-qualified SHA
helper import and import formatting were repaired for reusable test imports;
development source versions are explicitly pinned and audited for that limited
change. Fresh inference used the final pinned implementation and weights.
The source manifests, exclusion list, immutable complete receipts, and frozen
decision are required to reproduce this exact panel. A new selection is a new
evaluation. No customer media or model checkpoints are redistributed.
