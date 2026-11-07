# Face-swap transition development comparison

The frozen candidate located **55 of 60** swaps, compared with **19 of 60** for
the current check, on 120 previously consumed development clips. Both flagged
21 of 60 genuine/control clips for review. All ten original advancement
requirements passed. This qualifies the unchanged method for a separately frozen
fresh-source confirmation, not a serving/default or overall-confidence change.

## What was actually compared

The [prospective protocol](video-v16-sface-development-protocol.md), source panel
and implementation were published before any project real-face inference. The
genuine-only operating point was published before the manipulated development
clips were scored. Sixty genuine calibration clips come from 30 earlier v5 source
groups; 120 development clips come from 30 disjoint v6 source groups. These are
consumed development sources, not unseen confirmation subjects. Shared subjects
and overlap with upstream model training are unknown.

Both arms use the same native timestamps, scheduled neighboring frames, detected
faces, track association, candidate time windows and shared legacy fallback.
The current consecutive representation was recomputed and checked against its
historical reference. The legacy frame classifier and its track exhibits were
held as reference receipts. This was not a complete fresh pipeline rerun.

The Apache-2.0 SFace representation measures a bounded change in face appearance:
the middle interval differs from both flanks, while the two flanks resemble one
another. It is not an identity lookup, a lip-sync check, a calibrated probability
of AI generation, or a pixel-level edit mask. The candidate replaces the selected
consecutive range when eligible; otherwise the declared current/legacy fallback
applies. An unavailable candidate preserves the current result.

## Results

| Outcome | Current | Candidate |
| --- | ---: | ---: |
| Correctly located swaps / 60 | 19 | 55 |
| Missed edit locations / 60 | 41 | 5 |
| InSwapper locations / 30 | 13 | 28 |
| UniFace locations / 30 | 6 | 27 |
| Genuine/control review flags / 60 | 21 | 21 |
| Original-only review flags / 30 | 9 | 9 |
| Ordinary-edit review flags / 30 | 12 | 12 |
| Mean temporal overlap on edited clips | 0.250886 | 0.760788 |

A correct location means the first displayed range overlaps the known edit with
temporal intersection-over-union of at least 0.30. There were 36 newly correct
locations, none lost, and no added or cleared genuine flags. Lighting, jump-cut
and recompression controls remain in the denominator. Both arms have 108 eligible
cases, 12 unavailable cases and no failed cases. Unavailable cases are retained;
they do not imply authenticity.

Source-group paired bootstrap intervals (5,000 samples, fixed seed 20261107)
were [0.399262, 0.620456] for the overlap improvement and [0.466250, 0.733333]
for the localization-rate improvement. The paired genuine-flag difference is
zero on this panel; this is not a guarantee of zero future added false alarms.
The unchanged **21/60 genuine review rate is high**. Many flags are shared fallback
cues, and they must not be represented as AI verdicts or confidence calibration.

## Engineering checks and limits

All 180 source/native-clock bindings and current-check replays passed. For each
assessed face, actual native alignment and input bytes matched the independently
transcribed native contract; repeated and direct-graph feature comparisons stayed
within the original numerical limits. A separate private, model-free calculation
verified 11,676 native vector byte bindings, searched every allowed window and
reproduced all 1,968 retained windows. Its maximum component difference was
4.440892098500626e-16; the additional diagnostic limit was 1e-10 and did not
replace any frozen model or quality requirement.

Calibration and development finished in 231.69 and 478.67 seconds, respectively,
with no guard termination and successful child exits. Sampled process-tree memory
lower bounds were 1,065,844,736 and 1,068,007,424 bytes. Fifteen and eighteen
transient process-sampling errors mean these are incomplete sampled lower bounds,
not measured true peak memory.

Real face vectors, frame imagery and private packet byte hashes remain private.
The [public final-range packet](evaluation-panels/video-v16-sface-final-ranges.json)
contains source IDs, known edits, predicted ranges, statuses, aggregate outcomes
and run profiles. The [private arithmetic aggregate](evaluation-panels/video-v16-sface-private-review-aggregate.json)
discloses what was checked without exporting biometric vectors. Research data
retain the source catalogue's separate dataset terms; software licensing does
not grant additional rights in depicted people.

## Reproduce the public arithmetic

Only Python's standard library is needed:

```sh
python3 backend/scripts/reproduce_sface_result_v16.py review \
  docs/evaluation-panels/video-v16-sface-final-ranges.json \
  docs/evaluation-panels/video-v16-sface-panel.json /tmp/new-sface-range-review.json
python3 backend/scripts/reproduce_sface_result_v16.py tamper \
  docs/evaluation-panels/video-v16-sface-final-ranges.json \
  docs/evaluation-panels/video-v16-sface-panel.json /tmp/new-sface-range-tamper.json
```

The local independent arithmetic checked all 120 source labels/ranges, 240
per-arm overlaps/location flags, 154 aggregate fields and ten advancement gates.
All ten deliberate packet alterations were refused. It does not rerun native
inference, select the operating point, or recompute bootstrap intervals. Outside
human reviewers remain zero; automated review is not community validation.

## Next decision

Freeze source-disjoint confirmation with the same point and method. Recompute
the baseline on those fresh clips rather than claiming historical parity where
there is no reference. Require better localization without a higher allowed
false-alarm count, and complete the local evidence/export/prepaid flow before
changing the default. Do not add this appearance cue to overall AI confidence.
