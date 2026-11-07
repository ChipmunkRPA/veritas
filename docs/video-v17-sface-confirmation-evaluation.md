# Fresh-source face-transition confirmation

The unchanged v16 candidate plus its declared fallback located **60 of 60**
known swap intervals, compared with **25 of 60** for the current check. Both
flagged the same **26 of 60** genuine/control clips for review. All ten fixed
confirmation requirements passed. The method qualifies for local integration
testing; this report records no serving/default or overall-confidence change.

This is a bounded time-range localization result on a small, constructed research
panel. It is not perfect deepfake detection, population accuracy, a calibrated
AI probability, independent human validation, or validation for court use.

## Prospective selection and execution

The [protocol](video-v17-sface-confirmation-protocol.md),
[source plan](evaluation-panels/video-v17-sface-source-plan.json),
[prepared panel](evaluation-panels/video-v17-sface-prepared-panel.json) and
[complete method freeze](evaluation-panels/video-v17-sface-freeze.json) were
published as `220de3b309f1e2acdf39ac4a1dc1b694ae36ca8e` before fresh baseline
or candidate inference. The private lock SHA-256 is
`ec6e884ecdbc64c452e4ea48901a61423015ffa64b1116a5db5f36785aaf6f1b`.
It binds 433 source files, 314 installed files, exact weights, tools, configuration
and source packets. All 27 existing native-contract/method tests passed first.

Thirty new RTFS-10K source-video groups provide 30 originals, 30 InSwapper
insertions, 30 UniFace insertions and 30 ordinary edits. Selection excluded all
132 previously consumed source-video IDs and 264 donor filenames. All 90 acquired
source files and their matched versions passed byte-duplicate checks against prior
research media. All 120 planned clips were ready; no source was replaced or omitted.
Known edit positions and surrounding decoded frames were checked by the unchanged
boundary-preparation helper. Controls include 13 recompressions, seven jump cuts
and ten lighting changes.

Source-video IDs and donor filenames are disjoint, but shared people, undisclosed
near-duplicates and upstream training overlap remain unknown. Dataset declarations
and source metadata do not independently establish depicted-person consent or all
underlying rights. The publisher's separate CC BY-SA 4.0 dataset terms apply.

The commercial-admitted baseline frame classifier, face detection and tracking
were rerun on every clip through isolated production stages. The current
consecutive representation and new appearance representation were then computed
on the same native frames/tracks. No historical-reference parity was claimed on
these fresh sources. Both methods and the previously genuine-calibrated internal
point remained unchanged; no fitting used the confirmation results.

## Complete comparison

| Outcome | Current | Candidate plus declared fallback |
| --- | ---: | ---: |
| Correctly located edits / 60 | 25 | 60 |
| Missed locations / 60 | 35 | 0 |
| InSwapper locations / 30 | 11 | 30 |
| UniFace locations / 30 | 14 | 30 |
| Genuine/control review flags / 60 | 26 | 26 |
| Original-only review flags / 30 | 13 | 13 |
| Ordinary-edit review flags / 30 | 13 | 13 |
| Mean edited-clip temporal IoU | 0.250312 | 0.848440 |

A correct location means temporal intersection-over-union of at least 0.30
between the first displayed range and the known insertion. That is an internal
localization criterion, not a public authenticity threshold. There were **35
newly correct locations, none lost, no new genuine flags and none cleared**.

The new appearance cue itself supplied 58 edited-clip ranges and no genuine-clip
ranges. Two edited clips lacked sufficient candidate tracked context and were
correctly located by the predeclared current fallback. All 26 genuine flags also
came from shared fallback. Thus, **60/60 is the combined method**, not 60 successful
new-model assessments. Candidate windows were available on 116 clips and unavailable
on four, with zero failures. Both aggregate arms' `status_counts` fields describe
that candidate-window availability; the separate baseline completed all 120.
Unavailable checks remain in the denominator and never imply an original file.

The 26/60 genuine review rate is high. Those are review cues rather than AI
verdicts; the result does not fix the legacy fallback's false alarms. Lighting
controls account for six flags, cuts for three and recompressions for four.
This study improves *where to inspect*, without establishing authenticity or
overall confidence calibration.

Source-group paired bootstrap intervals (5,000 repetitions, seed 20261107) were
[0.485969, 0.706125] for the mean IoU improvement and [0.433333, 0.733333] for
the localization-rate improvement. The genuine-flag difference is zero on this
panel; the [0, 0] paired interval is not a zero-future-false-alarm guarantee.

## Native and numerical evidence

Every applicable native alignment/input/repeated-feature/direct-graph check
passed under the unchanged tolerances. A separate private, model-free review
verified 7,815 native float32 vector byte bindings, searched every allowed
source/track/cadence window and reproduced all 1,392 retained windows. Maximum
separate component difference was 4.440892098500626e-16, within the additional
1e-10 arithmetic diagnostic limit. This additional review did not relax the
original model or quality requirements. No face vectors, imagery or per-frame
biometric hashes are published.

The baseline phase finished in 559.12 seconds and the candidate/current phase in
484.86 seconds, with successful exits and no guards. Sampled process-tree RSS
lower bounds were 2,236,121,088 and 1,041,121,280 bytes. Seventeen and thirty
transient sampling errors mean these are incomplete lower bounds, not true peaks.
The two phases perform different work; their times do not establish an app speedup.

## Public reproduction

The [final-range packet](evaluation-panels/video-v17-sface-final-ranges.json)
contains all 120 source-bound final ranges, known boundaries, statuses and
aggregate results. The [private-review aggregate](evaluation-panels/video-v17-sface-private-review-aggregate.json)
describes its separate verification without exporting the vectors.

Only Python's standard library is needed for the public arithmetic:

```sh
PYTHONPATH=backend python3 -m scripts.reproduce_sface_confirmation_v17 review \
  docs/evaluation-panels/video-v17-sface-final-ranges.json \
  docs/evaluation-panels/video-v17-sface-prepared-panel.json \
  docs/evaluation-panels/video-v17-sface-source-plan.json /tmp/new-sface-confirmation-review.json
PYTHONPATH=backend python3 -m scripts.reproduce_sface_confirmation_v17 tamper \
  docs/evaluation-panels/video-v17-sface-final-ranges.json \
  docs/evaluation-panels/video-v17-sface-prepared-panel.json \
  docs/evaluation-panels/video-v17-sface-source-plan.json /tmp/new-sface-confirmation-tamper.json
```

Local replay checked source/donor exclusions, all 240 per-arm overlaps/location
flags, 154 aggregate fields, eight additional paired/fallback counts and ten
fixed requirements. All twelve deliberate alterations were refused. This does
not repeat native inference or bootstrap fitting and does not prove unseen
people. Outside human reviewers remain zero. Linux replay is a separate automated
check; its actual receipt must be recorded before claiming completion.

## Local release requirements

Integrate the identical method and fallback behind a rollback setting, verify
production-adapter parity on the retained sources, and test real prepaid worker
execution through ordinary evidence views, source-frame navigation, PDF and JSON.
Keep customer model-name filtering and overall confidence unchanged. Full-clip
face swaps without original flanking context, lip synchronization, long clips and
other generator families remain outside this result's validated scope.
