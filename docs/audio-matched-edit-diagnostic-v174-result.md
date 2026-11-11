# Ordinary cross-source edits remain a major speech false-alarm source

The guarded local CPU run completed 414 cases from 207 matched pairs.
Both arms returned scores in 204 pairs. The scorer made 409 raw forwards
and six original-side stability forwards in 255.66 seconds, with 2.77 GiB
maximum sampled process-tree RSS and zero guard or sampling errors.
[Frozen method](audio-matched-edit-diagnostic-v174-method.md) · [Verified aggregate](evaluation-panels/audio-matched-edit-diagnostic-result-v174.json)

| Background | Ordinary false AI flags | AI inserts flagged | AI inserts missed | Of missed inserts, leaned original | Of missed inserts, inconclusive or unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 51/69 | 66/69 | 3/69 | 0 | 3 |
| 0 dB | 34/69 | 41/69 | 28/69 | 0 | 28 |
| +5 dB | 37/69 | 50/69 | 19/69 | 2 | 17 |
| Total | 122/207 | 157/207 | 50/207 | 2 | 48 |

For ordinary edits the other outcomes were four original, 78 inconclusive
and three unavailable. Ordinary false flags fell from 45/69 with a hard
join to 44/69 with a 20 ms fade and 33/69 with a 50 ms fade, pooling noise
conditions. AI-insert flags fell from 55/69 to 53/69 and 49/69 across those
same joins. Smoothing affected both outcomes. These paired interventions
do not identify a unique cause of the false flags.

Neither repetition cue found a repeated span in any of the 414 cases.
The repetition comparator therefore changed no decisions. Its earlier
benefit applied to same-source repeated copies. The matched genuine
donors here supply different speech. Keep repetition available as
inspection evidence, and leave it out of the speech-origin decision for
this edit class. This study measured whole-clip tendencies. It supplied
no localization claim and used no holdout audio.
[Same-source diagnostic](audio-short-edit-diagnostic-v173-result.md)

All 23 comparison blocks share a connected source component. The
next origin comparison should score the splice interior independently
of its joins, then compare both against untouched material from the same
host. Measure localization on exact sample support and keep ordinary edit
false alarms separate from missed synthetic inserts. Recent studies give
examples of interior supervision and document how boundary-focused scores
can transfer differently across editing conditions and corpora.
[Segment-aware learning](https://arxiv.org/html/2601.21925v1) · [Matched boundary interventions](https://arxiv.org/html/2610.04479v1#S3.SS2) · [Cross-corpus diagnostic](https://arxiv.org/html/2605.29531v3)
