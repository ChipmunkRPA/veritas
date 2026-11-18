# Face transition evidence on videos with sparse backgrounds

The current single-transition check uses off-face texture to separate a
persistent change in the face from a cut between scenes. In some eligible
clips, too few coarse tiles around the face provide a usable comparison.
We tested two additions that run when the existing coarse reading lacks
context. One looks at a finer grid of background tiles. The other compares
small brightness patterns below both face boxes over nearby source frames,
requiring texture, spatial spread, and temporal consistency. Both use the
existing face locator and appearance weights. Broad scene changes are screened
before either addition is considered.

This experiment uses the general observation that nearby frames and different
spatial scales can offer complementary evidence. The implementation and weights
are the project's own existing components and the new measurements described
above. The study reused no weights or code from the recent papers.
[MARS](https://arxiv.org/abs/2611.12508) · [MoSTE](https://arxiv.org/abs/2611.12443)

## Held results

Development used 36 files and was sealed before scoring the 246 held deliveries.
The holdout included 96 labeled single transitions and 150 ordinary controls,
including exposure changes, focus changes, jumps, and cuts. Direct and tandem
encoding are separate deliveries of the same sources. Each source contributed
to only one framing group. The three groups were native framing, a tighter crop
about 1.60 times the height of the face, and a crop about 1.35 times its height.
The native framing selection qualified four of 96 unused publisher groups and
zero of 12 independently sourced webcam groups. Three of those webcam groups
also contributed ordinary controls in the tightest crop. Distinct captures and
published donor keys were tracked. Participant overlap remains unknown.

An edit counted as located within one frame of its known boundary. The original
operating point was frozen. The separately calibrated motion reading used the
second-highest development group maximum with a floor of 0.06. Four ordinary
groups produced a usable motion reading during development, leaving this
calibrated point at the floor. Every delivery was checked against its native frame
timestamps and file hash. The nearest available cut backgrounds were chosen
within each framing group. Distances varied. An independent reconstruction
verified all 30 ordinary cut deliveries against their source pictures and splice
positions. The input and reconstruction hashes are in the receipt.

| Framing | Existing locations | Finer tiles | Off-face motion at existing point | Off-face motion at separate point |
| --- | ---: | ---: | ---: | ---: |
| Native, 32 edits | 0 | 17 | 31 | 31 |
| 1.60× crop, 32 edits | 2 | 19 | 32 | 32 |
| 1.35× crop, 32 edits | 0 | 15 | 0 | 0 |
| All 96 edits | 2 | 51 | 63 | 63 |
| Alerts among 150 ordinary deliveries | 0 | 0 | 0 | 3 |
| Edit alerts outside the one-frame band | 0 | 0 | 1 | 1 |
| Deliveries without a usable reading, out of 246 | 187 | 95 | 98 | 98 |

Both additions kept the two existing locations. At the existing point, motion
located 61 additional deliveries across eight source groups. Its single misplaced
alert was in the native group. The separate motion operating point found the
same edits and added three ordinary alerts from two native groups, so the
existing point is preferable in this comparison. Neither addition flagged an
ordinary cut. Fine tiles remained useful in the tightest crop where motion had
too little room outside the face. The 15 successes in that crop came from two
source groups, so broader coverage for tightly framed clips still needs testing.

These results apply to the exact short, standardized video profile and the
two manipulation renderers used here. The figures describe location accuracy,
without an estimate of AI authorship. The released analysis still uses its
current check. The stronger options are ready to test in the source-bound
browser, PDF, JSON, and signed-original workflow.

The compact receipt contains file digests, source groups, reference frames,
and the numeric scores used to reproduce this table. The recount needs
Python 3.11 or later and no model weights. [Evaluation receipt](evaluation-panels/video-face-sparse-transition-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_sparse_transition \
  docs/evaluation-panels/video-face-sparse-transition-2026-11-18.json.gz
```
