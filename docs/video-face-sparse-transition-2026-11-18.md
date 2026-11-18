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
without an estimate of AI authorship.

## Fresh check of the combined selector

The combined selector keeps any qualifying existing result. Otherwise it checks
motion and finer tiles at the existing face comparison point and uses the
qualifying motion result first. If two alternative results differ by more than
one frame, it withholds the transition. On the earlier held set, this
retrospective selection located 78 of 96 edits, misplaced one, and raised no
alerts among 150 ordinary deliveries. Where both additions identified an edit,
they agreed on the exact frame in all 36 deliveries. The combination was chosen
after those results and was then frozen for the confirmation below.

The confirmation used 54 direct H.264 files from six previously unscored
camera captures and distinct published donor keys. The ordinary captures had
previously been screened for native framing eligibility. The crop boundaries
and edits were new to the measurements. There were 24 deliberately staged edits
from two publisher renderers and 30 ordinary controls, including six cuts. All
edits switched at frame 34. Person overlap remains unverified, and the test
used prepared crops with a single edit position.

| Framing | Existing | Finer tiles | Motion | Combined |
| --- | ---: | ---: | ---: | ---: |
| 1.60× crop, 12 edits | 0 | 9 | 4 | 12 |
| 1.35× crop, 12 edits | 0 | 4 | 0 | 4 |
| All 24 edits | 0 | 13 | 4 | 16 |
| Alerts among 30 ordinary controls | 0 | 0 | 0 | 0 |

Each found boundary matched the reference frame exactly. Combined gains
covered four groups, including one group in the tighter crop. The serving
code reproduced every outcome and preserved the same witness type and face
separation scores. On the local CPU, the median measurement took 1.77 seconds
and all 54 completed within the 12-second budget. This timing excludes model
loading and the source probe. The combined review was unavailable on 20 of
the 54 files, six of which contained ordinary cuts.
[Fresh confirmation receipt](evaluation-panels/video-face-sparse-transition-confirmation-2026-11-18.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_sparse_confirmation \
  docs/evaluation-panels/video-face-sparse-transition-confirmation-2026-11-18.json
```

## Local application

Enable `VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_REVIEW` and
`VERITAS_VIDEO_FACE_SINGLE_SPARSE_CONTEXT_REVIEW` in development to use the
additional checks on new Full or Deep analyses of the qualified four-second
profile. The additional setting defaults off. The report shows the source
timestamp and the before, transition, and after frames, and names the
background measurement that supported the location. JSON preserves the
method for the coarse and exact comparisons. The signed-original check
reconstructs both background comparisons from native frames and verifies
that the source is one continuous shot. Existing records use their original
verification procedure. The check contributes zero weight to AI confidence.

Seven isolated USD 9.99 simulated checkouts tested both additions, ordinary
footage, a cut, and the previous face transition paths. All cases opened in
the browser and verified their PDF, JSON and signed package. Both additions
reported 1.36 seconds at frame 34, with source frames 33, 34 and 45 shown
consistently across the browser and reports. No analysis started before its
simulated payment. [Local application receipt](evaluation-panels/video-face-sparse-transition-app-2026-11-18.json)

The compact receipt contains file digests, source groups, reference frames,
and the numeric scores used to reproduce this table. The recount needs
Python 3.11 or later and no model weights. [Evaluation receipt](evaluation-panels/video-face-sparse-transition-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_sparse_transition \
  docs/evaluation-panels/video-face-sparse-transition-2026-11-18.json.gz
```
