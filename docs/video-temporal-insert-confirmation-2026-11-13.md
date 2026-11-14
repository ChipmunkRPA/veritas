# Confirmation of inserted-footage localization

## What the check measures

The check looks for footage present in a submitted video between pictures
that also occur, in order, in an earlier copy. It scores a single proposed
insert by comparing the pictures before and after that interval. The finding
identifies a physical edit relative to the supplied earlier copy. It carries
zero weight in the AI confidence estimate. Inserting ordinary footage can
produce the same finding.

## Confirmation panel

The [frozen plan](evaluation-panels/video-temporal-insert-confirmation-plan-2026-11-13.json)
chooses examples 2, 81, 97, 91, 82, 71, 70 and 72 from the same publisher as
the [initial comparison](video-temporal-insert-native-2026-11-13.md). The
sixteen source downloads total 345,702,960 bytes. Every member was checked
against the publisher's CRC and native picture clock. Source overlap checks
compare the new source files and decoded pictures with each other and with
the eight original and generated source clips from the initial comparison.
This checks for duplicates in the evaluated material. It cannot establish
independence by generator, camera, or publisher. The Zenodo record lists the
dataset under CC BY 4.0. The comparison uses no model weights. The repository
contains metadata, clock and picture commitments, and outcomes, without
redistributing media pixels.
[Dataset record](https://doi.org/10.5281/zenodo.20548572) ·
[Publisher methodology](https://arxiv.org/html/2606.07090v1#S3.SS2)

Each source contributes two generated inserts at prescribed early and late
positions, two ordinary inserts of matching lengths and positions, two
same-length generated replacements, two wrong earlier references and one
unchanged control. Insert lengths are 30, 54, 72 or 102 frames. Native source
and delivered clocks are exactly 24 fps. Delivered picture streams use H.264
and a 480-pixel long side, with no soundtrack. Independent FFmpeg frame hashes
and timestamps verify that the decoder kept the intended clock. The runner
also checks that the rendered pictures correspond to the authored prefix,
insert or replacement, and suffix. Reference comparison results are reported
against the existing app with all its local picture options enabled.

The paired method and its alignment thresholds were frozen before this panel
was scored. Both edges must fall within three frames of the authored interval
to count as within tolerance. Local advancement requires at least eight exact
generated ranges, at least 12 of 16 generated ranges within three frames,
success in at least six of eight source groups, no generated range outside
that tolerance, at least eight of 16 ordinary inserts within tolerance, and
no reports on unchanged deliveries, wrong references or equal-length
replacements. These are evaluation criteria for the bounded feature.

## Results

| Condition | Pairs | Existing picture comparisons assessed | Insert ranges reported | Exact ranges |
|---|---:|---:|---:|---:|
| Generated insert | 16 | 0 | 16 | 16 |
| Ordinary insert | 16 | 0 | 13 | 13 |
| Unchanged delivery | 8 | 8 | 0 | — |
| Generated equal-length replacement | 16 | 16 | 0 | — |
| Wrong earlier reference | 16 | 0 | 0 | — |

All eight planned source groups passed the overlap and native-clock checks.
The paired method found both generated edges exactly on every source and
every render. It withheld three ordinary insertions because several possible
starts fit the pictures. The existing app declined ten generated insertion
pairs because their timelines differ and the other six at its duration limit.
All six local advancement criteria passed.

The 72 rows cluster into eight source groups and use one publisher's footage.
These results cover compatible clips with earlier copies, one additional
interval, and recognizable retained pictures on both sides. They do not
estimate single-upload deepfake accuracy, performance across publishers, or
performance after changes to the frame rate or geometry. Equal-length
replacement controls assess false insert reports. The existing picture
comparison measured decoded difference ranges on all 16 such controls.
[Confirmation ledger](evaluation-panels/video-temporal-insert-confirmation-2026-11-13.json)

## Local app

Set `VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_VIDEO_REFERENCE_PICTURE_INSERTION=true` on the local backend. Sign
in, open a completed video analysis, and choose **Compare with an earlier
video**. Use matching frame dimensions with a 480-pixel long side, upright
H.264 with `yuv420p`, and exact 24 fps starting at timestamp zero. The
feature can compare up to 360 verified frames. The earlier copy must provide
at least 24 pictures on each side that satisfy the alignment checks together. The
result gives the submitted time range and screenshots taken before, during,
and after the inserted footage. It also gives corresponding screenshots from
the earlier video on both sides and retains native timestamps in JSON and
PDF. The comparison uses the existing authenticated earlier-copy route and
does not start a paid analysis or alter the existing analysis result.

## Replay

Use new paths for `$confirmation_cache` and `$receipt` and the prior native
analysis media cache for `$preflight_cache`. Keep the two caches separate.
The public ledger can also be recounted offline with Python's standard
library. This mode verifies the roster, boundaries, witness times, frame
clock commitments and totals. Replay with the media recomputes the content
commitments and output from the actual pictures.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.video_temporal_insert_native acquire "$preflight_cache"
python3 -B -S -m scripts.video_temporal_insert_confirmation acquire "$confirmation_cache"
backend/.venv/bin/python -B -m scripts.video_temporal_insert_confirmation score "$confirmation_cache" --preflight "$preflight_cache" --write "$receipt"
python3 -B -S -m scripts.video_temporal_insert_confirmation review "$receipt"
cmp "$receipt" docs/evaluation-panels/video-temporal-insert-confirmation-2026-11-13.json
```

Add `--verify-app` to `score` to exercise the local opt-in comparison on the
same deliveries and require matching intervals, source ticks, screenshot
roles, screenshot hashes, and zero contribution to AI confidence.
The two complete local runs reproduced the ledger byte for byte, including
the run that verified app results on all 72 pairs. The offline recount agrees
on Python 3.11 and 3.12. It rejects changes to a native clock commitment, a
witness timestamp, an authored split, or the wrong-reference media binding.
