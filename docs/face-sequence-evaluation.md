# Denser sampling and face sequence comparison

This local experiment compares two independent changes to the video review
range. Both use the same frozen source clips and preserve the current review
range when their candidate has no usable result. Neither changes whole-video
confidence as part of the experiment.

## Test inputs and weights

The [RTFS-10K release](https://huggingface.co/datasets/godmodes/rtfs-10k)
provides original clips and paired outputs from two swap methods. The source
selection excludes the 72 source-video IDs and the donor filenames from three
earlier comparisons. Thirty new source groups each provide an original, two
swaps inserted at the same known boundaries, and an ordinary edit of the
original. The first ten source groups are for development. Twenty different
groups, totaling 80 clips, are held out.

All inserted swap frames were verified by decoded pixel digests. Ordinary
edits comprise an eight-frame jump cut, a temporary lighting change, or
recompression. In the prepared jump-cut controls, the encoder also repeats
five neighboring frames. The remaining picture content matches the original
after removing the specified eight frames and accounting for the repeats.
The controls contain no inserted face-swap frames. Source identities and
donors are disjoint by the publisher's metadata. Overlap with pretraining is
unestablished.

The current and denser arms use the same two pinned frame-check weights from
the [MIT-licensed release](https://github.com/selimsef/dfdc_deepfake_challenge/releases/tag/0.0.1).
The consecutive-frame arm uses the publisher's pinned [Apache-2.0 feature weights](https://huggingface.co/facebook/dinov2-small/blob/ed25f3a31f01632728cabb09d1542f84ab7b0056/README.md)
and the existing MIT-licensed face detector. The runner verifies the installed
weight hashes and checks the project's commercial policy before loading.
No checkpoint with research-only or unresolved release terms is loaded for
the comparison.

## Comparisons

- **Current.** The full profile's face frames and tracks. The first interval
  is shown when the current rule finds one. Otherwise the range spans the
  current peak frame and its selected neighbors.
- **Denser sampling.** The same face scorer and tracks, with additional native
  frames near three separated score peaks. This is tested at the current
  interval rule and again with a development-only rule chosen from a small
  threshold grid. Both variants use the current range as a fallback.
- **Consecutive crops.** Each coarse sample is represented by three actual
  consecutive aligned face crops. The check measures change in the inner
  face across a potential interval and both flanks, discounts change in the
  outer region, and discounts disagreement between the flanks. Its window
  calculation uses no frame-classifier scores and does not use the denser
  schedule. It applies the frozen development threshold and uses the current
  range as a fallback.

A correct location requires the first displayed range to overlap the known
insert by at least 30% intersection over union. A miss has zero overlap.
Genuine clips count as false alarms when the method supplies a review range.
The local adoption rule requires more localized edits on the held-out split
and at most one additional genuine clip flagged overall and one additional
ordinary-edit control flagged. This absolute tolerance is 2.5 percentage
points overall on the 40 genuine controls. The test compares review cues,
and the counts do not describe whole-file AI conclusions.

## Held-out results

| First review range | Localized swaps (40) | Genuine clips flagged (40) | Ordinary edits flagged (20) | Mean overlap |
| --- | ---: | ---: | ---: | ---: |
| Current | 7 | 15 | 8 | 14.3% |
| Denser sampling | 5 | 15 | 8 | 10.9% |
| Consecutive crops | 12 | 17 | 9 | 24.5% |

Development selected the existing interval thresholds for the denser arm, so
the fixed-threshold and development-selected rows agree. It tested a median
of 29.5 additional scored frames on 76 held-out clips. The sequence arm
returned measurable windows on 72 clips. Four clips had no applicable frame
score in the current check and remained in the totals. No baseline inference
failed. The sequence threshold was `0.005256394110620025`.

Consecutive crops recovered two additional InSwapper intervals and three
additional UniFace intervals and retained every interval correctly located by
the current check. Its two extra false alarms came from one source group,
the untouched clip and its lighting-change control. Both flagged 2.292–4.125 s
of genuine footage. In a separate source group, the check flagged 2.583–3.625 s
against a known UniFace insert at 2.500–3.750 s. The positive examples and the
false alarms share the constraints of this paired, short-clip benchmark.

Denser sampling lost two localizations at the same false-alarm count. The
sequence arm exceeded the frozen allowance of one additional flagged genuine
clip. The local app continues to use the current review check.

The local per-clip receipt is `backend/data/independent-evaluation/video-v5/heldout-summary.json`.

| Record | SHA-256 |
| --- | --- |
| Input manifest | `82b7b77de9d8dbdd4e29ed72403828bc334643d997180be36fe62fc8b3fac341` |
| Development operating points | `0867bd713cdfa49b5995e3501033bd6785d5dc74f8df3cac38b0757b50ca43af` |
| Held-out current scores | `c19b9f9377cf1eee57af63bfb84c769ffb6275e8f10f136e50a63e6bef9b066c` |
| Held-out candidate scores | `7f44cddc4062195a3593960dceb7594f0fcda1508fd14bf1120a87b2af35877d` |
| Held-out summary | `4b8eb91988d05fca4c94238ba1d019c1bfd45785f91c87d5319d9d0439fabb09` |

## Reproduction

Run from `backend`. The prior manifests in `video-v2`, `video-v3`, and
`video-v4` supply the exclusion list for the fresh source acquisition.

```sh
trial=data/independent-evaluation/video-v5
.venv/bin/python scripts/prepare_face_swap_upgrade_eval.py "$trial/sources" \
  --groups 30 --exclude-manifest "$trial/excluded.json"
.venv/bin/python scripts/prepare_face_interval_eval.py "$trial/sources/manifest.json" "$trial/intervals" \
  --development-groups 10 --hard-controls
for split in development test; do
  PYTHONPATH=. VERITAS_DEVICE=mps .venv/bin/python scripts/score_face_interval_eval.py \
    "$trial/intervals/manifest.json" "$trial/current-$split.jsonl" --split "$split"
  PYTHONPATH=. .venv/bin/python scripts/score_face_sequence_candidates.py \
    "$trial/intervals/manifest.json" "$trial/current-$split.jsonl" "$trial/candidates-$split.jsonl" --device mps
  if [ "$split" = development ]; then
    PYTHONPATH=. .venv/bin/python scripts/summarize_face_sequence_eval.py \
      "$trial/intervals/manifest.json" "$trial/current-development.jsonl" \
      "$trial/candidates-development.jsonl" "$trial/operating-points.json"
  fi
done
PYTHONPATH=. .venv/bin/python scripts/summarize_face_sequence_eval.py \
  "$trial/intervals/manifest.json" "$trial/current-test.jsonl" "$trial/candidates-test.jsonl" \
  "$trial/operating-points.json" --test-output "$trial/heldout-summary.json"
```
