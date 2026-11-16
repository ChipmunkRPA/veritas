# Calibrate paired face readings on fresh recompressed clips

The earlier temporal experiment found a large sensitivity loss on
strongly recompressed face swaps. This follow-up tests whether
combining the two existing face readings differently can recover
those detections. It uses the same native crops and verified weights
as the app's six-frame review. [Temporal results](video-whole-face-temporal-transfer-2026-11-16.md)
· [Earlier clip-level pooling comparison](video-whole-clip-pooling-transfer-2026-11-11.md)

## Fixed design

The 14 previously assigned development groups supplied two deliveries
of each authentic original for calibration. The next 24 groups in
the locked source order supplied the fresh held originals and the
two published face swaps for each. They share no recorded target
link, donor key or source-file digest with the preceding 36 groups.
The publisher supplies the swap labels. The evaluation measures
whole-clip swaps at 640 × 360 on the same codec panel as the previous
study. All 48 held target-to-swap comparisons preserve the same
source sampling pattern. [Publisher source and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

Center and scale each of the two checks using the distribution of
development original medians. Compare four ways to aggregate those
normalized readings.

1. Keep the larger of the two separately calculated medians.
2. Take the larger of the two readings of each sampled face and
   calculate the median across positions.
3. Take the average of the two readings of each sampled face and
   calculate the median across positions.
4. Blend that median with the lower quartile to reward agreement
   that persists over more sampled pictures.

Each alternative has its own paired operating points. Score each
development original using the other 13 groups, retain its larger
response of the two encodes, and use the 90th and 80th percentiles
of those group maxima. All candidate parameters and the held
delivery manifest were locked before the new held model readings.
The candidates share one calibration across both delivery levels.
Same-frame calculations need four jointly measured faces.

## Held results

| Method at the tighter point | Direct swaps flagged /48 | Tandem swaps flagged /48 | Direct ordinary flags /24 | Tandem ordinary flags /24 |
| --- | ---: | ---: | ---: | ---: |
| Existing paired check at its saved point | 35 | 15 | 4 | 4 |
| Recalibrated clip maximum | 11 | 1 | 0 | 0 |
| Same-frame maximum | 11 | 1 | 0 | 0 |
| Same-frame agreement | 12 | 1 | 0 | 0 |
| Persistent agreement | 14 | 1 | 0 | 2 |

At the wider point, same-frame agreement adds one ordinary tandem
alert. Every swap total remains the same across the two candidate
points. All candidate positives overlap existing alerts. One source
group lacked face coverage, accounting for two unassessed swaps and
one unassessed original per delivery across all methods. The tighter
candidate settings give up too many of the existing swap detections
for the proposed sensitivity improvement.

An exploratory comparison varied the cutoff separately on each
already examined held delivery set, at a budget of four ordinary
flags. Same-frame agreement reaches 38 direct and 21 tandem swap
flags, versus the current 35 and 15. This estimates possible
discrimination under an in-sample cutoff. It supplies no deployable
operating point. The prospective points remain the record of the
actual test. A separate representative calibration set would need
to recover the gain before the paired detector changes.

## Replay

Recompute all candidate points, current flags and matched changes
from the public numeric receipt. It contains exact sample clocks,
response values and digests. The dataset media and face boxes stay
local.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-paired-face-calibration-2026-11-16.json.gz
python3 -B -S -m scripts.video_face_pair_calibration verify "$receipt"
python3 -B -S -m scripts.audit_video_face_temporal verify-pair-diagnostic "$receipt"
```

Use the pinned publisher file index and source-restore commands in
the [native replay](video-whole-face-temporal-transfer-2026-11-16.md#replay).
Run the existing face check for the `development` split of that source
package and the `held` split of the follow-up. Restore the follow-up
roster by passing this page's receipt to the same restore command.

```sh
python3 -B -S -m scripts.video_face_pair_calibration freeze \
  "$previous/delivered/manifest.json" "$previous/reference-dev.jsonl" "$run/delivered/manifest.json" "$run/freeze.json"
"$runner" -B -m scripts.video_face_temporal_reference "$run/delivered" "$run/reference.jsonl" held
python3 -B -S -m scripts.video_face_pair_calibration report \
  "$run/delivered/manifest.json" "$run/reference.jsonl" "$run/freeze.json" "$run/result.json.gz"
```
