# Checking retained footage around a reported insertion

The earlier-copy video comparison now verifies both retained sections before
publishing a single insertion range. A sustained local mismatch or a section
that fits better at another frame offset can indicate that one interval fails
to explain the submitted track. In that case the browser and supplemental PDF
show **The retained pictures did not support a single insertion**. JSON
records the same status and both support summaries. A measured single range
keeps its exact before, during and after pictures, source ticks, and exclusive
end. The comparison tests the picture track and carries zero AI-confidence
weight.

## Method and results

The existing proposal compares all retained pictures at offsets zero and
the net count of added pictures. Its whole-section percentiles can hide
brief stretches of mismatched pictures. The extension scans overlapping
eight-picture windows in both retained sections. It checks the worst median
luma mismatch against 0.035 and the largest mean improvement from any other
between-copy offset against 0.020. Both are internal normalized error
units. The candidate can withhold a proposal; its accepted boundaries come
unchanged from the existing algorithm. The experiments also run each check
alone. The offset check rejected one additional incorrect development report
compared with the median check. Both agreed on the held results.
[Earlier multiple-insertion evaluation](video-temporal-multiple-inserts-2026-11-13.md)
· [Research on video copy localization in time](https://openaccess.thecvf.com/content/CVPR2022/papers/He_A_Large-Scale_Comprehensive_Dataset_and_Copy-Overlap_Aware_Evaluation_Protocol_for_CVPR_2022_paper.pdf)

| Input | Prior correct single ranges | Correct single ranges retained | Incorrect single ranges, before → after |
| --- | ---: | ---: | ---: |
| Development on four cached sources | 4/4 | 4/4 | 12 → 0 |
| Held on eight different cached sources | 8/8 | 8/8 | 30 → 0 |
| Public conversational transfer, four sources | 12/12 | 12/12 | 2 → 0 |

The eight held groups contributed 80 comparisons. Seventeen of the 30
incorrect earlier results grouped separate insertions. Thirteen combined
insertions with ordinary replacement or removal, where one interval cannot
explain both timelines. The 40 conversational comparisons used ordinary
insertions from a separate excerpt of each recording. The 12 correct cases
include four with a global brightness grade. None of the 44 controls across
both held banks for unchanged, wrong-reference, and static/near-static source
conditions gained a single-insertion finding. The eight cached groups had
prior exposure in the earlier comparison. The four conversational recordings
had prior use for lip-sync testing but were disjoint from the fitting sources
and had no earlier insertion fit. Both held banks passed the frozen
retention and false-report gates. [Held ledger](evaluation-panels/video-insert-retention-held-2026-11-13.json)
· [Public transfer ledger](evaluation-panels/video-insert-retention-public-2026-11-13.json)
· [Development ledger](evaluation-panels/video-insert-retention-development-2026-11-13.json)

Replaying the original 72-delivery single-insertion confirmation recovered
all 29 previously correct ranges under the original rule. With the stricter
check, 27 of 29 remained measured, including 15 of 16 generated and 12 of 13
ordinary insertions. Both newly unassessed deliveries came from one original
source and the same exact [80, 110) frame interval. The eight-picture
median mismatch of 0.03713 exceeded the fixed 0.035 limit in the retained
section after the edit. Both passed the alternate-offset test. This is a
loss of correct coverage under the stricter comparison. It shows why an
unassessed pair can still contain a single insertion. The same frozen
check was used for this replay.
[Historical regression ledger](evaluation-panels/video-insert-retention-regression-2026-11-13.json)

The scope is an earlier copy and an analyzed H.264 pair on the same verified
24 fps clock, with matching geometry and a 480-pixel long side. The check
may abstain if normal changes to the retained pictures compete with the match.
Multiple-insertion localization remains experimental. The app continues to
support byte-for-byte offline replay of existing version 12 comparisons.
No model weights are required. The released ledgers contain scalar
measurements, native clocks, and hashes without any source pictures.

The isolated local app verified both outcomes after simulated payment. The
ordinary insertion matched frames [31, 61), or 1.292 to 2.542 seconds to
three decimals. Chrome loaded all five original witness images and
downloaded the same JSON the endpoint returned. The mixed-replacement
case displayed the unassessed explanation with no insertion pictures.
Both PDF downloads agreed with the JSON, and both exported comparisons
replayed from the two source files. The earlier-copy request reused
the existing completed analysis.

## Independent replay

Verify the published ledgers with Python's standard library.

```sh
export PYTHONPATH=backend
for bank in development held public; do
  python3 -B -S -m scripts.video_insert_retention_transfer review \
    "docs/evaluation-panels/video-insert-retention-${bank}-2026-11-13.json"
done
python3 -B -S -m scripts.review_video_insert_retention_regression \
  docs/evaluation-panels/video-insert-retention-regression-2026-11-13.json
```

The full comparison requires Python with NumPy and the app's video probe,
plus FFmpeg with libx264 and ffprobe. Recreate the earlier source caches by
following the [native intake](video-temporal-insert-native-2026-11-13.md#replay)
and [held intake](video-temporal-insert-confirmation-2026-11-13.md#replay).
Alternatively, acquire just the four [public recordings and attribution](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)
into `$public_source`, saving each as its listed ID plus `.webm`. The runner
verifies each exact source digest and reads the fixed offsets from that
manifest. Use a new path for each receipt. The encode and readback bind the
comparison to the authored picture sequence and its 24 fps clock.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_insert_retention_transfer development "$native_cache" --write "$dev_receipt"
backend/.venv/bin/python -B -m scripts.video_insert_retention_transfer held "$confirmation_cache" --write "$held_receipt"
backend/.venv/bin/python -B -m scripts.video_insert_retention_transfer public "$public_source" --write "$public_receipt"
backend/.venv/bin/python -B -m scripts.review_video_insert_retention_regression "$new_regression_receipt" \
  --native-cache "$confirmation_cache" --preflight "$native_cache"
```
