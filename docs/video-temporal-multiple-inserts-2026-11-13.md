# Multiple inserted ranges in an earlier-copy video comparison

The existing comparison can fit one run of additional pictures. If a clip
adds footage at more than one point, that check can either withhold a result
or group distinct edits into one reported interval. This experiment tested an
ordered alignment that can locate exactly two or three ranges. It treats the
earlier picture stream as retained in order and gives the finding zero
weight in the AI estimate. Both ordinary and generated footage can be added.

## Method and frozen panel

The comparison builds a matrix of differences between verified 96 by 96
grayscale pictures at each candidate frame offset. A dynamic program
minimizes the summed difference across retained pictures while allowing
two or three forward jumps. Each jump adds at least 16 submitted pictures
and must be bounded by at least 24 retained earlier pictures, including
between jumps. It checks that each segment has temporal detail and a close
match, and that no competing local offset better matches a 12-picture span.
It withholds ranges when alternatives within the fixed cost tolerance
disagree by more than three frames on an endpoint or disagree on how many
ranges exist. Against an incumbent single result, the new route must also
improve the total match cost by a fixed minimum. The method applies the
published idea of mapping video matches in time. It uses the repository's
existing luma features and requires no learned weights.
[Video copy localization and evaluation](https://openaccess.thecvf.com/content/CVPR2022/papers/He_A_Large-Scale_Comprehensive_Dataset_and_Copy-Overlap_Aware_Evaluation_Protocol_for_CVPR_2022_paper.pdf)

The original four cached recordings were used for development. The eight
held recordings passed the earlier native input and overlap checks and were
disjoint from development recordings. All held edit recipes are new. The
footage itself was already used in the single-insert study, and all groups
come from one publisher. The source deposit declares CC BY 4.0. The
published measurements contain no source footage, evidence images, or
model weights.
[Source deposit](https://doi.org/10.5281/zenodo.20548572) ·
[Prior source verification](video-temporal-insert-confirmation-2026-11-13.md)

The panel uses 150 earlier pictures per source, delivered as H.264 at
24 fps with a 480-pixel long side and no soundtrack. The ten conditions per
source include two generated insertions, three generated insertions, two
ordinary insertions, an unchanged clip, one generated insertion, a wrong
earlier copy, insertions mixed with replacement, insertions mixed with
removal, and two insertions each on a static and near-static earlier clip.
The static reference repeats one source picture. The near-static version
adds a small periodic luma offset. Each authored segment was checked against
its source pictures after encoding. Independent FFmpeg frame hashes bind
the held results and before, during, and after evidence to native clocks.

The frozen criteria require at least six of eight complete results for each
generated condition and at least six sources where both conditions pass.
They also require at least five of eight ordinary results, no newly reported
negative controls, no positive result with a missing or displaced range, and
no disruption to the existing single result. A complete result accounts for
every authored range, each edge within three frames. Reports from the
incumbent on mixed and static controls are recorded separately.
[Frozen recipes and criteria](evaluation-panels/video-temporal-multiple-inserts-plan-2026-11-13.json)

## Held results

| Condition | Pairs | Existing complete | Proposed multi-range exact | Proposed multi-range withheld |
|---|---:|---:|---:|---:|
| Two generated inserts | 8 | 0 | 3 | 5 |
| Three generated inserts | 8 | 0 | 3 | 5 |
| Two ordinary inserts | 8 | 0 | 3 | 5 |

The same three source groups, 97, 91 and 72, account for all nine exact
multi-range results. The other five source groups lack sufficient temporal
detail under the frozen check. The candidate made no new reports across
48 unchanged, wrong-reference, mixed-edit and static or near-static
controls. All eight correct single-insert results were preserved.

Across the 24 multiple-insert positives, the existing method reported one
incorrect interval on 17 pairs. Opting into the multiple-insert alternative
would still leave nine incorrect intervals from the incumbent fallback.
The incumbent also reports an insertion interval on 13 of 16 mixed
insertion/replacement or insertion/removal controls, where no one interval
explains the relationship between both clips. The multi-range check failed
the held coverage and interval-accuracy gates and remains experimental.
Lowering motion thresholds on these same held clips would need another
source-disjoint test. The results suggest studying finer temporal features
that can distinguish subtle real motion from encoding noise and allowing
both picture additions and deletions while testing each reported operation.
[Held ledger and evidence clocks](evaluation-panels/video-temporal-multiple-inserts-held-2026-11-13.json)

## Replay

Use the public earlier-copy source cache populated by
[`video_temporal_insert_confirmation acquire`](video-temporal-insert-confirmation-2026-11-13.md#replay).
Use a new `$receipt` output path. The separate `review` command requires
only the Python standard library and checked-in source manifests.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.video_temporal_multiple_inserts plan > "$regenerated_plan"
cmp "$regenerated_plan" docs/evaluation-panels/video-temporal-multiple-inserts-plan-2026-11-13.json
backend/.venv/bin/python -B -m scripts.video_temporal_multiple_inserts score "$confirmation_cache" --partition held --write "$receipt"
python3 -B -S -m scripts.video_temporal_multiple_inserts review "$receipt"
cmp "$receipt" docs/evaluation-panels/video-temporal-multiple-inserts-held-2026-11-13.json
```
