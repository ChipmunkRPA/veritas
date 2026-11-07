# Image crop v6: reject under frozen fresh-image requirements

6 November 2026. **Keep both current format-specific checks.** Cropping recovers
more native edits, but also adds too many genuine-image highlights. The JPEG
gain is small and also exceeds the false-highlight allowance. Do not adjust the
gates after these results or enable this prototype.

The [source plan](evaluation-panels/fused-crop-v6-source-plan.json), all48 new
source pairs, runtime, weight digests and20 method/test pins were frozen and
published as `589f81a` before any fresh detector forward. All78 selected tests
passed. Twelve source-identity-selected consumed replays reproduced every native
map, scalar and highlight-gate decision exactly (maximum difference0), before
the fresh run. All192 planned input outcomes then completed: no failures or
scope exclusions, four crops per input (768 crop forwards), and the current
format-specific checks on the same inputs. The reviewer reconstructed all source
bytes, crop coordinates/RGB commitments, raw arrays and displayed metrics.

| Format,48 genuine +48 edited images | Current genuine highlights | Crop genuine highlights | Current overlapping edits | Crop overlapping edits | Mean edited IoU, current → crop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native PNG | 2 | 9 | 12 | 20 | 0.18395 → 0.23355 |
| JPEG75 | 3 | 6 | 12 | 13 | 0.10344 → 0.10371 |

Native gains eight located edits but adds seven genuine highlights, exceeding
the frozen allowance of one. Its mean IoU gain is0.04959 (source-paired95% interval
[0.01369,0.09556]); that pixel-overlap gain does not satisfy the false-highlight
requirement. It still misses28/48 edited images.

JPEG gains one located edit but adds three genuine highlights. Its mean IoU gain
is0.00027 (interval[0.00000,0.00081]) and it still misses35/48 edits. Neither
format qualifies for local integration. These are component review cues and
changed-pixel overlap measurements, not whole-file AI verdicts or calibrated
probabilities. The48 sources are shared across formats and are now consumed.

The [aggregate summary](evaluation-panels/fused-crop-v6-confirmation-summary.json)
provides per-format/per-generator counts and uncertainty.
The [complete result](evaluation-panels/fused-crop-v6-confirmation-review.json)
retains all192 case outcomes, per-generator metrics, native replay, raw-map/RGB
commitments and unchanged gates. No source image, raw mask, checkpoint or private
path is redistributed. No overall-confidence, API/UI/PDF/JSON or serving default
was changed. Outside human reviewers:0.

This rejects the tested maximum-of-four-crops method. The improvement seen on
the24-source development panel did not retain its genuine-image tradeoff on
fresh data. Whether lost context, texture, processing history or another factor
caused particular false highlights has not been independently established.
Future methods need new source-disjoint confirmation and must exclude all288
consumed TGIF originals plus the eight historical web examples. Preserve this
failed result and both original check paths for comparison.
