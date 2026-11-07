# Region-aligned calibration: completed, rejected

Matching calibration to displayed regions did not fix the candidate's missed
image edits. It still located0/48 edits in both PNG and JPEG75, compared with12/48
for the current components. Keep current image checks and preserve the earlier
v8 rejection. Neither format advances to fresh confirmation or service integration.

## What changed and what the test establishes

The display requires an8-connected region covering at least0.1% of its entire
inspection grid. The old point statistic was a single pixel maximum. The new
statistic finds the highest response plateau that still forms a large enough
region. It keeps the original raw response map, reliability exclusions, component
ranking, two-region cap and source-pixel projection.

The [protocol](image-region-calibration-v9-protocol.md) and full source/response
freeze were public before computing these region levels. The new method was
tested only on the same312 already-consumed v8 inputs:60 genuine calibration
sources and48 disjoint original/edit pairs, each in both formats. No model
download, new neural forward, new photo or customer upload was involved. This
is a new calibration-development hypothesis, not a revised v8 decision or fresh
accuracy estimate. Model/source licenses and native checks remain those pinned
in [v8](image-patch-drift-v8-native-protocol.md).

Compute only the120 genuine levels first, then choose each format's second-largest
eligible level with strict `>` eligibility. Actual display/pixel replay confirmed
one of60 calibration originals highlighted per format. Publish both points before
development. The new raw inspection points were0.06192006170749664 PNG and
0.05865628272294998 JPEG. These values are inspection units, not percentages,
AI probabilities or guarantees about future false alarms.

## All development outcomes

| Format / component | Located edits /48 | Missed edits /48 | Genuine highlighted /48 | Mean edit IoU |
| --- | ---: | ---: | ---: | ---: |
| PNG current | 12 | 36 | 2 | 0.183951 |
| PNG region-calibrated candidate | 0 | 48 | 1 | 0 |
| JPEG75 current | 12 | 36 | 3 | 0.103440 |
| JPEG75 region-calibrated candidate | 0 | 48 | 1 | 0 |

The candidate missed every edit in each of the three generator families. Mean
genuine highlighted area was0.042815% PNG and0.027760% JPEG, averaged over all48
genuine development images. Located means any selected-pixel overlap with the
known edit; IoU separately penalizes poor coverage and wrong pixels. A genuine
highlight is a regional false alarm, not an AI-file verdict. Empty regions never
become an original-file conclusion.

Both formats passed3/6 requirements: full completion, separate implementation
checks and the genuine-highlight budget. Both failed the fixed minimum four
additional located edits, nondecreasing mean IoU and no family overlap loss.
The baseline is the current selected PNG/JPEG component, not the failed v8 arm.
No point, area, layer, arm or region union was selected after seeing results.
The [complete review](evaluation-panels/image-region-calibration-v9-review.json)
retains every family/format measurement and requirement.

## Independent implementation and resource checks

All120 calibration and192 development outcomes completed without errors or
resource guards. Every new level has a separate8-neighbour graph certificate:
a sufficiently large component exists at its plateau, and none exists strictly
above it. This proves the maximum feasible plateau on that retained response
grid. All312 original source/input/archive bindings, plateau certificates and
display/source-pixel calculations were independently replayed. Both genuine-only
points and all120 actual calibration displays reproduced exactly. The original
neural/native prerequisites were not relabelled as new inference.

The34 selected model-free tests include exhaustive small graphs, plateaus,
diagonal connections, exclusions, isolated spikes and exact agreement with the
actual display immediately below/at/above its boundary. Separate processes
used the same6GiB/1.5GiB/30-minute guards. Calibration took6.587s with sampled
RSS lower bound142,966,784 bytes; point fitting3.157s/198,590,464 bytes;
development14.946s/218,103,808 bytes; full review9.966s/203,620,352 bytes.
All had zero observation errors. These are sampled lower bounds, not true peaks.
New records retain hashes/certificates/counts; large prior response grids remain
read-only rather than being duplicated.

## Public reproduction and remaining diagnosis

The [public packet](evaluation-panels/image-region-calibration-v9-public-scalars.json)
contains all312 new scalar/certificate outcomes and original response commitments,
but no source media, raw masks, full response grids or real-image embeddings.
The source data retains its separate terms; a code license grants no photo rights.

```sh
PYTHONPATH=backend python -S -m scripts.reproduce_image_regions_v9 review /tmp/regions-review.json
PYTHONPATH=backend python -S -m scripts.reproduce_image_regions_v9 tamper /tmp/regions-tamper.json
```

Outputs refuse overwrite. Standard-library review refits both points, checks312
source/certificate bindings and120 genuine display audits, recomputes384
development metric sets, all family aggregates and12 advancement requirements.
All14 deliberate changes are refused. This portable path checks scalar and
certificate consistency; the private-grid/source reconstruction above ran
separately. An actual completed Linux receipt is required before claiming
separate-host agreement. Outside human reviewers remain zero.

This rules out the pixel-maximum/display-area mismatch as a sufficient fix in
the tested configuration. It does not isolate the remaining cause. The next
useful diagnosis is whether raw maps rank edited locations above background,
and whether those same locations already rank highly in genuine originals.
That paired control can distinguish some semantic/texture effects from an
edit-specific response. Such a diagnostic needs its own declared method, cannot
rescue these gates, and cannot be called fresh validation. Difficult audio,
lip-sync/general video, writing and outside human testing remain open work;
GCP stays deferred.
