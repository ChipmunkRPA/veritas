# Image crop localization v6: completed development comparison

6 November 2026. Both formats meet the predeclared development count tradeoff.
**Advance to a separately frozen fresh comparison; keep the service unchanged.**
These 24 source pairs have already been used for development and cannot establish
independent accuracy. No outside reviewer participated.

The method and source review were published as `2184dd9` before extraction.
The [immutable protocol](fused-crop-v6-protocol.md) and
[prospective freeze](evaluation-panels/fused-crop-v6-development-freeze.json)
bind all thirteen implementation/protocol files, exact consumed selection and
three existing commercially permitted checkpoints. No new weights were acquired.

## Complete results

All 96 planned source/format outcomes completed: 96 assessed, zero failures,
zero out-of-scope inputs. Each ran four crops, giving 384 crop forwards. The
current native/JPEG components were freshly reproduced on the same inputs.
Every source byte, crop RGB commitment, native map and displayed baseline metric
was reconstructed before gate selection. Actual crop scores and map commitments,
all case outcomes and per-generator summaries appear in the
[public result](evaluation-panels/fused-crop-v6-development-review.json).
Raw images, masks, weights and model-map arrays are not redistributed.

| Format, 24 genuine + 24 edited images | Current genuine highlights | Crop-method genuine highlights | Current overlapping edits | Crop-method overlapping edits | Mean edited IoU, current → crop method |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native PNG | 2 | 3 | 10 | 12 | 0.20651 → 0.23965 |
| JPEG75 | 1 | 2 | 3 | 10 | 0.03674 → 0.10435 |

The native method still misses 12/24 edits; the JPEG method misses 14/24.
Misses contribute zero IoU. One additional genuine highlight per format meets
the declared count allowance, but is still a false review cue. These are
component-level regions, not whole-file AI verdicts or calibrated probabilities.

The exact genuine-only crop gates are `0.5474082231521608` for native and
`0.9998180270195008` for JPEG75. They are internal response selection rules, not
accuracy claims. The unusually high JPEG gate does not imply high scientific
confidence. Source-paired bootstrap intervals for mean IoU improvement are
[0.00000, 0.08291] and [0.02059, 0.12364], respectively. They summarize this
development panel and do not correct its reuse or model-selection effects.

Native Firefly/FLUX each recover one edit; native SD2 gains no edit and adds
the one false genuine highlight. JPEG gains four Firefly, one FLUX and two SD2
edits. Preserve every family, including the weak native SD2 result. Formats share
the same 24 source groups. Recorded durations include concurrent audio research
and do not establish controlled speed or memory capacity.

## Verification and next work

All 77 initial regression checks passed. Eleven additional reviewer checks pass
against unchanged frozen method files: changed inputs, labels, RGB commitments,
crop coordinates, arrays, metrics, code pins and incomplete/duplicated outcomes
are refused; failed extraction remains recorded and cannot select a gate.
One older production-grid test double had stopped accepting the existing
threshold keyword. It was repaired and now verifies disagreement detection for
both 0.5 and 0.7; production component code was not changed.

A source-only plan selects 48 new pairs, sixteen per generator, excluding all
240 earlier TGIF source IDs plus eight historical web examples. Its exact SHA-256
is `024d8837c9ec07601f3a41efcc91d90679d9c31d3c27f9c66bcfd49e97b2b3a5`.
[Public source plan](evaluation-panels/fused-crop-v6-source-plan.json).
Acquisition and source/duplicate checks are preparation only. Freeze the
confirmation runner, manifests, gates and rejection rules and publicly verify
that freeze before any fresh detector forward. No fresh outcome is claimed here.

The primary reference measures changed RGB pixels between each supplied original
and edit, with author masks retained separately. This is not independent expert
annotation of semantic edit boundaries. Unknown pretrained-source overlap and
cross-dataset generalization remain unresolved. Confirmation must improve edit
overlap/mean IoU at the predeclared genuine-highlight allowance before any local
integration. Overall confidence remains unchanged even if localization improves.
