# Image comparison: a whole-generation gain, no local-edit recovery

Completed 7 November 2026. The candidate supplied additional whole-image AI indications without additional genuine-image flags on this consumed development panel. It did not recover a single localized edit. The original mixed-image advancement rule passed **9 of 10 requirements**, so this study does **not** authorize a default-model replacement, fresh-confirmation advancement under its original protocol, or a new overall-confidence contribution.

This is a promising result for one task and a failed broad advancement decision. It would be inaccurate to describe it as either a proven production upgrade or evidence that the model is uniformly worse. The commercial weight/source review remains valid; limited detection scope, rather than a licensing or native-runtime failure, prevents adoption here.

## Comparison and provenance

The [protocol](image-smaug-v7-development-protocol.md), [source-bound input panel](evaluation-panels/image-v7-smaug-development-panel.json) and [full freeze](evaluation-panels/image-v7-smaug-development-freeze.json) were published at `664c0ccd08516893dc81b78a283e3741a49ba14b` before project-model scoring. The genuine-only [operating point](evaluation-panels/image-v7-smaug-genuine-operating-point.json) was published at `0783ceb30b124f94fb044677f5261b096decbd5f` before development scoring. No point was relaxed after seeing errors.

The baseline was the actual current full image pipeline, including its commercial local configuration, 17 enabled checks, current calibrated fuser, current policy and current customer interpretation. All 224 analyses completed through publication. The 48 calibration inputs were 24 untouched source groups in native/JPEG forms; the remaining 176 development inputs comprised 44 genuine and 44 AI-containing sources per format. The source pool was already consumed by earlier studies; pretrained-model training overlap is unknown. Native/JPEG copies are paired observations, not independent extra sources.

## Results

The supplement preserves the baseline and adds only positive candidate cues. A low candidate response never supplies an original-file vote. These are counts of statistical AI indications, not newly calibrated overall conclusions.

| Development subset | Native baseline → supplement | JPEG75 baseline → supplement |
|---|---:|---:|
| Fully generated images, 20 sources | 14 → 20 | 18 → 20 |
| Localized edits, 24 sources | 1 → 1 | 0 → 0 |
| All AI-containing sources, 44 | 15 → 21 | 18 → 20 |
| Genuine sources flagged, 44 | 0 → 0 | 0 → 0 |
| AI sources without a statistical AI indication, 44 | 29 → 23 | 26 → 24 |

The candidate alone flagged 20/20 whole generations, 0/24 localized edits and 0/44 genuine sources in each format. It added six native and two JPEG whole-generation indications, spanning two generation families in each format. Native retained one statistical AI indication on an edited source already present in the baseline; the candidate did not create it. There were no candidate-positive conflicts with an existing original tendency.

The failed requirement was the fixed minimum of four additional AI sources **in each format**: JPEG added two. Both formats passed the genuine false-alarm allowance, and every completion/native requirement passed. Neither format met the separately declared local-edit usefulness requirement. We retain the adverse decision and do not redefine a whole-generation success as localized-edit support.

Seven native edited sources were already provenance-led; these are reported separately from statistical detection, not recast as model recoveries. On the 24 edited sources, the current customer interpretation was original for 10 native and 16 JPEG copies, inconclusive for six native and eight JPEG copies, and AI for one native copy; the remaining seven native copies were provenance-led. The candidate leaves this local-edit gap unresolved.

Across all 88 development inputs per format, the baseline had 16 inconclusive and four unavailable native outcomes; JPEG had 12 inconclusive and one unavailable outcome. These remain separate from failed analyses, of which there were none. Statistical indication counts do not replace saved customer conclusions or provenance records.

## Uncertainty and limits

The paired component bootstrap gives a native AI-indication gain of 13.6 percentage points, with a 95% interval of 4.5–25.0 points; JPEG gives 4.5 points, interval 0.0–11.4 points. These are consumed-panel comparisons, not prospective service accuracy. The all-zero observed false-alarm differences produce a degenerate bootstrap interval; that is not evidence of a zero future false-alarm rate. With zero flags among 44 independent genuine sources, a two-sided 95% exact binomial interval still has an upper bound of approximately 8.0%. All 20 whole-generation successes likewise have a lower exact bound of approximately 83.2%, not a guarantee of perfect detection.

The model produces a whole-image response from three crop views. Their coordinates identify the pixels inspected; they are **not** edit masks or proof that a crop was manipulated. No per-file confidence calibration, new public threshold, customer report/API/PDF change, Stripe transaction, deployment, or GCP resource was introduced.

## Verification and public replay

All 224 retained stage/result packet hashes, all canonical result hashes, all input bindings, all frozen customer interpretations and every exact 17-check set were checked. All 12 pipeline stages produced outputs on every input. The 3,808 check outcomes were 3,015 `ok` and 793 `not_applicable`, with no failed/skipped/timed-out checks. The [integrity receipt](evaluation-panels/image-v7-baseline-completion-integrity.json) records this separately from inference replay. Intermediate image exhibits were removed only after successful verification under the prospective storage plan; their digest/size manifests remain. This was not a retained PDF/image-asset or paid customer-flow replay.

All 26 predeclared family/label/format/role native probes passed source/crop/coordinate correspondence and single-crop batching. Maximum raw-logit difference was 9.5367431640625e-7 and maximum sigmoid difference was 2.0372681319713593e-10, inside the unchanged 1e-4/1e-5 tolerances. [Native receipt](evaluation-panels/image-v7-native-replay.json). Every original model run exited zero with no resource guard. The baseline memory sampler had one observation error; memory maxima are sampled lower bounds. Baseline git metadata contains two checkout HEADs as research-only files were committed during the run; all 422 frozen scoring-source hashes, configuration and assets remained unchanged.

The public [scalar response packet](evaluation-panels/image-v7-cue-responses.json) includes all 224 planned IDs, source/label/group/role bindings, three crop logits, raw responses, baseline directions and expected aggregates. It includes no raw images, face embeddings, customer files or model weights. The [release manifest](evaluation-panels/image-v7-cue-release.json) pins its source and bytes.

From this checkout, with standard Python only:

```sh
PYTHONPATH=backend python -m scripts.reproduce_image_cues_v7 review \
  docs/evaluation-panels/image-v7-cue-responses.json /tmp/new-image-review.json \
  --panel docs/evaluation-panels/image-v7-smaug-development-panel.json
PYTHONPATH=backend python -m scripts.reproduce_image_cues_v7 tamper \
  docs/evaluation-panels/image-v7-cue-responses.json /tmp/new-image-tamper.json \
  --panel docs/evaluation-panels/image-v7-smaug-development-panel.json
```

The separate implementation reproduced all 418 count fields across format/family/scope tables, all ten advancement requirements, source bindings and 224 outer reductions. All 14 deliberate changes were rejected. An optional separate NumPy implementation reproduced all four declared component-bootstrap intervals exactly. The standard-library and interval reviews do not repeat neural inference. The Linux workflow runs the scalar/tamper review separately; outside human review remains zero until an actual person participates.

## Next development decision

Keep the current serving checks and preserve the candidate as a research asset with a demonstrated whole-generation strength on this panel. Do not spend a reserved confirmation panel under the failed original gate or silently retune it. The next image method must address regional editing explicitly and establish how regional evidence can support a calibrated file assessment alongside whole-generation checks. It must preserve genuine-source controls, comparison denominators and a separately frozen fresh confirmation before adoption. Improving only the headline or converting misses into abstentions would not establish better detection. Audio in strong noise, face-swap/lip-sync localization, writing and actual outside collaboration remain unfinished parts of the improvement program; GCP stays last.
