# Regional patch drift: consumed-source development, frozen before project inference

The six authored synthetic native checks passed with zero normalization and
intermediate-layer/full-hook difference. Their successful receipt is bound into a
separate project-source freeze. This study is development, not fresh confirmation:
all selected photos were consumed in earlier experiments. No customer upload is used.

## Selection and roles

From the initial72 genuine TGIF originals, sort each of firefly,sd2,flux-fill by
SHA256 of `patch-drift-v8-calibration:` plus source ID and take20 per family.
These60 original sources alone set operating points. Development uses all48
original/edit pairs from the failed v6 crop confirmation,16 pairs per family.
Calibration and development source IDs must be disjoint. Both original PNG and
whole-image JPEG75 are used:120 calibration and192 development inputs,312 total.
JPEG differences are not relabelled as edits; original author masks remain fixed.
Pretraining overlap and source-image rights are not established by this split.
Research photos and real-image feature vectors are retained privately.

## Fixed method

Keep every setting from the separate native contract: Apache-2 DINOv2 register
base backbone,86.6M parameters,448-square bilinear RGB,0.2 Haar attenuation with
float32 perturbed pixels,ImageNet normalization,one-based block10 final-normalized
patch tokens,NumPy float32 cosine drift and reflect3-square median. No fine-tuning,
reference original at inference or edit mask enters the method. The two declared
readouts are raw median drift and within-image `(map-median)/IQR`, float32-epsilon
scale floor. No other layer, perturbation, normalization or map ensemble is searched.

Use the existing aspect-correct screening grid, bilinear map interpolation and
current reliability exclusions. For each readout/format, take each genuine
calibration image's maximum over reliable cells. Require at least48 of60 eligible
images, all120 forwards complete, and a positive operating point equal to the
second-largest eligible maximum. Eligibility is strictly greater than that point;
ties do not get highlighted. Thus at most one calibration original can exceed it.
No edited example selects a point. Publish the point receipt before development.

Encode `round(clip(score/(2*point),0,1)*255)`, clamping its high bit so bytes>=128
if and only if the original float32 cell is strictly above its point. This encoding
is a relative inspection index, not a calibrated probability or confidence score.
Keep8-connectivity, current minimum component area and ordering, and two-region
per-map cap. Project the actual selected component pixels with OpenCV nearest
neighbor, exactly as the v6 comparison. This is explicitly a component comparison;
full production packet admission and Pillow projection are separate adoption work.

## Comparator and native audit

Recompute all192 current-check baseline masks/metrics from their exact retained
native scalar arrays and matching source bytes: TIDAL PNG with its original
native-cell gate; full-image FUSED JPEG with its original image gate. Do not use
rejected crop maps. Require exact matching old receipts. This is a scalar-component
replay, not a new full-app neural baseline run. Pin array/prediction/source hashes.

Before seeing scores, select one development input per family and reference label
by minimum SHA256 of `patch-drift-v8-native:` plus item ID; audit both formats,
12 inputs total. They repeat a full native forward/hook and retain private patch
features. Require zero hook difference, exact byte commitments, separate float64
cosine versus original float32 response within5e-7, and exact independent median,
readout, display-mask and pixel-metric replay for every completed case. All312
inputs must complete without failures, and both stages must finish without a guard.
Memory guards:6GiB sampled RSS,1.5GiB available RAM/disk,30minutes per stage.
Sampled maxima are lower bounds; report observation failures. Retain failed runs.

## Advancement gates fixed before project scores

Each format/readout qualifies only if all of these hold on its96 development
inputs: all48 edits and48 genuine originals complete; at least4 more edits have
nonzero intersection with the known edit than the current check; mean edit IoU
is not lower; no family loses an overlapping edit; genuine highlighted images are
at most current+1; all source, synthetic, native and independent replay checks pass.
Report false-highlight area, missed edit count and family breakdown too. A region
overlapping one changed pixel counts as located but is not precise attribution;
IoU is therefore a separate gate. Failures or unavailable cells remain misses,
not original-file verdicts. There is no image-level AI decision in this experiment.

If both readouts qualify, choose raw first; otherwise choose the sole passing arm.
Formats may qualify separately. No post-result union or rescue. A selected arm
requires a separately frozen fresh-source confirmation before service integration.
No source from a failed candidate's unused confirmation reserve may be spent here.
Keep old checks, overall confidence, billing and customer reports unchanged during
development. GCP remains last. Public automated replay is not outside human review;
outside reviewers are currently zero and no outreach is authorized.
