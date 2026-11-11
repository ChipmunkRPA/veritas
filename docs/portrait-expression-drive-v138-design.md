# Stronger second-generator construction: full expression driving

Historical pre-execution design: implemented arithmetic/CPU bridge with authored
checks, **not yet qualified by actual neural execution at this design stage**.
See the later [v140 qualification method](portrait-expression-qualifier-v140-method.md)
for the subsequent contract; actual results must be reported separately.
It is not imported into the customer
application, included in confidence or used for benchmark generation. The
unchanged frozen v133 CPU aperture-response qualification remains separate.

A single mouth-opening ratio cannot represent lip rounding, protrusion or
horizontal motion. Replacing that scalar with a value measured 320 ms earlier
therefore does not establish phonetic retiming. Before calling a second
construction path a temporal articulation test, use richer motion information
and independently measure its achieved response. No transcript/phoneme or
audio-alignment claim follows merely from a changed picture.

The new research bridge reuses the same retained, checksum-bound MIT human
animation source/weights through the existing safe/strict loader. It adds no
weights, InsightFace code or identity recognizer. Run CPU float32 only. A current
source crop supplies appearance, canonical 21-point structure, rotation, scale,
translation and expression. Another crop from the **same recording/person**
supplies only its expression displacement. Driver identity, rotation, translation
and scale are not substituted.

If source keypoints are `K_s = scale_s * (canonical_s @ rotation_s + exp_s) + t_s`,
the driven keypoints are `K_d = K_s + scale_s * (exp_d - exp_s)`. A separate NumPy
reference verifies this tensor equation before rendering. Equal expressions must
leave keypoints unchanged. Different full expression vectors remain distinct
even when their vertical displacement/opening is equal. The latent points are
learned animation controls, not asserted anatomical landmarks. Model inference
may entangle expression and pose; the parameter equation alone does not prove
anatomical isolation or faithful speech content.

The expression path uses source appearance, two motion-extractor calls, warping
and decoding: five top-level module forwards per output, with no lip-ratio-head
forward. The existing loader still retains that unused head for the separately
tested aperture path. Real qualification must hook/count these calls, verify
repeatability and source/driver correspondence, compare against the scalar path,
and preserve the correct 256-to-512/native pixel-centre mapping and exact mouth
composite support. Cached source meshes may be reused only after their actual
qualification and byte identities are checked.

No pretrained forward was run by the authored arithmetic tests. Actual execution
requires a separately frozen qualification, successful v133 baseline, the same
900-second/6-GiB neural limits and sufficient initial RAM. Preserve failed stages;
no fallback, retry or source substitution. Only after actual response, achieved
motion and ordinary-control checks may this path enter the new-placement,
multi-generator transfer comparison with the existing detector heads/calibrations
fixed. Keep raw images, learned motion vectors and generated benchmark media
private. No detector/default, Actions, outreach, payment or deployment changes.
