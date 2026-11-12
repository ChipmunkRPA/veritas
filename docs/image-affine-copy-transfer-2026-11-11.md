# Source-pixel evidence for rotated and resized copy pairs

The dense comparison found useful regions on both sides of **10/24 sharp PNG copies**, up from 5/24 with sparse boxes, and **8/24 sharp JPEG75 copies**, up from 4/24. It also improved feathered copies in both deliveries. None of the 240 ordinary variants received a pair. [Audited counts and protocol](evaluation-panels/image-affine-copy-transfer-2026-11-11.json)

| Delivery | Copy condition | Sparse boxes | Dense comparison |
| --- | --- | ---: | ---: |
| PNG | Sharp | 5/24 | 10/24 |
| PNG | Feathered | 2/24 | 5/24 |
| JPEG75 | Sharp | 4/24 | 8/24 |
| JPEG75 | Feathered | 0/24 | 5/24 |

## Inputs and method

The two panels have disjoint credited photographers and photographs. All 24 photos had earlier project exposure. The new fixtures used two fixed rotation and scale combinations, including one downsize and one upsize. Each group supplied an original, global contrast, mild blur, a whole-image resize round trip, and a local tone adjustment. Every condition was encoded as PNG and JPEG75. An authored copy counts as useful when the areas overlap both known native masks at intersection over union of at least 0.30 and recall of at least 0.50. The controls reuse the same photos across treatments. They do not estimate performance on arbitrary repeated objects or real user uploads. [Earlier source qualification](image-background-patch-transfer-2026-11-11.md) · [Later source qualification](image-wavelet-region-transfer-2026-11-11.md)

The local comparison uses the current sparse feature proposals. For each of the first three distinct pairs, it brings Area B into Area A's grid and checks local luma correlation, texture, and photometric agreement. It retains a connected region with sufficient support near the original sparse box. A short morphological closing fills small gaps in that mask. The second mask is the corresponding projection into Area B. These are undirected comparison areas. The method cannot detect a copy that the sparse check fails to propose. A natural repetition can resemble a copy, and the result carries zero weight in AI authorship confidence.

The predeclared rule required at least three additional useful feathered pairs from three source groups per format, at most two lost useful baseline locations, and zero additional ordinary flags. Both deliveries passed. No previously useful pair was lost. The separate standard-library auditor recomputed all 336 outcomes and 136 pair-side overlap records using integer counts. The production implementation reproduced all 336 input bindings and all 34 dense pair masks exactly.

This experiment concerns geometric copying within one image. Finding a locally generated replacement without an earlier copy remains a separate weakness. The recent Inpainting Exchange study identifies why an image-level detector can score background artifacts even when the edited region is small. [Primary study](https://arxiv.org/html/2602.00192v2)

## Local evidence and reproduction

Enable `VERITAS_IMAGE_AFFINE_COPY_REVIEW=true` in development for images with a long edge of 513–1,024 pixels and a short edge of at least 384 pixels. New analyses record `image-evidence-13`. The evidence view, PDF and JSON use the same two native crops and corresponding masks. A local worker smoke with simulated Stripe confirmed that payment preceded analysis. One selected feathered positive was localized, its matched original reported no pair, and all four plates matched the PDF and signed source replay.

From the repository root, provide the two retained intake paths. The resulting case journals contain source-specific commitments and can remain private. The public aggregate carries counts and protocol commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_affine_copy_transfer freeze "$earlier" "$later" "$run"
backend/.venv/bin/python -B -m scripts.image_affine_copy_transfer run "$earlier" "$later" "$run"
backend/.venv/bin/python -B -m scripts.review_image_affine_copy_transfer "$run"
```
