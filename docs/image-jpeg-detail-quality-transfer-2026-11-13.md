# Fine-detail loss in matching JPEG75 and JPEG85 copies

The earlier-copy comparison can now measure local loss of fine luminance
detail in two standard JPEG compression profiles. It uses the existing
JPEG90 measurement and exact native block alignment. It compares only
upright, unprofiled, baseline RGB JPEGs of the same dimensions and the
same standard quality, with 4:2:0 sampling and both sides from 512 to
768 pixels. A finding describes local softening. It contributes zero
AI-confidence weight. [Existing JPEG90 measurement](image-jpeg-detail-loss-2026-11-13.md)
· [JPEG specification](https://www.itu.int/rec/T-REC-T.81/en)

## Matched delivery study

The 34 source photo groups came from the earlier JPEG90 confirmation bank.
Their rights were established in the retained photographer and publisher
intakes. The same sources across qualities permit a paired comparison
of format changes. The source photos had prior exposure, so these
results establish a delivery extension rather than fresh photo or
generator generalization. The current operating criteria were frozen
on the previous JPEG90 study. No tuning used these new deliveries.
[Source photo and editor intake](image-reference-same-jpeg-confirmation-2026-11-13.md)

Each photo supplies a weak, moderate and strong local blur at each
quality. Truth uses exact changed pixels before encoding. Each quality
also has 306 controls covering untouched copies, tandem JPEG saves,
whole-image blur, contrast, local sharpening, a different photo, a
shifted image, mismatched JPEG quality and a feathered generated edit.
Only pairs for which the current earlier-copy route produces no region
can use the additional check. An added region is useful if it covers
at least half the true changed pixels with at least 0.30 intersection
over union.

| Delivery | Added weak regions meeting overlap | Added moderate regions meeting overlap | Added strong regions meeting overlap | Moderate or strong source groups added | Added control highlights |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG75 | 7/34 | 11/34 | 1/34 | 11 | 0/306 |
| JPEG85 | 7/34 | 11/34 | 1/34 | 11 | 0/306 |

JPEG75 produced 40 added masks, with 19 meeting the overlap rule.
JPEG85 produced 37 added masks, with 19 meeting it. Some regions
cover too little of the known local operation for useful localization.
All previously selected regions stay available, including the
incumbent's highlights on some ordinary processing controls. This
comparison measures detail loss between supplied copies. None of the
68 generated edit controls gained a finding from this method.
[Evaluation ledger](evaluation-panels/image-jpeg-detail-quality-transfer-2026-11-13.json)

## Local app and replay

Enable `VERITAS_IMAGE_REFERENCE_JPEG_DETAIL_LOSS_EXTRA_QUALITIES=true`
alongside the existing earlier-copy and JPEG detail-loss options in a
local development instance. In **Compare with an earlier copy**, the
app shows the earlier and submitted crops, how many tiles directly
support the result, and the exact selected mask. The JSON proof records
the measured quality. The comparison PDF and JSON export preserve the
same crop and mask evidence. The earlier JPEG90 format remains readable.

Use the 34-source qualification paths described in the JPEG90 study.
The native runner rebuilds the original and submitted bytes and the
exact native truth. The standard-library reviewer independently verifies
the source matrix, geometric counts, routing and fixed decisions in the
public ledger.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_jpeg_detail_quality_transfer review \
  docs/evaluation-panels/image-jpeg-detail-quality-transfer-2026-11-13.json
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_export \
  "$export_json" "$earlier_jpeg" "$submitted_jpeg"
```
