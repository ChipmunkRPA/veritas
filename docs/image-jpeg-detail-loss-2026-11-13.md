# Local loss of fine detail in matching JPEG copies

The earlier JPEG study found more ordinary blur after adding a coarse
measurement of frequency differences. This follow-up measures their
direction. It checks whether fine texture became weaker between two
matching JPEGs while broader image structure stayed similar. Use of
separate frequency bands follows the general forensic motivation in
FASA. This implementation uses the delivered JPEG coefficients and no
learned model, paper code or checkpoint. [Earlier JPEG comparison](image-jpeg-ac-difference-2026-11-13.md) ·
[FASA methodology](https://arxiv.org/html/2604.12341) ·
[JPEG specification](https://www.itu.int/rec/T-REC-T.81/en)

Both images must be upright, baseline 4:2:0 JPEG90 with the same pixel
dimensions, 512 to 768 pixels on each side. The existing framing check
requires at least 85% of complete 16 × 16 decoded RGB tiles to match and
requires distinctive matches across the image. Within each complete tile,
the detail comparison reads and dequantizes the original luminance
coefficients. It compares four aligned JPEG blocks in two bands. It
requires useful texture in the earlier tile, agreement in the broader
coefficients and stronger attenuation in the finer band. A bounded
closing operation connects small gaps in supporting tiles. The report
distinguishes the number of directly measured tiles from the full
highlight. The finding carries zero AI-confidence weight.

## Local evaluation

The operating point was chosen on 14 separate development photographs.
The unchanged candidate then ran on 544 pairs from 34 held photographs,
disjoint from development by recorded original, photographer and visual
identity. The photographs had already appeared in previous project
studies. The new paired deliveries include the publisher footprint with
weak, moderate and strong blur, plus local sharpening, strong global
blur, global sharpening, resize, repeated JPEG encoding, shifts and
wrong references. The source photos carry individual CC BY 2.0
attribution and the publisher's contributions carry CC BY 4.0.
[Held source qualification](image-reference-same-jpeg-confirmation-2026-11-13.md) ·
[Publisher rights and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Useful localization requires 50% coverage of exact pre-encode changed
pixels and at least 0.30 intersection over union. The comparison adds
fine-detail regions only when the current route selected none, preserving
any earlier highlight. The frozen decision required useful moderate or
strong regions from at least eight additional sources, at most two
other control flags, and zero whole-image or incompatible flags.

| Paired condition | Current useful locations | With detail-loss check |
| --- | ---: | ---: |
| Weak localized blur | 0/34 | 8/34 |
| Moderate localized blur | 1/34 | 14/34 |
| Strong localized blur | 10/34 | 11/34 |

The addition found useful moderate or strong regions on 13 different
photographs. Across the three strengths it returned 38 new highlights,
22 useful under the overlap rule. The other 16 fell below the overlap
or coverage requirement. The detail check reported zero findings on
all 442 other deliveries, including 68 known generated edits. The new
result supplies evidence of local softening. Tests of standalone AI image
detection and broader JPEG carriers remain separate.

Two complete native scores reproduced the case masks and measurements.
The app comparison reproduced all 38 fallback masks and preserved
incumbent highlights on candidate-positive pairs. The authenticated
comparison returns native crops, an overlay, a PDF and a replayable
JSON export. Both source file hashes are bound before comparison.

## Local use and replay

On the local backend, enable `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true`
and `VERITAS_IMAGE_REFERENCE_JPEG_DETAIL_LOSS=true`. Sign in and open a
completed image analysis, then choose **Compare with an earlier copy**.
The comparison uses supported JPEG pairs and displays **Fine detail loss**
when measured. A confirmed change is evidence about differences from the
supplied copy.

Use the seven retained source rosters from the [paired confirmation](image-reference-same-jpeg-confirmation-2026-11-13.md)
with the `freeze`, `score` and `replay` operations. The public scalar
ledger independently recounts the outcome and checks case commitments
and invariants without third-party Python dependencies or source images.
The local export verifier also recomputes and checks the crop pixels,
overlay, source hashes and evidence coordinates.
[Scalar ledger](evaluation-panels/image-jpeg-detail-loss-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_jpeg_detail_loss_confirmation review \
  docs/evaluation-panels/image-jpeg-detail-loss-2026-11-13.json
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_export \
  "$export" "$earlier" "$submitted"
```
