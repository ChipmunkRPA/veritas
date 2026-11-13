# Earlier-copy comparison of an integer-shifted JPEG

An image can be moved a few pixels on a canvas, leaving a border that has no earlier counterpart. The existing integer alignment can identify that movement but cannot reproduce the JPEG delivery at the unknown edge. This comparison uses the central part of the shared scene. It reports the two native comparison areas, exact block support and the local pixel differences that survived the existing compression checks. The result carries zero weight in AI confidence. [Existing integer alignment and its JPEG limits](image-reference-comparison-2026-11-11.md#integer-crops-and-translations) · [JPEG coding standard](https://www.itu.int/rec/T-REC-T.81/en)

## What the check covers

The earlier image must be a single-frame upright RGB8 PNG and the submitted file a baseline RGB JPEG with standard quality-75 tables and 4:2:0 sampling. Both images must have identical dimensions with each side between 256 and 768 pixels. Offsets must be whole-pixel translations with at least one nonzero axis and no more than 24 pixels on either axis. The comparison first uses the existing corroborated integer alignment. When that is unresolved, a rounded phase-correlation estimate nominates one candidate, which still has to pass all exact support checks. Prior work motivates excluding unknown areas during registration. This implementation uses no research code or weights. [Exact JPEG proof and supported carriers](image-reference-exact-jpeg-2026-11-12.md) · [Research on excluding unsupported pixels from registration](https://iacl.ece.jhu.edu/proceedings/cvpr2010/papers/1815.pdf)

The shared region is trimmed to complete 16 × 16 JPEG-grid cells at least 16 native pixels from its edges. The encoder is run twice on that geometry with contrasting values in the unknown margins. The two decoded simulations must agree byte-for-byte throughout the interior. At least 65% of its complete blocks must exactly match the submitted image, with 24 unique detailed blocks and support in six of nine grid zones. The difference check uses only that measured interior. The mask and original crops preserve both source coordinate systems. Margins outside the measured bounds are unassessed. Small local changes, ordinary retouching, resampling or a different compression history can change the result.

## Frozen local comparison

The two panels use 20 development photos and 12 separate confirmation photos from the earlier credited patch studies. Both sets had prior project exposure. Confirmation contains six Kandinsky and six Stable Diffusion 1.5 sources, distinct by photographer and original image from the development set. For each photo, use a fixed offset selected by its frozen roster position. Construct one sharp and one feathered edit from the source-derived masks. Encode the shifted originals with black, white and mirrored fills beyond the original scene. Separate controls use an intermediate encode, a fractional shift, a rotation, an unrelated image, or unsupported JPEG tables. Useful localization requires 30% intersection over union and 50% recall against the exact pre-encode edit, in addition to the correct signed offset. [Source qualification and licenses](image-three-class-patch-fresh-2026-11-12.md) · [Development source credits](image-three-class-patch-probe-2026-11-12.md)

| Confirmation case | Earlier integer check | Added check |
| --- | ---: | ---: |
| Sharp edits usefully located | 0/12 | 12/12 |
| Feathered edits usefully located | 0/12 | 12/12 |
| Unchanged interiors correctly aligned | 0/36 | 36/36 |
| Highlights in those unchanged interiors | 0/36 | 0/36 |
| Recompression, incompatible and unsupported views admitted | 0/60 | 0/60 |

The new comparison localized all 40 sharp and feathered development variants and assessed all 60 ordinary fills without highlights. All retained coordinates in both panels matched the authored offsets. In confirmation, the existing integer match nominated 35 of 60 accepted offsets. Phase correlation nominated the remaining 25. Each still passed the same exact JPEG proof. The test measures specific JPEG deliveries and authored offsets. Two variants from the same source are correlated, and the result gives no independent evidence about the earlier copy's authenticity or the origin of a difference. [Frozen comparison and exact replay](evaluation-panels/image-reference-shifted-jpeg-transfer-2026-11-13.json)

## Local use and replay

Set `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` and `VERITAS_IMAGE_REFERENCE_SHIFTED_JPEG=true` on the development server. Open a completed image analysis and use **Compare with an earlier copy**. The browser and supplemental PDF show the compared areas and the excluded margins. The JSON binds the offset, JPEG anchors, both original crop coordinates and full-size mask. This supplemental check uses a separate earlier upload. It leaves the saved analysis result intact.

Use the earlier and submitted source files to replay the exported JSON and verify both crops, the mask and the coordinates.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_oriented_resize_export \
  "$export" "$earlier_png" "$submitted_jpeg"
```

The native evaluation ran and replayed all 320 pairs from the retained prior research directories. The complete earlier-copy production path reproduced all 120 confirmation cases, including the exact mask commitments and matching coordinates for every retained comparison. The authenticated endpoint and report tests verified both source crops and the mask against the browser export, rejected a tampered offset, and confirmed that the fallback is unavailable when switched off. Raw sources and individual rows remain local. Supply the same inputs documented for the [earlier exact JPEG study](image-reference-exact-jpeg-2026-11-12.md) and a fresh scratch directory.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_shifted_jpeg_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --output "$run" --publication "$receipt" || exit 1
done
```
