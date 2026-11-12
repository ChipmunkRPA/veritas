# Earlier-copy comparison after a crop and resize

The local earlier-copy comparison now supports a corroborated crop and resize for submitted color JPEGs with the tested quality-75 quantization tables and 4:2:0 chroma sampling. The fallback supports earlier images up to 768 pixels on the longest side, submitted images up to 845 pixels on the longest side, and crop resize ratios from 0.72× to 1.17× after rounding. It shows the earlier native area used, the two source crops, and the submitted-grid difference overlay. Pixels outside the earlier area are unassessed. The comparison measures differences between the supplied files and carries no AI confidence weight. [Frozen study and readback](evaluation-panels/image-reference-crop-resize-transfer-2026-11-11.json)

## Frozen method

Use mutual scale-invariant keypoint matches for a provisional axis-aligned transformation. Search within two native pixels of each proposed crop edge. Compare nine distributed tiles and require both tile groups to rank the same integer crop first. The existing proportional-resize comparison checks that crop and retains supported differences. If alignment passes, its recorded fractional residuals provide a final check. The reference crop for each reported region maps submitted pixel edges with rational scaling and adds the earlier crop's native offset. This uses existing OpenCV and Pillow code paths with no model weights. [Scale-invariant matching](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) · [Recent research on reference-based evidence](https://arxiv.org/abs/2611.10908)

Freeze the candidate and role assignments before measuring eight SD1.5 development photos and two 12-photo holdouts. The held panels have distinct source and credited creator groups and each has six SDXL and six Kandinsky sources. The photos had previous project exposure. The sharp and feathered masks measure exact changes on the submitted grid before encoding. Nuisance controls include untouched crops, earlier JPEG processing and ordinary global contrast. Local blur supplies a known ordinary local edit. The incompatible controls use rotation, a half-native-pixel shift, nonproportional resizing, or an unrelated image. [Development sources](image-bg-real-insertions-2026-11-11.md) · [First held panel](image-background-patch-transfer-2026-11-11.md) · [Second held panel](image-wavelet-region-transfer-2026-11-11.md)

A useful region reaches 30% intersection over union and 50% recall. Each held panel and delivery needed at least four new useful feathered regions, six exact untouched crops, at most four nuisance highlights across at most three sources, and no incorrect or incompatible accepted crop. The existing comparison found zero useful sharp or feathered crop-and-resize regions on these deliveries.

## Held outcomes

| Panel and delivery | Useful sharp /12 | Useful feathered /12 | Exact untouched crops /12 | Nuisance highlights /36 | Incorrect or incompatible crops |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier PNG | 5 | 5 | 7 | 1 | 1 |
| Later PNG | 10 | 10 | 11 | 2 | 0 |
| Earlier JPEG75 | 4 | 4 | 7 | 3 | 0 |
| Later JPEG75 | 10 | 10 | 11 | 3 | 0 |

All six JPEG75 nuisance highlights followed an earlier lossy encode. The JPEG75 branch also highlighted 11 of 24 ordinary local blur views, which contain a real difference. No untouched crop highlighted in either format. One fractional PNG produced a wrong crop despite a 0.997 tile correlation and would have highlighted 11,632 pixels. The PNG branch stays experimental.

A separate local readback rebuilt all 480 held deliveries, verified 199 native masks, checked 122 earlier region mappings with independent rational arithmetic, and reproduced the one incompatible PNG acceptance. Local API tests exercise the JPEG fallback, exact export and PDF crops, altered coordinates, and exclusion of PNG and JPEG quality 90 from the new branch.

An isolated local run used the real Quick analysis worker with simulated Stripe transport. Checkout and an unpaid event left the analysis queue empty. The worker started after the verified $9.99 USD payment. The comparison required authentication and a completed analysis. The PDF and independently reviewed JSON reproduced the exact earlier area at x 24, y 14, 328 × 227 pixels.

## Replay

Run from the repository root with the three retained qualified intake folders and a new output path.

    export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    for stage in freeze development confirm; do
      backend/.venv/bin/python -B -m scripts.image_reference_crop_resize_transfer "$stage" \
        "$development" "$earlier" "$later" "$output" || exit 1
    done
    backend/.venv/bin/python -B -m scripts.review_image_reference_crop_resize_transfer \
      "$earlier" "$later" "$output" "$output/audit.json"
