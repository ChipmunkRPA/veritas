# Structural contour continuity on small generated image edits

The new check recovered the exact region on four authored pattern tests, including feathered and JPEG75 deliveries. It localized zero of 96 sharp or feathered generated photo deliveries. The two-distance arm highlighted 4 of 240 ordinary PNG views across two source groups and 3 of 240 ordinary JPEG75 views from one source group. The candidate remains experimental. [Audited results and frozen settings](evaluation-panels/image-contour-continuation-transfer-2026-11-11.json)

## Method

Recent research describes localized edits that preserve the surrounding photo and weaken low-level forensic cues. Boundary-aware work studies edge artifacts at multiple scales under JPEG compression. These papers motivated testing a separate structural cue with no pretrained weights. The implementation uses NumPy and OpenCV and is released under Apache-2.0. [Semantic Manipulation Localization](https://arxiv.org/abs/2604.10132) · [FUSED study](https://arxiv.org/html/2608.28302v1) · [Boundary-aware study](https://doi.org/10.1016/j.knosys.2026.116750)

Measure native luminance gradients along the tangent of each candidate rectangle side. Count a contour when the same signed edge persists at two locations outside the rectangle, allowing five fixed slopes. Track whether it continues at two projected locations inside the rectangle. Compare the loss with parallel crossings farther outside that side. The score requires three sides to agree. The second arm requires the same sides to show the effect at both distances. This measures structure that stops or fades, which can arise in ordinary scenes or conventional pastes.

Set the strict points using the second-highest of eight development source maxima or 0.10, whichever is greater. Development controls include the unchanged image, image-wide processing, local processing, blur, contrast, sharpening, and four traditional pastes. The method, operating points and success criteria were fixed before the holdout run. The 24 held photos have disjoint source and recorded creator groups from development and split equally between SDXL and Kandinsky. These retained panels had previous project evaluation exposure. Each supplies two generated edits and ten ordinary controls per format. The added controls include sharp and feathered conventional pastes from the same photo or another qualified genuine photo. Both deliveries use the same native changed-pixel masks recorded before compression. [Development intake](image-bg-real-insertions-2026-11-11.md) · [Earlier qualification](image-background-patch-transfer-2026-11-11.md) · [Later qualification](image-wavelet-region-transfer-2026-11-11.md)

A useful region requires at least 30% intersection over union and 50% recall of changed pixels. The PNG reference is the existing sharp-boundary check. The JPEG reference is the existing aligned double-quantization check. The two-distance arm needed at least two additional feathered locations per panel and delivery, with at most four extra ordinary highlights across at most three source groups.

## Held results

| Format | Check | Useful sharp /24 | Useful feathered /24 | Ordinary highlights /240 | Photo groups highlighted /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Format reference | 12 | 0 | 51 | 19 |
| PNG | Short crossing | 0 | 0 | 68 | 11 |
| PNG | Two distances | 0 | 0 | 4 | 2 |
| JPEG75 | Format reference | 0 | 0 | 0 | 0 |
| JPEG75 | Short crossing | 0 | 0 | 65 | 11 |
| JPEG75 | Two distances | 0 | 0 | 3 | 1 |

All seven two-distance highlights were conventional pastes. The two-distance arm produced a measurable proposal on all 96 positive views, but only two highest-ranked proposals had useful geometry before applying the cutoff. Relaxing the cutoff would still miss most regions. The existing PNG reference found nine useful sharp Kandinsky insertions and three sharp SDXL insertions. Each result applies to this controlled 24-photo panel and its two delivery formats.

The held CPU run took 72.4 seconds. A separate readback rebuilt and checked all 576 deliveries, verified 1,728 native masks and overlap counts, and checked 192 unthresholded region overlaps. Four unit tests verify authored geometry, abstention with insufficient contour orientations, and sample direction.

## Replay

Supply the retained qualified development and confirmation intakes, the earlier blended-boundary rows journal, and a fresh output path. The local journals retain per-view geometry and hashes. The public aggregate contains counts and commitments.

    export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    for stage in freeze fit confirm; do
      backend/.venv/bin/python -B -m scripts.image_contour_continuation_transfer "$stage" \
        "$development" "$earlier" "$later" "$boundary_rows" "$output" || exit 1
    done
    backend/.venv/bin/python -B -m scripts.review_image_contour_continuation_transfer \
      "$earlier" "$later" "$output" "$output/audit.json"
