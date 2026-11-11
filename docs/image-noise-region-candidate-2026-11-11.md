# Noise-guided localization on sharp and blended image edits

The new model located two of 12 feathered PNG edits and one of 12 feathered JPEG75 edits at the publisher’s 0.5 map setting. It also highlighted all 72 PNG controls and 58 of 72 JPEG75 controls. The points fitted on separate controls found zero feathered edits. The existing regional checks remain in standard analyses. [Audited aggregate](evaluation-panels/image-noise-region-candidate-2026-11-11.json)

## Method and licensing

The checkpoint combines EfficientNet features with a learned noise residual. The publisher labels the source and released checkpoint Apache-2.0. Its model card names CASIA2 training. The individual training-image rights and overlap with these source photos were unverified. The comparison ran offline on the Mac CPU using the pinned checkpoint, ImageNet normalization and the full aspect-preserved view capped at 448 pixels on its shorter side. The runner checks the source archive, source files and checkpoint digests and loads weights only. [Pinned source and license](https://github.com/juhenes/ngiml-infer/tree/9cf1535e050e97490e11634ae2faa6ae55c8611d) · [Pinned checkpoint and model card](https://huggingface.co/juhenes/ngiml/tree/027d6f86b7761eafd0e237cc52a8e1070f05339e) · [Local runner](../backend/scripts/image_noise_region_candidate.py)

Eight development photos supplied 64 control views across PNG and JPEG75. The first point with a highlight on at most one source group was sealed per format before the paired comparison. PNG needed a point above the map’s maximum possible value, effectively disabling its output. JPEG75 selected 0.95. These points measure map selection in this study.

The paired evaluation made 192 model forwards on twelve different source photos, six with SDXL and six with Kandinsky patches. Each photo supplied both a sharp and a feathered edit, plus six controls in each format. The photos were used for the earlier boundary experiment. Their encoded views and native edited-pixel counts matched the existing comparison exactly. Both reference routes and the boundary method were read from their sealed results. An independent readback matched all input digests and recomputed all 768 overlap and recall measurements from their pixel counts. [Source photos and controls](image-boundary-blend-transfer-2026-11-11.md)

| Delivery | Check | Sharp edits localized /12 | Feathered edits localized /12 | Controls highlighted /72 | Originals highlighted /12 |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Current regional route | 0 | 0 | 11 | 2 |
| PNG | Noise-guided, publisher point | 2 | 2 | 72 | 12 |
| PNG | Noise-guided, control-fit point | 0 | 0 | 0 | 0 |
| JPEG75 | Current regional route | 1 | 0 | 7 | 1 |
| JPEG75 | Noise-guided, publisher point | 1 | 1 | 58 | 10 |
| JPEG75 | Noise-guided, control-fit point | 1 | 0 | 13 | 2 |

A useful region covers at least half the edited pixels and has at least 0.30 intersection-over-union with the true edit. A control highlight means any selected region. Formats and controls reuse the same source groups. The stricter JPEG point preserved one useful sharp result while almost doubling control highlights compared with the current route. Smoother local edits remain the priority for a different candidate on additional source photos.
