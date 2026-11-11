# Local JPEG grid differences in PNG images

Chipmunk Verify checks supported PNGs for local differences in earlier JPEG block alignment. A qualifying region appears with an original crop and a source-aligned overlay in the analysis, PDF and JSON evidence. Ordinary cropping, a separate save or compositing can leave the same trace. This check contributes zero weight to the AI confidence score.

The check uses no model weights. It measures the sparsity of twelve spatial-frequency coefficients for each of the 64 possible JPEG block alignments. It compares each 64-pixel native window with the dominant image alignment and checks for stronger pixel steps along the alternate block boundaries. At least two edge-connected 32-pixel cells must support the same alternate alignment. The app measures single-frame RGB PNGs at least 256 pixels on each side and up to 1,048,576 pixels total. When the earlier grid is too weak to measure, the check records that limitation.

An eight-source development set supplied 120 control views. Both candidate settings were fixed before running two 12-source confirmation panels. The panels used 24 disjoint source photos previously qualified for local evaluation. Each photo contributed five controls and two edits, rendered as PNG, JPEG 90 and JPEG 75, for 504 confirmation views. The edits placed a sharp or feathered patch with a shifted prior JPEG grid. The controls covered the original, an image-wide grid shift, ordinary local contrast or blur, and a generated patch on the dominant grid. A useful location required at least 30% intersection over union and 50% recall against the changed pixels. The control count measures flags for the specific compression cue. It is not a false-alarm estimate for AI authorship.

| Delivery | Check | Sharp locations | Feathered locations | Control flags |
| --- | --- | ---: | ---: | ---: |
| PNG | Coefficients alone | 6/24 | 7/24 | 1/120 |
| PNG | Coefficients and block boundaries | 4/24 | 6/24 | 0/120 |
| JPEG 90 | Coefficients alone | 0/24 | 0/24 | 1/120 |
| JPEG 90 | Coefficients and block boundaries | 0/24 | 0/24 | 0/120 |
| JPEG 90 | Existing aligned double-compression check | 23/24 | 23/24 | 60/120 |
| JPEG 75 | Coefficients alone | 0/24 | 0/24 | 0/120 |
| JPEG 75 | Coefficients and block boundaries | 0/24 | 0/24 | 0/120 |
| JPEG 75 | Existing aligned double-compression check | 4/24 | 4/24 | 27/120 |

The existing aligned double-compression comparison is inapplicable to PNG. The production check uses the corroborated PNG result. The earlier panel supplied eight of its ten useful locations. The later panel supplied two, both feathered. The test covers constructed shifts and a fixed prior JPEG quality pair, so it does not establish coverage of ordinary edits or arbitrary processing. A clean or skipped result leaves other edits unresolved.

The original experiment and the production-shared implementation reproduced identical development points and confirmation journals. The [aggregate receipt](evaluation-panels/image-local-grid-transfer-2026-11-11.json) includes counts by delivery, control role and panel, plus the frozen implementation and journal commitments. The [local runner](../backend/scripts/image_local_grid_transfer.py) accepts the existing development and confirmation directories.

Related primary work includes the [IPOL implementation of non-aligned JPEG grid detection](https://www.ipol.im/pub/art/2020/283/) and the [IPOL ZERO study on local grid origins](https://www.ipol.im/pub/art/2021/390/article.pdf).
