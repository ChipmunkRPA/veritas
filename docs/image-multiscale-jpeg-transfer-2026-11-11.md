# Local recompression curves after JPEG delivery

At JPEG 90, requiring a second compression cue reduced the aligned double-compression check's control flags from 60 to 19 of 120. Useful edit locations fell from 46 to 30 of 48. At JPEG 75, the combination removed all eight useful locations. These results support further study of the precision and recall trade-off. Serving continues to use the existing JPEG check. [Audited counts](evaluation-panels/image-multiscale-jpeg-transfer-2026-11-11.json)

The experiment measured the average RGB change on re-saving the original JPEG at seven qualities from 45 to 95. Each native 32-pixel cell supplied a curve of these reconstruction errors. The multiple-quality cue removed the mean log error from each curve before comparing a 3 × 3-cell neighbourhood with its surrounding ring. A single-quality arm made the same spatial comparison at quality 90 without curve normalisation. The combined arm retained the existing aligned double-compression box only when the median multiple-quality response among its covered cells exceeded a fixed review point. Each fitted point used the second-highest of eight disjoint development source maxima, with a fixed minimum. The two development JPEG formats supplied 80 control views. No inference weights were used. [Related image-level recompression study](https://arxiv.org/html/2607.06615v1) · [Aligned and non-aligned block-grain method](https://flore.unifi.it/handle/2158/628298)

The two 12-source confirmation panels used the exact JPEG75 and JPEG90 files from the earlier prior-grid evaluation. These JPEGs were new to the recompression method. Each format has 24 sharp shifts, 24 feathered shifts, and 120 controls including original, whole-image shift, ordinary contrast, blur and an aligned generated patch. A useful location required at least 0.30 intersection over union and 0.50 recall against the changed pixels before final compression. Controls measure flags for the constructed compression-history cue. They include ordinary edits and are not a general false-alarm population for AI detection. [Source qualification and prior-grid controls](image-local-grid-transfer-2026-11-11.md)

| Final delivery | Check | Sharp locations | Feathered locations | Control flags |
| --- | --- | ---: | ---: | ---: |
| JPEG 90 | Single-quality response | 0/24 | 0/24 | 26/120 |
| JPEG 90 | Multiple-quality response | 1/24 | 1/24 | 33/120 |
| JPEG 90 | Existing aligned check with response corroboration | 16/24 | 14/24 | 19/120 |
| JPEG 90 | Existing aligned check | 23/24 | 23/24 | 60/120 |
| JPEG 75 | Single-quality response | 0/24 | 0/24 | 35/120 |
| JPEG 75 | Multiple-quality response | 0/24 | 0/24 | 50/120 |
| JPEG 75 | Existing aligned check with response corroboration | 0/24 | 0/24 | 6/120 |
| JPEG 75 | Existing aligned check | 4/24 | 4/24 | 27/120 |

The standalone multiple-quality map contributed two useful JPEG 90 locations, with more control flags than its single-quality comparison. Requiring corroboration reduced JPEG 90 flags in both confirmation panels and missed useful locations in both. The separate audit matched all 336 encoded inputs, changed-pixel commitments, and baseline measurements to the prior sealed journals. It also recomputed intersection over union and recall from the integer counts for all 1,344 measurements. [Local comparison runner](../backend/scripts/image_multiscale_jpeg_transfer.py)
