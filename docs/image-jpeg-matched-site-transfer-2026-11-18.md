# Does a JPEG cue follow a relocated edit?

A local model-free study tested whether two native-pixel measurements could follow the same inpainting operation when it moved between comparable locations in one photograph. Across two JPEG qualities, neither measurement nor their combination ranked a useful region first on any of the 64 edited deliveries. Each variant's highest-ranked box almost always overlapped the highest-ranked box on the untouched photograph. The measurements stay out of serving.

## Frozen comparison

Eight untouched photos set the operating points. Two separate eight-photo banks tested transfer. All 24 photos came from previously retained, individually licensed CC BY 2.0 sources. The banks shared no photographer, image or likely near duplicate. The minimum 256-bit thumbnail sketch distance was 96. The photographs had prior evaluation exposure. The placements, delivered JPEGs and measurements were new in this study. [Source collection and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Before scoring, a deterministic salted selection chose two 64 × 64 regions per held photo using only coarse native luma, structure and color. Both sites shared the same JPEG sampling alignment and were at least 160 native pixels apart. Each operation was rendered independently at A and B with the same six-pixel feather. The held files covered untouched re-encoding, local conventional blur, local brightness adjustment, and locally generated inpainting. The two JPEG copies of each variant used qualities 75 and 90, with identical baseline 4:2:0 settings within quality. Ground truth came from the exact changed RGB pixels before re-encoding. The editor authored test inputs. All three measurements used only the delivered JPEG. [Pinned editor release](https://huggingface.co/andraniksargsyan/migan/blob/1538c13/migan_pipeline_v2.onnx), [publisher weight terms](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS)

- **Edge continuation** compared neighboring fine luminance residuals along and across the direction of the local coarse edge.
- **Chroma alignment** compared co-located fine luminance and color residuals with a one-pixel displacement along the edge normal.
- **Joint** took the weaker of the two readings after dividing each by its own frozen clean reference point.

Each component measured the core against a surrounding ring inside coarse brightness and gradient-strength strata and required the direction of the core/ring contrast to agree on two interleaved pixel sets. The scanner used a fixed 16-pixel grid and excluded neighbor comparisons crossing an 8-pixel JPEG block boundary. The operating point for each quality and measurement was the second-highest untouched calibration maximum. A useful region had to include at least half of the changed pixels and have at least 0.30 intersection over union with the changed-pixel mask. For the paired placement diagnostic, each edited site had to beat its alternate site in the corresponding file by more than the untouched site's advantage plus the frozen margin. The margin was 0.002 for each component and 0.05 for the joint score. The diagnostic used the known placement; the blind scan had no site coordinates. [Pixel-Inconsistency Modeling for Image Manipulation Localization](https://arxiv.org/abs/2310.00234), [EDGER](https://arxiv.org/abs/2605.12002)

## Held result

Each row covers 16 photographic sources and 32 inpainting files. A false location is a thresholded inpainting result outside the useful-region criterion. The three paired intervention columns count stronger re-pointing to the known sites among 16 summaries per operation.

| JPEG quality | Measurement | Useful first regions | Untouched flags | False locations | Top repeats untouched location | Inpaint pairs | Blur pairs | Brightness pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | Edge continuation | 0/32 | 1/16 | 2/32 | 30/32 | 3/16 | 3/16 | 0/16 |
| 75 | Chroma alignment | 0/32 | 4/16 | 7/32 | 30/32 | 2/16 | 1/16 | 0/16 |
| 75 | Joint | 0/32 | 1/16 | 2/32 | 30/32 | 1/16 | 1/16 | 0/16 |
| 90 | Edge continuation | 0/32 | 1/16 | 2/32 | 30/32 | 2/16 | 5/16 | 0/16 |
| 90 | Chroma alignment | 0/32 | 3/16 | 5/32 | 30/32 | 2/16 | 1/16 | 0/16 |
| 90 | Joint | 0/32 | 1/16 | 2/32 | 31/32 | 1/16 | 0/16 | 0/16 |

The repeated location means at least 0.50 intersection over union with the untouched file's top box. Every file had a measured candidate and all 16 paired sites per row were measurable. The highest-scoring region often stayed fixed even though the intervention moved. Some measurements responded at the supplied sites, including for conventional blur. The frozen release rule required at least three correctly localized thresholded A/B pairs across both held banks and limited ordinary, conventional and wrong-location alerts. Every row had zero such pairs. This result is bounded to one inpainting tool, one region size and the two tested JPEG deliveries.

## Independent recount

The independent local reader verified 240 delivered JPEG byte hashes, native dimensions, 4:2:0 sampling and the quantization tables, exact changed-mask membership, native score maps, top positions and both banks' metrics. The public receipt preserves anonymous per-group native-score scalars, mask areas, intersection sizes and whole-run commitments. It contains no images, box coordinates, source hashes, creator identifiers, or asset paths.

From the project root, run the standard-library recount. It derives the six calibration points, localization, control and release counts from the scalar receipt. Visual localization and native detector reruns require the separately held originals and are outside the public replay's scope. [Anonymous scalar receipt](evaluation-panels/image-jpeg-matched-site-transfer-2026-11-18.json.gz)

```sh
python3 backend/scripts/review_image_jpeg_matched_site_transfer.py
```
