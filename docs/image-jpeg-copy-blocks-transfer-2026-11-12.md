# Small copied regions in one JPEG

The local image workflow can show two small regions from the same JPEG when their encoded brightness blocks match exactly. This fallback runs after the existing non-mirrored copy check finds no qualifying pair. It shows the original crops and their source positions in the normal evidence view, standard PDF and JSON. Ordinary repetition can also match, and this check contributes zero weight to the AI estimate. [Frozen comparison and outcomes](evaluation-panels/image-jpeg-copy-blocks-transfer-2026-11-12.json)

## Method

Read the submitted JPEG's native quantized 8 × 8 luminance coefficients. Look for matching blocks with at least four nonzero non-constant coefficients and exactly two occurrences in the encoded image. At least eight such blocks need to agree on the same displacement. A reported pair needs a separate rectangle on each side with exact encoded equality across every retained block, evidence in all four quadrants, and dimensions of 24–128 pixels on both axes. The earlier implementation's decoded-patch and geometric checks keep priority. The new method uses no learned weights. It adapts the published block-matching idea by using the original encoded luminance grid directly. [Original block-matching study](https://ws2.binghamton.edu/fridrich/Research/copymove.pdf)

The held panel uses 18 source photos outside the previous translation test and eight repeated-detail controls from that test. All 26 have distinct recorded source groups and photographer profiles. The rights-reviewed photo archive retains CC BY 2.0 and CC BY-SA 2.0 terms. The fixed recipes author 40 × 48 and 64 × 64 same-image copies at grid-aligned and deliberately off-grid displacements. No test outcome entered source, location or threshold selection. The controls include untouched publisher JPEGs, matched re-encodes, global contrast, local blur, resampling, and naturally repeated detail. Source media and exact pair coordinates stay local. [Source qualification and rights](image-fresh-source-cache-v39-results.md) · [Earlier translation comparison and repeat controls](image-translation-patch-transfer-2026-11-11.md)

| Delivery | Authored copy | Current /18 | Encoded blocks alone /18 | Current with fallback /18 |
| --- | --- | ---: | ---: | ---: |
| JPEG75 | 40 × 48, grid-aligned | 3 | 16 | 15 |
| JPEG75 | 64 × 64, grid-aligned | 11 | 17 | 17 |
| JPEG75 | 40 × 48, off-grid | 0 | 0 | 0 |
| JPEG75 | 64 × 64, off-grid | 4 | 0 | 4 |
| JPEG90 | 40 × 48, grid-aligned | 2 | 18 | 17 |
| JPEG90 | 64 × 64, grid-aligned | 10 | 18 | 17 |
| JPEG90 | 40 × 48, off-grid | 1 | 0 | 1 |
| JPEG90 | 64 × 64, off-grid | 9 | 0 | 9 |

Useful means at least 0.50 intersection-over-union for both rectangles and at least half the exact changed pixels on the edited side. The fallback added useful results on 15 distinct 40 × 48 source groups and eight distinct 64 × 64 groups across both qualities. It added zero incorrect locations. The 186 ordinary deliveries produced no additional pair. Of those, 171 met the brightness-block input profile and 15 original publisher JPEGs were outside it. Deliveries reuse source photos. Existing pairs remain as measured, even where the encoded comparison could have supplied a more accurate location. These counts concern authored translations on real photos and provide no evidence about AI authorship.

## Local app

Set `VERITAS_IMAGE_JPEG_BLOCK_COPY_REVIEW=true` in the development environment before starting the API and worker. The new branch admits readable upright color JPEGs with 4:2:0 chroma and a short side of at least 384 pixels and a long side of 513–1,024 pixels. The measured comparison covers full 8-pixel blocks. Copies that are rotated, scaled, blended, or shifted off that grid can remain unresolved. Existing saved evidence remains readable.

Newly generated reports record the two crop positions and the exact encoded measurement. The original-media verifier independently reads the luminance coefficients in both committed rectangles and checks the source hash and crop pixels. A completed check with no pair is shown as such, with no implication about authenticity.

The isolated local Full worker ran one held positive and its matched unchanged JPEG with simulated prepaid checkout. The positive returned two exact 32 × 40 pixel crops at the frozen positions. Signed original replay reproduced 20 matching encoded brightness blocks per crop. The unchanged JPEG returned a completed check with no pair. Both runs matched the API and JSON and verified their standard PDF and signed package. The PDF scan also confirmed that the generated face-scope limitation retained its explanation without identifying the underlying model.

## Reproduce

For a clean checkout with public sources, use the [eight-photo JPEG replay](image-jpeg-public-replay-2026-11-12.md). It checks both JPEG fallbacks and their native crop evidence on its own public panel.

Set `cache` to the retained `image-fresh-source-cache-v39/private-groups.json` and `previous` to the archived earlier translation `panel/freeze.json`. Set `run` to an unused directory for the first stage. The runner freezes source and delivery hashes before testing both methods. Review reproduced all 330 exact input deliveries, paired regions and coefficient commitments.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_jpeg_copy_blocks_transfer \
    "$stage" "$run" "$cache" "$previous" || exit
done
```
