# Broken contour continuation in a single JPEG

This check looks for two aligned, same-polarity sections of a scene edge that do not continue through the intervening image. It uses coarse signed gradients in one delivered JPEG. Earlier residual-encoder and boundary approaches measure different signals. Image-inpainting research documents the difficulty of preserving edge and texture continuity. This trial implements its own bounded measurement and uses no third-party detector code or weights. [Structure and texture reconstruction study](https://www.sciencedirect.com/science/article/pii/S1568494624014455) · [Inpainting color and structure study](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Towards_Enhanced_Image_Inpainting_Mitigating_Unwanted_Object_Insertion_and_Preserving_CVPR_2025_paper.pdf) · [Earlier JPEG residual result](image-jpeg-texture-rank-transfer-2026-11-12.md)

Measure Gaussian-smoothed luminance and evaluate eight possible line orientations at every eighth interior pixel. Each orientation and gap length checks seven samples on both sides at the same perpendicular position. Six samples on each side must support the same signed edge, with comparable strength. The central gradient must stay below 45% of the weaker flank. Rank the remaining candidates by the geometric mean flank strength, penalizing interior support and longer gaps. The selected square is centered on the gap and extends eight pixels beyond each half-length. This records a measured interruption without assigning authorship or intent.

A known 88-pixel break in a horizontal edge produced the expected region at JPEG75 and JPEG90. Reversing gradient polarity and rotating the scene preserved the expected response. A continuous edge and a flat image returned no region.

## Source-separated pilot

Twelve earlier credited photos supply ordinary calibration views. The held comparison uses two separate retained BG-REAL groups of 12 and 16 photos. Each contributes equal numbers of LaMa removals and Stable Diffusion 1.5 insertions, for 14 examples from each family. The prior project already evaluated these photos under different methods. Source photos, publisher recipients, and credited photographers are disjoint across the three roles. Exact and perceptual duplicate checks passed. The individual photos carry CC BY 2.0, and publisher contributions carry CC BY 4.0. The asset lineage names Apache 2.0 LaMa and CreativeML OpenRAIL-M Stable Diffusion 1.5. The evaluation uses the published media without running the generators. [Publisher data and licenses](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Earlier twelve-source qualification](image-jpeg-signed-transitions-2026-11-12.md) · [Later sixteen-source qualification](image-jpeg-texture-rank-transfer-2026-11-12.md)

Each photo contributes an unchanged original, matched local tone, blur and same-photo copy, a resize round trip, a previous JPEG encode and a conventional opaque annotation. The latter can naturally interrupt a line and tests a likely source of false flags. The eight later SD1.5 groups add local and global autoencoder controls. Sharp and eight-pixel feathered positives use the known native changed-pixel mask before final common JPEG encoding. The candidate and current aligned luma check see the same quality-75 and quality-90 files. Useful localization requires 30% intersection over union and 50% edited-pixel recall. The candidate was frozen before scoring. Each quality uses the second-highest ordinary calibration source maximum, with a strict greater-than rule.

## Held results

| Quality | Contour useful sharp / feathered, out of 28 each | Current luma useful sharp / feathered, out of 28 each | Contour / current ordinary flags, out of 212 | Contour wrong edit locations |
| --- | ---: | ---: | ---: | ---: |
| JPEG75 | 0 / 0 | 0 / 0 | 42 / 1 | 12 |
| JPEG90 | 0 / 0 | 0 / 0 | 10 / 0 | 4 |

The contour check found a measurable break in every held delivery. The highest-ranked location was useful on zero edited views even before applying the operating point. An independent enumeration confirmed that at least one candidate square could have met the useful-location criterion for 50 of the 56 distinct native edit variants before applying the edge-support tests. All six geometry misses were LaMa variants. The remaining 50 misses include every Stable Diffusion edit. The ordinary contour flags affected seven photo groups at JPEG75 and two at JPEG90. The annotation controls alone supplied six flags at JPEG75 and two at JPEG90, and unchanged originals supplied five and one respectively.

The frozen advancement rule required useful feathered findings on three additional photos across both generator families, with at most two added ordinary groups and four wrong edit locations. Both qualities fell short. The measured edge interruption also occurs in genuine scenes and ordinary retouching. This check stays offline.

## Replay

The CPU run measured 168 ordinary calibration deliveries and 536 held deliveries. The local readback rebuilt all 704 delivery and truth bindings and independently recalculated 1,240 native overlaps. The source photos and individual results remain local. [Aggregate and replay commitments](evaluation-panels/image-jpeg-contour-gap-transfer-2026-11-12.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_jpeg_contour_gap_transfer "$phase" \
    --archive "$archive" --encoder "$prior_encoder" --earlier "$earlier_f5" \
    --later "$later_f5" --output "$run" --publication "$receipt" || exit 1
done
```
