# Earlier JPEG evidence after a crop and quality-90 delivery

The tested comparison recovers crop-and-resize evidence for standard RGB 4:2:0 quality-90 submissions with an upright earlier JPEG encoded with standard quality-82 or quality-95 tables. The existing route left those crop-and-resize pairs unassessed. Accepted results identify the earlier native crop and show differences on the submitted grid. Differences can result from ordinary processing and carry zero AI-authorship weight. [Earlier crop comparison](image-reference-crop-resize-transfer-2026-11-11.md)

The source selection used 24 distinct recorded contributors from the verified licensed photo cache. The selection excluded entire linked components when a source JPEG or decoded RGB hash appeared in the retained exposure ledgers. Original sources with a long side above 768 pixels supplied a centered native crop without resizing. Each earlier copy was authored from those pixels at quality 82 or 95. Every main submitted delivery used a separate JPEG encoding at quality 90 after an asymmetric crop and a proportional resize. The fixed scales were 0.74×, 0.86× and 1.13×. The same source bindings produced the known local tone edits and nuisance controls. Those edits measure the localization of visible changes. The sample does not estimate AI detection performance. [Source credits and aggregate](evaluation-panels/image-reference-jpeg90-transfer-2026-11-12.json) · [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) · [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/)

The existing keypoint, crop, registration, and JPEG-delivery checks remained fixed. Exact crop recovery requires all four earlier native edges to match. A useful location also needs at least 30% overlap relative to the union and at least 50% recall against the exact pre-encode local change. Each quality stratum had to recover eight clean crops and six useful feathered locations, with no incorrect accepted crop and at most four nuisance highlights across three sources. [Crop registration method](image-reference-crop-resize-transfer-2026-11-11.md) · [Reference-based forensic research](https://arxiv.org/abs/2611.10908)

| Held outcome | Earlier quality 82 | Earlier quality 95 |
| --- | ---: | ---: |
| Clean pairs with exact crop and no differences | 8/12 | 10/12 |
| Sharp tone edits usefully located | 9/12 | 10/12 |
| Feathered tone edits usefully located | 8/12 | 10/12 |
| Nuisance views highlighted | 2/48 | 3/48 |
| Incompatible pairs incorrectly accepted | 0/48 | 0/48 |

The incompatible pairs used fractional translation, rotation, anisotropic resizing, or an unrelated image. Three nuisance highlights followed an independent earlier delivery and two followed global contrast. Clean re-encodes from the actual earlier pixels and the additional prior-recompression controls showed no highlights. The quality-87 controls remained outside the crop fallback. The quality-75 comparison served as a separate regression control. The aggregate includes the independently rebuilt delivery, coordinate and mask counts. Photographs, derived files, and private per-file records stay local.

## Replay

Use the study commit and the retained v39 source cache and exposure ledgers. Set `run` to a new local directory. The source cache credits identify the work and license. The scripts regenerate only the disclosed transforms.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze measure replay; do
  backend/.venv/bin/python -B -m scripts.image_reference_jpeg90_transfer "$stage" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_reference_jpeg90_transfer "$run" "$run/independent.json"
```
