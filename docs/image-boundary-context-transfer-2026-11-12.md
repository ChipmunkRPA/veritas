# Color continuity and repetition around small image boundaries

The contextual checks cleared three additional patterned-panel flags after the earlier flatness rule. Seven screenshot flags remained across four groups. The repetition check also removed one correctly localized conventional paste and two correctly localized repeated-texture pastes. Those three locations came from the experimental broad fallback. Every existing fixed-size crop was preserved. The frozen result keeps the contextual checks offline. [Audited case ledger](evaluation-panels/image-boundary-context-transfer-2026-11-12.json)

| Held outcome | Current fixed check | Broad fallback | Prior interior flatness | Flatness and context |
| --- | ---: | ---: | ---: | ---: |
| Conventional sharp paste, correct location | 0/12 | 5/12 | 5/12 | 4/12 |
| Repeated-texture paste, correct location | 1/12 | 4/12 | 4/12 | 2/12 |
| Local tint, correct location | 0/12 | 6/12 | 6/12 | 6/12 |
| Feathered paste, correct location | 0/12 | 0/12 | 0/12 | 0/12 |
| Ordinary photo and printed-sign controls with a box | 12/56 | 12/56 | 12/56 | 12/56 |
| Incumbent-negative screenshots with an added box | 0/14 | 13/14 | 10/14 | 7/14 |

## Frozen context measurements

The earlier distributed-side point and native box geometry remain fixed. Only a newly selected broad fallback box receives context measurements. The color-continuity check fits a shared RGB transfer slope and per-channel offset on three sides. It tests the fit on the held fourth side using pixel pairs immediately across that edge. At least three held sides must meet the frozen texture, slope, color, gain and median-error conditions. The test borrows the source-over relationship from alpha compositing. Nearby image colors are used as a proxy for the occluded background. A single image generally leaves the background and foreground layers ambiguous. [Source-over compositing](https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/Compositing%20Digital%20Images.pdf) · [Recent alpha-composited layer decomposition study](https://arxiv.org/abs/2505.21541)

The repetition check removes a Gaussian-smoothed version of the luminance. It measures normalized correlation at one and two small displacements inside the candidate box. Both interior responses must reach 0.82 and the same displacement outside the box must have correlation at most 0.50. The separate and combined arms use the same responses. The previous flatness arm keeps its published 35% interior rule. [Texture periodicity from autocorrelation](https://www.sciencedirect.com/science/article/pii/S0167865597000305) · [Previous flatness comparison](image-small-boundary-interior-transfer-2026-11-12.md)

## Held photos and screenshot controls

The source roles used 12 target photos, 12 distinct donors and 12 separate screenshot background photos from a credited Wikimedia Commons roster. The retained photos had earlier project exposure. No context measurements or scores entered source selection. One duplicate photographer alias and a source depicting children were left out. Four separately attributed Flickr photos depict road signs, a dairy truck or a storefront. Their original evaluation views and resize round trips give eight printed-sign controls. The photo views compare exact sharp pastes, repeated native donor tiles, localized tint and feathered donor pastes against unchanged and ordinarily processed controls. [Wikimedia source roster](image-affine-native-scale-2026-11-11.md) · [Printed-sign source credits](evaluation-panels/image-small-boundary-control-stress-2026-11-12.json)

The 36 offline Chromium controls use opaque, translucent and periodic panels at varied native positions over real photos. Half have no caption. The fixed-size check flagged 22 of 36 and its measurements were sealed before the contextual scoring. The 14 remaining views came from only five groups, below the frozen eight-group minimum. The broad fallback flagged 13 of those views. The repetition check removed three caption-free patterned-panel flags and the prior flatness rule removed three opaque-panel flags. None of the four added translucent-panel flags was removed. Their candidate boxes exactly matched the CSS panel positions, so the color fit failed even with the authored boundary coordinates.

The contextual response can recognize periodic material while missing the distinction between a repeated graphic and a repeated pasted texture. The tested method measures boundary context and contributes zero AI confidence. No source image, browser capture or derived crop pixels are redistributed.

## Verify and reproduce

The native replay verified 36 original capture hashes, 118 analyzed PNG deliveries, 83 arm-specific crop records and 27 context measurements. It independently recomputed the RGB transfer fits and periodic lags. The public reader checks the sealed decisions with the Python standard library.

```sh
python3 -B -I -S backend/scripts/review_image_boundary_context_transfer.py \
  docs/evaluation-panels/image-boundary-context-transfer-2026-11-12.json
```

Use the retained credited `$archive`, the exact sign JPEGs in `$cache/<encoded_sha256>.bin`, and a new `$run` directory. The browser step uses the local Playwright installation and the Chromium version recorded in the ledger.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
node backend/scripts/capture_image_boundary_context.cjs plan "$run"
backend/.venv/bin/python -B -m scripts.image_boundary_context_transfer prepare "$archive" "$cache" "$run"
node backend/scripts/capture_image_boundary_context.cjs capture "$run"
for stage in freeze incumbent candidate; do
  backend/.venv/bin/python -B -m scripts.image_boundary_context_transfer "$stage" "$archive" "$cache" "$run" || exit
done
backend/.venv/bin/python -B backend/scripts/review_image_boundary_context_transfer.py "$run"
```
