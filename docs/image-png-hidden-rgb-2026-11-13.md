# Retained RGB behind fully transparent PNG pixels

A PNG can retain RGB samples at coordinates where alpha is zero. An ordinary viewer composites those coordinates as fully transparent. Editing and export software may deliberately preserve their colors for future work. This offline check measures where fully transparent pixels contain structured variation. It produces native positions and counts and does not display, export or store a rendering of the hidden colors. It contributes zero AI-authorship weight. [PNG specification](https://www.w3.org/TR/png-3/)

## Frozen measurement

The bounded reader accepts static, noninterlaced, 8-bit RGBA PNGs. It verifies every chunk's CRC, IDAT order, complete zlib stream, row filters and exact raster extent before decoding. Animated, unsupported color and orientation carriers, trailing bytes, and unknown critical chunks are declined. The detector only considers 16 × 16 native windows consisting entirely of alpha-zero pixels and at least five pixels away from visible samples. It requires variation between enough neighboring samples along each axis and a connected set of six or more overlapping windows. The output is their exact union and native rectangular bounds, with a binary mask commitment. A fully transparent constant color, a thin interpolation fringe, or isolated colored pixels are outside that measurement.

## Local result

Eight earlier photos provide development inputs. Twelve distinct confirmation photos provide the exact same three controlled alpha masks and six ordinary controls per photo. Both photo groups were used previously for unrelated local JPEG studies. Their rights and source disjointness come from the retained source commitments. Five additional Wikimedia Commons PNG originals are attributed to five CC0 contributors. [Development source rights](image-three-class-patch-probe-2026-11-12.md) · [Confirmation source rights](image-three-class-patch-fresh-2026-11-12.md)

| Confirmation condition | Measured files | Files with marked areas | Useful extent |
|---|---:|---:|---:|
| Large alpha mask with RGB retained | 12/12 | 12/12 | 10/12 |
| Small alpha mask with RGB retained | 12/12 | 11/12 | 5/12 |
| Photo cutout with background RGB retained | 12/12 | 12/12 | 9/12 |
| Opaque, scrubbed, constant-color, partial-alpha, fringe and isolated-pixel controls | 72/72 | 0/72 | — |
| Native public transparent-image controls | 5/5 | 1/5 | — |
| Malformed or unsupported authored PNGs | 0/9 | 0/9 | — |

Useful extent requires 30% intersection over union and recovery of half the known alpha-masked pixels. The fixed rule required ten large and six small useful extents, no additional ordinary negative highlights and no malformed admissions. The small-area result fell short. This experiment stays offline.

The published cactus cutout contains 23,232 fully transparent pixels in the four marked areas. Its original PNG and attribution were verified by publisher hash. This benign export demonstrates that retained RGB can accompany ordinary background removal. The check reports structured variation and gives no evidence about why it was retained or what its hidden pixels depict. [Ordinary cutout](https://commons.wikimedia.org/wiki/File:Cactus_flower_extracted.png)

## Replay

The offline replay reconstructs 194 exact PNG deliveries, checks each mask against the actual alpha plane and reproduces all measured regions and results. The public receipt contains aggregate coordinates-free measurements and published source attribution. Any future user-facing inspection should label the finding neutrally and disclose coordinates or counts only. Revealing pixel colors would require a separate, deliberate inspection action. [Frozen result and replay ledger](evaluation-panels/image-png-hidden-rgb-2026-11-13.json)

Use the exact earlier and later patch-head source directories and the five Commons originals named in the receipt. Store the originals in the private public-controls directory as `blank.png`, `blue.png`, `cactus.png`, `flowers.png` and `strawberry.png`.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_png_hidden_rgb_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --public "$public_controls" --output "$run" \
    --publication "$receipt" || exit 1
done
```
