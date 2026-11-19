# Image residuals compared at similar brightness and texture

Neither of the two local residual readings correctly located an edit on
the held files, even before applying its operating point. The candidates
also added highlights to ordinary files. These findings keep both methods
experimental.

## Local measurement

Each JPEG is divided into native 16-pixel cells. A cell is compared with
distant parts of the same file selected for similar brightness, coarse
detail, color and direction of texture. At least 12 such cells across
four separate neighborhoods are required. The first reading measures a
drop in fine luma detail. The second also measures changes in the local
red-to-green and blue-to-green residual relationships. Both omit samples
that cross an eight-pixel block boundary. The highest-ranked rectangle
must have measurable cells over at least three quarters of its area.

Brightness conditioning takes inspiration from prior research on sensor
noise. These inputs are compressed color images. Their measured details
include effects of texture and processing. The test treats them as image
observations. [Signal-dependent noise research](https://arxiv.org/abs/2010.16211)
· [Nov. 4 Inpainting Exchange revision](https://arxiv.org/html/2602.00192v2)

Ten separately assigned camera-photo groups supplied 240 development
deliveries. Each operating point is the second-highest source maximum
among ordinary development files left clear by the existing JPEG and
regional checks, set separately for JPEG75 and JPEG90. The held camera
panel has 12 distinct photos with LaMa and MI-GAN removals. It includes
originals, ordinary blur, color edits, copies, and conventional inpainting.
The separate scene-text panel has 14 more source groups with staged
removals. The original references in this panel can have unknown origin.
All groups have prior exposure in other project experiments. No held group
entered point selection for either reading. [Input panels and rights](image-masked-reconstruction-2026-11-17.md#held-inputs)

| Held panel | JPEG quality | Existing useful / known edits | Added useful, each arm | Added ordinary alerts, luma / combined |
| --- | ---: | ---: | ---: | ---: |
| Camera photos and matched controls | 75 | 3/48 | 0 | 8/120 · 10/120 |
| Camera photos and matched controls | 90 | 0/48 | 0 | 9/120 · 13/120 |
| Scene-text removals | 75 | 0/28 | 0 | 14/70 · 3/70 |
| Scene-text removals | 90 | 0/28 | 0 | 19/70 · 10/70 |

Useful means that the highest-ranked region covers at least half the
pre-encode changed pixels and reaches 0.30 intersection over union. Among
96 edited camera deliveries the top combined region overlapped its own
untouched counterpart's region by at least 0.50 IoU in 58 cases. Among 56
edited scene-text deliveries the count was 48. The local comparison still
followed many structures that were already present before editing.

A retrospective check supplied the true location and required comparable
readings across at least 70% of the edited pixels. Even with that known
area, neither arm exceeded the strongest ordinary treatment by 0.20 in
any of the 40 fully matched camera deliveries or 28 fully matched
scene-text deliveries. The fixed-area diagnostic also shows overlap
between coarse appearance and the changes produced by ordinary processing.

The separate native audit rehashed 772 deliveries and their masks and
reconstructed all 1,544 candidate crop intersections. The anonymous
receipt retains scalar responses, group assignments and overlap counts.
[Numeric receipt](evaluation-panels/image-conditioned-residual-2026-11-19.json.gz)

```sh
python3 -B -I -S backend/scripts/review_image_conditioned_residual.py \
  docs/evaluation-panels/image-conditioned-residual-2026-11-19.json.gz
```
