# Does a JPEG's repeating residual pattern locate a small edit?

Among 64 staged inpainting deliveries, the best result was one useful
location. It also flagged three of 16 untouched photos and ten of 64
conventionally retouched copies at that delivery quality. The top box
usually stayed in the same place as on the untouched JPEG when the edit
moved elsewhere. These measurements remain in research.

## Frozen comparison

The August 2026 residual-pattern study estimates periodic traces from
the submitted image and compares local residuals to that reference.
This experiment independently tests a related idea with two separate
local measurements, one for signed two-pixel phase and one for
eight-pixel residual magnitude. It uses no code, weights or data from
that paper. [Residual-pattern study](https://arxiv.org/html/2608.20558v1#S4)

Decode the native RGB raster, form luminance and red-minus-green and
blue-minus-green channels, and subtract a 1.2-pixel Gaussian-smoothed
view. Divide into native 8 × 8 tiles. Remove separate row and column
means and add back the tile mean. Divide by the channel's residual RMS
plus one pixel level. For the signed view, average the residuals by
two-pixel phase. For the magnitude view, cap absolute residuals at
three normalized levels and remove additive row and column means again.

Scan 64-pixel squares at 16-pixel steps with a full 32-pixel surrounding
ring. Separate tiles by coarse detail above or below the image's median,
then by checkerboard tile parity. In each supported detail stratum,
compare the phase vector of each core half with the opposite ring half.
Require at least six tiles per core and ring half and cosine similarity
of at least 0.35 between the two ring halves. Use the smaller nonnegative
drop from ring similarity across the two comparisons and keep the
stronger supported detail stratum. The two spatial halves are separate
tile sets that share the same scene and decode.

The retained matched-placement bank has eight clean calibration photos
and two held banks of eight photos each. Photo groups and photographers
are separate across banks. It uses two equally sized edit placements per
held photo, one untouched JPEG, and both placements of the matching blur,
brightness and inpainting operations. The working comparison uses the
same earlier deliveries, so this is fresh to the phase measurements only.
The score on the second-highest untouched calibration photo sets each
strict operating point. A useful box needs at least half the exact changed
pixels and 0.30 intersection over union. The frozen advance rule required
at least three fully localized A/B pairs across both banks with separate
bounds on untouched, conventionally retouched and wrongly marked edits.
[Retained bank, license and source controls](image-jpeg-matched-site-transfer-2026-11-18.md)

## Held results

| Quality | Measurement | Useful locations before point | Useful locations after point | Wrong inpaint flags | Untouched flags | Conventional flags | Inpaint top boxes overlapping untouched top |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | Signed phase | 0/32 | 0/32 | 0/32 | 0/16 | 0/64 | 26/32 |
| 75 | Magnitude phase | 2/32 | 1/32 | 5/32 | 3/16 | 10/64 | 29/32 |
| 90 | Signed phase | 0/32 | 0/32 | 0/32 | 0/16 | 0/64 | 23/32 |
| 90 | Magnitude phase | 0/32 | 0/32 | 0/32 | 0/16 | 0/64 | 29/32 |

Top box overlap with the untouched top uses a 0.50 intersection over
union. All 224 held files had a proposed region in both views. When
the method was given the two exact edit sites, it could compare both
positions across all three paired deliveries for just one to four of
16 groups, depending on quality and measurement. No measurement
located a complete A/B pair at its operating point. These results
bound the value of this phase comparison on the tested JPEG profiles
and edit size.

## Independent replay

A direct-sum implementation verified 240 native JPEG commitments and
encoding profiles and exact overlap against 192 changed-pixel mask
deliveries. It recomputed 740 retained numeric readings to within
6.67 × 10⁻¹⁶ and reproduced all 636 unsupported site readings.

The public receipt includes anonymous group and role labels, hexadecimal
scores, pixel counts and a whole-study commitment. Media, individual
source digests, exact coordinates and creator identifiers stay local.
The public checker re-derives the four operating points, overlap,
false-location and paired-placement results without media or model
libraries. [Anonymous receipt](evaluation-panels/image-jpeg-phase-consistency-2026-11-18.json.gz)

```sh
python3 -I -S backend/scripts/review_image_jpeg_phase_consistency.py
```
