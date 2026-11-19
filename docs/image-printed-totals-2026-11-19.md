# Printed totals in PNG screenshots

The optional Full and Deep local check compares a printed subtotal plus
printed tax with a single printed total. When the same three values can be
read repeatedly and the difference is at least two minor currency units, it
shows the calculation and gives links to each amount in the original.
The Standard PDF presents three exact original-image crops and the amount
positions. JSON and the signed package carry the matching coordinates.
The newsroom PDF withholds the amounts and crops.

The check looks for explicit English labels in an aligned set of rows using
one printed currency. It skips ambiguous amounts, other money rows or
adjustments, common tax-included wording, and unsupported images. A missing
finding can mean the reader could not measure the screenshot. A difference
can come from an ordinary adjustment or data-entry error and carries zero
weight in the AI assessment.

## Fresh controlled comparison

Two deliberately edited amounts and thirteen controls for each of sixteen
layouts were exported to PNG and JPEG quality 75. Controls included balanced
totals, coordinated changes, fees, discounts, inclusive tax, mixed currency
and duplicated summaries. A qualifying detection required all three values
to match the known printed values and every crop to cover at least 90% of
its amount. The frozen gate required at least 25 of 32 qualifying detections
with zero wrong alerts. A separate pixel comparison confirmed all 32 PNG
interventions were confined to their amount regions.

| Carrier | Exact detections | Unassessed edited images | Control alerts | Unassessed controls |
| --- | ---: | ---: | ---: | ---: |
| PNG | 25/32 | 7/32 | 0/208 | 170/208 |
| JPEG quality 75 | 20/32 | 12/32 | 0/208 | 177/208 |

The earlier run is retained in the receipt. Five intended mixed-currency
controls per carrier rendered the same symbol, so that run could not satisfy
the planned control test. The repeat used new held fonts and a new layout
seed, confirmed the authored signs before OCR, and used the exact same
reader. The production reader reproduced all 240 PNG alert decisions.
Only PNG met the release requirement. These authored screenshots measure
constrained arithmetic and localization. Real-world layout coverage and
AI authorship accuracy still need separate evaluation.

The production reader recovered all 75 amount values from the 25 flagged
PNGs. Each linked amount rectangle covered at least 90% of its authored
reference rectangle. Five simulated checkout runs also verified the two
positive PNG cases, a balanced PNG, an ordinary fee example, and an
excluded JPEG. Standard PDFs embedded all six original crops, and all
five signed packages verified against their originals. Chrome loaded those
same six crops and matched their links to the JSON coordinates.

Enable this inspection in development with
`VERITAS_IMAGE_PRINTED_TOTAL_REVIEW=true`. Review the published counts with
`make replay-evidence`.
[Anonymous per-image evaluation receipt](evaluation-panels/image-printed-totals-2026-11-19.json)
