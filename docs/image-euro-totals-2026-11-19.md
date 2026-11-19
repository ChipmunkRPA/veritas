# Printed euro totals with explicit tax treatment

The local Full and Deep image review checks supported PNGs that explicitly
say whether value-added tax is already included in the subtotal or added
to it. An explicitly deducted fixed rebate enters the arithmetic when it
appears in the same summary. When the amounts disagree, the finding shows
the calculation and links every term to the original pixels. The browser,
Standard PDF, and JSON contain matching positions. The newsroom PDF
withholds the values and crops. The observation carries zero weight in the
AI assessment.

The check uses saved pixel content and two readings of each amount, first
in the page and then on a tight crop. Amounts need a printed euro symbol or
currency code. Point, comma, and apostrophe digit grouping are read as
printed. The calculation uses the same separators as the printed total.
The check requires an explicit tax basis and a supported row
sequence. It can decline images with other adjustments, conditional
discounts, unreadable amounts, or ambiguous layout.

## Publisher comparison

The test used the publisher's fictional invoices in InvoiceBenchmark.
Each pinned PDF was rendered upright to a PNG with a 1,680-pixel long
side. The candidate received only the PNG. The source PDF text and its
publisher-supplied values separately verified the recognized amounts and
their declared tax treatment. A matching amount rectangle covers at least
90% of the source amount's horizontal text span and intersects the same
printed line. The 11 development cases guided the text-reading changes.
The 29 primary held cases and 23 additional cases were reserved before
reading any of their images. [Publisher release and MIT terms](https://huggingface.co/datasets/jngb-labs/InvoiceBenchmark/blob/f0699d8c94468fb2fefaa22f669684b22e68c3d3/README.md)

| Evaluation group | Existing exact findings on inconsistent files | New exact findings on inconsistent files | New alerts on balanced files |
| --- | ---: | ---: | ---: |
| Primary held group | 0/10 | 9/10 | 0/19 |
| Additional tax, discount, and document cases | 0/8 | 2/8 | 0/15 |

The new check declined one of ten inconsistent primary files and five of
19 balanced primary files. In the additional group, it declined six of
eight inconsistent files and 13 of 15 balanced files. The installed reader
reproduced all 63 development and held alert decisions. Raster-verified
source crops accompanied all 18 findings. It also reproduced all 240
earlier PNG alert decisions. The figures measure one
publisher's fictional layouts with arithmetic errors. Broader layout
coverage, scanned paper and deliberate AI editing need separate evidence.

```sh
python3 -B -I -S backend/scripts/review_image_euro_totals.py \
  docs/evaluation-panels/image-euro-totals-2026-11-19.json
```
