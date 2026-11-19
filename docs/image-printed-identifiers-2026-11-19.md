# Printed identifier comparisons in screenshots

The local Full and Deep image checks can compare a printed reference,
record, or invoice ID with an explicitly labeled confirmation of the
same field. Chipmunk Verify shows both text readings and their original
pixel crops when the values differ. Each crop links to its exact position
in the original image. The Standard PDF and JSON record the same values
and coordinates. Redacted PDFs withhold the values and both crops.

This observation identifies a mismatch between two visible fields. A
typing error or an ordinary correction can explain it. The result does
not assign the error to one of the two lines and carries zero weight in
the AI assessment.

## Scope and method

The optional local check accepts upright, opaque PNGs and JPEGs up to
2,048 pixels per side and 3 million pixels. It reads explicit English
labels, including `Reference ID` and `Confirm reference ID`, on separate
text lines, with a colon and a value consisting of two to five ASCII
letters, a hyphen, and six to twelve digits. It requires exactly one
recognized entry and confirmation for the field, then recognizes each
line again in its own crop. An ambiguous pair is withheld. The two
readings use the same local optical character reader and trained data.
This comparison uses no third-party service or new weights.

Recent text-image research distinguishes a localized visual trace from a
semantic contradiction. A separate exploratory October study also found
that its observed edit-supported concerns came from identifier disagreements.
These studies motivated adding a narrow, source-bound consistency comparison.
We used neither study's trial media or model weights.
[Grounded text-image forensics study](https://arxiv.org/html/2609.02097v1) ·
[Identifier disagreement study](https://arxiv.org/html/2610.30758v2)

## Held results

Six authored layouts were used for development. Sixteen other layouts used
different font families and included alternate alignments and dark-on-light
and light-on-dark renderings. Each group had two edited values and six
controls. The controls included equal fields, unrelated labels, multiple
records, an old ID shown alongside the current one, local blur, and a
correction applied to both fields. Location success required each source
crop to cover at least 90% of its corresponding authored value box.

| Held carrier | Exact values | Met location rule | Ordinary flags |
| --- | ---: | ---: | ---: |
| PNG | 32 / 32 | 31 / 32 | 0 / 96 |
| JPEG quality 75 | 32 / 32 | 32 / 32 | 0 / 96 |

The production reader reproduced all 352 development and held outcomes,
including the 86 reported pairs and their measured value coverage. The
trial is a controlled test of visible disagreement. It does not estimate
deployment-wide false alarms or accuracy at identifying AI-generated forms.
The public reviewer recounts the anonymous outcomes using Python's standard
library. A signed original-file replay checks that both crops match the
submitted pixels. It does not authenticate the issuer or rerun recognition.
[Anonymous evaluation receipt](evaluation-panels/image-printed-identifiers-2026-11-19.json)

Four browser runs completed after simulated checkout. A PNG and a JPEG
showed both correct values and four source links in total. The synchronized
and multiple-record controls showed no mismatch. The Standard PDF and JSON
matched the original crop pixels. The newsroom PDF withheld both values
and their crops. All four signed packages verified against the uploads.

```sh
python3 -I -S backend/scripts/review_image_printed_identifiers.py \
  docs/evaluation-panels/image-printed-identifiers-2026-11-19.json
```
