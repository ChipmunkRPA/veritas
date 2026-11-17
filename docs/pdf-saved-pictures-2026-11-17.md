# Picture changes across saved PDF versions

The local document analysis now compares a visible picture at the same
placement across verified saves retained inside one uploaded PDF. A finding
shows the earlier and submitted page numbers, the measured pixel rectangle,
the submitted picture with the differences marked, and an unmarked crop.
The PDF report and JSON export use the same coordinates. The earlier image
stays in the submitted file. The report identifies the earlier save by
byte range and SHA-256.

PDF incremental saves can retain earlier versions. The check follows the
same cross-reference chain as the saved-wording review, then checks picture
coverage independently. It requires a unique horizontal picture placement,
visible in both saves, with pixel-for-pixel support from detailed tiles
throughout the picture. The existing image comparison measures local color
differences inside that aligned area. Covered, moved, resized, and heavily
recompressed areas can be unassessed. Ordinary editing and AI editing can
both produce a local difference. This evidence carries zero AI confidence
weight. No new model weights are needed.
[PDF incremental update specification](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.5_v6.pdf)
· [Page-object comparison research](https://arxiv.org/abs/2507.00827)

## Local comparison

The first 24 source photos were taken from two prior screenshot panels.
An initial comparison flagged 47 of 48 recompressed controls. Exact
surrounding-tile support withheld all 48 on replay. A separate confirmation
used 24 different photographs and contributors, with staged local edits
made from two licensed removal systems. Each pair produces 18 local PDF
deliveries across PNG and JPEG picture streams. A correctly located area
covers at least half of the staged rectangle with at least 0.30 intersection
over union. All encodings and saves from a photograph stay together.
[Source attribution and editor licenses](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)

| Held delivery, per encoding | Compared | Correctly located | With a highlight |
| --- | ---: | ---: | ---: |
| Staged AI-edited patch | 24/24 | 22/24 | 24/24 |
| Same patch, followed by another comment save | 24/24 | 22/24 | 24/24 |
| Deliberate local tone adjustment | 24/24 | 22/24 | 24/24 |
| Metadata-only and outside-picture comments | 48/48 | — | 0/48 |
| Recompression, global recoloring, visually covered picture or full rewrite | 0/96 | — | 0/96 |

The earlier saved-wording comparison reported zero changes on the same
deliveries. The new result shows an observed picture change within the
file's retained history. Full rewrites in the panel left no earlier save
to compare. The measured scope uses at most five saves, eight pictures per
page and 24 pages per save. The browser and PDF tell the reader how many
comparisons they can inspect. JSON retains the compared and unassessed
counts by page for each earlier save.

## Reproduce the review

The [earlier-panel receipt](evaluation-panels/pdf-saved-pictures-replay-2026-11-17.json.gz)
and [confirmation receipt](evaluation-panels/pdf-saved-pictures-held-2026-11-17.json.gz)
contain source attribution, known edit rectangles, PDF and retained-prefix
hashes, and the measured positions. An independent recount can verify the
deliveries and source-group separation using standard-library Python.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.pdf_saved_picture_panel verify \
  "$panels/pdf-saved-pictures-held-2026-11-17.json.gz" \
  --prior "$panels/pdf-saved-pictures-replay-2026-11-17.json.gz"
```

For the first panel use the two [resized screenshot replay](image-reference-scaled-screenshot-2026-11-17.md#replay)
directories. Generate the confirmation pairs using the image-removal
intake in the [source study](image-jpeg-frequency-bands-2026-11-17.md#replay-and-independent-comparison).
Run from the repository root with backend dependencies and Playwright
Chromium installed.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=backend/.venv/bin/python
"$python" -B -m scripts.image_jpeg_band_sources prepare \
  "$intake" "$editors" "$out/confirmation-pairs" --partial --as-pairs
"$python" -B -m scripts.pdf_saved_picture_panel run "$out/replay" "$first" "$second"
"$python" -B -m scripts.pdf_saved_picture_panel run "$out/held" "$out/confirmation-pairs" \
  --prior "$out/replay/results.json.gz"
"$python" -B -m scripts.pdf_saved_picture_app "$out/held" "$out/local-app"
```

The app replay uses four simulated USD 9.99 checkouts. It verifies zero
analysis jobs before payment, identical customer assessments with the option
enabled and disabled, browser picture positions, PDF pixels, and signed
exports against the uploaded files. The existing development setting
`VERITAS_DOCUMENT_PDF_SAVED_WORDING_REVIEW=true` enables the comparisons.
[Local app receipt](evaluation-panels/pdf-saved-pictures-app-2026-11-17.json)
