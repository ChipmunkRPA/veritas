# Copyable text behind solid covers in a PDF

Chipmunk Verify records when words are still copyable behind a solid
rectangle in the rendered page content. The evidence view, Standard PDF
and JSON export give the affected page and up to four locations in
unrotated PDF points. They omit the concealed words. The app excludes
the whole affected page from writing checks and withholds a
whole-document authorship assessment. Eligible text on other pages can
still be checked individually.

The check compares glyph positions with the copyable text and the order
of drawing operations. A later opaque rectangular fill must cover a
complete word. The app verifies that the final rendered region is a
uniform color consistent with the fill. The check records how many pages
it examined. It covers simple rectangular fills in page content.
Redaction annotations, image covers, complex graphics and split or
partially covered words need other checks. An empty finding does not
establish that a PDF is securely redacted. The observation carries zero
AI-authorship weight. [MuPDF text and painting sequence](https://pymupdf.readthedocs.io/en/latest/functions.html#Page.get_bboxlog)

## Local verification and replay

The test inputs use authored documents from PyMuPDF and ReportLab.
They include black and white covers, a cover over one word in a longer
line, rotated/cropped pages, translucent highlights, text drawn above
a rectangle, applied redaction and a review annotation. Local checks
compare legacy extraction with current extraction. The simulated paid
flow verifies that the app can publish page evidence even if all text
was excluded. The Standard PDF and JSON use the same recorded location.
The source verifier checks it against the submitted PDF and rejects
changed coordinates.

```sh
cd backend
.venv/bin/python -m pytest -o addopts='' -q \
  tests/test_document_pdf_covered_text.py \
  tests/test_document_pdf_replacement.py \
  tests/test_document_hybrid_pdf_coverage.py
```

Use the existing original-document verifier on the signed evidence
package and the exact submitted PDF to replay the source extraction.
[Independent testing instructions](independent-testing.md)
