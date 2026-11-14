# PDF copyable text and glyph text

Chipmunk Verify compares two extractions of each PDF page. One reads the
copyable text, including PDF replacement text. The other reads the glyph
text without replacements. A difference limits whole-document writing
assessment. It carries zero AI-authorship weight.
[PyMuPDF extraction flags](https://pymupdf.readthedocs.io/en/latest/vars.html#text-ignore-actualtext)

The comparison ignores whitespace, Unicode format controls, and equivalent
ligature or accent encodings. It keeps number, symbol, and word changes.
The report shows the checked page count and up to four locations on each
different page, in unrotated PDF points. Inserted text can lack a glyph
location. Large or ambiguous changes may retain only the page number.
The browser, PDF, and JSON use the same evidence record. Original-file replay
recreates it from the submitted PDF without a writing model. This comparison
uses the PDF text representations. It does not read page pixels or identify a
provider.

## Local validation

The authored test compared seven variants against a PDF with the same
rendered page pixels. It found all four content changes (an amount, an
omission, an insertion, and a long replacement) and reported no difference
for the three unmodified, matching, or spacing-only controls. Another test
retained a box on rotated page 14, rejected a modified evidence location,
and checked the same finding in the generated report and JSON. A simulated
prepaid analysis exercised the actual worker path.

The six published PDF Association text-technique fixtures all completed the
comparison. The first five produced no copyable-text discrepancy. PDF106,
which uses replacement text to correct an OCR layer, produced one page-level
difference with incomplete locations and also registered invisible text.
That is a real difference with a documented accessibility purpose. These
fixtures establish that the check reports that ambiguity. They do not measure
the frequency of such discrepancies in customer PDFs.
[PDF Association text fixtures](https://github.com/pdf-association/techniques-for-accessible-pdf/tree/main/fundamentals/2-text)
· [PDF106 and the OCR correction](https://pdfa.org/techniques-for-accessible-pdf/actualtext-provides-correct-extractable-characters-in-place-of-ocr-errors/UA1_Tpdf-G2_06/)

| Publisher fixture | Git blob | Copyable-text difference |
| --- | --- | --- |
| G2_01 | `27db26802f50e83e0e9ac5c78dd5539d22770db0` | None |
| G2_02 | `bc7854333f6baf80478bd7e47882db381e8caa46` | None |
| G2_03 | `261170bef76fb4c3f4d863d073e128f6538c02d2` | None |
| G2_04 | `951ef674f5145b73481e298db261b1d1f9653e56` | None |
| G2_05 | `1142dc341e77af1f33bee9925bbec9e602ee80a7` | None |
| G2_06 | `25a22c163cb7ad928b21365171747bc1da2ae8b0` | Page 1 |

## Replay

Run from `backend` with the tested local dependencies.

```sh
python -m pytest tests/test_document_pdf_replacement.py
```

Download each fixture from the PDF Association repository using its blob
above. For example, to check PDF106 on macOS:

```sh
gh api repos/pdf-association/techniques-for-accessible-pdf/git/blobs/25a22c163cb7ad928b21365171747bc1da2ae8b0 \
  --jq .content | base64 -D > /tmp/pdf106.pdf
python -c 'from pathlib import Path; from app.detectors.text.extract import extract_pdf; print(extract_pdf(Path("/tmp/pdf106.pdf")).meta["pdf_text_visibility"])'
```
