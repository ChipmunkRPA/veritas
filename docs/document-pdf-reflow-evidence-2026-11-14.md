# Follow a changed sentence across PDF page breaks

Open **Compare with an earlier draft** and select **Include experimental
paired writing check** on a retained PDF analysis. The comparison can
match a changed sentence when it or its two unchanged neighbors cross
adjacent native text blocks or a page break. The earlier and analyzed
sentences retain their exact source offsets. The evidence view shows
the relevant source pages and links to highlighted word boxes in the
ordinary wording changes. The comparison PDF and JSON use the same
geometry. [Existing PDF sentence comparison](document-pdf-independent-anchor-2026-11-14.md)

The check still needs short English drafts with complete selectable
text, upright single-column pages and a verified reading order. Across
a page break, the previous body block must end near the bottom and the
next must begin near the top of a page of the same size. The blocks'
left margins and typical word-box heights must agree. The rule leaves
wide gaps, mismatched text sizes and ambiguous broken words outside the
extended mapping. Two unchanged neighbors must match uniquely in both
drafts. Existing sentence mappings are preserved when the extended
mapping adds coverage. The two writing responses are uncalibrated and
carry zero weight in the overall authorship assessment.
[PDF text extraction and coordinates](https://pymupdf.readthedocs.io/en/latest/app1.html)

The local option uses `VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW=true`
with the existing earlier-draft and paired writing settings.

## Reflowed source comparison

The prepared panel verified source hashes against the earlier
Gates Open Research and Wellcome Open Research body study. The
21 ordinary revisions are the actual dated publisher changes.
Two separately generated groups add 16 rewrites apiece. The
evaluation puts the exact retained prose into straightforward
single-column PDFs and places a page break inside the central sentence.
The page layouts are authored for the test. All 95 draft pairs reproduced
the exact source wording and reading order before the mapper ran.
[Original sources and terms](writing-journal-body-transfer-2026-11-11.md)

| Submitted pair | Old exact mapping | New exact mapping across both source pages |
| --- | ---: | ---: |
| Ordinary published revision | 0/21 | 21/21 |
| Generated revision, first family | 0/16 | 16/16 |
| Generated revision, second family | 0/16 | 16/16 |
| Unchanged wording with a break | 0/21 | 0/21 |
| Revision with an unsupported gap | 0/21 | 0/21 |

No mapped revision targeted the wrong sentence. The unchanged controls
also produced zero changes in the ordinary word comparison. These results
measure the location of edited wording in known layouts. Authorship
accuracy requires a separate evaluation.
[Numeric reflow receipt](evaluation-panels/document-pdf-reflow-geometry-2026-11-14.json)

## Local application verification

Three separate source works went through the local worker and comparison
endpoint after simulated USD 9.99 payment. No jobs started before payment.
Two writing responses completed in each comparison. All three reports
showed page 1 in the earlier version and pages 1 and 2 in the analyzed
version, with no shared increase at the paired review level. The verifier
replayed the source bytes and 32 word boxes across the three pairs. The
browser test shows both page locations, and the rendered PDFs keep the
linked source crops with their wording changes. No internal model names
appeared in these customer outputs.
[Local application receipt](evaluation-panels/document-pdf-reflow-app-2026-11-14.json)

## Independent replay

Verify a downloaded comparison JSON against the original two PDFs
using the existing source verifier. It recomputes the mapped context
and source page regions and checks the saved response arithmetic.

```sh
cd backend
.venv/bin/python -B -m scripts.review_document_revision_export comparison.json earlier.pdf analyzed.pdf
```

Recount the public reflow panel with standard-library Python.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.pdf_reflow_revision_transfer review \
  docs/evaluation-panels/document-pdf-reflow-geometry-2026-11-14.json
```

The native replay uses the pinned `scored/cases.json` from the earlier
study and the DejaVu Sans TTF whose hash appears in the receipt. All
prose and rendered documents stay local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.pdf_reflow_revision_transfer transport \
  "$source_cases" "$dejavu_ttf" "$transport_receipt"
backend/.venv/bin/python -B -m scripts.pdf_reflow_revision_transfer app \
  "$source_cases" "$dejavu_ttf" "$transport_receipt" "$smoke_test_directory"
```
