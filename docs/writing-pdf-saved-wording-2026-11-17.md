# Wording changes in a PDF’s retained saves

Chipmunk Verify can compare the submitted PDF with an earlier version stored
within the same upload. The normal evidence view identifies added, replaced,
or removed wording and shows an unmarked and a highlighted crop of the
current text. The PDF report and JSON record the matching previous and
current page positions. Earlier wording stays in the original file.
Reports preserve its location and the hash of the retained revision,
including when a later save applied a redaction.

## What the check measures

Some PDF applications append new objects and a cross-reference section
when saving. The recorded `/Prev` link points to an earlier section.
Chipmunk Verify follows that link through the stored save boundaries and
checks the version count and the SHA-256 of each identified byte prefix.
It compares the nearest retained version with a demonstrated change to the
submitted version. A later comment or metadata change can therefore sit
between the two compared saves. Signature metadata, form field contents, and
annotations are outside the comparison of native page wording. The finding
records save order and page locations and carries zero AI confidence weight.
[PDF Association format reference](https://pdfa.org/download-area/cheat-sheets/Basics.pdf)
· [PDF Association presentation on incremental saves](https://pdfa.org/presentation/in-defense-of-the-incremental-save/)
· [PyMuPDF document API](https://pymupdf.readthedocs.io/en/latest/document.html#Document.version_count)

Native pages are matched by retained page identity. The check uses word
positions on the unrotated page and excludes unsupported text layers and
pages without reliable matching text. It records missing page matches and
changes in text order separately. It accepts classic cross-reference tables
and cross-reference streams. It checks up to five saves of up to 24 pages
from a PDF no larger than 24 MiB. A full rewrite can discard earlier bytes.
The check distinguishes an unavailable comparison from one with agreeing
page wording. No new model weights are used.

## Held-out evaluation

Six Federal Register publications supply the development set. The held set
uses 24 documents from three other issues, with no overlapping source
documents. The deterministic working copies retain the first publisher page
and rebuild a second page from its body words so that the exact edited word
position is known. Half use a cross-reference stream. Eight have a rotated
working page. The untouched publisher files supply an additional control
with the original government signing updates. The benchmark checks their
native page text, without relying on signature trust as an edit label.
[Official Federal Register archive](https://www.govinfo.gov/help/fr)
· [GPO signature workflow](https://www.govinfo.gov/about/authentication)

| Held PDF case | Result |
|---|---:|
| Staged replacement, insertion, or deletion located on the correct page | 72/72 |
| Staged replacement followed by an ordinary comment save located | 24/24 |
| Added page located | 24/24 |
| Metadata, comment-only, or form saves with a wording flag | 0/72 |
| Unmodified publisher files with a wording flag | 0/22 compared, 2 over the page limit |
| Reordered text, requiring an unassessed wording status | 24/24 |
| Edited full rewrites with only one save retained | 24/24 |

These are 264 held deliveries with 24 underlying publications. The edits are
controlled tests. The results describe detection of retained file changes and
cannot estimate authorship accuracy or accuracy on PDFs rewritten elsewhere.
[Exact held results](evaluation-panels/pdf-saved-wording-held-2026-11-17.json.gz)
· [Development results](evaluation-panels/pdf-saved-wording-development-2026-11-17.json.gz)

## Local app and reproducible check

Enable `VERITAS_DOCUMENT_PDF_SAVED_WORDING_REVIEW=true` on the development
backend. Open **Changes across saved PDF versions** in a completed
document analysis. A verified asset-bound AI declaration continues to end
the automatic checks early. In the simulated $9.99 flow, payment confirmation
precedes analysis. The replay verifies browser text, exact source positions,
current crop pixels in the PDF, and the corresponding JSON in a signed
package. An applied redaction test confirms that the retained private token
is absent from the public output. [Local app receipt](evaluation-panels/pdf-saved-wording-app-2026-11-17.json)

The regular writing pipeline timed out on the 7,793-word publisher file
`2025-10477`. The saved-version comparison completed independently on that
file. The repeat app run used a 906-word signed publisher control.

The public source roster lists the original URLs and SHA-256 values. Download
each listed PDF to `publisher/<name>.pdf` and use the same renderer version
recorded in the benchmark. The held replay verifies 264 inputs and their
parent hashes, checks 526 retained-prefix hashes, and confirms 168 earlier
and later word boxes with Poppler. The released reader reproduced the frozen
results after the portable binding was separated from the PDF reader.
[Source roster](evaluation-panels/pdf-saved-wording-sources-2026-11-17.json)
· [Development readback](evaluation-panels/pdf-saved-wording-development-audit-2026-11-17.json)
· [Held readback](evaluation-panels/pdf-saved-wording-held-audit-2026-11-17.json)

```sh
export PYTHONPATH=backend
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
receipt="$panels/pdf-saved-wording-held-2026-11-17.json.gz"
roster="$panels/pdf-saved-wording-sources-2026-11-17.json"
python3 -B -S -m scripts.pdf_saved_wording_panel review "$receipt"
"$runner" -B -m scripts.pdf_saved_wording_panel prepare "$publisher" "$roster" confirmation "$run/held"
"$runner" -B -m scripts.verify_pdf_saved_wording_panel "$receipt" "$run/held" "$run/audit.json"
npm --prefix frontend run build
"$runner" -B -m scripts.pdf_saved_wording_app "$run/held" "$run/app"
```

Use `development` and its receipt for the six-source replay. Put Poppler’s
`pdftotext` on `PATH` for the coordinate check. Working copies and publisher
PDFs stay in the local replay folder. For an individual signed package, run
`python -m app.cli verify report.zip --public-key signer.pem --source-document upload.pdf`
to match the locally supplied original against the saved-version evidence.
