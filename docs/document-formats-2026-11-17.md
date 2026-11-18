# Document formats and signed source records

Chipmunk Verify accepts Word (`.docx`), OpenDocument Text (`.odt`), PDF,
Rich Text (`.rtf`), plain text, Markdown, HTML, and email (`.eml`). These use the same local
writing pipeline and produce PDF and JSON reports. The document upload
limit is 25 MB in the standard configuration.

| Format | Writing examined | Source position in reports |
| --- | --- | --- |
| DOCX | Body paragraphs and tables, with separate supported Word revision and visibility evidence | Word paragraph and available run coordinates |
| ODT | Body paragraphs, headings, lists, and tables | Paragraph and character position in the extracted text |
| PDF | Selectable text and supported local OCR | Page and region where mapped |
| RTF | Extracted writing from standard Rich Text files | Paragraph and character position in the extracted text |
| EML | The selected plain-text or HTML message body | Decoded text and supported HTML source positions |
| TXT | Decoded Unicode text | Decoded character offsets |
| Markdown and HTML | Text extracted from the markup | Paragraph and character position in the extracted text |

For email, attachments can be submitted separately. Word and ODT use body
text for writing checks. Document packages are read locally without opening
Office, fetching linked content, or executing macros.

Export Google Docs, Apple Pages, and older Word (`.doc`) files as DOCX or
PDF before uploading. Export Outlook (`.msg`) files as EML or PDF. Keep the
original file if its native provenance matters.
Checkout rejects Word and ODT files without readable body text and checks
the text extracted from RTF, Markdown, HTML and HTML email bodies before
starting payment. Scanned documents can be uploaded as PDF for local OCR.
Rich Text files with binary data should be saved as DOCX or PDF.
The local batch tool also accepts native DOCX and ODT packages.

The local development app can check ODT for earlier published wording when
automatic review and the GovReport archive are enabled. When a dated U.S. GAO
highlight qualifies, the report pairs each difference with the earlier
wording and records its character range within the extracted body paragraph.
The web view, PDF and JSON preserve these ranges. Offline verification
recomputes them from the original ODT and retained transcript. The comparison
contributes zero weight to the AI assessment.

In a local format check, all three edited ODT copies of archived GAO
highlights identified the exact replacement. All three unchanged copies
showed zero wording differences, and the original-file replay passed for
all six.

The local app replay used the same 224-word sample in all seven formats.
Three writing checks finished per format and every result remained
inconclusive. All PDF and JSON exports opened successfully. The browser
accepted each format and displayed its result. A later local replay verified
the evidence packages against the originals in all seven formats. The Word
replay checked exact body paragraph locations, and the PDF replay checked
the mapped page images. All seven jobs followed simulated payment.

## Earlier OpenDocument drafts

After a document analysis, **Compare with an earlier draft** accepts ODT
alongside Word DOCX, plain text, and PDFs with selectable text. Each version
can be up to 1 MiB. For ODT, the comparison checks extracted body paragraphs,
headings, lists and tables. The browser and comparison PDF identify the same
paragraph offsets and mark the same words as the JSON export. These ranges
count Unicode characters in the extracted text and within each listed
paragraph. ODT offsets use extracted reading order. Word and PDF retain
their own native positions.

The comparison uses exact unchanged sentences to constrain word matching.
When a uniquely matched sentence moves, its existing words stay unmarked.
An additional copy can still appear as an insertion. The comparison
identifies changes between the two supplied files and carries zero weight in
the automated AI assessment.

| Replay with the earlier draft supplied | Rewritten sentences containing a marked word | Human wording changes located | Restyled originals with wording marks |
| --- | ---: | ---: | ---: |
| Earlier GAO paraphrase panel | 148/149 | — | 0/55 |
| Fresh BEA and Census paraphrase panel | 33/33 | 17/17 | 0/22 |

Both ODT-to-ODT and TXT-to-ODT tests gave the same submitted token ranges.
In the GAO panel, 123 marked tokens outside the known rewrite positions
came from four additional copies of earlier sentences. In the fresh
panel every marked token in an admitted AI rewrite stayed within a
labeled rewritten sentence. These counts measure version comparison
with both originals. The [standalone writing trial](writing-economic-paraphrases-2026-11-17.md)
uses a different localization criterion and receives only the submitted file.
[Earlier panel replay](evaluation-panels/writing-odt-draft-gao-2026-11-17.json.gz) ·
[Fresh panel replay](evaluation-panels/writing-odt-draft-economic-2026-11-17.json.gz)

The local paid replay purchased one simulated USD 9.99 analysis and queued
zero jobs before payment. The completed analysis accepted the earlier ODT
in Chrome. The browser download matched the API JSON and the comparison PDF
showed the same wording and offsets. Offline replay verified both original
hashes and 68 marked token positions across both versions. The comparison
created zero additional analysis jobs and preserved the assessment.
[Local checkout and browser receipt](evaluation-panels/writing-odt-draft-app-2026-11-17.json)

An offline reviewer can verify the downloaded comparison against the two
actual files with `scripts.review_document_revision_export`. Use
`scripts.writing_odt_draft_transfer` with the original result plan from
either linked writing study to reproduce the aggregate range counts. Both
commands take file paths and provide usage through `--help`.

## Content Credentials

The offline verifier checks DOCX and ODT for the standard embedded C2PA
store. It requires cryptographic validation of the active collection hash,
which covers the document's archive entries and central directory. To skip
writing checks, the signature must also be trusted on the bundled official
list, explicitly declare AI involvement, and have zero validation failures.
Supporting evidence identifies the signer and gives the validation result.
The verifier reads the original submitted byte stream. The
[supported PDF signature path](pdf-content-credentials-2026-11-13.md) remains
available through its existing option. Pasted text, editable metadata, and
credentials on attachments cannot authenticate the document body.
[C2PA ZIP-format specification](https://spec.c2pa.org/specifications/specifications/2.5/specs/ContentCredentials.html#_embedding_manifests_into_zip_based_formats) ·
[ZIP support in the SDK](https://github.com/contentauth/c2pa-rs/pull/499)

The regression tests sign an authored DOCX and ODT with a temporary test
certificate. Both are checked through simulated $9.99 payment, the analysis
worker, PDF, JSON and offline verification of the evidence package. The
writing models receive no work on the accepted signed files. Matching
unsigned documents and documents whose contents changed after signing
continue to the writing checks. Production trust uses the existing C2PA list.
The test certificates establish format compatibility. Public vendor-issued
DOCX and ODT exports remain unverified in this review.

The tested runtime uses `c2pa-python 0.39.0` (SDK 0.92.0), which reproduced
the readings on all 21 earlier [public provider files](public-content-credentials-2026-11-17.md).
The application handles native Word and ODT files as supplied. For older Word
`.doc` files, save a copy as DOCX or PDF before uploading.

## Independent replay

From the project root, run the offline reviewer on the original document
with the backend environment. The receipt identifies the exact file bytes,
the SDK version, the trust snapshot, the package binding, and how the
credential gate would handle the file. The optional `--expect-sha256` argument
checks that two reviewers received identical bytes. You can share the receipt
with a review after checking certificate fields for personal information.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_public_content_credentials \
  --file original.docx
```

Two upstream C2PA SDK test documents can serve as unsigned controls. For both,
SDK 0.92.0 reports `state: absent`, `document_binding: false`, and
`pipeline_action: continue_detection`.

| File | SHA-256 | Upstream download |
| --- | --- | --- |
| `sample1.docx` | `a83ac04b948ad09d547c69db35012de116ffd0cf0d8abe163baba05ad8176342` | [DOCX fixture](https://raw.githubusercontent.com/contentauth/c2pa-rs/c2pa-v0.92.0/sdk/tests/fixtures/sample1.docx) |
| `sample1.odt` | `630749f3a2bf55740b901f86c3b3ca79b6072dc9d461a2aec1e471f9088e8542` | [ODT fixture](https://raw.githubusercontent.com/contentauth/c2pa-rs/c2pa-v0.92.0/sdk/tests/fixtures/sample1.odt) |

Run the signed, unsigned, and changed-document controls locally with
`backend/.venv/bin/python -B -m pytest -q backend/tests/test_office_credentials.py`.
They generate fresh test certificates and exercise both the receipt command's
verification logic and the app's report flow with simulated payment.
