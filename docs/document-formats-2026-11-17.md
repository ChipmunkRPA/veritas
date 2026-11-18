# Document formats and signed source records

Chipmunk Verify accepts Word (`.docx`), OpenDocument Text (`.odt`), PDF,
plain text, Markdown, HTML, and email (`.eml`). These use the same local
writing pipeline and produce PDF and JSON reports. The document upload
limit is 25 MB in the standard configuration.

| Format | Writing examined | Source position in reports |
| --- | --- | --- |
| DOCX | Body paragraphs and tables, with separate supported Word revision and visibility evidence | Word paragraph and available run coordinates |
| ODT | Body paragraphs, headings, lists, and tables | Paragraph and character position in the extracted text |
| PDF | Selectable text and supported local OCR | Page and region where mapped |
| EML | The selected plain-text or HTML message body | Decoded text and supported HTML source positions |
| TXT | Decoded Unicode text | Decoded character offsets |
| Markdown and HTML | Text extracted from the markup | Paragraph and character position in the extracted text |

For email, attachments can be submitted separately. Word and ODT use body
text for writing checks. Document packages are read locally without opening
Office, fetching linked content, or executing macros.

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
accepted each format and displayed its result. The ODT, Markdown and HTML
evidence packages also verified against their original files. All seven jobs
followed simulated payment.

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
