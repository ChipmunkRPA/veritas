# Document formats and signed source records

Chipmunk Verify accepts Word (`.docx`), OpenDocument Text (`.odt`), PDF,
PowerPoint (`.pptx`), LibreOffice Impress (`.odp`), Excel (`.xlsx`),
OpenDocument Spreadsheet (`.ods`), CSV, TSV, EPUB, Rich Text (`.rtf`), plain
text, Markdown, HTML, XHTML, and email (`.eml`). These use the same local
writing pipeline and produce PDF and JSON reports. The document upload
limit is 25 MB in the standard configuration.

| Format | Writing examined | Source position in reports |
| --- | --- | --- |
| DOCX | Body paragraphs, tables, filled form controls, supported saved fields, and supported linked footnotes and endnotes in standard and Strict Open XML DOCX. Separate revision and visibility checks cover supported structures. | Word body paths, paragraph characters, available run coordinates, and the note's paragraph characters and reference |
| ODT | Body paragraphs, headings, lists, and tables outside hidden and conditionally displayed sections | Decoded body paragraph number and character range |
| PDF | Selectable text and supported local OCR | Page and region where mapped |
| PPTX | Editable text boxes and tables on visible slides | Slide in presentation order, text block, and decoded character range |
| ODP | Saved text in supported slide shapes and tables. Hidden slides, hidden layers and speaker notes are excluded. | Slide in the deck's stored order, text block, and decoded character range |
| XLSX | Literal text on visible worksheet tabs, rows and columns | Workbook sheet order, row, column and saved cell character range |
| ODS | Literal text on visible sheets, rows and columns. Linked sheets, formulas and calculated values are outside the check. | Workbook sheet order, row, column and saved cell character range |
| CSV and TSV | Text fields in UTF-8 or BOM-marked UTF-16 delimited files | Record, field and decoded character range |
| EPUB | XHTML text in the main reading order | Reading-order entry, text block, and decoded character range |
| RTF | Extracted writing from standard Rich Text files | Decoded text block number and character range |
| EML | The selected plain-text or HTML message body | Decoded text and supported HTML source positions |
| TXT | Decoded Unicode text | Decoded character offsets |
| Markdown | Text extracted from the markup | Paragraph and character position in the extracted text |
| HTML and XHTML | Text extracted from saved `.html`, `.htm`, or `.xhtml` markup | Original markup character ranges for uniquely mapped article passages, otherwise extracted-text positions |

ODS row and column positions follow the saved grid, including repeated or
hidden rows and columns before a finding. The writing check excludes
explicitly hidden groups and reads text saved in cells on locally stored
sheets. [OpenDocument spreadsheet schema](https://docs.oasis-open.org/office/OpenDocument/v1.3/OpenDocument-v1.3-part3-schema.html)

A local Chromium run completed Word, Excel, ODS, CSV and TSV writing
analyses after simulated USD 9.99 checkout. Each result displayed source
positions in the browser and exported both a PDF and JSON. All five signed
packages verified against their original files. A separate ODS saved by
LibreOffice reproduced the same four visible cell addresses and exact text
as its source workbook. Hidden sheets, rows and columns and a cached
formula stayed outside the extracted text.

XLSX also excludes cells with Excel's hide-value format applied directly.
Other display formatting, including inherited and conditional styles, can
differ from the saved writing. Existing signed reports replay using their
original extraction rules. [Excel's hide-value format](https://support.microsoft.com/en-us/excel/hide-or-display-cell-values)

For email, attachments can be submitted separately. Word reads supported
body and note text. ODT uses body text. Document packages are read locally
without opening Office, fetching linked content, or executing macros.

The writing check reads saved text in supported Word form controls and
skips placeholder text. A paragraph mixing inline controls and tracked
edits is outside the accepted-revision projection. Evidence for filled
controls preserves the Word body path and paragraph character range.
Word body text remains readable in exports without a default paragraph
style. The original-file verifier applies the extraction recorded when
the report was created. [Microsoft's Word control structure](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.sdtblock?view=openxml-3.0.1)

Word tables retain the saved row and column positions even when a cell is
wrapped in a form control, follows a placeholder, or starts after omitted
grid columns. Merged continuations and unsupported tracked table
structures are outside this check. The signed package records the
coordinate rule for original-file verification. [Microsoft's Word table grid reference](https://learn.microsoft.com/en-us/office/open-xml/word/working-with-wordprocessingml-tables) · [Word row controls](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.sdtrow?view=openxml-3.0.1)

Word passage locations include the saved label text within hyperlinks and
the exact offsets of the text that follows. The reader uses the stored text
and never opens the hyperlink destination. Signed reports record the
mapping rule so earlier evidence can replay with its original coordinates.

### Linked Word notes

The writing check reads supported saved footnote and endnote paragraphs
linked from ordinary visible Word body paragraphs. Its coverage records
how many eligible references led to readable note text. Findings can
identify the Word file's internal note ID, the note paragraph, its Unicode
character range, and the source body block that contains the reference.
The browser and PDF use the same positions as the JSON evidence. A signed
package can replay those positions against the original DOCX. Earlier
reports use the extraction rules recorded when they were created.

Links in tracked documents, hidden references, duplicate note IDs within
a note type, placeholders, fields, unlinked notes, and unsupported note
layouts are outside this check. Earlier-draft comparison uses Word body
text. [Word document parts](https://learn.microsoft.com/en-us/office/open-xml/word/structure-of-a-wordprocessingml-document) ·
[Word note types](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.footnoteendnotetype.type?view=openxml-3.0.1)

A Chromium test submitted standard and Strict DOCX, a file with readable
text only in footnotes, and a file with readable text only in endnotes.
Each ran after simulated checkout. The test verified note character
ranges and reference locations in the browser, PDF, and JSON, then
verified the signed packages against their original files. Two earlier
signed reports also replayed with their original reading rules.
[Word notes checkout and export test](evaluation-panels/document-word-notes-app-2026-11-18.json)

Word's **Strict Open XML Document** option also saves a `.docx` file.
Upload it directly for analysis or an earlier-draft comparison. The reader
maps the Strict body and style markup for text extraction and retains the
submitted bytes for Content Credentials, hashing, and original-file review.
[Microsoft's Word format reference](https://learn.microsoft.com/en-us/office/compatibility/office-file-format-reference) ·
[Open XML SDK Strict DOCX test fixture](https://github.com/dotnet/Open-XML-SDK/blob/431ab05cf160248cc3885a4a766026d4f8243792/test/DocumentFormat.OpenXml.Tests.Assets/assets/TestFiles/Strict01.docx)

New analyses also read the saved display text of supported simple fields in
ordinary body paragraphs, including dates, sequence numbers, and cross-references.
The evidence page, PDF and JSON show the retained values with zero-based Word
body and field indices and character ranges. These are the values stored
in the submitted file. Formulas, external references, nested fields, table fields,
and fields in tracked documents stay outside this check. The original-file
verifier follows the extraction policy recorded in each report.
[Microsoft's SimpleField reference](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.simplefield?view=openxml-3.0.1)
· [Open XML SDK field fixture](https://github.com/dotnet/Open-XML-SDK/blob/431ab05cf160248cc3885a4a766026d4f8243792/test/DocumentFormat.OpenXml.Tests.Assets/assets/TestDataStorage/v2FxTestFiles/wordprocessing/field/simple-field.docx)

New ODT and RTF reports show Unicode character ranges before the analysis
normalizes whitespace. ODT numbers readable body paragraphs in reading order.
The ODT reader leaves out hidden sections and sections whose display depends
on a condition. These sections also stay outside checkout's readable word
count and earlier-draft comparisons. Signed packages and draft exports
retain the extraction policy used for their source positions.
[OpenDocument 1.3 section visibility](https://docs.oasis-open.org/office/OpenDocument/v1.3/os/part3-schema/OpenDocument-v1.3-os-part3-schema.html#attribute-text_display)

RTF numbers decoded text blocks, including blocks separated by soft line
breaks. The web view, PDF and JSON use the same ranges. The signed package can
replay them against the original file.

Export Google Docs, Apple Pages, and older Word (`.doc`) files as DOCX or
PDF before uploading. Export Google Slides and Keynote as PowerPoint or PDF.
Export Outlook (`.msg`) files as EML or PDF. Keep the
original file if its native provenance matters. A PDF export can include
Word notes outside the native check, text boxes, headers and footers in
their rendered layout.
For older, macro-enabled or OpenDocument spreadsheets, save the sheets with
writing as macro-free XLSX, CSV or PDF.
Before creating a payment, checkout checks that writing files have at least
15 readable words and fit within the configured extracted-text limit, normally
120,000 characters. Some Word files can still produce native revision evidence
with fewer words. PDF checkout validates the file structure and allows scanned
pages for local OCR during the paid analysis. Text recognition may yield too
little writing to assess.
Rich Text files with binary data should be saved as DOCX or PDF.
The local batch tool also accepts native DOCX and ODT packages.

PowerPoint findings use the order shown in the presentation. Hidden slides,
hidden groups, speaker notes, charts, SmartArt, and image text are outside the
writing check. Text in hidden groups does not count toward checkout's readable
word minimum. Earlier signed reports use the visibility policy recorded at
the time of analysis.

Impress findings retain the slide number even when an earlier slide is
hidden. The reader uses saved slide text and drawing-page visibility. It
skips hidden or print-only layers, placeholders, and speaker notes. Animation,
custom show order, image text, embedded objects, and master-slide content are
outside this check. The report records offsets in the decoded text block, and
the signed package can reproduce them from the original ODP. A file with
fewer than 15 readable words in scope cannot start checkout.
[OpenDocument slide visibility](https://docs.oasis-open.org/office/OpenDocument/v1.3/os/part3-schema/OpenDocument-v1.3-os-part3-schema.html)

A local Chromium test completed an ODP upload and analysis after simulated
USD 9.99 checkout. The browser, PDF and JSON matched all ten text locations
on slides 1 and 3. The hidden slide at position 2, hidden layer, placeholder,
and speaker notes stayed outside the writing input. The signed package
verified against the original. Three additional LibreOffice regression files
confirmed text-box and shape extraction and rejection before checkout when
there were too few readable words.
[LibreOffice Impress regression fixtures](https://github.com/LibreOffice/core/tree/master/sd/qa/unit/data/odp)

EPUB findings use the original main reading order, skipping navigation and
optional entries. EPUB styling, image text, and SVG chapters are outside the
writing check. Slide and EPUB entry numbers in the web view and PDF are
one-based. JSON includes zero-based part and text-block indices and
half-open Unicode character ranges in the decoded text block.
The original-file verifier reconstructs these positions from the submission.
The writing check requires at least 15 readable words. Longer ebooks can
be submitted as excerpts within the configured writing limit. Checkout
rejects unreadable packages and protected EPUB chapter text before payment.
[Microsoft's slide order and text reference](https://learn.microsoft.com/en-us/office/open-xml/presentation/how-to-get-all-the-text-in-all-slides-in-a-presentation)
· [Microsoft's hidden drawing property](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.nonvisualdrawingproperties?view=openxml-3.0.1)
· [W3C EPUB 3.3 package specification](https://www.w3.org/TR/epub-33/#sec-pkg-spine)

An isolated local browser run accepted a 224-word fixture in all ten formats.
Ten simulated USD 9.99 payments preceded analysis. Each case produced PDF and
JSON exports and a signed package that verified against the original. The
PowerPoint and EPUB cases each preserved six text locations, including the
same decoded character ranges in PDF and JSON. An excerpt from the W3C sample
book yielded 54 mapped text blocks from five reading-order entries. The full
sample exceeded the configured writing limit.
[Local browser and export receipt](evaluation-panels/document-format-upload-app-2026-11-18.json)
· [W3C EPUB samples](https://idpf.github.io/epub3-samples/30/samples.html)

[Saved HTML evidence](writing-saved-html-scope-2026-11-18.md) preserves original
character ranges for uniquely mapped article passages without changing the text
sent to the writing checks. XHTML uses the same handling for analysis, signed
source replay and earlier-draft comparison.

A follow-up local browser replay covered DOCX, PPTX, EPUB and XHTML under
the revised checkout rules. Four simulated USD 9.99 payments preceded
analysis. Each case produced PDF and JSON exports with matching positions
and a signed package that verified against the original. Word preserved 17
exact positions, including two in a table. The isolated replay used generated
fixtures and the available CPU checks, with the optional large model pair
disabled. [Prepaid document position receipt](evaluation-panels/document-format-prepaid-positions-2026-11-18.json)

The next local replay completed analysis for all 13 advertised extensions,
including both Markdown extensions and all three HTML extensions. Every
fixture produced matching PDF and JSON evidence and a signed package that
verified against the original. All 13 jobs began after simulated checkout.
Separate scanned and hybrid PDF tests verified the recognized text and its
page positions against the original file.

The follow-up Chromium run submitted all 13 extensions through the upload
form, plus a Strict Word file and a scanned PDF. No analysis started until
its simulated USD 9.99 payment cleared. Both Word cases preserved 17 source
ranges, including two in a table, and kept the saved hyperlink positions.
PowerPoint preserved ten visible positions and left the hidden group out
of the analysis and both exports. All 15 signed packages verified against
their original files. The run used the local CPU writing checks with the
optional large model pair disabled.
[Browser, export, and original-file receipt](evaluation-panels/document-format-browser-checkout-2026-11-18.json)

After the Word table and OpenDocument visibility fixes, a local Chromium run
completed the document upload and simulated USD 9.99 checkout for all 16
advertised extensions and a separate Strict DOCX fixture. Every original
verified against its signed package, and the browser returned the same PDF
and JSON evidence. Both Word variants retained five evidence ranges from
filled table controls. The ODT fixture excluded a saved hidden section.
The first browser sweep failed to load one TSV evidence page. A fresh
session verified all 17 evidence pages and both exports without HTTP errors.
[Local format and evidence receipt](evaluation-panels/document-format-table-odt-2026-11-18.json)

### Spreadsheet writing

Excel writing checks read literal shared and inline text. Formula results,
numeric cells, hidden sheets, rows and columns, comments, charts, drawings
and images are outside this check. The reader preserves the saved workbook
order even when some sheets are hidden. It opens no links and runs no formulas.
CSV and TSV checks read text fields in consecutive records, including quoted
line breaks within a field. Headers and blank records count toward row numbers.
Numeric, link-only and formula-looking fields are skipped. Use comma-separated
CSV or tab-separated TSV for predictable import.

The web view and PDF show one-based sheet, row and column labels. JSON
records `part_index` as a zero-based workbook index for Excel, one-based rows
and columns for all three formats, and decoded Unicode character ranges.
The signed package can replay those ranges against the submitted original.
The writing results concern only the passages the app could read.
[Microsoft's shared text reference](https://learn.microsoft.com/en-us/office/open-xml/spreadsheet/working-with-the-shared-string-table)
· [CSV syntax reference](https://www.rfc-editor.org/rfc/rfc4180)

A local Chromium test submitted DOCX, XLSX, CSV and TSV through four simulated
USD 9.99 checkouts. Zero jobs started before payment. All four produced a PDF,
matching JSON and a signed package that verified against its original. Excel,
CSV and TSV each retained 18 exact cell locations. Text from hidden Excel
cells and a cached formula stayed out of the evidence. An XLSX written by
LibreOffice separately matched the source CSV at all six mapped cells.
The browser test used local CPU writing checks with the optional large model
pair disabled.

The local app also compares two saved HTML articles when an earlier copy is
available. It marks wording changes in both decoded sources and gives the
number of compared text blocks. [Earlier HTML drafts](writing-html-draft-2026-11-18.md)

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

The November 18 replay covered all eight formats in the table, including
Rich Text. Both writing checks finished for each original after simulated
USD 9.99 checkout. Each PDF rendered, each JSON export matched the app's
evidence, and each signed package verified against its original file. A
finding inside a filled Word field retained body path `1 / control / 0`
and its paragraph character positions in the report and JSON.

A follow-up replay tested all eight formats plus a separate scanned PDF.
All nine opened in the browser after nine simulated USD 9.99 payments, and
each signed package verified against the original. OCR completed on both
scanned pages. The Word fixture had no default paragraph style and included a
table. Nine saved character ranges each for Word, OpenDocument and Rich Text
matched their decoded text and appeared in the PDF. The same location
labels appeared in the browser. A header-only DOCX, a blank ODT, an invalid
RTF and a blank RTF were rejected before checkout.

The Strict DOCX compatibility replay covered 11 files across the eight
formats, including a scanned PDF and two Strict Word files. Every file
completed its applicable checks after simulated checkout, produced PDF and
JSON exports, and verified its signed package against the original. The
controlled Strict file preserved nine exact paragraph ranges. The Open XML
SDK fixture produced exact repeated-text locations and displayed a coverage
notice for unsupported tracked revisions. A header-only Strict file was
rejected before checkout. Local regression tests also check draft comparison
offsets and reject external Word body relationships and macro-enabled
packages disguised with a DOCX extension.

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

Rich Text can also be compared with an earlier draft. See the [Rich Text
format positions and historical revision tests](writing-rtf-draft-2026-11-18.md).

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
