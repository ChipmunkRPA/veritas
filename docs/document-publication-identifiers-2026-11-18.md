# Publication identifier evidence

Chipmunk Verify compares the final digit of a clearly labeled ISBN-10,
ISBN-13, or ISSN with the value calculated from the preceding digits.
This can surface a typing or transcription error in a single document.
The report identifies the printed value, the calculated digit, and the
corresponding original text range or PDF boxes. The check carries zero
weight in the AI assessment. An identifier can have a matching check
digit even when it was never assigned or appears beside the wrong title.
[ISBN agency reference](https://www.isbn-international.org/content/what-isbn/10) ·
[Library of Congress ISSN calculation](https://www.loc.gov/issn/basics/basics-checkdigit.html)

The check reads the mapped TXT body, ordinary Word body paragraphs,
selectable PDF text, the mapped EML body, uniquely mapped article
paragraphs in HTML, and the decoded ODT or RTF text blocks. It skips
explicitly canceled or invalid numbers, ambiguous continuations,
quoted email, and PDF pages where the visibility check identified
hidden, covered, or replaced text. The check stays local and makes no
publisher or identifier registry requests. [Library of Congress format for canceled and invalid identifiers](https://www.loc.gov/marc/bibliographic/bd020.html)

## What the local tests showed

The initial group contained seventeen prospectively selected public PDFs.
Nine complete sources could be examined within the byte and page limits.
Some sources needed a separately replayed original page to demonstrate
the native location. The initial parser missed the first value when an
ISBN-10 and ISBN-13 were explicitly labeled on adjacent lines. With
that boundary rule corrected, the group contained twelve mapped values,
including one printed discrepancy. Page 47 of a CDC Stacks metadata
example prints `92-95055-02-5` in an illustrative field. Its preceding
digits give `0` as the check digit. The adjacent ISBN-13 value matches
its own check digit. A render from the source PDF confirmed the visible
text. [CDC Stacks metadata example, page 47](https://stacks.cdc.gov/view/cdc/104571/cdc_104571_DS_EXCEL.pdf#page=47)

A separate confirmation used five official UK government and CDC PDFs
with the corrected rule frozen. Four were within the page limit and
contained six mapped values with matching check digits. Across both
groups, thirteen source ISBNs and ISSNs with matching digits also
supplied Word test cases with one changed last digit. All thirteen
changed values were found at their exact original positions. The
thirty-nine unchanged, unlabeled or explicitly invalid controls
contained no newly reported disagreement.
[Office of the Public Guardian report](https://assets.publishing.service.gov.uk/media/66a1092ffc8e12ac3edb03ff/opg-annual-report-and-accounts-2023-to-2024.pdf) ·
[Building Digital UK report](https://assets.publishing.service.gov.uk/media/6759c4004cbda57cacd34719/BDUK_Annual_Reports-Accounts_2024_-_Certified_copy.pdf) ·
[CDC recommendations report](https://stacks.cdc.gov/view/cdc/177225/) ·
[CDC surveillance report](https://stacks.cdc.gov/view/cdc/255816)

The benchmark measures checksum and position accuracy only. It gives
no estimate of how often AI-generated citations contain invalid digits
or how much this check changes citation verification as a whole. PDF
numbers with unreadable native digit encodings are left unchecked.
[Numeric evaluation receipt](evaluation-panels/document-publication-identifiers-2026-11-18.json.gz)

## Local app verification

Eight simulated Stripe checkouts covered DOCX, Strict DOCX, ODT, RTF,
HTML, EML, TXT, and PDF. The application started zero analyses before
payment and eight after the test webhook. Each case matched the exact
identifier and expected digit across the browser, JSON, and rendered
PDF. The signed package reproduced the original file's positions using
the local public key. The final TXT, PDF, and Strict DOCX checks ran with
local evidence only. [Local app verification receipt](evaluation-panels/document-publication-identifier-app-2026-11-18.json)

Seven app fixtures use project-written prose and a one-digit variation
of the ISBN from the Office of the Public Guardian's annual report.
The report identifies its terms as Open Government Licence v3.0 except
where otherwise stated. The PDF fixture preserves one original page
from the public CDC Stacks metadata example. The public receipts
contain identifier values, positions, hashes, and check results. The
source documents remain with their publishers under each source's own
terms. [Office of the Public Guardian report and licence](https://www.gov.uk/government/publications/opg-annual-report-and-accounts-2023-to-2024/opg-annual-report-and-accounts-2023-to-2024-web-version) ·
[CDC Stacks content and copyright](https://stacks.cdc.gov/Content%20and%20Copyright)

Run the repository's local regression with
`backend/.venv/bin/python -m pytest backend/tests/test_document_publication_ids.py`.
It exercises original positions in seven document formats, including an
HTML identifier split across a formatting tag and a quoted identifier
excluded from an email.
