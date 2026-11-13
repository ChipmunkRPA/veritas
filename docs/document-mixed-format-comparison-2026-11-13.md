# Compare drafts across TXT, Word and PDF

In **Compare with an earlier draft**, an earlier TXT, DOCX or PDF can be
compared with a completed local analysis in one of the other formats. Each
side retains its own exact text offsets. Word changes include body paragraph
positions. PDF changes include page word boxes and, where available, the
original and highlighted source crops. The browser and supplemental PDF
label both source formats. The downloaded JSON can be replayed using both
original files.

Cross-format PDF comparison supports up to 12 upright pages with selectable
text in a clear reading order. Multi-page PDFs need a single column per page.
Word comparison covers top-level body paragraphs.
The text check locates changed words and punctuation, ignoring reflow and
spacing. Selectable page numbers and running headers are compared as text.
It uses no model and assigns no authorship confidence.

## Public validation

The replay rebuilds ten pairs of PLOS Biology excerpts from the pinned
publisher XML files. Each pair preserves the published wording in the earlier
excerpt and uses an ordinary later variant with one sentence drawn from a
different part of the same article. It writes TXT, DOCX and single-page PDF
renditions, verifies that their token sequences survive rendering, and
compares each of the six ordered cross-format combinations. These are
constructed editorial changes in previously studied articles. Their dates,
bylines, provenance and CC BY 4.0 terms are recorded in the existing source
replay. [Publisher reconstruction and attribution](writing-word-public-source-replay-2026-11-12.md)

- All **60 edited comparisons** reproduced the exact changed words and
  punctuation of the TXT comparison.
- All **60 unchanged renditions** returned zero wording changes.
- The replay checked both source hashes for every comparison, including 44
  Word paragraph locations and 44 PDF source previews across the edited
  comparisons.

An independent rebuild from freshly downloaded publisher XML produced a
byte-identical result. A local Checkout test exercised the paid worker and
the existing result. Payment confirmation queued one analysis. Comparing
with an earlier draft left that result intact. A Chrome test verified the
paragraph position, PDF crop and the exact JSON and PDF downloads. The
authenticated tests also checked that the export rejects a modified
source coordinate. The initial release declined multi-page cross-format PDFs. An
initial replay font lacked Greek glyphs used by the source articles. The
final replay requires both glyph coverage and exact token equality after PDF
rendering. [Validation receipt](evaluation-panels/document-mixed-format-comparison-2026-11-13.json)

## Multi-page validation

The same ten publisher source pairs were rendered as three-page PDFs and
compared with their TXT and Word renditions in both directions. All 40 edited
PDF comparisons recovered the same changed tokens as the TXT baseline. All
40 unchanged PDF comparisons returned zero wording changes. The other 40
TXT-to-Word and Word-to-TXT comparisons also replayed as expected. Sixteen
edited PDF comparisons contained changed words on two pages. All 812 PDF
word regions and 60 source crops replayed from the original drafts, and a
second clean run reproduced the result byte-for-byte.

The app test exercised an actual local analysis worker behind simulated USD
$9.99 Checkout and opened its completed result in Chrome. Both page 2 and
page 3 crops loaded, and the browser downloads matched the response. A large
optional writing detector was skipped for its memory budget in this run.
The separate paired writing-model check still requires same-format drafts
and, for PDFs, one upright page per draft. The multi-page comparison tests
also reject rotation, missing text, detected multiple columns, and more than
12 pages. Multi-page comparison declines PDFs with explicitly hidden or
fully transparent text. Covering graphics and low-contrast text can still
affect how the source crops look. The reading-order guard accepts a limited
left-margin spread of 36 points on each page. Centered headings or heavily
indented material may fall outside scope. [Multi-page receipt](evaluation-panels/document-mixed-format-multipage-2026-11-13.json)

## Reproduce

Run from the repository root with the backend dependencies. The scratch
directories hold the public XMLs and generated drafts locally. Do not submit
the articles as attachments to an evaluation issue. The JSON result contains
source references, hashes and counts.

```sh
draft_replay=$(mktemp -d)
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.replay_document_mixed_formats \
  "$draft_replay/publisher" "$draft_replay/drafts" --download > "$draft_replay/result.json"
```

For the three-page rerun, add `--multipage` and use a new output directory.

To verify a JSON export from the local app using the two original drafts:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_document_revision_export \
  "$export" "$earlier" "$analyzed"
```
