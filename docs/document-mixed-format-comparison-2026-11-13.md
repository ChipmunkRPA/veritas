# Compare drafts across TXT, Word and PDF

In **Compare with an earlier draft**, an earlier TXT, DOCX or PDF can be
compared with a completed local analysis in one of the other formats. Each
side retains its own exact text offsets. Word changes include body paragraph
positions. PDF changes include page word boxes and, where available, the
original and highlighted source crops. The browser and supplemental PDF
label both source formats. The downloaded JSON can be replayed using both
original files.

Cross-format PDF comparison requires one upright page with selectable text
in a clear reading order. Word comparison covers top-level body paragraphs.
The text check locates changed words and punctuation, ignoring reflow and
spacing. It uses no model and assigns no authorship confidence.

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
source coordinate, and that a two-page cross-format PDF is declined. An
initial replay font lacked Greek glyphs used by the source articles. The
final replay requires both glyph coverage and exact token equality after PDF
rendering. [Validation receipt](evaluation-panels/document-mixed-format-comparison-2026-11-13.json)

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

To verify a JSON export from the local app using the two original drafts:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_document_revision_export \
  "$export" "$earlier" "$analyzed"
```
