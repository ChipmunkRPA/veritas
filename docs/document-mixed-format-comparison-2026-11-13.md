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

## Reordered paragraphs in matching TXT or Word files

The earlier-draft comparison can locate one unchanged body paragraph whose
position changed relative to the other matching paragraphs. It requires the
same wording and punctuation in both versions, ignoring whitespace, and
withholds the label for ambiguous swaps, repeated passages or quoted blocks.
The analysis view, comparison PDF and JSON include the paragraph text and its
exact character range in each version. Word also reports both native paragraph
positions. The ordinary wording comparison still reports actual text edits.
The reorder finding needs both drafts and contributes no AI-authorship score.

The local test wrapped excerpts from 16 previously unused GAO test reports
as TXT and Word, excluding IDs in the earlier GAO archive-comparison panels.
In each format, all 32 known reorder pairs returned the exact
earlier and later ranges. Across both formats, 208 controls produced no reorder
finding. These included unchanged text, added introductions, ambiguous adjacent
swaps, duplicate passages, edited wording, quotations and hidden Word text.
These controlled pairs verify the feature's defined geometry. [GovReport sources and terms](https://gov-report-data.github.io/)

Taking the first four eligible sentences as separate paragraphs makes the
expected range replayable. The earlier sequence is `A, B, C, D` and the
later sequence is `A, C, D, B`, with the reverse direction tested separately.
The controls keep the same excerpts and change only the tested condition.

## A paragraph that moved and changed

With two matching TXT or Word drafts, the comparison can now pair a
paragraph that crosses at least two other paragraphs and also changes
wording. It needs three or more other unique, ordinary body paragraphs
that keep their wording and relative order. Exact opening and closing
sequences must each contain at least five words, occur once per draft,
and together cover at least 60% of the tokens in the longer passage.
The browser and PDF highlight only changed words and punctuation inside
that passage. JSON carries the positions in both drafts, including native
paragraph numbers in Word. The check uses no model and assigns no AI
authorship weight. The usual wording comparison still applies when the
anchors are incomplete or the layout is ambiguous.

The distinction between a structural move and the wording edits inside it
follows the problem identified by Cuculovic and colleagues in their JATS
study. This implementation uses the existing local TXT and Word readers
and a separate exact-text rule. [JATS revision study](https://doi.org/10.1002/spe.3074)

We repackaged 72 pinned report excerpts from two published generator
families. Each positive kept the publisher's one-sentence rewrite and
both immediate unchanged neighbors within the moved paragraph. Two
other report sentences and one authored paragraph formed the rest of
the document. The test covers newly composed revisions from previously
available source shards. The report groups are distinct across the two
families. These are controlled rearrangements, with no natural-reorder
accuracy or single-upload origin estimate. [Benchmark source and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

| Measure, separately in each format | First generator, 48 reports | Second generator, 24 reports |
| --- | ---: | ---: |
| Exact paragraph in both drafts, with highlights confined to the changed sentence | 43 | 22 |
| Cases with highlights outside the changed sentence, previous → updated | 48 → 5 | 24 → 2 |
| Comparison outputs changed on controls | 0 / 240 | 0 / 120 |

The five controls per source cover a wording change in place, an adjacent
swap with an edit, a move with unchanged text, an unrelated replacement,
and a repeated anchor. The seven remaining staged positives per format
used the prior comparison and still marked words outside the revised
sentence. The receipt contains both sets of token ranges, target sentence
bounds and input hashes. [Comparison receipt](evaluation-panels/document-edited-relocation-2026-11-15.json.gz)

Two local analyses used simulated USD 9.99 checkout. The comparison
ran on the retained TXT and Word files after payment. The API, PDF and
JSON identified the same four changed tokens across both sides, and the
downloaded comparison replayed against both original files. The example
shows **logged six** changing to **recorded eight**, and the Word
coordinates move from paragraph 2 to paragraph 4. The worker skipped an
optional single-upload writing check because of its memory budget.
[Local worker receipt](evaluation-panels/document-edited-relocation-app-2026-11-15.json)

Replay the published numeric receipt with the standard library. Add
`--source-shards` with the pinned `test-reports_gpt-5.4.csv` and
`test-reports_qwen3-8b.csv` files to rebuild the reported offsets locally.
Use `scripts.review_document_revision_export` to check your downloaded
comparison against both original files.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.replay_document_edited_relocation \
  docs/evaluation-panels/document-edited-relocation-2026-11-15.json.gz
```

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
