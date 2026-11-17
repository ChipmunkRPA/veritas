# Compare earlier published wording in two-column PDFs

The normal document analysis can compare visible text in two separated PDF
columns with a dated source in the local GAO highlights archive. The results
show the original page, the compared column, and the wording from both
versions. Each difference carries its original extracted character range
and native PDF word boxes in JSON. The comparison contributes zero weight
to the AI assessment. [Source transcripts and CC BY 4.0 terms](https://gov-report-data.github.io/#license)

## Reading order and source proof

The check uses the original selectable word boxes. It looks for one clear
whitespace gap and verifies the reading order within each column. It can
join balanced columns in left-to-right order when the archived source
supports both sides. It compares unequal columns separately. A matched
range needs exact wording anchors in the submitted and archived text.
The two excerpts must have comparable lengths. That keeps a sidebar from
claiming a long intervening section elsewhere in the archived transcript.
[PyMuPDF native text layout](https://pymupdf.readthedocs.io/en/latest/app1.html)
· [Reading order and separated text streams](https://arxiv.org/abs/2607.01018)

The fallback skips pages with ambiguous columns and requires verified text
visibility. It sets aside bullets, equivalent punctuation encodings, and
standalone headings without an archive counterpart. A lost hyphen inside
a word needs matching wording and original boxes on different PDF lines to
qualify as a line wrap. The browser and report list only the compared
portions. The signed package replays the text and boxes against the
original byte-for-byte PDF. Older exports keep their original positions.

In `/coverage/published_wording_review/sections`, `submitted_range` uses
the projected column's code points. `source_blocks` and each difference's
`locations` use the uploaded PDF's original extraction. Word boxes use
unrotated PDF page points.

## Fresh confirmation

The final panel uses 36 unused matched GAO report pairs with 108 known
AI-polished sentences. Two publisher generator families contributed 18
pairs apiece. Each pair uses a balanced and a narrow-sidebar delivery of
the same sources. The reader independently checks the authored PDF text
before comparing the label positions. A sentence counts as located when
at least one verified wording difference overlaps it. This measures
whether a rewrite was localized, with exact original boxes for the
reported wording. [Pinned OpAI-Bench release](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/commit/072c03c7051409346aaba05168d183cf6d863ee9)

| Final confirmation | Earlier comparison | With PDF columns |
| --- | ---: | ---: |
| Known rewrite sentences located, balanced | 0 of 108 | 88 of 108 |
| Known rewrite sentences located, narrow sidebar | 0 of 108 | 86 of 108 |
| Unchanged reflows with a wording difference | 0 of 72 | 0 of 72 |
| Official publisher PDFs with a wording difference | 0 of 24 | 1 of 24 |
| Incorrect source attributions | 0 | 0 |

The earlier comparison skipped these multi-column inputs. The new method
matched 12 of the 24 native publisher pages to the archive. The one
remaining wording difference was a figure callout present in the original
PDF and absent from the retained transcript. Each layout had one edited
PDF whose extracted text could not be mapped exactly to the reflow labels.
Their known sentences remain in the denominator. All 36 three-column
controls stayed outside the comparison. [Final results](evaluation-panels/writing-pdf-columns-confirmation-2026-11-17.json.gz)
· [GAO source terms](https://www.gao.gov/copyright)

The first 24 publisher controls exposed nine apparent differences. Writing
and typography filters cut that count to one on the same set. A second
separate 24-page set showed three differences from archive dash formatting
and a long sidebar alignment. Fixing those cases preserved the 147 located
sentences of 220 in the paired reflows. The final panel above used separate
sources after those rules were fixed. The three panels share no report IDs.
[Initial results](evaluation-panels/writing-pdf-columns-initial-2026-11-17.json.gz)
· [First typography replay](evaluation-panels/writing-pdf-columns-typography-2026-11-17.json.gz)
· [Second panel before the last fixes](evaluation-panels/writing-pdf-columns-order-check-2026-11-17.json.gz)
· [Second panel readback](evaluation-panels/writing-pdf-columns-order-replay-2026-11-17.json.gz)

## Local app and independent replay

`VERITAS_DOCUMENT_GAO_PDF_COLUMNS_REVIEW=true` enables the comparison in
development with `VERITAS_DOCUMENT_GAO_AUTOMATIC_REVIEW=true` and the pinned
local GAO archive. In the browser, open **Earlier published wording** in a
completed analysis. The app replay checks both the edited PDF and its
unmodified counterpart, PDF and signed JSON, and an edited rerun with the
option disabled. The AI assessment agrees between the two edited runs.
All three simulated USD 9.99 purchases queued analysis only after payment.
[Browser and export receipt](evaluation-panels/writing-pdf-columns-app-2026-11-17.json)

The public roster contains source IDs, edit locations and hashes. Use the
pinned publisher shards under `$cache/default/test` and
`$cache/default/train`, the local GovReport pack, and DejaVu Sans with the
SHA-256 recorded in the final result. The same roster regenerated the exact
private source plan. The input verifier rehashed 204 PDF cases and 408
input and parent file bindings. [Source roster](evaluation-panels/writing-pdf-columns-sources-2026-11-17.json.gz)
· [Portable input manifest](evaluation-panels/writing-pdf-columns-bindings-2026-11-17.json.gz)
· [Archive preparation](writing-gao-archive-comparison-2026-11-13.md)

```sh
export PYTHONPATH=backend
module=scripts.pdf_column_wording_panel
panels=docs/evaluation-panels
runner=backend/.venv/bin/python
python3 -B -S -m "$module" review "$panels/writing-pdf-columns-confirmation-2026-11-17.json.gz"
"$runner" -B -m "$module" prepare "$cache" "$pack" "$run/plan.json.gz" \
  --roster "$panels/writing-pdf-columns-sources-2026-11-17.json.gz"
python3 -B -S -m "$module" acquire "$run/plan.json.gz" "$run/publisher"
"$runner" -B -m "$module" run "$run/plan.json.gz" "$pack" "$run/publisher" "$font" "$run/results"
python3 -B -S -m "$module" bindings "$run/results/result.json.gz" "$run/publisher"
python3 -B -S -m scripts.independent_review verify-inputs \
  "$run/results/manifest.json" "$run/results/file-index.json" "$run/verified.json" --root "$run/results"
npm --prefix frontend run build
"$runner" -B -m scripts.pdf_column_wording_app "$run/results/result.json.gz" "$pack" "$run/prepaid"
```
