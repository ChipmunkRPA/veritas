# Compare wording in Word table cells

Open a completed Word analysis and choose **Compare with an earlier draft**.
Upload another `.docx`. When both files have a simple table in the same
body position, the comparison can show changes inside matching cells.
Both tables must retain the same dimensions, column headings and unique
row labels. Each change shows the earlier and analyzed wording and its
table, row, column and paragraph positions. PDF and JSON use the same
ranges. This comparison reports wording changes and assigns no AI confidence.

Only top-level tables with plain paragraphs are eligible. A merged cell,
nested table, tracked markup or structural change can make that table
ineligible. The result identifies tables it could not compare. The check
can compare at most 96 cells, 2,000 tokens and 30,000 characters across
the tables in each file. Table position reporting is separately capped at
96 positions. Paragraph comparison continues independently. Cross-format
comparisons retain their existing Word body paragraph scope.

Table and row numbers are one-based. In JSON, `body_index` counts element
children of `w:body`, and `cell_index` counts element children of the cell,
including property elements. Both indexes start at zero. Excerpt offsets
count Unicode characters from the beginning of the cell, with two line
breaks between included paragraphs. Each paragraph also has its own local
range. The underlying source hashes bind both Word files. Earlier schema-1
Word exports still replay with their original paragraph-only scope.

## Local validation

We placed the 21 source-bound USWDS version pairs into matching 3×2 Word
tables. The cell in row 2, column 2 contained the exact source excerpt.
The previous paragraph comparison missed every table edit. The new check
found every edited case and matched all 253 changed token ranges in both
versions against the corresponding plain text comparison. All six
formatting-only controls produced zero wording changes. These are
constructed Word layouts around existing licensed editorial and generated
text. [Upstream writing test and provenance](writing-uswds-revision-transfer-2026-11-13.md)

| Text in matched Word tables | Previous comparison | Cell comparison |
| --- | ---: | ---: |
| Genuine historical revisions | 0/6 | 6/6 |
| Granite rewrites | 0/5 | 5/5 |
| OLMo rewrites | 0/4 | 4/4 |
| Unchanged wording with changed formatting | 0/6 | 0/6 |

Every comparison replayed from both generated Word files. The local
report test checked that the rendered PDF retained the native cell
coordinates. Authenticated-route and browser-component tests cover the
coordinates, exports, text formatting, unsupported tables and historical
JSON verification. [Local replay ledger](evaluation-panels/document-word-table-comparison-2026-11-13.json)

An isolated Chrome run used the ordinary local checkout path with simulated
payment. It showed the actual words in the earlier and analyzed cells and
downloaded the same JSON and PDF returned by the authenticated endpoint.
The downloaded JSON replay verified both sources and ten token ranges.
One optional statistical writing check abstained for memory. The comparison
used the native Word files directly.

## Reproduce

The standard-library reader verifies source-group coverage and recorded
counts. It does not read user files or load a model.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.replay_document_word_tables review \
  docs/evaluation-panels/writing-uswds-revision-transfer-2026-11-13.json \
  docs/evaluation-panels/document-word-table-comparison-2026-11-13.json
```

To repeat the native Word, PDF and source-range checks, rebuild the USWDS
writing inputs using the upstream study's local commands. In a fresh scratch
directory, place its `roster.json`, `lock.json`, `granite.json` and `olmo.json`.
Change `review` to `score`, give a new receipt path and add `--run "$run"`.
The script verifies the upstream text hashes before building Word files.

To verify a downloaded customer JSON using its two original Word files:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_document_revision_export \
  "$export" "$earlier" "$analyzed"
```
