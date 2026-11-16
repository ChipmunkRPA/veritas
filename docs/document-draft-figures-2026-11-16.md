# Figures and dates in earlier-draft comparisons

Open **Compare with an earlier draft** on a completed local document
analysis. When two versions contain a figure in the same uniquely matched
wording, the comparison can show the earlier and later printed values
side by side. Each highlight links to the exact character range and the
available Word paragraph, decoded email position or PDF page and word
boxes. The browser, comparison PDF and JSON use the same source ranges.
The comparison leaves the original analysis and its confidence intact.

The November audit of rewriting tools compared retained numbers, names
and quoted text in matched sources. This implementation takes a
narrower measurement from that design. It binds each figure to the words
around it and reports a value change when both versions support the
same location. It uses no model, external checker code or remote request
on customer material. [Humanizer fact audit](https://arxiv.org/html/2611.00359v1#S3.SS2)

## What the comparison measures

- Use three exact adjacent words on each side of a figure, or four words
  on the other side when a figure is at a sentence or paragraph boundary.
  The wording must identify a single location in each source. The
  comparison ignores case and spacing in the matched surrounding words.
- Compare decimal values, the same currency symbol, percentages,
  percentage points and supported calendar dates. For example, the
  same dollar amount can appear as `$1,000` or `$1.00 thousand`.
  ISO and full English month dates use the same calendar value.
- Inspect the decoded document or selected email body. Quoted email,
  attachments, ambiguous contexts, unrecognized number forms and
  unsupported PDF layouts can leave figures unmatched. Up to eight
  verified changes are displayed. Check the surrounding text when
  deciding whether a change is a correction, a new fact or a mistake.

## Comparison on published sources

We used the pinned GoHumanize test split under CC BY 4.0. It includes
54 matched federal prose and AI rewrite pairs from 30 articles, plus
200 matched historical prose and rewrite pairs. The staged panel
reuses the federal originals. A separate literal selection changes the
first eligible numeral or appends decimal zeros without changing its
value. Thirty-seven passages from 26 articles supply both stages.
Another check pairs each original with itself and another article.
[GoHumanize source and attribution](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset/tree/cc066df21227f5fdb73b513c330181f946ce82ef)

The October AI Humanizer Benchmark supplies 363 rewrites of 33
AI-styled source texts. The independent paper's MIT-licensed repository
supplies 75 judgments of altered numbers or dates, and 541 of retained
ones. A judgment concerns the original unit. It can include a figure
that was removed or spelled out, and supplies no exact location for a
replacement in the rewrite. The project evaluates published outputs
and does not run the rewriting services. [October benchmark and CC BY 4.0 terms](https://huggingface.co/datasets/ai-humanizer-benchmark/ai-humanizer-benchmark/tree/6306b2043736dd958cf39ee1d9a64956c54dd78a)
· [Independent judgments and terms](https://github.com/zwang-devv/humanizer-fact-audit/tree/b92fc42c49f82c8db9df1a72f31e515c74905f42)

| Evaluation | Three-word rule | Two-word rule |
| --- | ---: | ---: |
| Correctly located staged values | 15/37 | 21/37 |
| Flags on equivalent formatting | 0/37 | 0/37 |
| Flags on identical copies | 0/54 | 0/54 |
| Flags between different federal articles | 0/54 | 0/54 |
| Federal rewrites with at least one matched figure | 7/54 | 11/54 |
| Federal rewrites with a flagged difference | 0/54 | 0/54 |
| October rewrites with at least one matched figure | 11/363 | 18/363 |
| October rewrites with a flagged difference | 0/363 | 1/363 |

Neither rule paired figures in the 200 historical prose rewrites.
The sole shorter-rule result matched week 1 of a training plan with
week 9 where similar words appeared in the rewritten plan. It linked
different activities and did not match an independently labelled
altered unit. The app uses the three-word rule. The published rewrites
show that the comparison offers little coverage when wording is
substantially reworked. The existing wording comparison still shows
other changes between the supplied files. [Source-linked measurements](evaluation-panels/document-draft-figures-2026-11-16.json.gz)

## Local app verification

The isolated worker found the same two changes across all nine TXT,
Word and PDF pairings and a plain-text email comparison. The changed
date and percentage kept their exact source ranges. The two dollar
formats counted as one unchanged value. Chromium displayed the
highlighted PDF regions and downloaded both reports. The JSON matched
the API and the PDF matched byte for byte. Testing used four simulated
USD 9.99 purchases. No job queued before payment, and the earlier-draft
comparison preserved each completed analysis.
[Local app receipt](evaluation-panels/document-draft-figures-app-2026-11-16.json)

## Independent replay

The evaluation receipt records file and source-group hashes, the
published label counts and measured character offsets. To recalculate
the result on the published sources, place the four pinned files below
in one folder. The runner checks sizes and SHA-256 values before reading
the texts. The portable recount and the native evaluation use standard
Python. The export verifier independently reconstructs both ranges
from the exact submitted files.

- `test.jsonl` comes from the [pinned GoHumanize split](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset/blob/cc066df21227f5fdb73b513c330181f946ce82ef/test.jsonl).
- `samples.json` and `results.jsonl` come from the [pinned October benchmark cycle](https://huggingface.co/datasets/ai-humanizer-benchmark/ai-humanizer-benchmark/tree/6306b2043736dd958cf39ee1d9a64956c54dd78a/data/cycles/2026-10).
- `fact_units.jsonl` comes from the [original study's released judgments](https://github.com/zwang-devv/humanizer-fact-audit/blob/b92fc42c49f82c8db9df1a72f31e515c74905f42/data/fact_units.jsonl).

```sh
export PYTHONPATH=backend
module=scripts.evaluate_document_revision_values
receipt=docs/evaluation-panels/document-draft-figures-2026-11-16.json.gz
python3 -B -S -m "$module" review "$receipt"
python3 -B -S -m "$module" run "$sources" "$run/figures.json.gz"
backend/.venv/bin/python -B -m scripts.review_document_revision_export "$export" "$earlier" "$analyzed"
backend/.venv/bin/python -B -m scripts.replay_document_revision_values "$run/app" --browser
```
