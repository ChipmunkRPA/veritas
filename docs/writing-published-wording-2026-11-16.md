# Earlier published wording in normal document analyses

Enable `VERITAS_DOCUMENT_GAO_AUTOMATIC_REVIEW=true` in local development
with the verified GAO highlights pack. A completed analysis can now show
the publication date, the official report, earlier and submitted wording,
and the original locations directly in the ordinary results view.
Eligible uploads are TXT, mapped top-level Word paragraphs, and visible
selectable PDF text in a single column. The comparison runs on the
local copy of the archive within the existing paid analysis.
[Pack preparation and source terms](writing-gao-archive-comparison-2026-11-13.md)

The earlier report must have exact anchors supporting one bounded source
in the pinned collection. A competing anchor is extended in both directions
to measure the complete shared run, including words omitted by the unique
phrase index. A long exact run from a second source keeps the document
ambiguous. When an entire optional writing-style highlight already appears
in the dated transcript, allowing for spacing, the app shows it with the
published source. Highlights beyond this exact match continue to appear
with the writing checks. Wording differences can reflect transcription,
ordinary editing, or AI assistance. They contribute zero AI-score weight.
[GovReport edition and CC BY attribution](https://gov-report-data.github.io/#license)

## Local measurements

Forty-eight previously unused government-report groups provide exact
untouched text and short AI-polished versions. The source intake binds
the publisher-labeled changes to the same dated reports in GovReport,
excludes previous report IDs and text digests, and includes two generators
across two publisher splits. A useful location covers at least half of
the labeled text with at least 30% overlap relative to the combined range.
[Pinned publisher data and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

| Metric | Earlier 69 pairs | New 48 pairs |
| --- | ---: | ---: |
| Known edits located by the writing-style check | 0 of 212 | 1 of 161 |
| Known edits located by the published-wording check | 161 of 212 | 97 of 161 |
| Known edits located across both checks | 161 of 212 | 98 of 161 |
| Edited files with a qualified source | 59 of 69 | 37 of 48 |
| Untouched files with a qualified source | 61 of 69 | 43 of 48 |
| Untouched files with a located difference | 0 of 69 | 0 of 48 |
| Untouched files with a remaining style highlight | 3 → 1 of 69 | 2 → 0 of 48 |

The first run selected one source for two controls that inserted a passage
from another report. Extending the competing anchors withheld both earlier
selections and preserved every known location in the paired tests. A further
36 mixed-source controls made from separate source pairs received no single-source attribution.
Across all 108 controls, zero received a qualified single source. The corpus
covers a collection of government highlights. For writing outside that
collection, the style check supplies the available statistical evidence.
[Exact ranges and counts](evaluation-panels/writing-published-wording-2026-11-16.json.gz)

## Report and original-file replay

The normal PDF and JSON show the same publication and wording locations.
Text positions use zero-based Unicode code points. Word records its
paragraph positions. PDF records page numbers and original word boxes.
The signed package binds both the full retained archive transcript and
the submitted locations. Replay with the original regenerates the bounded
wording differences and rechecks the exact historical matches. Independent
source selection can be repeated against the pinned local archive pack.
[Paid local app replay](evaluation-panels/writing-published-wording-app-2026-11-16.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
runner=backend/.venv/bin/python
receipt=docs/evaluation-panels/writing-published-wording-2026-11-16.json.gz
python3 -B -S -m scripts.writing_published_wording_panel verify "$receipt"
"$runner" -B -m scripts.writing_published_wording_panel intake \
  "$publisher_cache" "$gao_pack" "$previous_validation_plan" "$previous_test_plan" \
  "$earlier_22_report_plan" "$run/fresh-plan.json.gz" --receipt "$receipt"
"$runner" -B -m scripts.writing_multi_passage_panel score \
  "$run/fresh-plan.json.gz" "$pinned_writing_checkpoint" "$run/fresh-readings.json.gz"
"$runner" -B -m scripts.writing_published_wording_panel compare \
  "$gao_pack" "$previous_test_plan" "$previous_test_readings" \
  "$run/fresh-plan.json.gz" "$run/fresh-readings.json.gz" "$run/result.json.gz"
```

The previous test and validation plans use the short-interval report intake.
The separate 22-report plan uses the economic-report intake and its pinned
third shard. Run `npm run build` in `frontend` for the live browser test.
Use `scripts.writing_published_wording_app` with the previous test plan,
the PDF font path, a new output directory and `--browser` to repeat checkout
and original-file verification.
[Previous report plans](writing-short-intervals-2026-11-16.md#replay-locally) ·
[Economic-report intake](writing-report-confirmation-2026-11-16.md#local-review-and-reproduction)
