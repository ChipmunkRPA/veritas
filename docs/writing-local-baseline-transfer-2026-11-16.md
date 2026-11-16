# Find writing changes relative to the surrounding document

The passage check can now consider how writing changes within the same
file. The existing check compares passages with readings from a collection
of ordinary documents. The added check compares passages with the lower
quarter of the readings in the submitted file and, where available, with
sentences on both sides. It uses the same model responses. The app keeps the
existing ranges and adds separate ranges up to the existing six-range limit.
At least one unselected sentence must separate all displayed ranges.

The earlier selector draws on research into local authorship transitions.
We kept its word-count calibration and tested the within-document shift and
the neighboring-sentence comparison separately and together. The
neighboring-sentence comparison requires two complete sentences on each
side and uses up to five per side. The method and the added range's operating
point were chosen on the 54 ordinary calibration works and the retained
comparison before the fresh panel's results were available. The fresh panel
supplied no fitting observations. [Authorship transition research](https://arxiv.org/html/2609.33787v1)
· [Length-dependent calibration research](https://arxiv.org/html/2611.10901v1)
· [Earlier passage check](writing-multi-passage-transfer-2026-11-16.md)

## Matched excerpts and results

The retained comparison uses 49 ordinary excerpts and their matched
rewrites. The fresh comparison uses 40 different pairs. Twenty come from
12 historical author groups and 20 from 16 Federal Reserve report dates.
All 28 fresh source groups are outside the earlier calibration and test
selection. Publisher identifiers and input checksums are pinned. The
publisher calls these files part of its training split. Our project used
these selections only for testing the fixed passage check. The classifier's
prior training exposure is unknown. The historical sources are limited to
U.S. public-domain Gutenberg works. The report sources are limited to the
Board's website with its published terms. Requested generator labels are
retained with their recorded identity status. [Publisher dataset card](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans)
· [Project Gutenberg source terms](https://www.gutenberg.org/policy/license.html)
· [Federal Reserve website terms](https://www.federalreserve.gov/disclaimer.htm)

Both layouts preserve the same non-whitespace characters. The second layout
replaces whitespace characters with spaces to test sensitivity to formatting.
The native text extractor and sentence segmenter run before scoring. A useful
highlight must cover at least half of a labeled passage and have at least
0.30 intersection over union, measured on non-whitespace characters.
Each highlight can match at most one label. All 160 fresh deliveries had
complete readings on 5,963 distinct inputs.

| Panel and layout | Labeled passages found, existing → combined | Ordinary files flagged, existing → combined | Unmatched ranges in mixed files, existing → combined |
| --- | --- | --- | --- |
| Retained · published | 24 → 36 of 98 | 6 → 7 of 49 | 4 → 5 |
| Retained · layout-neutral | 25 → 36 of 98 | 6 → 7 of 49 | 4 → 5 |
| Fresh · published | 17 → 22 of 55 | 10 → 10 of 40 | 3 → 5 |
| Fresh · layout-neutral | 17 → 20 of 55 | 10 → 10 of 40 | 3 → 5 |

On the fresh panel, the five extra matches in the published layout came
from five historical source groups. Two extra proposed ranges failed the
overlap requirement. Historical ordinary files stayed at zero alerts out
of 20. Federal Reserve originals still had ten alerts out of 20 and saw
no extra edit matches. The combined check still missed 33 of 55 labeled
passages in the fresh published layout. This is evidence for a local
review aid on this panel. Broader genres, languages, and documents with
no reliable local baseline require separate testing. [Retained receipts](evaluation-panels/writing-local-baseline-retained-2026-11-16.json.gz)
· [Fresh receipts](evaluation-panels/writing-local-baseline-fresh-2026-11-16.json.gz)

## Application scope

`VERITAS_TEXT_PASSAGE_DOCUMENT_BASELINE_REVIEW=true` enables the additional
ranges alongside `VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW=true` in development.
The settings require the candidate English writing path, the supported
three-sentence context, and complete positions for 12–128 sentences. The
check covers mapped plain text, native Word text, selectable PDF text, and
the mapped reply body in supported email. The added findings show the
exact source positions and why the passage was selected. They contribute
zero weight to the overall assessment. The JSON commitment records the
selection basis so the offline verifier can replay it from the same
responses, and verify the source locations when the original is available.
The previous formats remain readable. The new method runs on the existing
Apache-2.0 checkpoint with no new weights. [Checkpoint publisher's license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small)

## Local app replay

Eleven simulated USD 9.99 checkouts exercised a fresh matched pair in
plain text, Word, selectable PDF, plain email, and HTML email, plus an
additional run with the new writing rule switched off. Zero analyses
queued before payment. The browser displayed the same rewritten passage
in all five formats. Each signed package verified against the original
file. The PDF and JSON identify the same character range, Word paragraphs,
PDF pages, or decoded MIME body positions. JSON also includes the exact
PDF word boxes. All five ordinary files
returned zero ranges. The overall assessment agreed across both runs of
the plain text file. The replay isolates the passage check by switching
off the separate checkpoint agreement option. The offline verifier
passed without the analysis runtime or network. [Prepaid browser and export receipt](evaluation-panels/writing-local-baseline-app-2026-11-16.json)

## Replay

The receipt contains source references, recorded offsets, hashes, and
numeric responses. Keep the native texts, source fixtures, and weights
local. The portable reader reproduces the full method comparison and
verifies both application variants at serving precision. To rebuild the
fresh scores, follow the publisher card for the two pinned parquet files
listed in `FRESH_PINS` and use the pinned classifier snapshot recorded in
the previous study. The score command verifies the checkpoint files before
reading the test documents.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
module=scripts.writing_local_baseline_transfer
published=docs/evaluation-panels
python3 -B -S -m "$module" review "$published/writing-local-baseline-retained-2026-11-16.json.gz" --serving
python3 -B -S -m "$module" review "$published/writing-local-baseline-fresh-2026-11-16.json.gz" --serving
runner=backend/.venv/bin/python
"$runner" -B -m "$module" prepare "$publisher" "$run/held-plan.json.gz"
"$runner" -B -m scripts.writing_multi_passage_panel score "$run/held-plan.json.gz" "$snapshot" "$run/measured.json.gz"
python3 -B -S -m "$module" compare "$run/results.json.gz" --held "$run/measured.json.gz"
npm --prefix frontend run build
"$runner" -B -m scripts.writing_local_baseline_app "$run/held-plan.json.gz" "$run/results.json.gz" "$font" "$run/prepaid" --browser
```

`$font` is the path to DejaVu Sans for rendering the selectable PDF
test fixture. The prepaid test directory holds private copies of the
fixtures, PDF renders, and signed packages.
