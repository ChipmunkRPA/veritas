# Writing passage checks on historical editorial revisions

This panel measures how the local passage check behaves on chronological
editorial changes from *The Turing Way*. The handbook archives its edits in
Git and releases its chapter text under CC BY 4.0. The test uses an exact
parent commit and its child for each change. It retains exact source hashes
and positions. The authorship control rests on attributed historical Git
records from 2019 and 2020. Those records establish chronology and source
lineage. They cannot establish how every contributor wrote every line.
[Archive release](https://github.com/the-turing-way/the-turing-way/tree/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c) ·
[Release license](https://github.com/the-turing-way/the-turing-way/blob/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c/LICENSE.md)

## Source qualification

The intake keeps parent and child chapter files from before January 2021,
with 2019 considered first. It verifies chapter licensing in both commits,
removes Markdown structure and link destinations, and retains eligible prose
in chapter order with its paragraph breaks. It excludes code, lists, explicit
quotations and notices of separately licensed text. The earlier
and later excerpts have 12 native sentences, an exact matching set of 11
neighboring sentences, and one verified lexical change of at least two
words. The intake screens for exact normalized 12-word overlap with five
earlier writing plans and other selected works, and limits duplicate source
paths and contributors. Project exposure checks cover those recorded plans.

The first scout kept exclusively uninterrupted prose runs and admitted no
pairs. The revised normalization, using eligible chapter prose in reading
order, was frozen before detector inference. It admitted 15 pairs across
13 chapters and ten anonymous contributor groups. Two pairs date from 2019
and 13 from 2020. The receipts keep public version and file identifiers,
contributor cluster aliases, source offsets, and scores. The source prose
and exact generated text stay local. The chapter material is derived from
*The Turing Way*, copyright The Turing Way Community, under CC BY 4.0.
The changes and text normalization used by the test are described above.
[Publisher reuse guidance](https://book.the-turing-way.org/reproducible-research/licensing/licensing-documentation/) ·
[License text](https://creativecommons.org/licenses/by/4.0/)

## Local passage results

The comparison froze the classifier and the live passage settings before
scoring. It uses both the document-relative baseline and the existing
report confirmation step when eligible. A useful location overlaps at least
half the matched edit's non-whitespace characters at 0.30 intersection over
union. The score and matching criteria apply only to the passage check.

| Input | Assessed | Flagged | Alert covering the edited sentence |
| --- | ---: | ---: | ---: |
| Historical earlier version | 15/15 | 1/15 | 0/15 |
| Historical revised version | 15/15 | 1/15 | 0/15 |
| Matched Qwen sentence rewrite | 8/8 | 1/8 | 0/8 |
| Matched SmolLM2 sentence rewrite | 1/1 | 0/1 | 0/1 |

The same historical work produced the only ordinary alert before and after
editing. No historical pair acquired a flag after its actual edit. In the
separate generated lane, each offline editor attempted one revision at
the historical edit position in every earlier excerpt. The fixed geometry
and lexical admission rules accepted eight Qwen replies and one SmolLM2
reply and excluded the other 21 before scoring. The one generated alert
landed outside the edited sentence. This small, uneven sample supports
retaining the controls for development. It cannot estimate coverage across
other writing genres or generators. [Natural baseline receipt](evaluation-panels/writing-turing-history-baseline-2026-11-18.json.gz) ·
[Generated comparison receipt](evaluation-panels/writing-turing-history-ai-comparison-2026-11-18.json.gz) ·
[Apache-2.0 Qwen revision](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) ·
[Apache-2.0 SmolLM2 revision](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Word and PDF source positions

The separate format replay packaged both versions of the first two pairs as
DOCX and one-page selectable PDF. All eight files preserved exact extracted
text and the same 12 native sentence positions. Each changed sentence mapped
to original Word paragraph character offsets or source PDF word boxes. The
replay covers extraction and mapping. The table above is the plain-text
detector comparison. [Format replay receipt](evaluation-panels/writing-turing-history-formats-2026-11-18.json)

## Replay locally

The portable review recomputes passage selections and the overlap counts
from the saved numeric responses. It needs no model downloads.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.writing_turing_history_baseline review \
  "$panels/writing-turing-history-baseline-2026-11-18.json.gz"
python3 -B -S -m scripts.writing_turing_history_ai_comparison review \
  "$panels/writing-turing-history-ai-comparison-2026-11-18.json.gz"
```

To rebuild from exact text, use a fresh run directory and the pinned
publisher tag. The five input plans come from the
[earlier report study](writing-token-supervision-2026-11-17.md#reproduce-the-local-comparison)
and [economic study](writing-economic-paraphrases-2026-11-17.md).
The baseline and generated commands require the pinned local weights.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runner=backend/.venv/bin/python
mlx=backend/data/text-upgrade/mlx-env/bin/python
git clone --filter=blob:limit=60000 --no-checkout --single-branch --branch v1.0.1 \
  https://github.com/the-turing-way/the-turing-way.git "$run/source"
"$runner" -B -m scripts.writing_turing_editorial_history "$run/source" "$run" \
  "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" "$prior/plan/held.json.gz" \
  "$prior/transfer-bound.json.gz" "$economic/held.json.gz"
"$runner" -B -m scripts.writing_turing_history_baseline freeze \
  "$run/source.json.gz" "$run/lock.json.gz" "$run/frozen.json.gz"
"$runner" -B -m scripts.writing_turing_history_baseline evaluate \
  "$run/frozen.json.gz" "$run/source.json.gz" "$incumbent" "$run/result.json.gz"
for editor in qwen smol2; do
  "$mlx" -B -m scripts.writing_turing_history_ai_comparison generate \
    "$run/source.json.gz" "$run/$editor-replies.json.gz" --family "$editor"
done
"$runner" -B -m scripts.writing_turing_history_ai_comparison prepare \
  "$run/source.json.gz" "$run/ai-frozen.json.gz" "$run/qwen-replies.json.gz" "$run/smol2-replies.json.gz"
"$runner" -B -m scripts.writing_turing_history_ai_comparison evaluate \
  "$run/ai-frozen.json.gz" "$incumbent" "$run/ai-result.json.gz"
"$runner" -B -m scripts.writing_turing_history_packaging "$run/source.json.gz" "$run/formats"
```
