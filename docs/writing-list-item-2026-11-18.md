# Locate a rewritten item in a single checklist

This trial tests whether scoring each checklist item on its own adds
useful locations beyond the current writing check. The current check
segments normalized prose and reads several nearby sentences together.
On the ordinary lists in this panel it segmented five source items into
four to nine sentences. A new candidate uses each item’s original position
and only its own words. A second candidate compares the item’s response
with the three closest-length items in the same upload. Each candidate
can add one location. Neither receives an earlier version at inference.
[Local-evidence segmentation research](https://arxiv.org/abs/2609.33787)

## Frozen source and comparison

The inputs use five consecutive, same-indent items from dated *The Turing
Way* Markdown files. Each original and style-only control retains the
publisher’s wording. The style-only version changes just the list markers
to a numbered sequence. For the edited version a local editor rewrites one
interior item. Its Unicode character position stays tied to the scored text
through the app’s normalized source map. The private roster records the
publisher file, source line numbers, native byte hashes, and exact text.
The public receipt omits the text.

The first intake required nine words per item and reserved only six
calibration documents. It stopped before scoring. The second prospective
intake uses 7–66 words and rounds the number of calibration groups up.
It admitted 19 different chapter files across 16 chapter groups. Seven
groups supplied 10 calibration files and nine separate groups supplied the
nine held files. The admission procedure screens for 12-word overlaps with
six earlier prose plans, the recent editorial reserve and the other selected
files. The source text is derived from *The Turing Way*, copyright The Turing
Way Community, under CC BY 4.0. The test describes its extraction and
text transformations. Checkpoint pretraining exposure is unknown.
[Publisher archive](https://github.com/the-turing-way/the-turing-way/tree/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c) ·
[Terms](https://creativecommons.org/licenses/by/4.0/) ·
[Prior source panel](writing-turing-editorial-history-2026-11-18.md)

Each pinned local editor attempted two rewrites of the same item. The first
admissible rewrite per editor and source was kept. Complete-sentence,
length and edit-distance checks admitted five of 18 generator–source pairs.
Three came from Qwen and two from SmolLM2. Thirteen generator–source pairs
produced no admitted rewrite. The matched controls and edited versions all
passed the source-position checks before inference. Both new readings used
the current Apache-2.0 classifier. The 90th percentile of ordinary file
maximums fixed both internal review points before the held scoring. A
location must cover at least half the known edit’s non-whitespace characters
and reach 0.30 intersection over union.
[Editors and terms](writing-turing-editorial-history-2026-11-18.md#local-passage-results) ·
[Passage checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Held results

| Reading | Known items located | Ordinary source groups flagged | Edited files highlighted only outside the edit |
| --- | ---: | ---: | ---: |
| Current passage check | 0/5 | 0/9 | 0/5 |
| Current plus isolated item | 0/5 | 4/9 | 2/5 |
| Current plus item’s difference from nearby-length items | 0/5 | 3/9 | 2/5 |

In the no-cutoff diagnostic, the isolated reading ranked one target first
and the relative reading ranked two targets first. The isolated score
cleared its point at one labeled item while ranking another item higher in
that file. The relative score stayed below its point at every labeled item.
Both readings rose at the target relative to the matching ordinary text on
four of five rewrites. These increases did not produce a useful finding at
the frozen points. All nine marker-only documents retained the same isolated
item responses as their ordinary counterpart. Both candidates stay in
evaluation. Five rewrites from one publisher leave broader list behavior
unverified. [Held numeric receipt](evaluation-panels/writing-list-item-2026-11-18.json.gz)

## Replay

Read the numeric receipt with standard-library Python. It recomputes the
operating points, selected locations, and ordinary and edited counts.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_list_item_holdout review \
  docs/evaluation-panels/writing-list-item-2026-11-18.json.gz
```

Use the publisher checkout and seven input plans from the [Rich Text
revision study](writing-rtf-draft-2026-11-18.md#application-replay). The
plans include `source.json.gz` from the first historical study, the second
historical roster from the Rich Text study, and five earlier source plans.
The two generator steps use the pinned snapshots from the first historical
study. The classifier checks its snapshot hashes before opening the weights.
Use unused output paths.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
mlx=backend/data/text-upgrade/mlx-env/bin/python
study=scripts.writing_list_item_holdout
"$runner" -B -m "$study" reserve "$publisher" "$run/roster.json.gz" \
  "$historical/source.json.gz" "$rtf/frozen.json.gz" "$prior/plan/fit.json.gz" \
  "$prior/plan/calibration.json.gz" "$prior/plan/held.json.gz" \
  "$prior/transfer-bound.json.gz" "$economic/held.json.gz"
for editor in qwen smol2; do
  "$mlx" -B -m "$study" generate "$run/roster.json.gz" "$run/$editor.json.gz" --family "$editor"
done
"$runner" -B -m "$study" prepare "$run/roster.json.gz" "$run/input.json.gz" \
  "$run/qwen.json.gz" "$run/smol2.json.gz"
"$runner" -B -m "$study" calibrate "$run/input.json.gz" "$classifier" "$run/calibration.json.gz"
"$runner" -B -m "$study" evaluate "$run/input.json.gz" "$classifier" \
  "$run/calibration.json.gz" "$run/result.json.gz"
```
