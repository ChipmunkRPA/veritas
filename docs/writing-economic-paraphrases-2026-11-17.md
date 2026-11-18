# Short rewrites in official economic prose

This test measures whether a frozen writing check can find a small rewrite in
an otherwise unchanged document, using an outside source genre. The known
change comes from the generation procedure. Both detectors see only one
version at a time, without the original, editor identity, or edit positions.

## Sources and comparisons

The reservation inspected 184 published pages in fixed order. Twenty-two
distinct originals qualified from a target of 24: twelve Bureau of Economic
Analysis releases, six of its blog articles, and four Census Bureau releases.
The twenty-two works cover seven release families. The archive dates are
2017–2020. The exact held material is the retrieved HTML, whose publication
paths may retain later website corrections. The trial uses federal-duty
editorial prose and excludes marked quotations, contact lines, tables and
external links. The BLS archive returned HTTP 403 during acquisition.

[BEA reuse policy](https://www.bea.gov/help/faq/145) ·
[Census Bureau public-access policy](https://www2.census.gov/foia/ds_policies/ds027.pdf) ·
[Federal works provision](https://www.copyright.gov/title17/92chap1.html)

Each original supplies an exact 12-sentence excerpt. The intake checked 322
earlier project receipts for matching source or text hashes. It also excluded
normalized 12-word overlaps with the prior detector's four retained fit,
calibration, held, and local paraphrase text plans, and among admitted source
excerpts and donor text. The study retains all work groups with an unusable
editor response. Project exposure checks cover the recorded material.
Model pretraining overlap is unknown.

Two already installed Apache-2.0 editors each received one two-sentence
instruction for 11 distinct originals. The source text outside the replacement
remains unchanged. A replacement needed two to five usable sentences at precise
Unicode offsets. Copies of unchanged sentences inside those replacements
received no edit label. Fifteen edited pairs qualified, seven from one editor
and eight from the other. Seven unusable replies remained in the attempt
accounting. The trial scored 22 original files, 17 ordinary comparison files
with same-article human prose substitutions, and 15 edited files. The
ordinary comparisons test plausible changes in content and flow.

[Qwen weights and license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) ·
[SmolLM2 weights and license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

The comparison keeps the prior tuned joint medium operating point and the
current passage rule. A qualifying location covers at least half of a known
edit and reaches 0.30 intersection over union over non-whitespace characters.
Each known range can count once. The fit, operating point and local editor
checkpoints were fixed before the held files were scored. These readings
describe detection in this panel. Individual release families share publishers,
reporting conventions, and working subject matter.

[Frozen detector and detection criteria](writing-token-supervision-2026-11-17.md) ·
[New source and offset receipt](evaluation-panels/writing-economic-paraphrase-sources-2026-11-17.json.gz)

## Held results

| Reading | Correct locations | Misplaced highlights on edited files | Ordinary files flagged |
| --- | ---: | ---: | ---: |
| Current passage rule | 1/33 | 0 | 1/39 |
| Tuned joint medium point | 0/33 | 0 | 0/39 |
| Both readings | 1/33 | 0 | 1/39 |

The current rule located one of 18 known changed sentences in eight Qwen
edits and zero of 15 in seven SmolLM2 edits. The tuned reading found zero
in both groups. The single ordinary flag came from a publisher original.
All 17 same-source substitutions remained clear. Both readings assessed all
54 delivered files. The tuned result stays in evaluation. This second
transfer confirms that the earlier detector's gains on benchmark compression
edits did not carry to short paraphrases. Matched-original comparison remains
a separate feature with its own requirements.

[Fixed-point readings and source ranges](evaluation-panels/writing-economic-paraphrase-held-2026-11-17.json.gz)

## Replay

The source receipt retains both the original source-page digests and the exact
normalized excerpt positions. The held receipt omits the texts and retains
the original-file hashes, sentence and word positions, both detector readings,
and the accepted operating point. The portable review checks each delivery's
positions and reproduces the totals with two matching implementations.
The local freeze and earlier published freeze differ only in the implementation
hash metadata. The source receipt binds both files and lets the portable readback
verify that their full calibration content agrees after substituting that metadata.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.writing_economic_paraphrase_holdout review \
  "$panels/writing-token-supervision-calibration-2026-11-17.json.gz" \
  "$panels/writing-economic-paraphrase-held-2026-11-17.json.gz" \
  --sources "$panels/writing-economic-paraphrase-sources-2026-11-17.json.gz"
```

To recreate the exact inference, use the prior fit, checkpoint and source plans
as documented in the [existing comparison](writing-token-supervision-2026-11-17.md#reproduce-the-local-comparison).
The saved source receipt records the precise archive, article and editor pins.
The index is live, so a later acquisition can yield different bytes or source
links. Compare its receipt with the frozen one before treating it as a replay.

```sh
runner=backend/.venv/bin/python
study=scripts.writing_economic_paraphrase_holdout
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$runner" -B -m "$study" reserve "$out/intake" "$freeze" \
  "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" \
  "$prior/plan/held.json.gz" "$prior/transfer-plan.json.gz"
for editor in qwen smol2; do
  backend/data/text-upgrade/mlx-env/bin/python -B -m scripts.writing_report_sources generate \
    "$out/intake/$editor.json.gz" "$editor" "$out/intake/$editor-edits.json.gz"
done
"$runner" -B -m "$study" assemble "$out/intake" "$out/held.json.gz"
"$runner" -B -m scripts.writing_token_supervision_eval baseline \
  "$out/held.json.gz" "$incumbent" "$out/baseline.json.gz"
"$runner" -B -m "$study" run "$out/held.json.gz" "$checkpoint" "$prior/fit" \
  "$freeze" "$out/baseline.json.gz" "$out/held-results.json.gz" --device mps
```

## Compare another detector

The portable package binds 54 exact UTF-8 deliveries to 22 verified publisher
HTML files. To build it, the exporter reread those publisher bytes, checked the
normalized excerpt on each page, reconstructed all 17 same-article human
substitutions and checked all 15 admitted rewrites against the retained editor
replies. It also rechecked the seven unusable replies. The package holds file
hashes, source groups, truth ranges, whitespace positions and reference
predictions. The publisher pages, delivered prose and editor replies stay in
the local replay directory.
[Model-independent package](evaluation-panels/writing-economic-paraphrase-independent-2026-11-17.json.gz)

Submit offsets as half-open Unicode character ranges in the exact delivered
text. Keep the manifest commitment, case ID and input hash in each response.
Use an empty regions list for a successful check without a finding. Give a
reason and a null response for any unassessed input. The scoring command
validates the native ranges and then removes whitespace from their extents
before applying the original event matching rule. The reference results are
one of 33 for the current check and zero of 33 for the frozen tuned check.

```sh
review=$(mktemp -d)
panels=docs/evaluation-panels
packet="$panels/writing-economic-paraphrase-independent-2026-11-17.json.gz"
python3 -B -S - "$packet" "$review" <<'PY'
import gzip
import json
import sys
from pathlib import Path

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    receipt = json.load(stream)
directory = Path(sys.argv[2])
parts = {key: receipt[key] for key in ("manifest", "truth", "file_index")}
for key, value in (parts | receipt["predictions"]).items():
    (directory / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
PY
PYTHONPATH=backend python3 -B -S -m scripts.export_writing_economic_review score \
  "$packet" "$review/incumbent.json"
```

Rebuild the package and write the exact local input files from the verified
reservation above. The public receipts bind the earlier fit and the held
operating point. The command refuses different inputs or editor replies.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.export_writing_economic_review export \
  "$out/held.json.gz" "$out/intake" \
  "$panels/writing-token-supervision-calibration-2026-11-17.json.gz" \
  "$panels/writing-token-supervision-fit-2026-11-17.json" \
  "$panels/writing-economic-paraphrase-held-2026-11-17.json.gz" \
  "$panels/writing-economic-paraphrase-sources-2026-11-17.json.gz" \
  "$review/files" "$review/rebuilt.json.gz"
python3 -B -IS backend/scripts/independent_review.py verify-inputs \
  "$review/manifest.json" "$review/file_index.json" "$review/file-checks.json" \
  --root "$review/files"
```
