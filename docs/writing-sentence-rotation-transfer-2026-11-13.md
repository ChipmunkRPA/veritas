# Sentence rotation and localized writing responses

The three-sentence rotation check added **zero useful locations across 64
known rewrites**. It stays out of the app. The existing three-sentence
context check localized 23 of the 64 rewrites and highlighted 17 of the 32
untouched excerpts. The public decision ledger preserves all outcomes.
[Measurement ledger](evaluation-panels/writing-sentence-rotation-transfer-2026-11-13.json)

## Frozen comparison

Each excerpt contains five consecutive sentences from an English GAO
report. Each source work contributes the untouched excerpt and two
publisher-labeled edits, one from Gemini 2.5 Flash and one from GPT-5.4
Nano. A known edit occupies exactly one interior sentence, with identical
surrounding text in both variants. The calculation uses the pinned
commercially permitted checkpoint already used for local analysis.

The first alternative picks the strongest native interior context response.
The second subtracts the average response of two cyclic rearrangements of
the same three sentences from the response in the original order. The
candidate selects a single interior sentence when its response is unique
and clears its frozen point. Each point was set to the second-highest
untouched work maximum among 20 previously measured reports. The transfer
panel uses a separate bank of 20 previously measured source works. The
follow-up froze 12 previously unused works, four per possible edit
position, after excluding known source commitments and repeated normalized
12-word passages. Unknown training exposure and the focus on one genre
limit generalization. Rotating the sentences changes discourse order and
may change the detector response for reasons other than the edit.
[Earlier source and calibration study](writing-context-margin-transfer-2026-11-13.md) ·
[Publisher benchmark](https://arxiv.org/abs/2606.06481) ·
[Local-evidence methodology](https://arxiv.org/abs/2609.33787)

| Panel | Method | Useful rewrite locations | Wrong-location flags on edits | Untouched excerpts flagged |
| --- | --- | ---: | ---: | ---: |
| Transfer | Existing context windows | 10/40 | 4/40 | 8/20 |
| Transfer | Highest native interior response | 0/40 | 6/40 | 2/20 |
| Transfer | Sentence rotation | 1/40 | 3/40 | 3/20 |
| Fresh | Existing context windows | 13/24 | 5/24 | 9/12 |
| Fresh | Highest native interior response | 2/24 | 9/24 | 5/12 |
| Fresh | Sentence rotation | 0/24 | 2/24 | 0/12 |

All 96 evaluation excerpts were measured by all three methods. Both
alternatives only recovered locations the existing method already found.
Each missed the predeclared requirement to add four locations from three
source works across both generators, with at most two new ordinary flags.
The fresh panel also illustrates the high flag rate on untouched reports
at the existing diagnostic point. These measurements do not estimate a
customer-level probability or modify live confidence.

## Local application verification

Two isolated Full analyses of authored test documents verified the coverage
map after simulated $9.99 payment. The three-sentence and longer routes
recorded 313 and 933 non-whitespace characters with local responses,
respectively. The evidence API, JSON export, Standard PDF and signed
package agreed. Both packages verified against the submitted TXT
originals. A separate writing detector could not load within the isolated
CPU worker's memory budget, so neither run tested that detector.

## Replay

From the repository root, the first command recomputes the frozen
comparison using the public ledger with no model, source text or network.
The native rerun uses the retained publisher snapshot, the two earlier
source rosters, the pinned checkpoint and the application's Python
environment. Use a new trial path and the evaluation ledgers from the
revision publishing this study. The public ledger has model inputs'
hashes and extracted-character positions but no text.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python3 -B -S -m scripts.writing_sentence_rotation_transfer review \
  docs/evaluation-panels/writing-sentence-rotation-transfer-2026-11-13.json
backend/.venv/bin/python -B -m scripts.writing_sentence_rotation_transfer freeze "$trial" --cache "$publisher_cache"
for phase in point transfer fresh; do
  backend/.venv/bin/python -B -m scripts.writing_sentence_rotation_transfer "$phase" "$trial" --snapshot "$checkpoint" || exit 1
done
```
