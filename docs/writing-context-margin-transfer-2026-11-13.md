# Does sentence context transfer to fresh reports?

The earlier contextual check increased useful locations from 10 to 13 of 40 on
one report panel. This follow-up tests whether its sentence scores carry over
to previously unused reports. It also tests whether a sentence deserves a
highlight only when its score is high and clearly above both alternative
sentences. Neither version produced a highlight on the new panel.
[Earlier result](writing-typed-decision-transfer-2026-11-13.md)
· [New outcomes and source commitments](evaluation-panels/writing-context-margin-transfer-2026-11-13.json)

## Comparison

Keep the same typed question, both answer orders and the fixed Apache-2.0
compact checkpoint. Choose the highest-scoring of the three interior sentences
with its immediate neighbors in context. The separation check also requires
that the winner exceed the runner-up by a fixed margin. Recent authorship
segmentation research motivates checking local contrast around suspected
boundaries. This experiment uses a single-sentence comparison and its own code.
[Segmentation research](https://arxiv.org/html/2609.33787v1)
· [Checkpoint and source terms](writing-typed-decision-transfer-2026-11-13.md#inputs-and-measurement)

Use the second-highest genuine-work response from the earlier study's 20
untouched GAO excerpts to set each internal operating point. The context
point was 0.84975 and the separation point 0.49635 on the raw model scale.
The earlier journal-calibrated context point was 0.84095. The experiment
keeps that earlier point as a separate comparator. These points were fixed
before the new excerpts were scored.

The new panel uses 20 unused GAO works from the pinned OpAI-Bench release.
Each supplies one untouched five-sentence excerpt, one Gemini 2.5 Flash polish
and one GPT-5.4 Nano polish at the exact same sentence position. The two edits
retain the same surrounding sentences. The roster excludes prior work IDs and
normalized 12-word matches to earlier eligible originals. Unknown pretraining
overlap remains. The roster admitted 20 of 64 examined reports, with 6, 7 and 7
edits per generator in the three target positions. A planned second fill could
qualify only 12 more works under the existing source and geometry rules. The
first fill was fixed before inference. [Pinned dataset and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Benchmark paper](https://arxiv.org/abs/2606.06481)

## Held results

| Route | Gemini edits located | Nano edits located | Untouched excerpts flagged | Excerpts assessed |
| --- | ---: | ---: | ---: | ---: |
| Existing writing check | 5/20 | 5/20 | 8/20 | 60/60 |
| Earlier contextual point | 0/20 | 0/20 | 0/20 | 60/60 |
| Context calibrated on earlier GAO originals | 0/20 | 0/20 | 0/20 | 60/60 |
| Context with score separation | 0/20 | 0/20 | 0/20 | 60/60 |
| Existing check plus context with separation | 5/20 | 5/20 | 8/20 | 60/60 |

Useful localization requires covering at least half of the known edited
non-whitespace characters and at least 0.30 intersection over union. The
contextual model's highest score over all 60 new excerpts was 0.79375,
below both contextual points. Before applying the cutoff, it ranked the
known edit first on 17 of 40 edits, compared with 24 of 40 in the prior
panel. That imperfect ranking and score shift explain why a stable operating
point needs further testing across fresh material within a genre as well as
across genres. A lower point chosen from these outcomes would consume this
panel for development. The context and margin remain experimental and carry
zero weight in authorship confidence.

## Replay

The published ledger contains anonymous source commitments and sentence
positions, paired scores and decisions. The scalar replay uses the Python
standard library. It recomputes the points from the earlier public ledger
and the held outcomes from the saved scores.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_context_margin_transfer review \
  docs/evaluation-panels/writing-context-margin-transfer-2026-11-13.json
```

For a native rerun, use the same pinned publisher shards and model snapshots
as the earlier typed trial. The original archive of prior ledgers is available
at revision `0e56cf0`. The native selector reads the local set of
`writing-*.json` ledgers, so later ledgers can change exclusions. Copy
`backend/scripts/writing_context_margin_transfer.py` into a checkout of that
revision for an exact source freeze. The local trial evaluated 151 unique
candidate contexts and 217 existing-check contexts.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_context_margin_transfer freeze "$run" --cache "$report_cache"
"$mlx_python" -B -m scripts.writing_context_margin_transfer candidate "$run" --snapshot "$candidate"
backend/.venv/bin/python -B -m scripts.writing_context_margin_transfer incumbent "$run" --snapshot "$incumbent"
python3 -B -S -m scripts.writing_context_margin_transfer publish "$run" --output "$new_receipt"
```
