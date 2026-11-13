# Testing token surprisal dynamics on exact writing edits

Basani and Chen's DivEye measures changes in token surprisal across a
document. This experiment asks whether an ordered signal helps locate a
single rewritten sentence. It uses independently written features and
small regularized fits, with no DivEye code or weights.
[DivEye paper and method](https://arxiv.org/html/2509.18880v2#S3.SS2)

## Frozen comparison

A pinned Apache-2.0 causal language model, locally quantized to four
bits, reads the full submitted
excerpt once. For each exact native sentence, the measurement requires
at least 12 predicted tokens, full non-whitespace tokenizer coverage, and
at most 512 tokens in the full input. A sentence containing the leading
unpredicted token is unassessed. Each file needs at least three assessed
sentences. Twenty of 21 point-selection works qualified. The candidate receives neither the earlier draft nor any
annotation when scoring.

All three fitted heads receive the mean, standard deviation, skewness
and excess kurtosis of the same token surprisals. The ordered head adds
the mean absolute first difference, the standard deviation of second
differences and lag-one correlation. The permuted control adds the same
statistics after a deterministic within-sentence token permutation,
preserving each sentence's surprisal distribution. Each head receives
the absolute features and their differences from the file's feature
medians. The fits balance positive and ordinary sentences within each
source work and give each work equal weight.
[Pinned causal checkpoint and license](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

The fit uses 24 previously evaluated GAO works, each with an untouched
excerpt and two edited counterparts. A separate 21-work set of genuine
dated journal revisions sets each head's operating point at its
second-highest work maximum. The strongest sentence must strictly exceed
that point. The primary comparator is the first six current three-sentence
classifier contexts above its established research point. It does not
recreate the Full worker's complete set of evidence packets.

The held report intake excludes the source groups of both prior short
polish panels, earlier report ledgers, the Qwen report shard and shared
normalized 12-word source phrases. It admitted 20 works from the first
59 of 178 eligible IDs in the fixed intake order. Thirty lacked a common
isolated edit, seven failed publisher geometry checks and two had prior
or within-panel source text overlap. The holdout has 20 originals and
40 exact edits using Gemini 2.5 Flash and GPT-5.4 Nano. The separate
transfer measures 16 genuine forum versions, 12 genuine USWDS versions
and 36 eLife files including 16 admitted rewrites. Those texts had
prior project exposure. The new report works share publisher, genre
and edit generators with the fit. Model pretraining overlap is unknown.
[Earlier short-polish sources and controls](writing-word-edit-transfer-2026-11-13.md)
· [OpAI-Bench release and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Primary checkpoint and license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

A useful range covers half the edit and reaches 0.30 intersection over
union in non-whitespace characters. The frozen advancement rule requires
four additional useful report locations across both generators, at least
two more gains than either control head, at most one extra ordinary group
per held cohort, and at least 90% edited-report coverage. These research
points have no authorship-confidence interpretation.

## Held results

| Outcome | Current primary proxy | Static | Ordered | Permuted |
| --- | ---: | ---: | ---: | ---: |
| Useful Gemini report locations | 6/20 | 0/20 | 0/20 | 0/20 |
| Useful Nano report locations | 6/20 | 1/20 | 0/20 | 0/20 |
| Additional useful report locations over the primary | — | 0/40 | 0/40 | 0/40 |
| Wrong-only selections in edited reports | 10/40 | 2/40 | 4/40 | 1/40 |
| Untouched reports flagged | 11/20 | 1/20 | 2/20 | 0/20 |
| Ordinary forum files assessed | 16/16 | 5/16 | 5/16 | 5/16 |
| Ordinary USWDS files assessed | 12/12 | 10/12 | 10/12 | 10/12 |
| eLife files assessed | 36/36 | 0/36 | 0/36 | 0/36 |

The 40 held report edits were all assessable. The highest response
before applying the point landed on the true sentence in five edits
for both the ordered and static heads and six for the permuted head.
The sole static hit was already covered by the primary comparator.
Adding the ordered head would raise untouched report highlights from
11 to 12 and ordinary USWDS highlights from three to four, without
recovering a missed edit. Many external files lacked three qualifying
sentences. The candidate measured no eLife file and cannot support an
eLife performance claim. All three heads stay outside the app.
[Scalar measurements](evaluation-panels/writing-surprisal-dynamics-2026-11-13.json)

A first publication attempt found that some retained calibration IDs
still used their earlier development namespace. After correcting the
reader, a separate native rerun reproduced all 238 source inputs, feature
vectors and three operating points. The scorer and evaluation rules stayed
the same.

## Replay

For full native replay, use the [earlier native roster and pinned report
shards](writing-word-edit-transfer-2026-11-13.md#replay). The runner checks
the exact prior roster digest and reopens the publisher shards. Use the
locally verified causal and primary model snapshots and the existing
offline MLX setup. Start with an empty private run directory. The public
receipt contains scalar features, source positions and commitments.
It contains no source prose, token identities or raw logits.

```sh
export PYTHONPATH=backend:"$mlx_packages" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.writing_surprisal_dynamics_transfer freeze "$run" \
  --prior-roster "$prior_roster" --cache "$publisher_cache"
for phase in development fit held; do
  backend/.venv/bin/python -B -m scripts.writing_surprisal_dynamics_transfer "$phase" "$run" \
    --snapshot "$causal_snapshot" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_surprisal_dynamics_transfer baseline "$run" \
  --snapshot "$primary_snapshot"
backend/.venv/bin/python -B -m scripts.writing_surprisal_dynamics_transfer publish "$run" \
  --output "$run/receipt.json"
PYTHONPATH=backend python3 -B -S -m scripts.writing_surprisal_dynamics_transfer review "$run/receipt.json"
```

To independently verify the source geometry, run `review` with the backend
Python and supply `--prior-roster` and `--cache`. The public reader on its own
recomputes the points and outcomes from the released feature vectors and
fitted coefficients. Native review also recomputes the fitted coefficients.
