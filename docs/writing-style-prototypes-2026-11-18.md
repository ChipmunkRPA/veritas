# Style embeddings for short rewrites

This experiment tests whether a model trained to represent writing style
can locate a short AI rewrite in a single upload. The encoder was trained
to help reconstruct earlier wording from a paraphrase alongside a separate
semantic representation. The trial uses the released MIT weights and a
new local fit. It uses none of the paper's training data or detection heads.
[Paraphrase inversion research](https://arxiv.org/abs/2606.10099) ·
[Pinned encoder](https://huggingface.co/rrivera1849/LUSR/tree/ab7852825ad1be40112bbfdaf1f9e2315a1727b7)

## Fit and scope

The source commitment selects 24 works in each of eight combinations of
editing operation and generator. The selected sources yield 1,323 aligned
earlier and revised sentence pairs across 192 reports. Each sentence is
embedded separately with masked mean pooling. The direction from the mean
original style to the mean revised style uses the same variance scaling for
both groups. Scaling pools the within-group variances and shrinks them
halfway toward their median. Each eligible sentence receives the cosine
similarity to that direction after centering and scaling. A second reading
subtracts the median score of the other sentences in the same upload. It
requires at least three other eligible sentences. The model sees only the
one submitted version at evaluation. [Source and label audit](writing-token-supervision-2026-11-17.md)

Ninety-six unchanged reports from disjoint sources set the two internal
operating points at the 95th percentile of their file maximums. Each point
selected four calibration reports. The fit, source and baseline digests,
settings, and points were saved before the held readings. The SHA-256 of
that freeze is `d7b7c2168754297e568ef1f44bce5fd4a4b10f6b3b7a329c1d3d6df22d083938`.
An eligible sentence has 8–100 words. The check can return up to six
sentence ranges. A useful location covers at least half the labeled
non-whitespace characters and reaches 0.30 intersection over union. Each
label can count once.

## Results

Each cell gives correctly located edit sentences and ordinary files
flagged. Both readings assessed every file. All held panels are disjoint
from the fit and calibration sources. Earlier experiments also measured
these panels. The last two panels come from separate short-rewrite editors.
[GAO short rewrites](writing-token-supervision-2026-11-17.md#further-transfer-to-separately-generated-reports) ·
[BEA and Census short rewrites](writing-economic-paraphrases-2026-11-17.md)

| Panel | Current passage check | Style | Document-relative style | Current plus style | Current plus relative style |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held benchmark reports | 11/911 · 10/133 | 7/911 · 13/133 | 5/911 · 3/133 | 18/911 · 22/133 | 16/911 · 12/133 |
| GAO short rewrites | 5/149 · 0/55 | 0/149 · 2/55 | 0/149 · 1/55 | 5/149 · 2/55 | 5/149 · 1/55 |
| Economic short rewrites | 1/33 · 1/39 | 0/33 · 0/39 | 0/33 · 0/39 | 1/33 · 1/39 | 1/33 · 1/39 |

Adding the document-relative style result found five additional edits,
added three misplaced ranges, and flagged two additional ordinary files
on the benchmark. Neither style reading found a short rewrite at its
frozen operating point. Neither flagged any of the 17 economic controls
with same-article human-written substitutions. These controls were
constructed using existing publisher prose. Naturally occurring editorial
revisions need a separate test.

As an exploratory ranking check, the top reported style score overlapped a
known edit in 15 of 55 GAO rewrites and four of 15 economic rewrites. Uniform
selection among eligible sentences would be expected to locate 11.36 and
2.70 files respectively. At the tested ordinary alert rate this ordering
produced no useful transfer flags. The style check remains experimental.
[Evaluation receipt](evaluation-panels/writing-style-prototypes-2026-11-18.json.gz)

## Replay

The public reader recomputes the operating points and uses a separate
one-to-one overlap algorithm to reproduce the counts. The receipt stores
hashes, fitted numerical coefficients, source positions and scores. It
omits original text, model weights and per-sentence embeddings.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_style_prototypes review \
  docs/evaluation-panels/writing-style-prototypes-2026-11-18.json.gz
```

The native run used Apple Metal, PyTorch 2.13.0, Transformers 5.15.1,
and NumPy 2.2.6. Use the exact source plans and baseline outputs from the
preceding studies. Use empty output paths for the new freeze and result.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
runner=backend/.venv/bin/python
study=scripts.writing_style_prototypes
"$runner" -B -m "$study" develop "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" \
  "$encoder" "$run/frozen.json.gz" --device mps
"$runner" -B -m "$study" evaluate "$run/frozen.json.gz" "$encoder" "$run/result.json.gz" \
  "$prior/plan/held.json.gz" "$prior/held-results.json.gz" \
  "$prior/transfer-bound.json.gz" "$prior/transfer-results.json.gz" \
  "$economic/held.json.gz" "$economic/result.json.gz"
```
