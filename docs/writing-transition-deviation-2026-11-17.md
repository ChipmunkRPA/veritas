# Sentence transitions in short rewrites

A standalone check measures whether a sentence's relationships with its
neighbors differ from other relationships in the submitted text. The
experiment tests whether this can locate short paraphrases whose style
looks ordinary in isolation. The main reading combines semantic and
surface-style relationships. Two separate readings use each kind alone.
This adapts distance-aware transition deviations from a recent
document-level study, with an independently implemented local fit
and source-position evaluation. [Sentence transition research](https://arxiv.org/html/2608.26694v1)

## Fit and operating point

For each eligible sentence, compare its relationships with the sentences
one, two, and three positions on either side. The paragraph scope keeps the
whole neighborhood inside one paragraph. The first run skipped all 54
economic excerpts under that rule. The separate document-order scope
crosses paragraph boundaries and has its own fit and operating points.
Only the older GAO training and calibration groups supply those points.
The semantic reading uses cosine similarity between masked-mean
sentence embeddings. The surface reading uses eight quantities including
word length, vocabulary repetition, function words, digits, and punctuation.
For each distance, normalize using other relationships in the same
scope and at the same distance that exclude the candidate. The fit uses signed
deviations, absolute deviations and left-right differences. An eligible
sentence has 8 to 100 words, a full neighborhood and four comparison
relationships per distance. These are measurements of how a sentence
relates to its surroundings. Editing, quotations and changes in subject
can produce the same pattern. [Pinned Apache-2.0 encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

For fitting only, use 24 original-and-edited pairs in each of eight
combinations of the prior two editing engines and four operations.
Within those works, 981 aligned edits qualify for the paragraph fit.
Subtracting the original
features from the edited features controls for known subject and report
differences during fitting. Each work with eligible edits receives
equal total weight.
At inference, the check sees just one version. Each of 96 separate
unchanged calibration files supplies a maximum sentence response. The
95th percentile sets each reading's operating point before held
inference. Up to six sentences per file can qualify at that point. A
useful range covers half the known edited non-whitespace characters and
reaches at least 0.30 intersection over union. [Source, licensing and labels](writing-token-supervision-2026-11-17.md)

The held benchmark includes 133 original-and-edited document pairs.
Two previously reported local short-rewrite panels provide diagnostic
transfer, one using GAO and the other using BEA and Census originals.
They include 17 same-work human substitutions in the economic panel.
Their earlier results informed the focus on short rewrites, and no
source from either transfer panel enters this fit or calibration.
[GAO short rewrites](writing-token-supervision-2026-11-17.md#further-transfer-to-separately-generated-reports) ·
[BEA and Census rewrites](writing-economic-paraphrases-2026-11-17.md)

## Results

Each cell gives correct edit locations and ordinary files flagged. The
document-order reading measured all 430 evaluation deliveries. The two
fits returned the same 981 aligned training examples and the same
operating points because the development neighborhoods already fit in
one paragraph.

| Panel | Current passage rule | Semantic only | Surface only | Joint | Current plus joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held benchmark reports | 11/911 · 10/133 | 2/911 · 10/133 | 2/911 · 4/133 | 4/911 · 6/133 | 15/911 · 15/133 |
| GAO short rewrites | 5/149 · 0/55 | 3/149 · 6/55 | 0/149 · 1/55 | 0/149 · 0/55 | 5/149 · 0/55 |
| Economic short rewrites | 1/33 · 1/39 | 1/33 · 5/39 | 0/33 · 0/39 | 0/33 · 1/39 | 1/33 · 2/39 |

The joint check added four useful locations and seven misplaced
highlights in the held benchmark. It added zero locations in the two
short-rewrite panels. The paragraph-only results have the same
localization counts for the benchmark and GAO panels. That scope
evaluated 262 of 266 held benchmark files, and its semantic reading
flagged nine ordinary benchmark files. It skipped all 54 economic
deliveries. The document-order reading restored that coverage while
retaining the same low recovery under the frozen points.

As a separate ranking diagnostic, the joint model put the strongest
sentence inside a known edit in 22 of 55 GAO rewrites and four of 15
economic rewrites. A uniform selection among the same eligible
sentences would be expected to locate 18.48 and 4.55 respectively.
The semantic-only reading ranked a correct economic sentence first in
seven of the 15. It found one of the 33 labeled changes at its frozen
point and flagged five of 39 ordinary files. A post-run union of the
semantic and current locations found two extra GAO locations and one
extra economic location. The union added six ordinary GAO alerts and
five ordinary economic alerts. The new readings remain in evaluation.
[Results and replay inputs](evaluation-panels/writing-transition-deviation-2026-11-17.json.gz)

## Reproduce

The portable reader recomputes the operating points and the
non-whitespace character-range scores without source text, models or
Python packages. It uses a second matching implementation for the
localization count. The saved material includes pinned input digests,
position ranges, measured scores and fitted coefficients. It contains
no text or sentence embeddings.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.writing_transition_deviation review \
  docs/evaluation-panels/writing-transition-deviation-2026-11-17.json.gz
```

Run the CPU experiment with the pinned model and source plans from the
earlier studies. The local run used torch 2.13.0, transformers 5.15.1,
NumPy 2.2.6 and scikit-learn 1.9.0.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
study=scripts.writing_transition_deviation
for scope in paragraph document; do
  casepath="$run/$scope"
  mkdir -p "$casepath"
  "$runner" -B -m "$study" develop "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" \
    "$encoder" "$casepath/frozen.json.gz" --device cpu --scope "$scope"
  "$runner" -B -m "$study" apply "$casepath/frozen.json.gz" "$prior/plan/held.json.gz" "$prior/held-results.json.gz" \
    "$encoder" "$casepath/held.json.gz" --panel held
  "$runner" -B -m "$study" apply "$casepath/frozen.json.gz" "$prior/transfer-bound.json.gz" "$prior/transfer-results.json.gz" \
    "$encoder" "$casepath/gao.json.gz" --panel gao_rewrites
  "$runner" -B -m "$study" apply "$casepath/frozen.json.gz" "$economic/held.json.gz" "$economic/result.json.gz" \
    "$encoder" "$casepath/economic.json.gz" --panel economic_rewrites
  python3 -B -S -m "$study" publish "$casepath/frozen.json.gz" "$casepath/held.json.gz" "$casepath/gao.json.gz" \
    "$casepath/economic.json.gz" "$casepath/receipt.json.gz"
done
python3 -B -S -m "$study" bundle "$run/paragraph/receipt.json.gz" "$run/document/receipt.json.gz" "$run/receipt.json.gz"
```
