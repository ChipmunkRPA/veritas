# Token-sequence likelihoods for short rewrites

Can a small language model locate an AI rewrite in a single uploaded
document, without the earlier draft? This experiment learns which word
sequences are more likely in original sentences or revised versions. It
compares independent word probabilities, ordered token sequences, and
the same ordered score with extreme token contributions limited. The
last arm adapts a method from a recent study on detecting machine-written
text under distribution changes. This implementation and its sentence
localization evaluation are independent. [Research paper](https://arxiv.org/abs/2609.29935)

## Training and calibration

Training uses 5,434 aligned original and rewritten sentence pairs from
800 public benchmark documents, with 2,752 revisions from one generator
and 2,682 from the other. Each of the two models is a token-count table.
At inference the sequence model estimates a token's likelihood given up
to two preceding tokens, borrowing from the shorter history when there
are few observations. All three arms average the token log-likelihood
ratios over each eligible sentence. The clipped model caps each token's
contribution at ±0.5 natural log units. The shared vocabulary contains
3,477 tokens, with rare words grouped together. The trial uses no
downloaded model weights. [Sources, licensing and labels](writing-token-supervision-2026-11-17.md)

Each of 96 source-disjoint unchanged calibration files supplies a maximum
sentence score per arm. The 95th percentile supplies a separate internal
operating point for each score. The implementation, settings, data
digests and calibration readings were frozen before this run opened the
three diagnostic panels. Each arm can return at most six sentence ranges
per document. A useful range covers at least half of a known edit's
non-whitespace characters and has at least 0.30 intersection over union.
Each label can count once. All 430 deliveries had at least one eligible
sentence, with eligibility limited to 8–100 words per sentence.

## Results

Each cell gives correct edit locations and ordinary files flagged. The
ordinary economic controls include human substitutions to test whether
editing alone raises a signal. All three panels were previously used in
project diagnostics and this run supplies no fresh confirmation.
[GAO transfer](writing-token-supervision-2026-11-17.md#further-transfer-to-separately-generated-reports) ·
[Economic transfer](writing-economic-paraphrases-2026-11-17.md)

| Cases | Current passage check | Unigram | Ordered tokens | Clipped ordered tokens | Current plus clipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held benchmark reports | 11/911 · 10/133 | 9/911 · 9/133 | 9/911 · 20/133 | 1/911 · 1/133 | 12/911 · 11/133 |
| GAO short rewrites | 5/149 · 0/55 | 0/149 · 0/55 | 1/149 · 3/55 | 0/149 · 0/55 | 5/149 · 0/55 |
| Economic short rewrites | 1/33 · 1/39 | 0/33 · 2/39 | 0/33 · 5/39 | 0/33 · 2/39 | 1/33 · 3/39 |

The ordered score raised more ordinary alerts in every panel than the
unigram score. Clipping reduced alerts and recovered just one edit. Adding
it to the current rule added that location, three misplaced highlights and
one ordinary alert in the benchmark. It added zero locations in the short
rewrite panels and two ordinary alerts in the economic panel. All three
variants stay in evaluation. The result concerns this count-based model
and operating point on these reports. The paper tested a different task,
classifying generated text under distribution changes.

## Reproduce

The public receipt stores hashed inputs, internal operating points,
sentence coordinates and scores. The reader recomputes operating points
and independently counts edit locations without the source text or the
fitted token vocabulary. [Evaluation receipt](evaluation-panels/writing-clipped-markov-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.writing_clipped_markov review \
  docs/evaluation-panels/writing-clipped-markov-2026-11-17.json.gz
```

Rebuild the pinned native inputs and incumbent scores from the
[earlier report study](writing-token-supervision-2026-11-17.md#reproduce-the-local-comparison)
and [economic transfer study](writing-economic-paraphrases-2026-11-17.md).
Use empty output paths for the new freeze and result.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.writing_clipped_markov develop \
  "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" "$run/freeze.json.gz"
python3 -B -S -m scripts.writing_clipped_markov evaluate \
  "$prior/plan/fit.json.gz" "$run/freeze.json.gz" "$run/result.json.gz" \
  "$prior/plan/held.json.gz" "$prior/held-results.json.gz" \
  "$prior/transfer-bound.json.gz" "$prior/transfer-results.json.gz" \
  "$economic/held.json.gz" "$economic/result.json.gz"
```
