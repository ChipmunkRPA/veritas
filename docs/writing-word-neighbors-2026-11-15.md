# Can neighboring word scores locate short rewrites more precisely?

The existing experimental check reads one sentence at a time and scores
each word from its encoded state. This comparison tests whether an explicit
reading of the word immediately before and after each candidate improves
localization. It uses the same frozen Apache-2.0 encoder and a small local
head. The comparison draws on research that uses nearby context for sentence
decisions and exact source-aligned words for editing evaluation.
[Sentence context research](https://aclanthology.org/2025.emnlp-main.268/)
· [Word localization research](https://arxiv.org/html/2609.32625v1)
· [Pinned encoder and terms](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

## Comparison

The incumbent head sees each word's encoded state and its difference from
the sentence average. The new head projects each of three word states to
24 values and combines them before scoring the middle word. A control of
identical size displaces the left and right inputs by half the sentence.
A fourth arm averages every incumbent and adjacent-word score. The fitted
heads use the same 1,690 changed words and 15,624 other words, with equal
total weight for the two classes. Only the submitted version enters any
head during evaluation. [Earlier fit and source roles](writing-editorial-direction-transfer-2026-11-15.md)

All arms rank an interior sentence by its six strongest word scores. The
highest-scoring sentence supplies a highlight containing its strongest
60% of words. Useful overlap requires half the changed words and at least
0.30 intersection over union. Two points per genre come from disjoint
ordinary works. The receipt separately records correct sentence choices,
correct word locations, flags on the corresponding ordinary sentence, and
word locations when the true sentence is supplied after scoring.

## Fresh abstract transfer

The transfer froze 48 unused abstract work pairs, 24 per generator and
eight per interior sentence position for each generator. Each original
and rewrite preserves four identical neighboring sentences. A separate
36 ordinary works set the abstract points. The GPT-5.4 Nano and full
GPT-5.4 test shards supplied no eligible unused works under this geometry.
The final panel uses the Gemini test shard and the full GPT-5.4 development
shard. It excludes known work hashes and normalized twelve-word overlaps
against previously recorded abstracts in the source shards and the newly
admitted works before fitting. Checkpoint exposure to the original text
remains unknown. [Source benchmark and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)
· [Publisher's editing methodology](https://arxiv.org/abs/2606.06481)

## Results

Each cell shows useful word locations and flags on the paired ordinary
originals at the wider point. Both counts use the listed number of works
as their denominator. The dated Wikipedia row shows flags before and
after genuine editing, with 120 ordinary excerpts in each version.

| Panel | Incumbent | Displaced neighbors | Adjacent neighbors | Average with incumbent |
| --- | ---: | ---: | ---: | ---: |
| 112 retained news and report pairs | 1 · 31 | 1 · 34 | 0 · 30 | 0 · 35 |
| 84 separate news and report pairs | 2 · 23 | 2 · 25 | 3 · 30 | 1 · 29 |
| 12 retained essay pairs | 0 · 2 | 0 · 1 | 1 · 1 | 1 · 2 |
| 48 fresh abstract pairs | 1 · 14 | 0 · 6 | 1 · 6 | 1 · 10 |
| 120 dated Wikipedia pairs | 5 · 4 | 5 · 7 | 6 · 7 | 6 · 5 |

All 48 abstract pairs were assessed. The adjacent head found a different
rewrite from the incumbent, and both highlights also recurred in the
matching ordinary sentence. At the tighter point all four arms missed
every exact abstract rewrite. They flagged between one and five ordinary
abstracts. Giving the adjacent and incumbent heads the true sentence
after scoring yielded useful word locations on 16 of 48 cases each. On
the 112-work news and report panel, adjacent evidence increased correct
flagged sentence selections from 10 to 13 while losing the only useful
word highlight. The retained 112- and 84-work panels left six and five
matched pairs unassessed respectively. All 12 essays were assessed.

The candidate gained one net useful location on the 84-work panel and
added seven ordinary flags. It gained an essay location that also flagged
the matching ordinary sentence. The adjacent and averaged heads stay in
evaluation. The test measures localized rewrites on English excerpts, with
limited evidence about broad editing or other languages.

## Replay

The numeric receipt keeps per-word responses and source offsets. It binds
the retained roles to the preceding public receipt and includes the fresh
source commitments. Source text, embeddings, and fit weights remain local.
The reviewer recalculates operating points, selected word offsets,
paired gains and losses, and the exact-sentence comparison. The portable
review gave the same output on Python 3.12 and 3.14. A second local CPU
run reproduced the input plan and numeric receipt byte for byte.
[Numeric receipt](evaluation-panels/writing-word-neighbors-2026-11-15.json.gz)

```sh
export PYTHONPATH=backend
module=scripts.writing_word_neighbors_transfer
receipt=docs/evaluation-panels/writing-word-neighbors-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Rebuild the earlier frozen fit and test plan with the
[editorial direction replay](writing-editorial-direction-transfer-2026-11-15.md#replay).
Retrieve `default/test/abstracts_gemini-2.5-flash.csv` and
`default/dev/abstracts_gpt-5.4.csv` from publisher revision
`072c03c7051409346aaba05168d183cf6d863ee9` and place both under `$shards`.
The runner verifies each shard's hash and declared split.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze-abstracts "$shards" "$run/abstracts.json.gz" --reference "$receipt"
"$runner" -B -m "$module" run "$earlier_plan" "$run/abstracts.json.gz" "$encoder" "$run/result.json.gz"
cmp "$receipt" "$run/result.json.gz"
```
