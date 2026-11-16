# Test masked grammar patterns on localized rewrites

Grammar and punctuation sometimes separate fully generated text from human
writing in benchmark studies. Rewrites can preserve the original author's
language patterns. We tested whether a small classifier could locate a rewrite
using function words, punctuation, suffix classes and simple word counts.
Content word identities are masked. The fits use new code and coefficients
built from the retained local intake.
[Cross-domain linguistic features](https://aclanthology.org/2026.findings-aacl.10/)
· [Study of AI generation and editing](https://arxiv.org/html/2608.27855v1)

Three linear fits use 318 matched original and rewritten reports. One uses
counts, one adds ordered sequences of up to four masked tokens, and one
also penalizes score changes across 180 dated ordinary Wikipedia revisions.
Operating points for one- and two-sentence selections come from 72 separate
ordinary excerpts. Each excerpt contributes its strongest selection per
window width. The held test uses 96 previously unused six-sentence excerpts,
balanced across news, reports, GPT-5.4 and Gemini 2.5 Flash. The intake excludes
known work IDs and overlaps of twelve normalized words. GPT-5.4 contributed no
eligible works from its publisher test partition. Its fresh works come from the
publisher's train files and remain separate from the local fits and operating
points. The Gemini works come from the publisher's test files.
[Earlier fit intake](writing-editorial-direction-transfer-2026-11-15.md)
· [Publisher benchmark and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

## Held results

Every original and rewritten excerpt was assessed. A useful highlight covers
at least half the changed words and reaches 30% word-level intersection over
union. The exact sentence count applies to a separate comparison with the
publisher's affected sentences. The experiment uses two fixed operating
points per genre and window width.

| Fit and operating point | Useful highlights /96 | Exact sentence sets /96 | Ordinary flags /96 | Same useful location on original /96 |
| --- | ---: | ---: | ---: | ---: |
| Counts, tighter | 1 | 1 | 8 | 0 |
| Counts, wider | 2 | 1 | 18 | 0 |
| Grammar sequence, tighter | 0 | 0 | 6 | 0 |
| Grammar sequence, wider | 4 | 6 | 23 | 1 |
| Grammar sequence with ordinary revisions, tighter | 0 | 0 | 7 | 0 |
| Grammar sequence with ordinary revisions, wider | 2 | 2 | 21 | 1 |

The wider grammar sequence left 92 of 96 rewrites without a useful highlight.
Accounting for ordinary revisions reduced the number of original alerts by two
and the useful highlight count by two at that point. The three fits remain
evaluation probes. The [earlier-draft comparison](document-paired-blocks-2026-11-15.md)
uses the supplied versions to locate and verify exact wording changes.
[Numeric receipt](evaluation-panels/writing-function-sequence-2026-11-15.json.gz)

## Replay

The standard-library reviewer verifies the frozen operating points and
recounts the effects on source-aligned words. The native replay needs the
local plan from the earlier fit intake. The `sources()` list in
`scripts.writing_function_sequence_transfer` pins the size, partition and
hash of each of six CSV files. Place them under `test` and `train` in `$shards`.
The published receipt contains positions, measurements and masked feature
coefficients. Source excerpts remain in the private inputs.
[Pinned publisher files](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9/default)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.writing_function_sequence_transfer
receipt=docs/evaluation-panels/writing-function-sequence-2026-11-15.json.gz
runner=backend/.venv/bin/python
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$module" fit "$parent_plan" "$run/fit.json.gz"
"$runner" -B -m "$module" freeze "$shards" "$parent_plan" "$run/held.json.gz" --reference "$receipt"
"$runner" -B -m "$module" test "$run/held.json.gz" "$run/fit.json.gz" "$run/receipt.json.gz"
cmp "$receipt" "$run/receipt.json.gz"
```
