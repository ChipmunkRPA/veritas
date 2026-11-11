# Passage evidence on verified report rewrites

Use the matched reports retained by the v257 intake. Each group contributes one human seed and the two publisher-labeled edit stages from each of two generators. The shared human seed is scored once. Human wording outside the tagged rewrite regions provides the within-document control. The publisher's labels cover the rewritten sentence regions, including any wording the rewrite preserved. [Benchmark construction](https://arxiv.org/html/2606.06481v1) · [Pinned dataset](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

Use the same four frozen native passage methods as v196. Compare the current method, its canonical whitespace view, and the two existing unions with the interval head. Keep the existing research points and calibration maps. The models receive only document text and token positions. Project highlighted evidence back to the publisher's Unicode source positions, then report assessed coverage, useful locations, known regions recovered, human characters highlighted, and genuine reports flagged. Report the polish and compression stages separately because their coverage and editing operation both differ. [Previous passage comparison](text-gohumanize-transfer-v196-method.md)

Freeze the exact private case-journal hash, method code, points and intake before scoring. Use up to three complete source groups per CPU batch under the existing local guard. The worker retains native proofs and complete response journals privately. The readout binds the guarded receipts and reports aggregate counts. This run covers government-report excerpts with controlled publisher edits. It does not estimate performance on email, naturally co-written text, or other kinds of articles.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.text_opai_bench_score_v260 preflight
backend/.venv/bin/python -B -m scripts.text_opai_bench_score_v260 freeze
# Run each batch in the frozen batch count in order, starting at zero.
backend/.venv/bin/python -B -m scripts.text_opai_bench_score_v260 run 0
backend/.venv/bin/python -B -m scripts.text_opai_bench_score_v260 readout
```
