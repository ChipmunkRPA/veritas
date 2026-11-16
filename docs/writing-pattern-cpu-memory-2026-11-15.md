# Restore the writing-pattern check on a local CPU

The local Full analysis could skip the writing-pattern ratio because the
CPU could not reserve enough memory for both language models. The optional
earlier-draft test reproduced that skip while its separate paired checks
completed. [Previous local result](document-mixed-format-comparison-2026-11-13.md#paired-writing-responses-across-formats)

When a development CPU analysis cannot acquire the original joint lease,
the runner now uses one managed model lease at a time. It keeps a reserved
CPU buffer for the observer's logits while the performer runs. Each
rolling window uses the same source token IDs and the same FP32 loss
reduction as the original path. The two pinned checkpoints, sentence
ranges, explicit shared anchor and internal review point stay consistent.
Memory contention or the analysis deadline can still leave the check
unassessed. The routine may reload both models between windows.
[Method paper](https://proceedings.mlr.press/v235/hans24a.html)
· [Pinned base weights and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
· [Pinned instruction weights and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

## Local results

The original runner skipped a report excerpt with the model budget
set to 7,600 MiB. At that same limit the revised runner completed
the check for two originals, two known rewrites and an ordinary
988-token control that used two overlapping contexts. Both known
rewrites stayed on the non-AI-like side of the existing raw ratio
point. The three ordinary inputs drew no whole-document ratio alert.
All five customer results withheld a whole-document conclusion.
These samples establish recovered coverage.

For the original report excerpt, the previous CPU calculation used
the native model forwards and reproduced the exported document score
at its six-decimal storage precision, all six exact sentence ratios
and the same token-context hash. The native reference also loaded the
two models one at a time. A separate regression test exercised true
joint and staggered leases with controlled logits and matched every
score and rolling range. A forced timeout released both the temporary
reservation and the model lease without publishing a partial score.

The local fixture completed six simulated USD 9.99 checkouts with
zero analysis jobs before payment. The evidence API and JSON agreed
on completed coverage. Two Standard PDFs displayed the check with
its customer-facing label. The receipt records numeric measurements,
source hashes and rolling-context hashes.
[Native worker receipt](evaluation-panels/writing-pattern-cpu-memory-app-2026-11-15.json)

## Replay

Rebuild the paired source plan from the
[official writing test replay](writing-structured-range-transfer-2026-11-15.md#independent-replay).
The worker replay selects one report work and one news work by the frozen
group hashes before scoring. It also joins six ordinary excerpts from the
plan for a longer rolling-window control. The two pinned weights must be
available offline. The replay uses an isolated local database and
simulates payment.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=2
backend/.venv/bin/python -B -m scripts.replay_text_ratio_cpu_memory \
  "$source_plan" docs/evaluation-panels/writing-structured-range-transfer-2026-11-15.json.gz "$scratch/app"
```
