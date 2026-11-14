# Reading one sentence alone and with its neighbors

This local trial compares two ways to find an AI-edited sentence in
one submitted text. Each candidate scores each eligible sentence alone
and with its immediate neighbors. One candidate uses one writing
checkpoint, and the other uses both installed writing checkpoints.
The earlier draft is used during source preparation only. The method
borrows the distinction between sentence and context signals explored
in recent writing-localization research. It uses the project's pinned
Apache-2.0 checkpoints and its own fitting code.
[Global and local writing signals](https://aclanthology.org/2025.ijcnlp-long.188/)
· [Checkpoint terms and pins](writing-paired-center-transfer-2026-11-12.md)

## Sources and comparison

The pinned OpAI-Bench test split supplied 60 news works, 60 scientific
abstracts, and 48 essays. Each work supplied a genuine seed excerpt and
one labeled center revision from each of two generators. Exact inline
tags and publisher character positions agreed. All five sentences had
to map to the app's native segmenter and the four surrounding sentences
had to match between versions. We excluded the reportable work IDs
from the preceding paired study and removed normalized 12-word
overlaps among selected and excluded seeds. This produces constructed
five-sentence excerpts with one known revision and covers English only.
The shared authorship benchmark does not establish pretraining exposure.
[Benchmark methodology](https://arxiv.org/abs/2606.06481)
· [Pinned benchmark data](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

In each training genre, 24 separate works fit the two regularized
comparisons and 16 more genuine seed excerpts fixed both operating
points. Each arm uses the sentence and context response, their
difference, and the two responses relative to other interior
sentences. The highest qualifying sentence can become an inspection
location. All 48 essay works are out of genre and out of fit.
A qualifying location covers at least half of the known edit's
non-whitespace characters and has at least 0.30 intersection over
union. The comparator is the primary local writing check and its
current context rule.

The separately verified macOS worker accounting used six held essay
deliveries from two source works. All six finished after simulated
USD 9.99 payment. Both existing learned checks completed. The worker
displayed passages on both genuine originals and usefully located one
of four known rewrites. All six overall outcomes were inconclusive.
The evidence API and JSON export agreed on the source positions. The
larger checkpoint comparison below uses the primary context rule as a
proxy. The six native runs show why its counts should not be read as
whole-app accuracy.
[Native worker receipt](evaluation-panels/writing-local-context-worker-2026-11-14.json)

## Held results

| Held source | Known rewrites | Existing useful locations | Primary-only candidate | Two-check candidate | Genuine excerpts highlighted, existing / primary / two-check |
| --- | ---: | ---: | ---: | ---: | ---: |
| News | 40 | 12 | 0 | 0 | 10/20 · 0/20 · 0/20 |
| Scientific abstracts | 40 | 2 | 0 | 0 | 3/20 · 0/20 · 0/20 |
| Essays | 96 | 35 | 0 | 0 | 21/48 · 1/48 · 0/48 |

Both candidates assessed all held files. The primary-only candidate also
highlighted two essay rewrites at the wrong sentence. Both candidates lost
all useful locations at their frozen operating points, so they stay
outside the app.
[Frozen measurements](evaluation-panels/writing-local-context-transfer-2026-11-14.json)

## Reported check coverage

An isolated CPU run completed the passage classifier and skipped the
writing-pattern ratio under the local memory budget. On Apple GPU the
same submitted text completed both checks. The browser, Standard PDF,
JSON evidence and signed package now identify the completed or skipped
check using generic labels. The PDF and JSON matched in both simulated
prepaid runs and the signed exports replayed against the original text.
This records the available evidence and does not change the assessment.
[Coverage verification](evaluation-panels/writing-check-coverage-app-2026-11-14.json)

## Replay

The receipt binds the public benchmark selection, the
pinned model reference, original input hashes, source offsets and
raw responses. It includes no source prose. Model-free review
recomputes both fits, genuine calibration points and held counts.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_local_context_transfer review \
  docs/evaluation-panels/writing-local-context-transfer-2026-11-14.json
```

For a native rerun, place each genre's two CSVs in its own directory.
The source reader checks the six pinned shard hashes. Run the two
phases in order so the genuine work points are recorded before held
inference.

```sh
cd backend
.venv/bin/python -B -m scripts.writing_local_context_transfer freeze "$run/roster.json" \
  --news "$news" --abstracts "$abstracts" --essays "$essays"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m scripts.writing_local_context_transfer develop "$run/roster.json" \
  --output "$run/roster.development.json"
.venv/bin/python -B -m scripts.writing_local_context_transfer held "$run/roster.json" \
  --output "$run/receipt.json"
```
