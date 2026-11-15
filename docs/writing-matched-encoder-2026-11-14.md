# Fine-tuned sentence localization on matched report rewrites

This experiment asks whether a more precise single-upload check can find a
rewritten sentence inside an otherwise unchanged report. The compact candidate
stays offline. It identified too few exact locations to displace the app's
passage checks. Evaluating the existing measurements on these government
reports also exposed a high rate of ordinary-passage flags.

## What ran locally

The fit used a pinned, Apache-2.0 six-layer MiniLM encoder. We updated the
encoder and its head together for four epochs on Apple Metal. The head combines
the candidate sentence with its difference from the surrounding sentences.
The loss scores each sentence and compares an original with its corresponding
rewrite. Training used 930 complete pairs out of 960 prepared from distinct
government works. Each example had five sentences, with exactly one of the
three interior sentences changed and the other four unchanged. We checked that
the tokenizer represented every non-whitespace character. [Pinned encoder and
license](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

The two operating points were set from 72 separate ordinary development works.
They allow up to 10% and 25% ordinary flags on that development panel. These
are research settings for this detector. They carry no customer confidence
meaning. Held reports were separated by source identifiers and shared
twelve-word spans. The publisher test contributes 60 GPT-5.4 pairs and one
eligible Qwen3 pair after excluding previously used or overlapping works.
We also prepared 32 fully verified local rewrites on unused reports with
Qwen2.5-Instruct, 17 light edits and 15 paraphrases. A related checkpoint is
part of the existing writing-pattern measurement, so the local panel should
be read with that dependency in mind. The 74 journal versions use saved
outputs from the existing Full worker. [Pinned dataset and reuse
terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [DetectRL robustness paper](https://arxiv.org/abs/2410.23746)

## Results

| Held material | Existing checks | Candidate at 10% point | Combined at 10% point | Candidate at 25% point | Combined at 25% point |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact rewrite locations in reports | 49/93 | 4/93 | 53/93 | 8/93 | 56/93 |
| Any ordinary report flag | 89/93 | 8/93 | 89/93 | 26/93 | 90/93 |
| Known generated journal locations | 18/32 | 3/32 | 18/32 | 6/32 | 19/32 |
| Any ordinary journal flag | 33/42 | 3/42 | 33/42 | 18/42 | 38/42 |

Report baselines replay the current serving rules on newly measured responses.
Journal baselines are actual previous report results. A candidate location
must match the exact changed source sentence. A combined success requires
either check to locate the change using its own rule. The denominator includes
six ordinary and six rewritten reports that the candidate could not assess
within its input coverage limits. The one held Qwen3 pair cannot establish
transfer for that generator. These are targeted excerpt panels.

## Replay

The [numeric receipt](evaluation-panels/writing-matched-encoder-2026-11-14.json)
includes source hashes, held response values, output offsets and fit
commitments. It contains no source text, model weights or raw embeddings.
The replay recalculates the development operating points and held outcomes.
It can run using only Python's standard library from the repository root.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_matched_encoder review \
  docs/evaluation-panels/writing-matched-encoder-2026-11-14.json
```

A native rerun requires the six input artifacts pinned by the script and the
CSV shards named in its `SOURCES` table, alongside the prior
[sentence-head](writing-sentence-probe-2026-11-14.md) and
[topic-cue](writing-topic-response-2026-11-14.md) local rosters and journal
worker inputs. Set `PYTHONPATH=backend` and use the backend virtual environment.
Run `scripts.writing_matched_encoder freeze` to build the source roster.
Optionally run `local` on that roster to add fresh local rewrites, then run
`train` with the pinned encoder directory. Run
`scripts.writing_topic_response_transfer classifier` and `ratio` on the
identical final roster, and pass both outputs and the training measurements
to `scripts.writing_matched_encoder analyze`. Each command's `--help`
documents its paths.
