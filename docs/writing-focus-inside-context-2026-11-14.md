# Choosing a sentence inside an existing writing finding

We tested whether narrowing an already flagged context to one sentence
would help readers find short AI edits. Three policies use the frozen
[contextual token pooling fits](writing-context-pool-transfer-2026-11-14.md).
Each needs the winning sentence to sit inside an existing AI-like
context window. The first uses the combined head. The second also
requires the shorter head to choose the same sentence. The third
requires both conditions and a winning margin of more than .10 in
the combined fit's units. We selected that margin on the previously
exposed writing panel before the new test. All methods keep the
original context as evidence. The research points are internal and
give no authorship probability.

## Transfer to different sources and generators

The pinned OpAI-Bench test shards provided 34 new work groups, with
one genuine excerpt and one labeled single-sentence rewrite per work.
The parent fit used Gemini 2.5 Flash and GPT-5.4 Nano. This test uses
Qwen3-8B and full GPT-5.4, with each represented equally within each
domain and edit position. The intake excluded recorded source hashes
and matching normalized 12-word phrases. It checked publisher inline
tags and character labels, native sentence boundaries, and the four
unchanged sentences. Two initially selected pairs had no text change
after normalization. The review caught them before a result was
written, so the final intake required a changed normalized sentence.

The Qwen shard offered few unused works with the required five-sentence
geometry. The final set has 14 news works, 18 abstracts and two essays.
Both essay edits were at the second sentence. This is a limited test
of essay transfer. No checkpoint was refitted. The test uses the same
pinned Apache-2.0 classifier as the prior study and runs offline.
[Publisher methodology](https://arxiv.org/abs/2606.06481)
· [Pinned dataset and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Checkpoint terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Results

The current native context proxy flagged 10 of 34 edited files and 10
of 34 genuine ones. Its windows usefully covered six known edits.
The table counts exact reading locations inside those windows.
A wrong location is a different sentence in an edited file. A focus
on a genuine file is a narrowed suggestion on an already flagged
file. Every policy uses the same existing file flags.

| Focus policy | Exact edited sentence / 34 | Wrong sentence in an edited file / 34 | Genuine file given a focus / 34 |
| --- | ---: | ---: | ---: |
| Highest eligible native response | 5 | 5 | 10 |
| Combined token pooling | 4 | 3 | 7 |
| Pooling with both views agreeing | 3 | 2 | 5 |
| Agreement plus clear winning margin | 1 | 2 | 5 |

None selected a correct sentence the native policy missed. The combined
policy changed one location to a wrong sentence. All three remain
offline. The largest remaining gap on this panel was coverage. The
context proxy gave no flag at all on 24 of 34 edited excerpts, so a
rule that narrows existing findings cannot recover those cases.

## Writing check explanations in the app

The user-facing evidence now explains recognized reasons for a skipped
or inapplicable writing check. It covers analysis memory, an unavailable
check, verified credentials, the language decision, and missing extracted
text. The browser and Standard PDF use the same fixed wording for each
reason code, and the JSON includes the code. Older signed reports keep
their original evidence schema. No reason can include raw diagnostic
messages or model names.

A simulated prepaid Full analysis used the normal local CPU worker. The
passage classifier completed and the ratio check exceeded the configured
memory budget before inference. The API, JSON, and PDF agreed on the reason,
and the signed package verified against its submitted text. Separate
local tests used the pinned signed PDF to confirm the credential bypass
in Quick, Full and Deep, and checked unsupported-language handling.
[CPU checkout verification](evaluation-panels/writing-check-reason-app-2026-11-14.json)

## Replay

The [numeric receipt](evaluation-panels/writing-focus-inside-context-2026-11-14.json)
carries the input hashes, source offsets, model responses, rules and
results. Review checks the role and source commitments and recalculates
each comparison without models or source prose.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_focus_inside_context review \
  docs/evaluation-panels/writing-focus-inside-context-2026-11-14.json
```

To repeat native inference, place the six Qwen3-8B and GPT-5.4
`news`, `abstracts`, and `essays` test CSV files from the pinned
release's `default/test` directory in a separate cache. The runner
checks their hashes and the installed model assets.

```sh
cd backend
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m scripts.writing_focus_inside_context freeze "$run/roster.json" --cache "$cache"
.venv/bin/python -B -m scripts.writing_focus_inside_context run "$run/roster.json" --output "$run/result.json"
```
