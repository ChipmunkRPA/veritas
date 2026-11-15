# Isolate the effect of a sentence in a writing detector

An isolated sentence score ignores the rest of the excerpt. Here a
second reading asks how much the detector changes when one sentence is
hidden from the same excerpt. The trial keeps the token IDs and their
positions and masks attention to the tokens of one exact interior sentence.
It compares the full and masked raw logits. Both readings use the same
68M-parameter MIT-licensed classifier on the CPU. The attention difference
can also reflect topic, length or coherence. The two readings share
weights and can share errors. [Local evidence and authorship boundaries](https://arxiv.org/abs/2609.33787)
· [Pinned classifier and license](https://huggingface.co/Deckard618/textdet/tree/8b677e1bd3bc6421ae710352cda13ca65502147b)

Each of the three interior sentences is evaluated at its exact native
character range. A mask needs tokenizer coverage of every non-whitespace
character in the range and cannot cover neighboring words. The two
readings are ranked against ordinary essays by sentence position. A
third arm uses the smaller rank of the two readings at the same
position. Each arm has its own two operating points, using the maximum
per ordinary calibration work and leaving that work out of its reference
distribution. Ties withhold a location. These are comparison points for
evaluation. [Prior sentence and word method](writing-width-calibration-2026-11-15.md)

## Source separation

The 36 ordinary calibration essays and 36 retained original/rewrite pairs
come from the previous width study. For the separate test, the source
selection was frozen before detector inference. Each pair keeps four
neighboring sentences identical and replaces one interior sentence with
the publisher's exact rewrite. From four pinned test shards, the intake
admits at most six different works per generator and position and
excludes prior retained work identifiers and shared normalized 12-word
phrases. The initial effort to fill equal strata ran out of unused
position-qualified sources. The final source-only roster keeps 18
Gemini 2.5 Flash pairs and six GPT-5.4 pairs, split 3/1/2 by position
for GPT-5.4. GPT-5.4 Nano and Qwen3 do not contribute fresh pairs. The
retained bank already includes GPT-5.4 Nano. These are controlled
single-sentence edits in English student essays from one benchmark.
Overlap with model training corpora is unknown.
[Benchmark paper](https://arxiv.org/abs/2606.06481) ·
[Publisher's release terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

## Results

Every excerpt had complete coverage. Each cell gives the tighter and
wider points, in that order. The word columns use the previous refined
range inside the selected sentence for the retained panel, and the full
selected sentence for the separate panel. Useful overlap requires covering
at least half the changed words with at least 0.30 word intersection over
union. The two panels' word figures describe different evidence extents.

| Panel and check | Correct sentences | Wrong sentences on rewritten files | Ordinary files flagged | Useful word overlap |
| --- | ---: | ---: | ---: | ---: |
| Retained, isolated | 5 → 7 / 36 | 2 → 5 / 36 | 4 → 8 / 36 | 3 → 3 / 36 |
| Retained, hide each sentence | 1 → 2 / 36 | 2 → 10 / 36 | 2 → 8 / 36 | 1 → 1 / 36 |
| Retained, agreement | 5 → 5 / 36 | 4 → 8 / 36 | 5 → 11 / 36 | 2 → 2 / 36 |
| Separate, isolated | 4 → 4 / 24 | 2 → 4 / 24 | 2 → 6 / 24 | 1 → 1 / 24 |
| Separate, hide each sentence | 1 → 3 / 24 | 0 → 2 / 24 | 1 → 4 / 24 | 0 → 0 / 24 |
| Separate, agreement | 0 → 3 / 24 | 3 → 4 / 24 | 2 → 8 / 24 | 0 → 1 / 24 |

At the wider point the masking check gained three separate exact
locations and lost four. Two of the three correct locations also
flagged the same sentence in the matched untouched original. Agreement
added one GPT-5.4 location and one Gemini location, but lost three
Gemini locations and increased ordinary alerts. Neither method
enters the app.

The paired diagnostic compares the score of each sentence between
the original and the rewrite. It uses the known pair only for
analysis. Masking changed the score of at least one untouched
neighbor in all 24 separate pairs. Only three of the 24 had a rise
at the rewritten sentence that exceeded the largest absolute
change on an untouched neighbor. For the isolated reading, all
untouched scores reproduced exactly and 17 rewrites increased
the score of the edited sentence. Even that isolated increase is
insufficient to select the sentence reliably from one document.
[Numeric receipt](evaluation-panels/writing-attention-attribution-2026-11-15.json.gz)

## Replay

The public numeric receipt contains file digests, source positions, exact
changed-word indices, input coverage and raw model responses. The
portable replay verifies the isolated responses and their points against
the earlier receipt. It reconstructs the two other operating points,
the false flags, the paired changes in correct locations and the
localization counts. Python 3.12 and 3.14 produced identical reports,
including the paired diagnostic. A separate native replay rebuilt the
source roster and all model responses byte for byte. The source prose
and model weights stay outside the receipt.

```sh
export PYTHONPATH=backend
module=scripts.writing_attention_attribution
receipt=docs/evaluation-panels/writing-attention-attribution-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, use the four essay shards from the
[pinned test partition](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9/default/test)
and the same classifier files and ONNX Runtime 1.24.3 CPU build as the
[previous experiment](writing-independent-range-transfer-2026-11-15.md#independent-replay).

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/plan.json.gz" \
  "$essays/essays_gemini-2.5-flash.csv" "$essays/essays_gpt-5.4-nano.csv" \
  "$essays/essays_qwen3-8b.csv" "$essays/essays_gpt-5.4.csv"
"$runner" -B -m "$module" capture "$run/receipt.json.gz" "$run/plan.json.gz" "$model"
cmp "$receipt" "$run/receipt.json.gz"
```
