# Exact sentence likelihood with nearby and distant context

The current check and all four experimental rules located zero of 14 held one-sentence rewrites at their frozen operating points. At two matched false human groups, the combined ranker located one rewrite and the native-score ranker located zero. That retrospective result did not meet the frozen advancement criterion. [Results and independent replay](evaluation-panels/writing-conditional-context-transfer-2026-11-12.json)

## Frozen comparison

Reuse the 55 known-origin plain-text files from the earlier multi-paragraph study. The older 12 archived works supply fitting data. Ten separate works supply 24 held files, comprising ten originals and 14 exact rewrites. Eight held rewrites come from one generator and six from the other. Both sets had previous project exposure in other experiments. This conditional method had no earlier held responses. The retained study contains the publisher and generation terms and exact passage qualification. [Prior source qualification](writing-multiparagraph-localization-transfer-2026-11-12.md)

A pinned causal model predicts each target sentence after three prompts. The first contains its actual preceding sentence. The second contains a distant sentence from the same article with similar length. The third uses the same scaffold and an empty context. Every comparison requires identical target token IDs and complete coverage of the exact source sentence. Eligible foci contain 12–65 words and have both a preceding and a following sentence. Candidate conditions contain 8–80 words. The test borrows the idea of a matched and control condition from recent research. This implementation measures forward token likelihood on already available source text. [Matched conditional-response probing](https://arxiv.org/html/2610.38312v1) · [Pinned probe checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

The absolute arm uses the mean target log probability from the empty context. The conditional arm adds the improvement with the preceding sentence over both the empty and distant conditions. The native-score arm uses the existing context score. The combined arm uses all four features. Each uses the same standardized logistic fit. The fit uses 148 eligible original sentences and 17 eligible known replacements from the selection works. Each arm selects a single focus and uses a selection point permitting at most two of the 12 original work groups to flag. The earlier first-six native contexts are carried through as a separate comparison.

A useful location covers at least half of the edit and reaches 0.30 intersection over union in non-whitespace characters. The advancement criterion requires three added locations in separate held works across both generators, with at most one additional genuine work group flagged. Equal false-group comparisons show retrospective ranking on the same held cases.

## Held results

| At the frozen points | Current first-six contexts | Absolute | Conditional | Native-score rank | Combined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful locations on 14 rewrites | 0 | 0 | 0 | 0 | 0 |
| False groups among ten genuine works | 5 | 0 | 0 | 2 | 1 |
| Flagged only at an incorrect location | 5 | 0 | 0 | 3 | 2 |

All five arms assessed all 24 held files. Adding any candidate to the current selections preserved five false human groups and added zero useful locations. At the retrospective budgets of zero, one, two, and three false groups, combined locations were 0, 0, 1, and 1. The other fitted arms were zero at each budget. The one combined hit came from the same model family as the probe. The held technical prose and scripted edits are a limited test. The candidate remains experimental.

## Reproduce

The completed run scored 1,030 exact prompts and 27,102 observed target token positions, with 699 comparable foci. Two continuous-run attempts ended before preserving a complete readout. The batched version froze byte-identical case and prompt lists, bounded the runtime cache, and saved completed results between processes. The 1,030 forwards completed with a measured peak of 3,378,129,928 bytes in the MLX allocator. The independent reviewer reopens the original text, verifies 2,097 prompt occurrences, checks the token reductions, and uses a separate numerical optimizer to reproduce 72 outcome counts.

Use the retained prior article panel as `parent`, the pinned local causal checkpoint as `snapshot`, and an unused private path as `run`. The Metal stage uses the MLX versions pinned in the runner. Keep raw text, responses, and token IDs local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B -m scripts.writing_conditional_context_transfer freeze "$parent" "$run" "$snapshot"
while [ ! -f "$run/readings.json" ]; do
  python -B -m scripts.writing_conditional_context_transfer score "$parent" "$run" "$snapshot" || exit 1
done
python -B -m scripts.writing_conditional_context_transfer report "$parent" "$run" "$snapshot"
python -B -m scripts.review_writing_conditional_context_transfer "$parent" "$run"
```
