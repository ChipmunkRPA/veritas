# Test learned prompts on known sentence rewrites

The local experiment compares how well two learned prompts explain the
submitted wording. One prompt is fitted on ordinary sentences and the other
on their rewritten counterparts. At evaluation time both read the same
submitted sentence and its preceding sentence. The difference in their
average log likelihood ranks each native sentence. The original paired
version supplies labels for evaluation. The check reads one upload at a time.

The design adapts the likelihood-ratio comparison in SPADE to short
replacements and matched ordinary controls. The experiment uses the
project's existing Apache-2.0 1.5B causal checkpoint. The model remains
frozen, and the fit updates two 12-token prompt embeddings. No code or
weights from the paper enter the experiment. [SPADE, November 2026](https://arxiv.org/html/2611.16929v1)
· [Causal checkpoint and license](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

## Comparison

The prompts start from the same neutral initialization. In each source
pair, each prompt is fitted to reduce its own class's average negative
log likelihood and to favor it over the matched other class. Training uses
the exact changed sentence, with the same preceding sentence in both
versions. The final run uses four-bit affine weights with float32
activation and gradient calculations. The first, lower-precision run
encountered a nonfinite update and produced no fitted artifact.

The held comparison uses the four interior positions of each six-sentence
excerpt. The current classifier reads three sentences at a time around
each position. The three causal alternatives compare unconditioned
likelihood, fixed instructions, and learned prompts. An additional arm
takes the stronger of the current and learned-prompt responses at each
position after normalization. All methods can select one sentence.
Per-genre and per-position scales come from ordinary texts only. The two
operating points use leave-one-work-out ordinary maxima. The result counts
an edit as located when the selected sentence is the exact labeled sentence,
including the ordinary matched version when it identifies the same place.
The research scores carry no authorship-confidence meaning.

The pinned OpAI-Bench release supplies 48 fit works, 52 new held works
and separate ordinary calibration. The held sources include news and
research abstracts, with rewrites by GPT-5.4 and Gemini 2.5 Flash. The fit
uses Gemini examples only. The intake verifies both publisher character
and sentence labels, unchanged neighboring sentences and the native text
extraction. It checks past public evaluation work hashes and normalized
12-word overlaps within available source material. The source roster
records shortfalls explicitly. No Nano or essay rewrite qualified for
this held test. The tests cover one rewritten sentence in an otherwise
unaltered excerpt. Broader transfer, pretraining overlap and behavior on
arbitrary uploads remain unverified. [Publisher terms and benchmark](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

## First held comparison

| Method | Exact sentence, tighter → wider | Ordinary excerpts flagged, tighter → wider | Wider locations repeated at the same place in the matched original | Wider locations with at least 30% changed words |
| --- | ---: | ---: | ---: | ---: |
| Current passage classifier | 2 → 3/52 | 7 → 13/52 | 2 | 0 |
| Plain likelihood | 2 → 5/52 | 3 → 8/52 | 3 | 0 |
| Fixed prompts | 2 → 4/52 | 3 → 11/52 | 4 | 1 |
| Learned prompts | 3 → 8/52 | 3 → 13/52 | 5 | 2 |
| Current and learned readings | 2 → 7/52 | 7 → 13/52 | 3 | 1 |

At the wider point the learned prompts ranked 22 of 52 edited sentences
first before applying the ordinary reference point, versus 11 for the
current classifier. Five of their eight flags at the correct location
repeated at that location in the corresponding ordinary text. Of the
eight selections, only two cover a sentence in which at least 30% of
the destination words are inserted or replaced. This is a limit on
the precision of a whole-sentence highlight.

The pooled ordinary rate hides opposite behavior by genre. On Gemini
abstracts the learned method had one ordinary flag versus seven current.
On news excerpts paired with GPT-5.4 rewrites, it flagged ten matched
originals versus five current.

## Transfer to reports

The second test added 48 report works, 24 for each rewrite generator.
It used the same fitted prefixes and the same two operating rules.
Twenty-four separate ordinary reports supplied the genre calibration.
The learner's fit contained no report excerpts. The intake excluded
works and normalized source-phrase overlaps from the first comparison.

| Method | Exact sentence, tighter → wider | Ordinary excerpts flagged, tighter → wider | Wider locations repeated in the matched original |
| --- | ---: | ---: | ---: |
| Current passage classifier | 4 → 4/48 | 17 → 17/48 | 4 |
| Plain likelihood | 1 → 2/48 | 1 → 6/48 | 2 |
| Fixed prompts | 0 → 1/48 | 1 → 12/48 | 1 |
| Learned prompts | 1 → 1/48 | 0 → 4/48 | 1 |
| Current and learned readings | 2 → 4/48 | 9 → 16/48 | 4 |

The wider point across both panels locates **9/100** rewrites with
the learned method versus **7/100** current, with **17/100** versus
**30/100** matched ordinary excerpts flagged. The combined method
locates **11/100** and flags **29/100** ordinary excerpts. Seven of
those eleven correct locations repeat at the same place in the
matched original. None of the transfer arms flags an exact report
edit that clears this matched-original check at the wider point.
At the tighter point the learned reading has one GPT-5.4 report
location that clears the matched-original check and zero ordinary flags.

At the known position, the learned reading rises from original to
rewrite on 67 of 100 works, versus 65 for the current reading.
On reports both read higher on 30 of 48. Across the wider comparison
just two of the nine correct learned sentence flags cover wording
with at least 30% inserted or replaced words. The study supports
further evaluation of sentence ranking. The new learned reading
stays experimental.

## Replay

The public ledgers contain hashes, sentence offsets, word-overlap totals,
ordinary calibration and scores. The ledgers omit source text and
learned weights. The portable reader reproduces both outcomes without
the models. The native rebuild checks the publisher files against
their SHA-256 digests and matches every committed excerpt by its Unicode
range and text hash before scoring.
[News and abstract ledger](evaluation-panels/writing-prompt-contrast-held-2026-11-16.json.gz)
· [Report ledger](evaluation-panels/writing-prompt-contrast-reports-2026-11-16.json.gz)

```sh
export PYTHONPATH=backend
module=scripts.writing_prompt_contrast
panels=docs/evaluation-panels
python3 -B -S -m "$module" review "$panels/writing-prompt-contrast-held-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/writing-prompt-contrast-reports-2026-11-16.json.gz"
```

For native replay, use an Apple Silicon runtime with MLX 0.32.4,
mlx-lm 0.31.3 and Transformers 5.22.1. Put the publisher test shards at
`$sources/<genre>/<genre>_<generator>.csv`. Place the supplemental
GPT-5.4 news and reports train shards, abstracts dev shard and essays
test shard under `$additional` with the original filenames. The Gemini
reports test shard goes in the same supplemental directory. All
shards come from the pinned publisher revision and their digests are
checked during reproduction.

The local rebuild matched all 385 distinct excerpts. The refit reproduced
both learned prompts and all 32 training losses byte for byte. The learned
method assessed 794 of 800 held sentence positions and at least one
position in every held file.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MLX_ENABLE_TF32=0
"$python" -B -m "$module" rebuild "$receipt" "$sources" "$additional" "$run/inputs.json.gz"
"$mlx_python" -B -m "$module" train "$run/inputs.json.gz" "$causal_snapshot" "$run/prefix.safetensors" "$run/fit.json"
"$mlx_python" -B -m "$module" score "$run/inputs.json.gz" "$causal_snapshot" "$run/prefix.safetensors" "$run/causal.json" "$run/fit.json"
"$python" -B -m "$module" incumbent "$run/inputs.json.gz" "$passage_snapshot" "$run/current.json"
"$python" -B -S -m "$module" merge "$run/inputs.json.gz" "$run/causal.json" "$run/current.json" "$run/replay.json.gz"
```

The current passage classifier uses its pinned project release.
Both snapshots are bound by the local runner.
[Passage checkpoint and license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)
