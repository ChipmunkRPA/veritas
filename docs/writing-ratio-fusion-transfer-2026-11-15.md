# Combine writing checks at matched ordinary calibration points

The CPU recovery restored the writing-pattern ratio. That ratio missed
both known rewrites in its local verification. The earlier word
comparison found different useful highlights with the broad and
two-range rules. This study asks whether the measurements complement
each other when each proposed combination sets its own ordinary-file
review points. A recent document-level study motivates an anchored
blend. This evaluation uses its own sentence-level method and the
project's existing commercially licensed model assets.
[CPU recovery](writing-pattern-cpu-memory-2026-11-15.md)
· [Earlier word comparison](writing-structured-range-transfer-2026-11-15.md)
· [Anchored calibration research](https://arxiv.org/html/2610.41459v1#S3.SS6)

## Held sources

The first intake required the whole article to keep the same number of
sentences and admitted two new work pairs. Before native ratio inference,
the expanded intake linked six-sentence excerpts by unique unchanged
first and last sentences and verified both excerpts' exact native
segmentation. One or two adjacent interior sentences must change, and
all unlabeled sentences in the excerpt must match exactly. The intake
checks the publisher's sentence and character labels and derives changed
words by exact alignment. Changes elsewhere in the article can affect
its sentence count without changing an admitted excerpt.

The held set contains 113 original-and-rewrite pairs. Each combination
of news or reports and Gemini 2.5 Flash or GPT-5.4 Nano contributes
28 work pairs. The remaining pair is a Qwen3-8B report. Gemini and Qwen
come from the official test partition. Nano comes from the publisher's
training partition, with all recorded local fitting and evaluation
work groups excluded. The source replay also checks normalized
12-word overlap with the two reconstructed earlier excerpt plans,
identifiable exposed works in the new shards and other selected
originals. These works are held out for this project. Model pretraining
overlap remains unknown.
[Publisher source and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)
· [Benchmark paper](https://arxiv.org/abs/2606.06481)

## What each rule sees

The existing broad word rule picks an interior sentence by its highest
six word scores and highlights its strongest 60% of words. The two-run
rule uses the earlier transition cost and published word-profile scale.
The standalone ratio ranks exact sentences by the observed-to-cross
loss ratio or by how that ratio differs from the other interior
sentences. Each ratio-selected sentence displays the broad word
highlight from that sentence. The ratio uses at least 12 tokens and
complete non-whitespace character coverage.

Two combinations choose the strongest standardized proposal. One
uses the broad and two-run word checks. The other also uses both
ratio variants. The anchored blend adjusts each broad sentence
score by a fixed fraction of the relative ratio response at that
sentence. All proposals read one excerpt at a time. The original
and rewritten excerpts are paired only for label scoring.

The 36 earlier ordinary works per genre supply the median and
90th-percentile gap of each component's work maxima. A separate
leave-one-work-out point is computed for each finished rule at each
of the two ordinary budgets. The points were sealed before held
ratio inference. A useful highlight covers at least half the labeled
changed words and reaches at least 0.30 word intersection over union.
An ordinary file counts as flagged whenever the rule selects any
highlight. The receipt tracks useful locations, misplaced selections,
ordinary flags, within-pair collisions and missing coverage.

## Results

Each cell shows useful rewrite locations and ordinary files flagged out
of 113 each. Seven original excerpts and six rewrites lacked a complete
word proposal. They stay in the denominators. The calibration used 35
assessable ordinary news works and 36 assessable reports per rule.

| Rule | Tighter point | Wider point |
| --- | --- | --- |
| Broad word highlight | 5 · 13 | 7 · 28 |
| Up to two consecutive runs | 4 · 15 | 6 · 29 |
| Absolute writing-pattern ratio | 0 · 5 | 0 · 18 |
| Within-excerpt ratio | 0 · 13 | 1 · 32 |
| Strongest of the two word rules | 7 · 20 | 8 · 30 |
| Strongest of the word and ratio rules | 3 · 14 | 7 · 35 |
| Broad word score with bounded relative ratio | 2 · 11 | 6 · 28 |

At the wider point, the strongest word rule gained three useful
locations and lost two compared with the earlier broad check. It
flagged three new ordinary files and cleared one. Four useful
locations also flagged the corresponding sentence in the matched
original, compared with two for the broad check. The two word rules
did surface different edits. The count of useful locations with a
clear matched original at the selected sentence fell from five to
four.

The standalone absolute ratio found no useful word highlights at
either point. Its relative form produced one wider-point location,
which also flagged the matched original. The strongest rule with a
ratio gained and lost three wider-point locations and flagged seven
additional ordinary files net. The anchored blend found one fewer
useful location at the wider point with the same ordinary flag count.
All of these combinations remain experimental.
[Held measurements](evaluation-panels/writing-ratio-fusion-transfer-2026-11-15.json.gz)

## Replay

The portable reviewer binds the earlier source receipts, recalculates
calibration points and replays the held decisions. Only numeric
scores, hashes and source offsets are published. A separate arithmetic
recount checked all 14 rule and operating-point combinations against
the stored word ranges and paired ordinary decisions. Python 3.12 and
3.14 reproduced the same review. A second native run reproduced the
298 ratio records and the whole receipt byte for byte.

```sh
export PYTHONPATH=backend
module=scripts.writing_ratio_fusion_transfer
receipt=docs/evaluation-panels/writing-ratio-fusion-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, download the six exact files listed under `SHARDS`
at the publisher revision. Store them in `train/` or `test/` under
one directory. Rebuild the two earlier private source plans following
the [word-anchor procedure](writing-word-anchor-transfer-2026-11-15.md#replay)
and the [official-test procedure](writing-structured-range-transfer-2026-11-15.md#independent-replay).
Pass the original frozen receipt to preserve the earlier exposure
inventory. Native ratio inference uses Apple Metal and the two pinned
local causal checkpoints. The native run used PyTorch 2.13.0 and
Transformers 5.15.1. The word step uses the installed encoder and
previous trained checkpoint.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$shards_root" "$development" "$earlier_test" "$run/plan.json.gz" --reference "$receipt"
"$runner" -B -m "$module" verify-source "$shards_root" "$development" "$earlier_test" "$run/plan.json.gz"
"$runner" -B -m "$module" word "$run/plan.json.gz" "$encoder" "$checkpoint" "$run/word.json.gz"
"$runner" -B -m "$module" ratio "$run/plan.json.gz" "$run/word.json.gz" "$run/native"
```
