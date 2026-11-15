# Calibrate writing highlights by the length of a proposed range

The earlier test ranked spans from three words to whole sentences by the
same score. A short and a long span can have different score spreads and a
different number of places where they can occur. This test asks whether
calibrating ranges of the same length against ordinary documents helps
locate a rewrite. It uses the prior causal readings and separately tests
the same reader on student essays.
[Earlier word readings](writing-curvature-ranges-2026-11-15.md) ·
[Length-conditional calibration research](https://arxiv.org/abs/2611.10901)

## Method

Take the prior absolute token-predictability score and the smaller
contrast with the two surrounding word groups. For each score, use the
maximum eligible range per ordinary work at each width of 3, 5, 8, 12,
18 and 24 words. Score other complete sentences as a separate group.
For each genre and width, center the responses on the ordinary median
and scale by the gap from the median to the 90th percentile, with a
minimum scale of 0.25 and at least 12 ordinary works. Apply the same
frozen rule to both score types. Compare each alone and the larger of
the two adjusted signals. The file comparison uses the earlier tighter
and wider ordinary-work budgets. Each calibration work is left out of
its own length profile. Tied maxima withhold a range. A useful location
covers half the changed words and has word intersection over union of
at least 0.30. The scores express relative responses and have no
authorship-probability interpretation.

The retained data includes 144 separate ordinary calibration works,
309 held rewrites and 319 held ordinary files. The essay transfer adds
36 ordinary calibration works and 36 different held original and
rewrite pairs. The two generators contribute 18 distinct held works
each, six per interior sentence position. Each five-sentence pair keeps
the four neighboring sentences unchanged and changes one interior
sentence. The source selection excludes earlier recorded work hashes
and overlapping normalized 12-word phrases. An initial attempt to
match both generators on the same work and text positions yielded only
11 eligible works. The revised source plan was fixed before scoring.
The test isolates controlled edits and records work separation within
the project. Overlap with model training corpora is unknown.
[Pinned essay files](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c/default/test) ·
[Dataset terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

## Results

On the retained panels, calibrating the absolute score by width
improved useful ranges from 20/309 to 40/309 when given the known edited
sentence. Selecting from the full excerpt still found one useful
range at the wider point. The width adjustment replaced that one
location with a different one and reduced ordinary flags from 96/319
to 91/319. Width-adjusted contrast reached 34/309 with the target
sentence supplied, versus 21/309 before adjustment. Searching the full
essay excerpts found zero useful ranges at both points with every arm.

A separate comparison lets the earlier independent classifier select
the sentence and decide whether it qualifies for an alert. The new
word proposals only change the highlighted words. All retained sentence
cutoffs reproduced. Essays use the same ranking and ordinary budgets,
with their own calibration works. [Fixed sentence selector and license](writing-independent-range-transfer-2026-11-15.md)

| Word proposal inside the same selected sentence | Retained useful, tighter → wider | Essay useful, tighter → wider |
| --- | ---: | ---: |
| Previously trained paired head | 1 → 5 / 309 | — |
| Unadjusted predictability | 1 → 3 / 309 | 1 → 1 / 36 |
| Width-adjusted predictability | 2 → 6 / 309 | 3 → 3 / 36 |
| Width-adjusted contrast | 1 → 6 / 309 | 1 → 1 / 36 |
| Larger adjusted signal | 2 → 6 / 309 | 2 → 2 / 36 |

The independent classifier selected the correct sentence in 15 to 48
of the 309 retained rewrites and in five to seven of the 36 essay
rewrites. It flagged 38 to 110 of 319 ordinary retained files and four
to eight of 36 essay originals. The wider adjustment gained six and
lost five ranges against the existing paired proposal on retained files.
It supplied word highlights for 110 ordinary files, three more than
the paired head, under the same 110 sentence flags. On essays the
absolute adjustment gained three and lost one against the unadjusted
proposal. All three gains came from the same generator. At the tighter
point none also flagged the corresponding sentence in its matched
ordinary original. At the wider point one did. The results leave the
word-range candidates in evaluation.
[Numeric results and essay measurements](evaluation-panels/writing-width-calibration-2026-11-15.json.gz)

## Replay

The public receipt reconstructs the width profiles, sentence points,
calibrated comparisons, correct word overlaps and matched ordinary
alerts. It retains Unicode word coordinates and numeric model responses
for the 108 essay excerpts. The native inputs and weights stay local.
Separate native runs reproduced both essay measurements exactly.
Standard-library reviewers on Python 3.12 and 3.14 produced identical
readouts.

```sh
export PYTHONPATH=backend
module=scripts.writing_width_calibration_transfer
python3 -B -S -m "$module" review \
  docs/evaluation-panels/writing-width-calibration-2026-11-15.json.gz
```

For native replay use `essays_gemini-2.5-flash.csv` and
`essays_gpt-5.4-nano.csv` from the pinned test folder. Use the
[previous causal snapshot and MLX runtime](writing-curvature-ranges-2026-11-15.md#replay)
and the [previous ONNX checkpoint and tokenizer](writing-independent-range-transfer-2026-11-15.md#independent-replay).
The two capture commands can run in separate environments with the
matching dependencies and offline model settings.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
backend/.venv/bin/python -B -m "$module" freeze-essays "$run/plan.json.gz" "$gemini" "$nano"
"$mlx_python" -B -m "$module" capture-essays "$run/words.json.gz" "$run/plan.json.gz" "$causal_model"
"$ort_python" -B -m "$module" capture-sentences "$run/sentences.json.gz" "$run/plan.json.gz" "$sentence_model"
backend/.venv/bin/python -B -S -m "$module" assemble "$run/receipt.json.gz" "$run/words.json.gz" \
  --sentences "$run/sentences.json.gz"
```
