# Narrowing an existing audio alert by measuring changes in detector response

The current short-audio review reports a two-second interval when it
flags a recording. This study tests whether a second look can narrow
that interval to a useful one-second excerpt.
Each alternative can only move the displayed range inside that alert.
The measurements describe dependence on particular input features and
carry no AI-authorship weight.

Previous work used a derivative of the response to estimate which
frequencies contributed to it. This study measures the actual response
after a finite change to the input. Research shows that the replacement
values can themselves affect audio classifiers. This trial compares two
replacement rules with the same model and tests the inferred ranges on
genuine speech and matched splices. It uses neither code nor weights
from the cited paper.
[Earlier local derivative trial](audio-input-contribution-transfer-2026-11-14.md) ·
[Research on masking effects](https://arxiv.org/abs/2609.33486)

## Fixed local method

The four-second comparison takes one original context containing the
whole alert. Choose the context with more samples at its nearer end
outside the alert, preferring the earlier context on a tie. The
two-second comparison uses the exact crop that raised the alert. Both
check three one-second windows at half-second steps within the alert.
Each needs at least half detected speech. In each one-second window,
replace the upper 40 of 80 normalized Mel bands with either their
normalization mean or the median of the nearest available flanking
features. Both replacements taper over five feature frames at each
edge. The input outside the interval and the lower bands remain as
measured. The same full context and attention positions enter every
forward pass.

Each perturbation produces one signed change in the original context's
logit. The first candidate ranks the lower of the two drops. A second
candidate also requires the same unique first-ranked window under both
replacements. The timing comparator ranks each window by the median
of the original frame responses. Both finite candidates require a
positive drop. The strict and wider points use the top and second
genuine group scores in the development panel. That leaves only two
genuine calibration groups after conditioning on an existing alert.
When fewer than two groups support the same-winner rule, reuse the
two-fill points. The wider finite point reaches the zero floor. These
are small-sample research operating points.

The study uses the project's existing commercial-use-reviewed audio
weights. The model and original ranges in the public source receipts
are pinned. A useful location covers at least half the true edit and
reaches 0.30 temporal intersection over union. Genuine splices and
background-only files have separate counts. For each new useful range,
the review also tests whether a matched genuine recording keeps a
useful range at the edit's clock.
[Model asset audit](commercial-model-audit.md) ·
[Previously studied speech and noise banks](audio-shared-mel-context-2026-11-14.md) ·
[Primary splice sources](audio-short-edit-detector-training-2026-11-14.md) ·
[Separate SINE readers and terms](audio-narrow-flag-transfer-2026-11-14.md)

## Held results

The locked roster contains 44 development deliveries and 331 held
deliveries. The held inputs include 106 staged edits. The current
review flagged 31 and usefully localized seven. The other 75 had no
existing flag for a narrowing rule to use. Each native pass examined
91 distinct broad alerts across development and holdout. The underlying
recordings have earlier project exposure. This is a new perturbation
test on those retained inputs.

At the wider point the four-second comparison produced the following
location counts. Ordinary counts are refinements of existing flags.

| Held edit set | Useful existing | Useful after unperturbed frame ranking | Useful after both finite fills | Ordinary ranges changed by both fills |
| --- | ---: | ---: | ---: | ---: |
| Randomized stock, competing speech | 1/16 | 1/16 | 0/16 | 1 |
| Three generators, competing speech | 0/12 | 1/12 | 0/12 | 1 |
| Primary matched splices, competing speech | 4/48 | 7/48 | 4/48 | 12 |
| Later SINE readers, clean | 2/15 | 3/15 | 2/15 | 0 |
| Later SINE readers, competing speech | 0/15 | 1/15 | 0/15 | 0 |

The finite changes added zero useful locations, including at the strict
point. At the wider point they lost the existing useful stock location.
Requiring the same winner under both fills produced the same held counts
for the four-second comparison. The unperturbed frame ranking added six
useful locations at the wider point. Three had a matched genuine alert at
the same edit clock, and this ranking narrowed all 42 existing ordinary
alerts in the primary panel.

Using the exact two-second crop gave two additional useful locations in
the 48-edit primary panel at the wider point, bringing it from four to
six. One matched an ordinary alert at the edit clock. The other was
**1.500–2.500 s** for an edit at **1.680–2.150 s**. The rule also narrowed
ten ordinary alerts in that panel and two in the randomized stock panel.
It preserved the seven useful existing held locations across the full
roster and added none in the other held sets. The same-winner rule
selected only the colliding additional location. The strict points
added none. The original two-second forward matched all 27 archived
scores that were available for this check within 0.00000008 probability.

The unperturbed frame ranking in the two-second crop added five useful
locations, including two in the later SINE set. Four additions had no
matched genuine alert at the same clock. Even at the strict point,
it narrowed 33 existing ordinary alerts in the primary panel. Both
context lengths and the unperturbed comparisons remain in evaluation.

## Replay

The numeric receipts contain source digests, score changes and exact
sample ranges. The Python reviewer checks the source roster and replays
the decisions. The Node.js reviewer separately derives six cutoffs per
receipt and recounts results from the numeric readings. Both reviewers
reproduced the held counts in a clean workspace with only the public
scripts and receipts.
[Four-second receipt](evaluation-panels/audio-finite-window-attribution-2026-11-15.json) ·
[Two-second receipt](evaluation-panels/audio-finite-window-attribution-same-context-2026-11-15.json)

```sh
for receipt in docs/evaluation-panels/audio-finite-window-attribution*-2026-11-15.json; do
  PYTHONPATH=backend python3 -B -S -m scripts.audio_finite_window_attribution review "$receipt"
  node backend/scripts/review_audio_finite_window_attribution.cjs "$receipt"
done
```

For native replay, reuse the two checked stock source folders and the
three-generator input bank from the earlier derivative study. The `sine`
argument is the prepared later SINE roster with its original source WAVs
and isolated backgrounds. The `data` argument is the prepared primary
input bank with its `sources.json` and `pcm` directory. Keep publisher
recordings and model checkpoints in the local source folders.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_finite_window_attribution score "$run" \
  --archive "$archive" --sine "$sine" --data "$data"
```

Add `--context-seconds 2` and use a new output directory for the
same-context trial.
