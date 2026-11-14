# Testing a second check on missed sentence rewrites

The two current writing checks found different edits. At their raw
research points, taking every positive reading from either one
improved coverage and raised the number of genuine excerpts flagged
sharply. Three additional rules produced no useful locations on the
edits both checks missed. They remain experimental.

## Method

The primary classifier and the writing-pattern ratio use different
measurements. The classifier scores a sentence with its neighbors.
The ratio compares observed token surprisal with cross-entropy from
two existing language checkpoints. It reads the complete excerpt
and reduces its token measurements to each exact sentence. The study
requires coverage of all non-whitespace characters and at least
12 tokens in a sentence. It does not use the app's short-sentence
neighbor fallback. Inference used the existing Apache-2.0 checkpoints
offline at the app's Apple GPU precision. The two raw screening proxies
use the existing reference points of .50 and .9505. Neither point is
a public confidence threshold. Their union measures local screening
coverage, not the paid app's overall verdict.
[Base checkpoint and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
· [Instruction checkpoint and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

The three candidates use the previous token-pooling head, exact
sentence likelihood features, or both. The latter two use small
source-balanced logistic fits. A candidate is evaluated as a
supplement only when the two raw proxies give no positive location.
An exact selected sentence counts as useful. The broader context proxy
uses the existing 0.30 intersection-over-union and half-edit coverage
criteria. EditLens separates provenance and degree of rewriting.
ARB motivates checking human text with local rewrites alongside the
untouched source. No weights or code from either paper were used.
[EditLens](https://arxiv.org/abs/2510.03154)
· [ARB matched study](https://arxiv.org/abs/2607.29539)

The exact same source works and roles from the two preceding panels
were retained. These are new causal measurements on previously
exposed sources. The fit uses 36 designated news and abstract works.
Each of 24 separate genuine works supplies a potential calibration
maximum. Only one of those 24 had neither proxy flag, fewer than the
initially planned eight. Before any held causal inference, the study
amended calibration to use the whole genuine role, with 23 complete
cases. Each rule needs a score strictly above the highest or second
highest genuine maximum. The earlier restriction remains unvalidated.
[Parent cohorts](writing-context-pool-transfer-2026-11-14.md)
· [Separate generator transfer](writing-focus-inside-context-2026-11-14.md)

## Held measurements

| Local screening proxy | Useful locations / 132 | Genuine excerpts flagged / 66 | Transfer locations / 34 | Transfer genuine flags / 34 |
| --- | ---: | ---: | ---: | ---: |
| Current primary context | 22 | 21 | 6 | 10 |
| Current ratio on eligible exact sentences | 53 | 57 | 16 | 31 |
| Either check | 62 | 58 | 20 | 32 |

On the first held panel, the combined proxy gave no positive local
reading on 13 edited excerpts and eight genuine ones. The separate
generator panel had three such edited files and two genuine ones.
Across those 16 missed edits, no candidate cleared either fixed
operating point. The remaining no-flag subset is small in part because
the existing ratio proxy is broad. This makes the evidence a reason to
test better calibration for the ratio, rather than adding an unqualified
second flag.

The source audit also found identical normalized text in some related
deliveries. One labeled edit matched its own original in the first held
panel. Ten other groups had identical outputs from the two generators.
The fixed primary analysis retained the parent deliveries. A sensitivity
analysis removed copies and labeled edits identical to originals in all
roles, then recomputed the new fits and points. The first held denominator
fell to 121 distinct revised deliveries and the combined proxy located
53. Its 58 ordinary flags stayed the same. Candidate recoveries still
numbered zero. The second generator panel had no identical deliveries.
Unknown pretraining overlap and limited essay transfer remain.

## What the app shows on a missed edit

One known rewrite from the prior worker panel passed through local
simulated prepaid checkout. The CPU run recorded five writing contexts
and zero candidate passages. The PDF now says that the checks selected
no passages for review and that AI editing may still be present.
The browser uses the same wording. The API and JSON report the five
scored contexts and zero candidates. The signed package verified
against the actual source. The ratio check skipped under the CPU
memory budget in this run.
[Local checkout and signed report receipt](evaluation-panels/writing-no-selected-passage-app-2026-11-14.json)

## Independent replay

The [numeric receipt](evaluation-panels/writing-missed-passage-transfer-2026-11-14.json)
provides per-sentence causal reductions, input hashes, operating
points and held outcome counts without source prose. The
standard-library replay produced identical output on local Python
3.11, 3.12, and 3.14.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_missed_passage_transfer review \
  docs/evaluation-panels/writing-missed-passage-transfer-2026-11-14.json
```

Native replay needs an Apple GPU, the installed checkpoints, and the
private source rosters reconstructed from the two published intake
procedures. Run with new output paths.

```sh
cd backend
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m scripts.writing_missed_passage_transfer develop "$run/development.json" \
  --context-roster "$context_roster" --transfer-roster "$transfer_roster"
.venv/bin/python -B -m scripts.writing_missed_passage_transfer held "$run/result.json" \
  --context-roster "$context_roster" --transfer-roster "$transfer_roster" --development "$run/development.json"
```
