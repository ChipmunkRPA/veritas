# Locating a rewritten sentence from one upload at varying positions

This comparison asks whether reading the exact sentence tokens gives
better location evidence than attaching a three-sentence response to
its central sentence. The current Apache-2.0 classifier already encodes
every token in both directions. We apply its existing linear head to
the mean of the target sentence's contextual token vectors. In the first
arm, the model reads the sentence with one neighbor on each side. The
second arm also reads the full five-sentence excerpt. The two views
share the same classifier and can share errors. No other weights or
external inference service were used. The Pangram 4 paper motivates
testing multiple context views on source-aligned locations. It trains a
different, dedicated token head. [Pangram 4 technical report](https://arxiv.org/abs/2607.27183)
· [Checkpoint and terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Frozen design

The pinned OpAI-Bench release supplied 126 source works and 378
deliveries. The fit uses 36 works from news and scientific abstracts.
A further 24 genuine works in those genres set the operating points.
The held run uses 18 fresh news works, 18 fresh scientific abstracts,
and 30 essays. Essays are held out entirely. Every held work has one
genuine excerpt and two labeled edited versions, from Gemini 2.5 Flash
and GPT-5.4 Nano. The source edit rotates equally across positions two,
three and four in each cohort. Both labels must agree with the
publisher's inline tags and the app's native sentence boundaries.
The other four sentences must match. We excluded recorded work IDs
and normalized 12-word overlaps against those works and other admissions.
The public receipt contains no source prose. Most edits are light polish. Six of
the 132 held rewrites are compression, so the panel cannot establish
compression performance separately. It also cannot rule out pretraining
exposure. [Dataset methodology](https://arxiv.org/abs/2606.06481)
· [Pinned dataset and license](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

Both arms use fixed ridge-logistic fits. Each fit reads the pooled
AI-like log odds, the difference from the excerpt's other eligible
sentences, and the edited/humanized margin. The baseline is a proxy
using the first six native three-sentence windows at the existing
research point. Each candidate can return one exact sentence. A useful
location covers at least half of the labeled edit's non-whitespace
characters and reaches 0.30 intersection over union. The calibration
uses only genuine files. The conservative point exceeds the
second-highest genuine maximum. The alternate point matches the
proxy's 8/24 genuine flags on the calibration set. Held rates can
differ from that calibration count. These are local study operating
points without a public authorship probability interpretation.
The [numeric receipt](evaluation-panels/writing-context-pool-transfer-2026-11-14.json)
records exact input hashes, observed scores and frozen coefficients.

## Held results

| Cohort | Current proxy, useful | Short pooling, useful | Both contexts, useful | Genuine flags, proxy / short / both |
| --- | ---: | ---: | ---: | ---: |
| News | 8/36 | 4/36 | 7/36 | 5/18 · 6/18 · 7/18 |
| Abstracts | 0/36 | 0/36 | 3/36 | 1/18 · 0/18 · 2/18 |
| Essays | 14/60 | 21/60 | 29/60 | 15/30 · 19/30 · 19/30 |

The table uses the calibration-matched points. Both candidates assessed
all 198 held deliveries. The combined essay arm pinpointed 10/20,
9/20 and 10/20 edits at sentence positions two, three and four,
respectively. The proxy covered 8/20, 1/20 and 5/20. The combined arm
found 21 useful locations missed by the proxy and lost six locations it had found.
Keeping both would increase useful locations to 35/60 and ordinary
flags to 22/30, up from 15/30. Eight rewrites with no proxy flag would
gain only a misplaced location. At its conservative point the combined
arm found 3/60 useful essay locations, including one additional
location, and flagged zero genuine essays. The shorter arm also found
3/60 at its conservative point, including two additional locations, and
selected two misplaced ranges. Neither candidate met the prespecified
false-alarm and wrong-location criteria for an app trial. The wider
pooling is worth investigating as a ranking signal under a limit on
where existing evidence can be presented more precisely.

## Comparison with actual reports

The six first held essay works supplied six unchanged excerpts and six
rewrites. Each generator contributed one rewrite at each tested position.
The regular Full analysis worker completed both writing checks on Apple
GPU. The existing reports gave a useful location on 4/6 edited files,
compared with 1/6 for the proxy and 5/6 for the offline combined candidate.
The candidate added two locations the reports missed and lost one the
reports had located. The reports flagged 5/6 unchanged excerpts, compared
with 4/6 for the candidate. One of the four was new. All twelve
whole-document decisions were inconclusive. These twelve runs share six
works and are a check on the difference between the proxy and actual
customer evidence. They do not estimate the app's general accuracy.

The test ran through simulated USD 9.99 checkout, with zero jobs before
payment and one analysis per paid order. The evidence API and JSON agreed.
Each Standard PDF recorded the two completed writing checks, and each signed
package verified against its submitted source.
[Worker receipt](evaluation-panels/writing-context-pool-worker-2026-11-14.json)

## Reproduce

The standard-library review rebuilds both fits and the two genuine-only
points, checks the roster's role and edit-position balance, and recounts
locations and flags. All four outputs of the portable review suite
matched across the local Python 3.11, 3.12, and 3.14 runtimes. The suite
also includes existing image, video and audio comparisons.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_context_pool_transfer review \
  docs/evaluation-panels/writing-context-pool-transfer-2026-11-14.json
review_dir=$(mktemp -d)
python3 -B -IS backend/scripts/replay_public_evaluations.py "$review_dir/review.json" --suite short-local-evidence
```

Native replay accepts the same six pinned shards as the
[previous study](writing-local-context-transfer-2026-11-14.md#replay).
It checks the pinned model assets and freezes the source roster before
fitting or held inference. The two `--output` paths must be new files.

```sh
cd backend
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -B -m scripts.writing_context_pool_transfer freeze "$run/roster.json" \
  --news "$news" --abstracts "$abstracts" --essays "$essays"
.venv/bin/python -B -m scripts.writing_context_pool_transfer develop "$run/roster.json" \
  --output "$run/roster.development.json"
.venv/bin/python -B -m scripts.writing_context_pool_transfer held "$run/roster.json" \
  --output "$run/receipt.json"
```

With an Apple GPU and the app's pinned local weights installed, rerun the
actual checkout and report comparison against the published receipt.
The final argument is a new private output directory.

```sh
.venv/bin/python -B -m scripts.writing_context_pool_worker "$run/roster.json" \
  ../docs/evaluation-panels/writing-context-pool-transfer-2026-11-14.json "$run/worker"
```
