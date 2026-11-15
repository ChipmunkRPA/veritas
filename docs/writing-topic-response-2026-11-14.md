# Topic cues and false passage flags in formal writing

The previous paid-worker audit highlighted passages in 47 of 48
ordinary government excerpts. Twenty-four had at least one location
supported by the passage classifier. The writing-pattern ratio
supplied the only supporting response on the other 23. The reports
gave an inconclusive overall assessment on all 96 original and
rewritten uploads.
[Paid-worker audit](writing-sentence-probe-2026-11-14.md#comparison-with-the-shipped-reports)

## What the app shows

The browser and Standard PDF now underline ranges selected solely by
the writing-pattern check. The display calls those ranges
**Predictability measurement**. Formal, templated and edited human
writing can give similar readings. Each packet still supplies the
exact source range, the measured terms and the individual check
responses. The evidence API and JSON use
`review_basis: "writing_pattern_only"` on these packets. The new
presentation uses the already recorded response directions when
displaying older packets.

The local paid-flow test used one genuine government excerpt from
the fresh panel. The browser regression checked that the underline
covered the exact source characters. The isolated Full worker
verified payment before analysis, equality of the evidence API and
JSON, the PDF wording, and the signed source package.
[Local checkout receipt](evaluation-panels/writing-pattern-display-app-2026-11-14.json)

## Topic-conditioned experiment

We compared the exact same sentence in two conditions. The matched
condition uses six terms drawn from the other four sentences in the
paragraph. The neutral condition repeats one generic token to preserve
the same input length and target position. The target text is identical
in both runs. The scorer excludes tokens that directly repeat a matched
term and requires at least 12 remaining tokens. The frozen causal model
supplies four statistics on the loss reduction, plus a comparison with
the other interior sentences. This adapts the matched-versus-neutral
idea to local passage screening. It uses extractive cues, without
importing the paper's code, weights or trained head.
[Matched Conditional-Response Probing](https://arxiv.org/abs/2610.38312)
· [Base checkpoint and Apache-2.0 terms](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

We reused the previous government benchmark for 32 fit and 16
calibration work groups, preserving both deliveries from each work.
The new test uses 48 disjoint government works with one genuine and one
rewritten excerpt each. Gemini 2.5 Flash and GPT-5.4 Nano each supplied
24 rewrites, eight at each interior sentence position. The held
generators differ from those used for fit and calibration. Source selection excluded
earlier work IDs and normalized 12-word overlaps and verified the
publisher's changed-sentence labels. This is transfer to new works and
generators within one benchmark and genre.
[Publisher's methodology](https://arxiv.org/abs/2606.06481)
· [Pinned release and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

Three logistic heads compare conditional features, raw responses and
both together. Each policy keeps classifier-supported locations and
asks whether a score supports forwarding a ratio-only location.
Operating points come from genuine calibration maxima at two fixed
ranks. Useful coverage requires half the known edited sentence and
0.30 non-whitespace character intersection over union. The comparison
uses a replay of the app's local passage rule. On the earlier paid
panel it reproduced the presence or absence of a passage selection on
96/96 files and the useful-location outcome on 46/48 rewritten files.
Exact packet ranges matched on 56/96 files. The fresh-panel table is
an offline comparison.

| Held government excerpts | Useful locations / 48 | Ordinary excerpts flagged / 48 |
| --- | ---: | ---: |
| Current passage-rule replay | 22 | 44 |
| Keep classifier-supported locations | 13 | 24 |
| Topic only, looser point | 19 | 37 |
| Raw only, tighter point | 18 | 37 |
| Combined responses, tighter point | 17 | 35 |

The topic-only rule missed three correct locations the current
comparison found. It reduced ordinary flags by seven. Combining
the responses reduced ordinary flags by nine and lost five correct
locations. Ordinary false selections remained frequent on both
generator groups. The experimental selection rules stay offline.
[Numeric evidence and operating points](evaluation-panels/writing-topic-response-transfer-2026-11-14.json)

## Reproduce

Run the numeric review with standard-library Python to verify source
roles, scalar responses, calibration points and counts. The receipt
contains hashes, positions and measurements without source prose.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/writing-topic-response-transfer-2026-11-14.json
python3 -B -S -m scripts.writing_topic_response_transfer review "$receipt"
```

The native commands use a private working directory. Download
`reports_gemini-2.5-flash.csv` and `reports_gpt-5.4-nano.csv` from the pinned
release's `default/test` directory. Generate `earlier_roster` and the
`earlier_worker_app` state by following the previous sentence-head
worker replay. It supplies the exact source and saved detector
responses for development. The new native run uses Apple GPU inference
with the existing installed checkpoints.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_topic_response_transfer prepare "$run/roster.json" \
  --old "$earlier_roster" --shards "$shards"
backend/.venv/bin/python -B -m scripts.writing_topic_response_transfer topic "$run/roster.json" --output "$run/topic.json"
backend/.venv/bin/python -B -m scripts.writing_topic_response_transfer classifier "$run/roster.json" --output "$run/classifier.json"
backend/.venv/bin/python -B -m scripts.writing_topic_response_transfer ratio "$run/roster.json" --output "$run/ratio.json"
backend/.venv/bin/python -B -m scripts.writing_topic_response_transfer analyze "$run/roster.json" \
  --cue "$run/topic.json" --classifier "$run/classifier.json" --ratio "$run/ratio.json" \
  --prior-db "$earlier_worker_app" --output "$run/receipt.json"
```
