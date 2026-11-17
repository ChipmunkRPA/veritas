# Testing next-token entropy on short rewritten passages

A recent study measures peaks in a language model's next-token entropy for
whole-text classification. This comparison applies the same measurement to
native passages of one to three sentences inside mixed documents. It tests
the peak measurement directly, against nearby passages, and jointly with a
separate token predictability measurement. The scorer uses an installed
Apache-2.0 model with local four-bit quantization. [Entropy study](https://arxiv.org/html/2610.33767v1#S4)
· [Pinned model and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

## Source and operating points

The test divides the sum of local peak prominences by the entropy of the
same span. The nearby version compares that value with the medians of
eligible sentences on each side. The joint version requires both high
entropy prominence and high token predictability. Each comparison adjusts
for the number of sentences and tokens. The prediction pass uses the
extracted text with one initial end-of-text token. Windows need 24 to 192
verified tokens. Documents above 2,048 tokens stay outside this comparison.
Scores rank evidence and are not authorship probabilities.

Eighty-six earlier ordinary excerpts supply the reference scales and two
operating points. Each point uses maxima by source group. An additional
118 earlier matched pairs select a rule before held inference. The frozen
utility values two points per new correct location, minus three for each
additional ordinary file flagged and one for each misplaced addition.
The tighter predictability-only rule ranked first on this comparison. Its
utility was negative even on the earlier pairs. All four methods and both
operating points continue through the frozen held comparison.

The held test includes 48 GAO report pairs outside the previous evaluation
IDs, work digests and extracted text digests. Their ordinary text matches
the dated GovReport transcript, and the labeled rewrites come from two
OpAI-Bench generators. An additional 40 historical prose and Federal
Reserve pairs transfer from an earlier benchmark. Their source groups and
text digests are disjoint from this comparison's calibration and selection.
They have prior exposure to other project evaluations. Training exposure
in the evaluator checkpoints is unknown. [GovReport edition and terms](https://gov-report-data.github.io/#license)
· [OpAI-Bench data and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench)
· [Earlier matched prose and economic reports](writing-local-baseline-transfer-2026-11-16.md)

## Held results

The comparator uses the current writing passage intervals, the within-file
baseline and the economic-report review, with the production calibration
and six-range limit. Each candidate can add one source interval separated
from the existing ranges. A match covers at least half of the labeled
characters and reaches 0.30 intersection over union after whitespace is
excluded. Each range can match only one label. The current style checks
located 19 of 204 known changes and selected five of 88 ordinary files.
They located zero of the 149 changes in the fresh report pairs. These
counts cover the style checks and exclude the separate archive comparison.

| Added method | New locations, tighter → wider | Additional ordinary flags, tighter → wider | Unmatched additions in rewritten files, tighter → wider |
| --- | ---: | ---: | ---: |
| Token predictability | 0 → 0 | 0 → 8 | 0 → 4 |
| Entropy prominence | 0 → 1 | 7 → 11 | 4 → 9 |
| Entropy relative to nearby sentences | 1 → 2 | 7 → 9 | 4 → 5 |
| Entropy and predictability | 0 → 1 | 1 → 5 | 0 → 4 |

The frozen rule assessed 174 of 176 deliveries and added zero locations.
The nearby-sentence check assessed 168 deliveries. Its wider point added
two locations across the 204 labeled changes and flagged nine additional
ordinary files. Both new locations were in the fresh report pairs. The
small gains and larger ordinary-alert totals keep these checks in
evaluation. This test covers English prose and government reports with
known replacements, with limited evidence for other writing styles.

## Independent replay

The published receipts contain source identities, hashes, original
character positions, window-level measurements and outcomes. The local
replay verified 13,914 token windows across the 176 held deliveries. It
reread the original source positions and separately recounted the matches
with a second assignment algorithm. The earlier and held files contain
numeric evidence without source prose or customer data.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/writing-entropy-prominence
date=2026-11-17
python3 -B -S -m scripts.verify_writing_entropy_prominence \
  "$panel-older-$date.json.gz" "$panel-lock-$date.json" \
  "$panel-held-$date.json.gz" "$panel-results-$date.json" \
  "$panel-sources-$date.json.gz"
```

For native inference, use `scripts.writing_entropy_sources` with the pinned
mixed dataset shard, the OpAI report files and official GovReport pack
described in the earlier [short passage replay](writing-short-intervals-2026-11-16.md#replay-locally).
Its `--reference` argument takes this study's public source roster and
restores the earlier exposure list. The retained inputs are the earlier
[mixed passage plan](writing-multi-passage-transfer-2026-11-16.md#replay)
and the short passage plan. Call `add_transfer` with those rebuilt plans
and the [separate article plan](writing-local-baseline-transfer-2026-11-16.md#replay).
`scripts.writing_entropy_prominence capture` uses the pinned local MLX
runtime in the reading environment and accepts `--prior` with the
corresponding baseline reading files. Run `freeze` on the earlier
readings, then use `capture --lock` on the held plan. Add `--native` and
`--tokenizer` to the verifier to retokenize the exact original spans.
