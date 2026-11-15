# Measuring how neighboring sentences affect short rewrite findings

This test measures how each word's representation changes when nearby
sentences are hidden. Each proposed sentence uses a full-excerpt reading
and a second reading of the same upload. In the second view, the attention mask
hides all tokens outside the native sentence and keeps the original
special tokens and position numbers. Both views score the same words.
Counterfactual context probes and separate word-level measurement
motivated the experiment. This is an independent implementation using
the existing commercial-use-reviewed encoder and locally fitted heads.
[Matched conditional response research](https://arxiv.org/abs/2610.38312)
· [Word-level edit research](https://arxiv.org/abs/2609.32625)
· [Encoder terms and existing controls](writing-edit-occurrence-transfer-2026-11-15.md)

The earlier heads used the difference between an entire sentence and
its two neighbors. The new heads use each word's full-context state
minus its state in the isolated view. The same 52,369 input words,
ordinary editorial revisions, word labels, head sizes and selection
rules apply. Each fitted head sets its own two points on 72 untouched
report excerpts before opening the held set. A useful location covers
at least half of the exact changed words and reaches 30% word
intersection over union. The aligned source versions supply the
evaluation labels. Inference sees only the submitted version.

## Report rewrites

| Wider point | Useful local Qwen rewrites /32 | Useful GPT-5.4 rewrites /60 | Ordinary flags /93 |
| --- | ---: | ---: | ---: |
| Earlier direct check | 4 | 0 | 22 |
| Context response, direct check | 5 | 0 | 16 |
| Earlier edit-and-origin check | 4 | 0 | 22 |
| Context response, edit-and-origin check | 6 | 0 | 14 |
| Either edit-and-origin check | 7 | 0 | 27 |

The new edit-and-origin check found three new useful locations and lost
one. One of its six locations also highlighted the same sentence in the
matched ordinary version. Five ordinary reports gained flags and 13
earlier flags cleared. The union keeps all seven distinct Qwen locations
at the cost of five additional ordinary flags relative to the earlier
edit-and-origin check. Every GPT-5.4 comparison found zero useful
locations even though the encoder assessed 57 of 60 rewrites. The one
Qwen3-8B case also yielded no useful location.

## Separate news, abstract and journal tests

The transfer reuses the 116 work pairs and 72 ordinary calibration
excerpts in the existing word benchmark. The native roster checks
exclude work IDs and shared 12-word phrases across roles. Separate
news and abstract points use 36 ordinary files each. The report points
also run on exactly the same inputs to measure the effect of changing
genre. These sources were known to the project and outside both fitted
heads' source works. The two generators in this test also contributed
training outputs in the report domain.
[Source roles and terms](writing-word-localization-2026-11-14.md#separate-calibration-for-news-and-abstracts)

Each cell gives useful locations / ordinary flags at the wider
genre-specific point. Denominators are 56 rewrites and 56 ordinary
versions for news, and 60 of each for abstracts.

| Research check | News | Abstracts |
| --- | ---: | ---: |
| Earlier direct check | 3 / 18 | 5 / 15 |
| Context response, direct check | 0 / 13 | 3 / 18 |
| Earlier edit-and-origin check | 0 / 15 | 4 / 12 |
| Context response, edit-and-origin check | 1 / 13 | 2 / 11 |
| Either edit-and-origin check | 1 / 20 | 6 / 20 |

The new edit-and-origin head added one useful news location without
flagging the matching ordinary sentence. It gained two abstract
locations and lost four. Both earlier and new edit-and-origin heads
flagged two of four excerpts from the two verified eLife revision
pairs. Their direct heads flagged none. Reusing the report point
flagged 29–33 ordinary files per news or abstract combination. The
small gain on reports and the loss in abstract coverage leave the new
head in evaluation.
[Verified genuine journal sources](writing-edit-occurrence-transfer-2026-11-15.md#new-genuine-revision-comparison)

## Independent replay

The public records include source hashes, native positions, operating
points and response scores. The reviewer also reports paired gains,
losses, the union and same-sentence ordinary alerts. Source prose,
token states and fitted weights stay local. A second local run
reproduced the fitted heads and both receipts byte for byte.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
module=scripts.writing_context_response_words
primary="$panels/writing-context-response-fit-2026-11-15.json.gz"
transfer="$panels/writing-context-response-transfer-2026-11-15.json.gz"
python3 -B -S -m "$module" review "$primary"
python3 -B -S -m "$module" review-transfer "$transfer" "$primary"
```

Use the same [earlier fits and verified journal input](writing-edit-occurrence-transfer-2026-11-15.md#replay)
and [news and abstract rosters](writing-context-order-transfer-2026-11-14.md#replay-and-independent-evaluation).

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" run "$report_roster" "$encoder" "$word_weights" "$run/fit"
"$runner" -B -m "$module" transfer "$report_roster" "$matched" "$additional" \
  "$elife" "$encoder" "$word_weights" "$earlier_fit" "$run/fit" "$run/transfer"
```
