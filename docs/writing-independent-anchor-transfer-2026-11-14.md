# Reviewing one changed sentence when another edit cannot be matched

The optional local earlier-draft review can now assess a sentence whose
two unchanged neighbors uniquely identify it in both drafts. It can do so
when other revisions change the sentence count or cannot be matched.
Each displayed sentence shows its exact earlier and later locations,
the changed wording highlighted within those locations, and the two
before-and-after responses. The report specifies that other changes may
remain unchecked. These responses carry zero authorship confidence weight.

The app first tries its existing single-sentence and multi-sentence
reviews. The new fallback applies to short English TXT pairs, with
up to four independently matched, nonoverlapping sentence neighborhoods.
Both neighbors must share the target's paragraph on their own side.
Repeated anchor pairs and reordered or overlapping matched neighborhoods
are excluded. The fallback keeps the previous rule requiring both
checks to at least double their AI-like odds. It uses the existing
pinned commercial-use checkpoints.
[Checkpoint terms and earlier method](writing-paired-center-transfer-2026-11-12.md)

## Local validation on new source works

The pinned OpAI-Bench test split supplied 40 news works and 40 scientific
abstracts. Each work contributed one exact edited sentence from Gemini
2.5 Flash and one from GPT-5.4 Nano. The trial composed each edited
excerpt with a separate ordinary revision. The ordinary controls use
historical journal revisions, a recorded technical blog revision, and
same-work substitutions drawn from genuine archival prose. These are
constructed short draft pairs, with two known target positions each.
The news source reader checked both publisher sentence and inline
character labels. For abstracts, the clean text and inline tags
agreed with the character offsets. The publisher's separate sentence
array did not reproduce the shared clean seeds, so admission checked
the edit against the native sentence ranges.
[OpAI-Bench paper](https://arxiv.org/abs/2606.06481)
· [Pinned benchmark](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Ordinary source preparation](writing-sentence-anchor-transfer-2026-11-14.md#frozen-comparison)

| Held cohort | Generated edits | Assessable before → after | Exact locations before → after | New ordinary works flagged |
| --- | ---: | ---: | ---: | ---: |
| News | 80 | 11 → 80 | 3 → 9 | 0/15 assessed |
| Scientific abstracts | 80 | 14 → 71 | 1 → 4 | 0/15 assessed |

Both generator families contributed added locations in both genres.
The ordinary panels reused a roster of 57 works. Fifteen works in each
cohort had an assessed ordinary revision, which limits the false-alarm
comparison.
Measured at the new scope, the two-scale alternative added five exact
locations in each genre and no ordinary flags. On news, it fell short
of the six-edit advancement requirement. On abstracts it outperformed
the context-only arm by two locations but also fell short of that rule.
This first replication supports a local trial of the context-only
fallback. The context-only result remains weak on abstracts. Research
results from the two arms should not be added together, since they
can detect some of the same edits. Single-upload writing received
no accuracy claim from this test.

## Try locally

Set `VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW=true` alongside
`VERITAS_DOCUMENT_PAIRED_SEPARATE_REVIEW=true` and the existing paired
review settings. In a completed retained TXT analysis, open **Compare
with an earlier draft**, choose the earlier TXT, then select **Include
experimental paired writing check**.

A local simulated USD 9.99 checkout ran the full worker after payment.
In the authenticated comparison, the app located the known changed
sentence at Unicode positions [164, 230) and showed both exact wording
changes. The earlier setting returned the pair as outside scope.
The PDF and JSON agreed on the range and the original-byte replay
recomputed the comparisons. The app reports the readings without
checkpoint names.
[App receipt](evaluation-panels/writing-independent-anchor-app-2026-11-14.json)

## Replay

The public receipts contain positions, source hashes and responses.
Their readers recompute the saved decisions and coverage without
inference, and the draft export verifier checks a customer's own
originals. The news and abstracts source shards are pinned to the
same public release. Keep the benchmark files and earlier source
rosters locally.
[News scope experiment](evaluation-panels/writing-paired-scale-news-2026-11-14.json)
· [Abstracts scope experiment](evaluation-panels/writing-paired-scale-abstracts-2026-11-14.json)
· [News independent matching](evaluation-panels/writing-independent-anchor-news-2026-11-14.json)
· [Abstracts independent matching](evaluation-panels/writing-independent-anchor-abstracts-2026-11-14.json)

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.writing_independent_anchor_transfer review \
  docs/evaluation-panels/writing-independent-anchor-news-2026-11-14.json
python3 -B -S -m scripts.writing_independent_anchor_transfer review \
  docs/evaluation-panels/writing-independent-anchor-abstracts-2026-11-14.json
```

Both research commands also support `freeze` and `score` using the
corresponding `--news` or `--abstracts` directory and the retained source
rosters. The independent matcher takes the matching two-scale receipt
via `--prior`.
