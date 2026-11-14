# Predicting masked words with both sides of an edited sentence

This experiment tests whether an AI-rewritten sentence receives less lexical
support from its two real neighbors. The check reads a single uploaded excerpt.
For each interior sentence it samples up to eight evenly spaced content words
that can be mapped exactly. Each word has at most four tokenizer pieces, and
the same pieces must match across both inputs. The model predicts one piece
while that piece and following pieces of its word are masked. The earlier
pieces and other words stay visible. The exact sentence is read alone and again
with its two native neighbors. [Masked language model scoring](https://aclanthology.org/2020.acl-main.240/) · [Within-word masking](https://aclanthology.org/2023.acl-short.80/)

A small earlier-generation Apache-2.0 BERT model supplies the masked-word
readings. The scorer compares trained sentence-only features with an arm
that also receives the gain from real surrounding text. No primary detector
response, previous version or author identity enters candidate inference.
The mask model and native production classifier use different checkpoints.
[Original BERT release](https://github.com/google-research/bert#pre-trained-models) · [Pinned 45 MB weights and terms](https://huggingface.co/gaunernst/bert-mini-uncased/tree/0772d26477d98b5896774fa570b581100fb60dcc)

The fitting split uses 24 previously evaluated GAO works. Historical journal
pairs supply each internal operating point. A single-piece restriction covered
only 15 of the 21 initially selected ordinary works. Allowing multi-piece
words covered 16, below the required 17. Neither feasibility attempt scored
any held case or emitted a cutoff. The final procedure preserves those 21
sources and adds all 14 earlier F1000Research calibration works, each with a
fixed five-sentence view around its first unique shared center. The point is
the second-highest genuine work response before held scoring. These response
points do not estimate the probability of authorship. [Dated revision controls](writing-temperature-transfer-2026-11-13.md) · [Additional F1000 controls](writing-long-position-transfer-2026-11-12.md)

Twenty new GAO works came from a salted order of 158 eligible report IDs. The
intake processed 55 IDs, excluding five with unsupported publisher geometry,
three with normalized overlap and 27 without a same-position isolated edit.
Earlier matching reports and the Qwen shard were excluded before the order
was formed. Each work includes one untouched excerpt and exact Gemini and
Nano versions that independently polish the same sentence. All five sentences
have the same flanks across versions. The older GAO, eLife, forum and USWDS
cohorts appear separately as retrospective transfer. [Pinned publisher data and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) · [Comparable earlier report study](writing-surprisal-dynamics-transfer-2026-11-13.md)

## Results

| Held outcome | Existing context proxy | With sentence-only mask check | With two-sided mask check |
| --- | ---: | ---: | ---: |
| Fresh known edits located | 7/40 | 7/40 | 7/40 |
| Fresh untouched files flagged | 8/20 | 9/20 | 10/20 |
| Previously scored GAO edits located | 20/80 | 23/80 | 24/80 |
| Previously scored GAO originals flagged | 21/40 | 21/40 | 22/40 |

Each candidate independently assessed 38 of the 40 new edits and 19 of the
20 new originals. The two-sided arm flagged four of the new edited files
and selected the wrong sentence each time. Its four additional retrospective
locations came from three older works and both generator families. The
improvement did not repeat on the fresh works. The candidate assessed only
two of 16 genuine forum versions and four of 12 genuine USWDS versions. Its
three-center requirements excluded every eLife file. Neither arm met the
frozen advancement rule. Both remain outside serving.

## Replay

The final local run fitted 23 of 24 development works and set the points on
27 of 35 ordinary work maxima. It measured 6,038 distinct masked inputs in
fitting and point selection, 9,598 distinct masked inputs in assessment, and
205 unique incumbent contexts for the new reports. A separate direct audit
checked 33 multi-piece masks against the full, unpadded model. The largest
log-probability difference was 0.0000087. The scalar receipt contains source
commitments, native ranges and saved scores without model inputs, word
identities or source excerpts. The portable reader verified every rounded
calibration maximum, cutoff and held decision exactly on macOS and Linux.
The Linux refit differed in a few coefficients by at most `1.12e-16`.
The reader allows at most `1e-14` absolute drift in a coefficient and requires
both sets of coefficients to reproduce every recorded location and count.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_bidirectional_transfer \
  docs/evaluation-panels/writing-bidirectional-transfer-2026-11-13.json
```

To rebuild from retained source cases, use an empty run folder. The `--cache`
input uses the pinned OpAI-Bench release with the Gemini, Nano and Qwen test
report shards and the release metadata described in the prior report studies.
The three local case paths refer to the word-edit, surprisal-dynamics and
long F1000 position case files referenced above. Model snapshots are pinned
by every supplied file hash. The trained head and tokenizer can run offline.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=backend/.venv/bin/python
$python -B -m scripts.writing_bidirectional_transfer freeze "$run" --cache "$cache" \
  --prior-a "$word_edit_roster" --prior-b "$surprisal_roster" --additional-revisions "$f1000_cases"
$python -B -m scripts.writing_bidirectional_transfer development "$run" --snapshot "$masked_snapshot"
$python -B -m scripts.writing_bidirectional_transfer point "$run"
$python -B -m scripts.writing_bidirectional_transfer held "$run" --snapshot "$masked_snapshot"
$python -B -m scripts.writing_bidirectional_transfer baseline "$run" --snapshot "$primary_snapshot"
$python -B -m scripts.writing_bidirectional_transfer publish "$run" --output "$run/receipt.json"
```
