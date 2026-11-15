# Learning where one sentence was rewritten

Two small heads score an exact sentence from a five-sentence paragraph
using the existing frozen writing encoder. The question is whether
sentence-level supervision can recover edits that a rolling context
score misses. One head learns to classify individual sentences. The
other learns the direction of the same sentence's change between matched
original and edited examples. Both heads read only the submitted text at
evaluation time.

The encoder reads all five sentences together. Each of the three interior
sentences contributes its mean token vector and the difference from the
mean of its two neighbors. The tokenizer must cover every
non-whitespace character in each sentence and may not combine content
from two sentences in one token. The candidate selects one interior
sentence using its unique highest response. The cutoff comes from
whole-file maxima on genuine calibration essays, news and abstracts.
The tighter cutoff allows at most six flags on 63 calibration files.
The looser cutoff matches the incumbent proxy's ordinary-flag count
in calibration. These are internal screening points.

Local comparison with neighboring sentences follows the evidence
principle used in recent change-point work. This study learns a
single-sentence response and evaluates short interior edits.
[LA-CPD](https://arxiv.org/abs/2609.33787)
· [Pinned encoder and Apache-2.0 terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Sources and decisions

The development roster contains 153 fit works from news and abstracts
with Gemini 2.5 Flash and GPT-5.4 Nano revisions. Another 63 genuine
works set the points. The held set contains 73 different works, each with
an original and one rewritten delivery. GPT-5.4 provides 49 revisions
and Qwen3-8B provides 24. The three previously studied genres had too few
unused Qwen works with the required geometry. Before any head fitting,
we added 48 government-report works, evenly split between both held
generators and all three interior positions. The government genre is
absent from training and cutoff selection.

The intake checks the publisher's inline tags and character offsets,
four unchanged neighboring sentences and one changed sentence. It
excludes source groups and normalized 12-word overlap identified in
earlier panels and within the new source roster. These are new source
works from the same benchmark family. Pretraining exposure is unknown.
The held edits comprise 37 polishing, 23 paraphrase and 13 compression
deliveries. The journal audit separately retains 42 dated ordinary
versions and 32 constructed rewrites from its previous paid worker.
[Benchmark methodology](https://arxiv.org/abs/2606.06481)
· [Dataset licensing and provenance](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)
· [Journal source provenance](writing-journal-body-transfer-2026-11-11.md)

A useful location covers at least half the known edit and has at least
0.30 non-whitespace character intersection over union. The new heads
return an exact sentence. The incumbent proxy uses the full text of each
native context that crosses its existing research point.

| Held evaluation | Useful locations | Ordinary files flagged |
| --- | ---: | ---: |
| Incumbent context proxy | 17/73 | 33/73 |
| Proxy plus paired head at the tighter point | 21/73 | 36/73 |
| Proxy plus per-sentence head at the looser point | 29/73 | 57/73 |
| Existing report on journal panel | 18/32 | 33/42 |
| Journal report plus paired head at the tighter point | 18/32 | 35/42 |

All held benchmark and journal cases yielded exact token coverage. On the
73 benchmark edits the paired head at the tighter point gave nine
locations outside the annotated sentence. It added three useful locations
in government reports, across both generators, and one in essays.
The two writing genres used to train the heads had no useful additions
from this arm. The looser per-sentence head added more locations and a
much larger number of ordinary flags. The editor examples show why
source-matched and human revision controls matter when assessing slight
writing changes. No score in this study establishes authorship.

[Sentence-head measurements and offsets](evaluation-panels/writing-sentence-probe-2026-11-14.json)

## Comparison with the shipped reports

All 96 government excerpts went through the local Full worker with
simulated USD 9.99 checkout. No analysis started before payment. Both
writing checks completed on every analysis, the evidence API and JSON
matched, and a signed package from each generator verified against its
submitted text. All 96 overall results were inconclusive.

| Government excerpts | Useful locations / 48 | Ordinary files flagged / 48 |
| --- | ---: | ---: |
| Shipped passage cards | 20 | 47 |
| Paired head alone at the tighter point | 8 | 9 |
| Shipped cards plus tighter paired head | 24 | 47 |
| Shipped cards plus looser per-sentence head | 26 | 47 |

The tighter paired head added one Qwen location and three GPT-5.4
locations. It also selected the wrong sentence in six edited excerpts.
The union appears to preserve the ordinary flag count because the
shipped reports already select passages on almost every ordinary
excerpt. The head has not demonstrated useful standalone separation
on that genre, and it added no useful locations to the dated journal
reports at the tighter point. Both heads remain offline.

[Paid worker comparison](evaluation-panels/writing-sentence-probe-worker-2026-11-14.json)

## Replay

A scalar replay checks source roles, every position, both genuine-only
operating points and the held outcomes. The published ledger has no
source text, token vectors or candidate weights.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/writing-sentence-probe-2026-11-14.json
python3 -B -S -m scripts.writing_sentence_probe review "$receipt"
```

Native replay needs the pinned news, abstracts and essays shards from the
[earlier context study](writing-context-pool-transfer-2026-11-14.md#reproduce)
and the separate generator files from the
[focus study](writing-focus-inside-context-2026-11-14.md#replay).
Download `reports_gpt-5.4.csv` and `reports_qwen3-8b.csv` from the same
pinned release's `default/test` directory. The runner checks the file
hashes. `journal_uploads` points to the `app/uploads` output of the journal
worker replay. The script requires an exact hash match on all 74 local
source files. Keep the roster and learned heads private during replay.
The native study used Apple GPU inference with network access disabled
for model loading.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_sentence_probe freeze "$run/roster.json" \
  --archive "$shards" --test "$held_shards" --uploads "$journal_uploads"
backend/.venv/bin/python -B -m scripts.writing_sentence_probe extend-reports "$run/roster.json" \
  --reports "$report_shards" --output "$run/frozen.json"
backend/.venv/bin/python -B -m scripts.writing_sentence_probe develop "$run/frozen.json" \
  --output "$run/development.json"
backend/.venv/bin/python -B -m scripts.writing_sentence_probe held "$run/frozen.json" \
  --development "$run/development.json" --output "$run/result.json"
```

Use the same frozen roster for the Full worker replay. It runs simulated Stripe
checkout for each government report, compares the evidence API with the JSON
export and verifies one signed source package for each generator.

```sh
worker=docs/evaluation-panels/writing-sentence-probe-worker-2026-11-14.json
python3 -B -S -m scripts.writing_sentence_probe_worker review "$receipt" "$worker"
backend/.venv/bin/python -B -m scripts.writing_sentence_probe_worker run \
  "$run/result.json" "$run/worker" --roster "$run/frozen.json"
```
