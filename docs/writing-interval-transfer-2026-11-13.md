# Finding longer edited passages in a single report

We tested whether a short interval ranker can localize an edited passage
from a single submitted document. It scores consecutive sentences jointly
and keeps their exact decoded character ranges. No earlier draft is used
by the ranker. GigaCheck motivates treating a span as one proposed object.
OpAI-Bench illustrates why cumulative editing and compression need their
own readout. This implementation uses neither project's model weights.
[Interval localization research](https://aclanthology.org/2026.findings-acl.213/) ·
[OpAI-Bench](https://arxiv.org/html/2606.06481v1)

## Frozen method

Two interval rankers use all same-paragraph runs of one to twelve app
sentences with an available flank, at least 60 non-whitespace characters,
and at most 80% of a document's characters. The first uses primary
classifier responses from the app's native three-sentence contexts.
The combined ranker adds the independent checkpoint's response for
each exact isolated sentence. Both use the mean log odds, the difference
from the document median and the stronger adjacent two-sentence flank,
and the interval's share of the document sentence count. They output one
top interval per file, with source order resolving a tie.

The rankers are small standardized logistic fits, with five or eight
coefficients including the intercept. An interval is a positive example
if it covers at least half of all annotated non-whitespace edit characters
and reaches 0.30 intersection over union. Each work and its positive and
negative examples receive balanced weights. The 105 previously consumed
changed report versions and their 20 originals supply fit examples. Sixteen
separate unchanged reports supply each method's second-highest maximum.
The candidate must strictly exceed that point. This operating point is for
the study and yields no authorship confidence. The checkpoint weights
are unchanged and carry Apache 2.0 terms.
[Earlier report inference and checkpoint terms](writing-revision-trajectory-2026-11-13.md)

## Fresh same-publisher reports

The source roster uses Gemini 2.5 Flash and GPT-5.4 Nano from the pinned
OpAI-Bench release. It excludes every earlier Qwen test-shard work and
each retained Gemini/Qwen comparison source ID. A normalized exact
12-word check excludes overlap with earlier works and other admitted
sources. We requested the first 48 remaining shared IDs in salted-hash
order and stopped after 20 admissions. Five of the 27 examined works
had conflicting published tag, sentence, or character representations.
Two had unsupported native text geometry. The 20 admitted works have
one shared original and six exact changed versions each. The stages are
polish, paraphrase, and compression. Matched originals identify source
groups and byte equality and do not enter candidate inference. This is
a fresh source transfer within the same publisher and prose domain.
Training exposure remains unknown.
[Pinned dataset and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[Government report source project](https://gov-report-data.github.io/)

The fresh baseline is the first six native primary contexts at the same
research point as the previous proxy. The consumed development report
also carries the earlier Full worker's exact evidence packets as a
separate comparator.

## Results

| Fresh same-publisher holdout | Primary proxy | Primary-only interval | Two-stream interval | Primary plus two-stream |
| --- | ---: | ---: | ---: | ---: |
| Strict locations in changed versions | 16/120 | 0/120 | 1/120 | 16/120 |
| At least half of one annotated region covered | 81/120 | 1/120 | 2/120 | 81/120 |
| Untouched originals with a highlight | 14/20 | 0/20 | 0/20 | 14/20 |
| Untouched characters highlighted | 15,038 | 0 | 0 | 15,038 |
| Changed versions assessed | 120/120 | 120/120 | 120/120 | 120/120 |

The two-stream interval's only strict location was in a Gemini compression
version, already covered by the primary check. Its other selected range
came from a GPT-5.4 Nano paraphrase and failed the strict criterion. Both
rules assessed every untouched report. The rankers added zero strict
locations, so they remain outside the product. An interval selected with
knowledge of the annotations could meet the strict rule in 117 of the
120 changed versions. That is an upper bound from range geometry and
does not measure detection performance.

On the consumed development bank, the combined ranker found six strict
locations missed by the Full worker across three reports and both earlier
generator families. A union with the primary proxy produced 10/105 strict
locations against 4/105 for the proxy. The candidate alone highlighted
2/20 untouched originals. The primary-only interval added zero strict
locations. That development improvement did not repeat on the fresh works.

## Replay

Local CPU inference ran on 1,752 distinct native context inputs and
1,063 distinct isolated sentence inputs. A second fit reproduced the
frozen file byte for byte. The portable reviewer recomputes the fit and
the results from the numeric receipt. The native reviewer also rebuilds
the publisher source roster, its exact geometry, and every input hash.

Use the two exact local classifier snapshots from the previous report
study and an empty scratch directory. Save the pinned `README.md` and
`croissant.json`, plus `reports_gemini-2.5-flash.csv`,
`reports_gpt-5.4-nano.csv`, and `reports_qwen3-8b.csv` from
`default/test/` in a separate publisher cache. The Qwen shard is used
only for prior source exclusion. The runner verifies dataset and weights
before native use. The [public result ledger](evaluation-panels/writing-interval-transfer-2026-11-13.json)
carries hashes, positions, numeric responses, and decisions. Publisher
text stays local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -B -S -m scripts.writing_interval_transfer freeze "$run"
backend/.venv/bin/python -B -m scripts.writing_interval_transfer intake "$run" --cache "$cache"
backend/.venv/bin/python -B -m scripts.writing_interval_transfer primary "$run" --snapshot "$primary_snapshot"
backend/.venv/bin/python -B -m scripts.writing_interval_transfer independent "$run" --snapshot "$second_snapshot"
python3 -B -S -m scripts.writing_interval_transfer report "$run" --output "$run/result.json"
python3 -B -S -m scripts.writing_interval_transfer review "$run/result.json"
backend/.venv/bin/python -B -m scripts.writing_interval_transfer review "$run/result.json" --cache "$cache"
```
