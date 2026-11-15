# Word-level writing evidence on matched revisions

This experiment tests whether a single-upload detector can point to changed
words inside a known AI rewrite. It compares those locations and ordinary
alerts against the app's existing writing checks. A word match measures
changed wording against the earlier text. Determining who authored an
individual word requires separate evidence.

## What changed

The prior sentence detector learned one label for each target sentence. This
trial starts from the same Apache-2.0 encoder and supervises individual words.
A monotonic, exact, case-folded word alignment between earlier and later
versions marks inserted or replaced words inside the annotated sentence.
Punctuation-only differences and deleted words are outside that submitted-word
label. At inference the model receives only the submitted text. It scores each
word from its token representations and their differences from the sentence's
mean. Word pieces share their word's weight. The fit also compares the edited
words with the strongest corresponding earlier words. The study implemented
this locally on the qualified report pairs. [Word-level editing research](https://arxiv.org/abs/2609.32625) ·
[Pinned Apache-2.0 encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

On each five-sentence excerpt the six strongest word scores rank each of the
three interior sentences. A unique winning sentence can be highlighted as a
whole or narrowed to its highest-scoring 60 percent of words. A useful
location covers at least half the changed words and reaches 0.30 word
intersection over union. The baseline gets credit when any of its displayed
ranges meets that rule. Pooled selected-word counts include all baseline
ranges. Tight and wider points use genuine calibration excerpts only. All
points are experimental operating settings.

## Reports and ordinary journal revisions

Training used 882 matched report pairs with complete word coverage. Fit
reports contain Gemini 2.5 Flash and GPT-5.4 Nano edits. A separate 72-file
ordinary report set supplied the points before held inference. Assessment
uses the previous held 93 report pairs and 74 dated journal excerpts. The
report baseline replays the existing classifier and writing-pattern selection
rules from previously measured responses. Journal baseline locations come
from saved results of actual local prepaid Full analyses. [Report source roles and terms](writing-matched-encoder-2026-11-14.md#what-ran-locally) ·
[Actual journal analysis](writing-editorial-worker-audit-2026-11-14.md)

| Measure | Existing | Tight word display | Wider word display |
| --- | ---: | ---: | ---: |
| Useful changed-word locations in report rewrites | 10/93 | 3/93 | 4/93 |
| Ordinary report excerpts given a highlight | 89/93 | 1/93 | 15/93 |
| Useful changed-word locations in journal rewrites | 12/32 | 1/32 | 2/32 |
| Ordinary later journal versions given a highlight | 16/21 | 0/21 | 3/21 |

All four wider report locations were new and came from local Qwen rewrites.
Keeping the existing locations alongside them would yield 14/93, with zero
new ordinary report flags. The head found zero useful locations on 60 held
GPT-5.4 edits and the one Qwen3 edit. In the report rewrites the wider whole
sentence readout selected 56 changed words out of 334 marked words. The word
readout selected 41 out of 206, with the same four useful locations. The
head left six paired report excerpts, ten ordinary journal versions and nine
journal rewrites unassessed under its word-coverage requirements. Ordinary
journal excerpts are dated publisher revisions and serve as expected-genuine
controls.

## Separate calibration for news and abstracts

The same fitted head was scored on two other retained panels with **56 news**
and **60 abstract** work pairs. These test genre change using the same two
generator families as training. The panels had earlier project exposure.
The source audit found no shared work IDs or normalized 12-word sequences
between these panels and the primary study, or across the additional source
roles. To calculate genre-specific points, use 36 separate ordinary works per
genre, then hold the points fixed across both test panels. The experiment
uses benchmark genre labels. Automatic genre selection is outside its scope.
[Transfer source and rights checks](writing-context-order-transfer-2026-11-14.md#method-and-results)

| Wider word display | News useful locations / ordinary highlights | Abstract useful locations / ordinary highlights |
| --- | ---: | ---: |
| Existing writing checks | 2/56 · 52/56 | 3/60 · 55/60 |
| Report calibration | 2/56 · 27/56 | 4/60 · 28/60 |
| Separate calibration for each genre | 2/56 · 13/56 | 3/60 · 14/60 |

The genre-specific points retained two new news locations and added a new
ordinary news highlight to the existing checks. They retained three new
abstract locations with zero additional ordinary abstract highlights. Carrying
over the report points added one more abstract location and doubled the total
ordinary highlights from this head in abstracts. Three news pairs were
unassessed by the word head. Each abstract pair was assessed. The additional
locations support keeping the head for further evaluation. Its low standalone
recall on publisher GPT-5.4 and historical journal material keeps it outside
customer confidence scoring.

## Independent replay

The numeric receipts contain source hashes, word ranges, changed-word labels,
scores and operating points. Source prose and fitted weights stay local.
Standard-library Python can recount both comparisons. With the native rosters
it can also reconstruct every word range and changed-word label without
loading a model. A second local CPU run reproduced the exact fitted weight
digest, calibration points and both panels' recorded responses.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/writing-word-localization-2026-11-14.json
transfer=docs/evaluation-panels/writing-word-localization-transfer-2026-11-14.json
python3 -B -S -m scripts.writing_word_localization review "$panel"
python3 -B -S -m scripts.writing_word_localization review-transfer "$transfer" "$panel"
```

To reproduce the model run, restore the native report and journal roster and
pinned encoder from the [sentence study](writing-matched-encoder-2026-11-14.md#replay).
Use the matched and additional native inputs from the [genre transfer](writing-context-order-transfer-2026-11-14.md#replay-and-independent-evaluation).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
module=scripts.writing_word_localization
runner=backend/.venv/bin/python
"$runner" -B -m "$module" run "$roster" "$encoder" "$run/fit"
"$runner" -B -m "$module" transfer "$roster" "$encoder" "$run/fit" "$matched" "$additional" "$run/transfer"
python3 -B -S -m "$module" review "$panel" --roster "$roster"
python3 -B -S -m "$module" review-transfer "$transfer" "$panel" --roster "$roster" --matched "$matched" --additional "$additional"
```
