# Locate separate rewritten passages in one document

This test measures whether a single submitted document can support two
separate, correctly located writing findings. It compares isolated
sentences and the current classifier's three-sentence inputs with
an interval selector that can return several source ranges. The
selector draws on change-point research.
It uses an existing classifier and evaluates source layout separately.
[Local-evidence change-point research](https://arxiv.org/html/2609.33787v1)
· [Earlier single-range comparison](writing-sustained-response-2026-11-13.md)

## Pinned sources and scope

The pinned benchmark contributes 49 held excerpts, each with two known
generated replacements and an otherwise source-matched original.
They come from 36 historical prose excerpts across three author
groups and 13 Federal Reserve excerpts across six report dates.
Another 54 ordinary excerpts from four separate author groups and
three separate report dates supply the calibration. The held
selection tracks the publisher's Claude and GPT backend labels.
For requested-only labels, the launched model identifier is recorded
and the serving checkpoint remains unspecified.
[Pinned benchmark and attribution notes](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans/tree/c8a59b129c34cfd9c67f739ae049abfff26fc2d3)

Source text is limited to the recorded U.S. public-domain Gutenberg
texts, three pinned Standard Ebooks editions of older works, and
Federal Reserve Beige Book prose from the Board website. The source
metadata pins the original source links and recorded terms. The
publication includes extraction hashes, geometry and scalar
responses. The document text and model weights stay local.
[Gutenberg source terms](https://www.gutenberg.org/policy/license.html)
· [Standard Ebooks terms](https://standardebooks.org/contribute#copyright-and-licensing)
· [Federal Reserve website terms](https://www.federalreserve.gov/disclaimer.htm)

Every mixed span covers exactly its published Unicode range. All
unchanged spans match the source text at the publisher's source
offsets. Each mixed and ordinary document has its published layout
and a publisher-supplied neutral layout. Across the 152 layout pairs,
the text agrees after removing whitespace, and 79 pairs produce a
different list of scored inputs after the app's text extraction and
sentence segmentation. The held median is 56.5 sentences. The range
geometry could match both labeled passages in all 49 published
versions and 48 of 49 neutral versions given the true locations.
This last measurement describes possible placement using the
available sentence boundaries.

## Four comparisons

1. **Current context responses.** Take the current passage classifier's
   response on each native three-sentence window. After calibration,
   combine adjacent qualifying center sentences.
2. **Isolated sentence responses.** Measure only the center sentence
   using the same checkpoint. Then apply the same selection rule.
3. **Intervals from isolated sentences.** Give each candidate range a
   sustained-response score with a small bonus for a rise or fall at
   the two outside edges. Use up to 40 sentences per range. Select up
   to six ranges that maximize total excess above the point, with at
   least one unselected sentence between ranges.
4. **Intervals from both readings.** Estimate a sentence response from
   the overlapping context scores, regularized toward the isolated
   readings. This treats a context response as an approximate average
   of its sentence scores. Use the same interval selection.

All four comparisons use the current passage classifier's pinned
checkpoint and tokenizer. The study
independently normalizes each reading by genre and by input word count.
Ordinary validation work maxima over both layouts set the tighter and
wider comparison points for each method. A useful range covers at
least half of the labeled non-whitespace characters and reaches
0.30 intersection over union. One reported range can match only one
labeled passage. The evaluation measures both located passages,
unmatched proposed ranges and flags on ordinary counterparts. Each
candidate uses the text of a single submitted version. The comparison
uses the raw context responses and measures range selection separately
from the app's document verdict.

## Held results

All 304 deliveries produced finite responses on all 16,240 distinct
inputs. Each layout has 98 labeled passages and 49 matched originals.
The table shows correctly located passages, documents with both passages
located, and ordinary excerpts with at least one proposed range. Each
original is counted once per layout.

| With source genre supplied | Published layout | Neutral layout | Passages found in both |
| --- | ---: | ---: | ---: |
| Current context, tighter | 6 / 0 / 2 | 11 / 0 / 2 | 6 |
| Isolated sentences, tighter | 0 / 0 / 3 | 2 / 0 / 3 | 0 |
| Isolated intervals, tighter | 34 / 6 / 2 | 33 / 6 / 1 | 28 |
| Combined intervals, tighter | 62 / 21 / 3 | 58 / 18 / 3 | 57 |
| Current context, wider | 15 / 2 / 7 | 17 / 3 / 7 | 15 |
| Isolated sentences, wider | 0 / 0 / 9 | 4 / 0 / 8 | 0 |
| Isolated intervals, wider | 44 / 10 / 9 | 44 / 12 / 7 | 41 |
| Combined intervals, wider | 66 / 24 / 7 | 67 / 26 / 6 | 62 |

At the tighter point, the combined intervals added 44 correctly located
historical passages and 12 correctly located report passages in the
published layout. The added ordinary flag occurred in one historical
excerpt. Unmatched proposed ranges in mixed excerpts fell from 98 to 14.
On the neutral layout the method added 49 new matches and lost two,
with the same additional ordinary flag.

| Using one calibration for both genres | Published layout | Neutral layout | Passages found in both |
| --- | ---: | ---: | ---: |
| Current context, tighter | 5 / 0 / 0 | 6 / 0 / 0 | 5 |
| Isolated intervals, tighter | 9 / 0 / 0 | 9 / 0 / 0 | 9 |
| Combined intervals, tighter | 14 / 2 / 2 | 14 / 2 / 2 | 14 |
| Current context, wider | 11 / 3 / 4 | 12 / 4 / 4 | 11 |
| Isolated intervals, wider | 15 / 5 / 5 | 16 / 5 / 5 | 15 |
| Combined intervals, wider | 24 / 7 / 6 | 25 / 7 / 6 | 24 |

The pooled comparison needs no known genre. At the wider point, the
combined method added 14 matched passages and lost one in the published
layout. Unmatched suggestions in mixed excerpts fell from 48 to four per
layout. Three of its 24 published-layout matches also had a flag at the
matched location in the original. Ordinary excerpts acquired three flags
and cleared one. The two-layout gain holds across both source collections.
Many labeled rewrites remain unlocated. The genre-specific gains depend
on knowing the source genre. These samples cover two English prose
collections, six recorded generator backends and two text layouts.
The evidence supports evaluating a supplemental local review with
the pooled calibration. Authorship calibration on typical customer
documents needs broader independent evidence.

A separate audit rebuilt the 304 deliveries from the private source plan
and independently checked 17,290 context input hashes, every selected
range and its overlap with the labels. The portable replay uses the
public numeric receipt. [Measured cases and locked protocol](evaluation-panels/writing-multi-passage-transfer-2026-11-16.json.gz)

## Replay

Fetch the `validation.parquet` and `test.parquet` files from both
`data/mixed` and `data/mixed_layout_neutral` of the pinned release.
Save them in the source directory as `mixed-validation.parquet`,
`mixed-test.parquet`, `mixed_layout_neutral-validation.parquet`, and
`mixed_layout_neutral-test.parquet`. The intake checks all four
SHA-256 digests. The receipt records enough information to audit the
arithmetic without the model or the text.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/writing-multi-passage-transfer-2026-11-16.json.gz
python3 -B -S -m scripts.writing_multi_passage_selection "$receipt"
```

To reproduce the native measurements, use the project environment with
PyArrow, PySBD, and the classifier snapshot at revision
f1795c86806e6838d4afa33d0b1427f8430c9615. The script verifies the
tokenizer, model, configuration and notices by hash. Use new output
files. [Publisher checkpoint and license](https://huggingface.co/wasitaigeneratedcom/tropa-mini-ai-text-detector)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.writing_multi_passage_panel prepare "$sources" "$run/plan.json.gz"
"$runner" -B -m scripts.writing_multi_passage_panel score "$run/plan.json.gz" "$snapshot" "$run/measured.json.gz"
python3 -B -S -m scripts.writing_multi_passage_selection "$run/measured.json.gz" --write "$run/result.json.gz"
```
