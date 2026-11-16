# Testing AI rewrite detection against historical edits

This experiment tested whether ordinary human revisions help a
single-upload detector recognize AI rewriting. All three variants
used the same frozen Apache-2.0 encoder, same AI-rewritten report
pairs and same sentence selection. One added the earlier version of
each historical Wikipedia excerpt to training. A second added the
revised version of those same excerpts. The matched augmentation
compares the effect of ordinary editing with the effect of adding
ordinary text from Wikipedia. Recent work on mixed-source writing
distinguishes the origin of wording from the origin of ideas. Here
the labels come from exact changes in known pairs.
[Mixed-origin research](https://arxiv.org/html/2608.27380v1)
· [Pinned encoder and license](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

## Source separation and calibration

The Wiki Neutrality Corpus supplied the original and revised
sentences. Wikimedia's public history provided the page ID, parent
revision and timestamp for each selected pair. Dates precede 2020.
The archive contains the publisher's aligned extracts. The intake
selects 180 training articles and 120 test articles by revision hash.
Article IDs and normalized 12-word sequences stay separate across
selected articles. Each selected revised sentence contains one to
six inserted or replaced words. Both versions of every qualified
Wikipedia pair enter the exact-token check. Each article appears
once. The numeric receipt includes its historical revision ID for
attribution and review.
[Dataset paper](https://ojs.aaai.org/index.php/AAAI/article/view/5385)
· [Publisher archive and mirror](https://github.com/rpryzant/neutralizing-bias#data)
· [Wikipedia content reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)

Two of 320 eligible AI training groups shared a normalized 12-word
sequence with calibration or evaluation text. Both were removed
before fitting. Both versions had complete sentence token coverage
and at least one usable target word in 297 AI work pairs. The shared
training inputs contain 1,690 marked AI-changed words and 15,624
other words from the same works. The matched Wikipedia additions
provide 5,224 earlier words or 5,300 revised words. The fit keeps the
AI-positive weight constant. It assigns one quarter of the weight to
the added Wikipedia class in each augmented fit. The sole feature
pair is the exact embedded word state and its difference from the
sentence mean. The training and inference use only the individual
sentence being measured.
[Earlier ordinary-revision fit](writing-edit-occurrence-transfer-2026-11-15.md)

Each variant uses separate news and report settings fixed on the same
36 ordinary excerpts per genre. A result highlights the top 60% of
words in the interior sentence with the highest mean of its six
strongest responses. A useful location covers at least half the known
changed words and reaches 0.30 word intersection over union. The
held English panels use previously measured pairs and two generator
families. Six of 112 pairs and five of 84 pairs lacked complete word
coverage and remain in the denominators.
[112-pair panel](writing-layer-transfer-2026-11-15.md)
· [84-pair panel](writing-structured-range-transfer-2026-11-15.md)

## Results at the wider research setting

| Variant | Useful locations on 112 pairs | Matched originals flagged | Useful locations on 84 pairs | Matched originals flagged | Earlier Wikipedia flagged | Revised Wikipedia flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AI reports only | 1/112 | 31/112 | 2/84 | 23/84 | 5/120 | 4/120 |
| Added original Wikipedia wording | 1/112 | 41/112 | 2/84 | 27/84 | 0/120 | 0/120 |
| Added revised Wikipedia wording | 0/112 | 39/112 | 2/84 | 23/84 | 0/120 | 0/120 |

Both ordinary-text additions cleared the same Wikipedia flags. The
measured reduction was shared by both versions of the training prose.
On the 84-pair panel the revision-trained variant found two useful
locations and lost the two found by the AI-only variant. Combining
the revision-trained result with the existing two-range check raised
useful locations from 4/84 to 6/84 and ordinary flags from 11/84 to
31/84. At the tighter point the same combination raised useful
locations from 4/84 to 5/84 and ordinary flags from 7/84 to 17/84.
The 112-pair head trained with historical revisions gained no useful
locations over the earlier two-location result. Even without the
calibration cutoff, its top proposal only reached useful overlap on
two of 112 rewrites. The fit variants remain experimental.
[Numeric receipt](evaluation-panels/writing-historical-editorial-negatives-2026-11-15.json.gz)

## Replay

The public numeric replay checks source article separation, cutoff
arithmetic, exact overlap, and the combination with the prior fixed
check. Python 3.12 and 3.14 returned identical review output. A
second native run reproduced the source plan and measured receipt
byte for byte. The receipt contains codepoint coordinates, source
hashes, historical revision identifiers and numeric results. Source
prose remains local. Native replay needs the official corpus
archive, the prior extended report roster and the two pinned
English input plans. The source command downloads only public
history metadata after the candidate revision list is frozen.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
source=scripts.writing_editorial_negative_source
module=scripts.writing_editorial_negative_transfer
receipt=docs/evaluation-panels/writing-historical-editorial-negatives-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$source" freeze "$wiki_archive" "$run/wiki-pool.json.gz"
"$runner" -B -m "$source" fetch "$run/wiki-pool.json.gz" "$run/metadata" "$run/wiki.json.gz"
"$runner" -B -m "$module" freeze "$extended_roster" "$layer_plan" "$range_plan" "$run/wiki.json.gz" "$run/input.json.gz"
"$runner" -B -m "$module" run "$run/input.json.gz" "$encoder" "$run/result.json.gz"
cmp "$run/result.json.gz" "$receipt"
```
