# Testing alignment preference on rewritten sentences

The Alignment Imprint study compares how a pretrained language model and its
instruction-tuned version read the same token. Its weighted statistic asks
whether the difference is unusually strong compared with alternative tokens at
the same preceding text. We independently implemented the categorical mean and
variance using the full shared vocabulary of two installed Apache-2.0
checkpoints. The test sums those contributions over source-aligned ranges of one
or two sentences inside a paragraph. The preceding text comes from the single
submitted excerpt. A secondary arm removes the self-information weight while
keeping the same inputs and exact coordinates. These measurements rank
passages. [Alignment Imprint](https://arxiv.org/html/2604.16923v2#S3.SS2)
[Base checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)
[Instruction-tuned checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)

The source plan admitted 25 groups from published NASA Earth Observatory and
NOAA NCEI material with site metadata dated before 2021. The source excerpts use
government-published writing. They exclude material identified as linked media,
credits or quotations. Selection checked 12-word overlaps with three previous
source plans. The ordinary comparison uses an unchanged excerpt and a hard
control exchanging two similarly sized sentences from that same article. Both
sentences in each hard control come from the original publication. Two local
Apache-licensed editors from other model families rewrote a single marked
sentence in each held source. The remaining wording stayed byte-identical to
the prepared excerpt. Admission requires a complete sentence, the same set of
written numbers, a bounded length and word-edit distance, and exact alignment
within the otherwise unchanged sentence sequence. Checkpoint pretraining
exposure and factual fidelity beyond those mechanical checks are unknown.
[NASA reuse terms](https://science.nasa.gov/earth/faq/)
[NOAA reuse terms](https://www.noaa.gov/office-education/outreach-communication/faq)
[SmolLM2 terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)
[Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/tree/707f574c62054322f6b5b04b6d075f0a8f05e0f0)

Calibration used 13 source groups, with the remaining 12 held separately
within each publisher. The threshold for each method was the second-highest
ordinary work maximum across untouched and rearranged text. Held comparisons
allow one additional candidate disjoint from the existing evidence. A useful
location must cover at least half the edit and reach 0.30 intersection over
union in non-whitespace characters. The comparator uses the installed writing
classifier's first six eligible three-sentence card centers and the existing
locally configured document passage interval selector. It excludes the
perplexity-ratio cards, separate one-sentence position model, and overall
decision. The advancement rule needs new locations from both editor families
in at least two independent source groups with at most one additional
ordinary group flagged. This small evaluation cannot establish a general
false-alarm rate.

## Held results

The initial editor prompt produced no admissible rewrites. It frequently
borrowed details from neighboring sentences. A second intake gave each editor
only the target sentence and allowed four fixed attempts. This change took place
before reading the held detector outcomes. The selected original texts, hard
controls, measurement rules, and all 26 genuine calibration readings reproduced
exactly. The second intake admitted 8 SmolLM2 and 11 Granite rewrites. Five
generator–source pairs remained outside the comparison. Every held file received
both readings, and every rewritten sentence had at least one eligible interval.

| Reading | Correctly located rewrites | Unchanged files flagged | Rearranged human files flagged |
| --- | ---: | ---: | ---: |
| Existing classifier locations and passage interval selector | 1/19 | 10/12 | 9/12 |
| Existing plus weighted alignment | 1/19 | 11/12 | 11/12 |
| Existing plus unweighted alignment | 1/19 | 10/12 | 9/12 |

The weighted addition produced five new locations in ordinary files, including
three previously unflagged deliveries from two source groups. It added four
locations outside the changed sentence in rewritten files. On its own, it
selected a location in three untouched and three rearranged files and localized
zero edits. Its highest-ranked interval overlapped three of the known edits
before applying the operating point. The unweighted arm had no selections at its
operating point. The current comparator's single localization came from the
Granite edits to a NASA excerpt. The results leave both trial readings outside
the app. The large gap between candidate coverage and correct ranking also
matches the general difficulty reported for realistic web rewrites in the
October 2026 PAWN++ test. [PAWN++](https://aclanthology.org/2026.wac-1.25/)

## Replay

The [numeric receipt](evaluation-panels/writing-alignment-preference-2026-11-18.json.gz)
includes source URLs, source-file and input hashes, exact ranges, window
statistics, calibration points and decisions. It omits article prose, generated
replies, full token distributions and local paths. Recompute the reported
selection and overlap counts without model weights or network access.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_alignment_preference review \
  docs/evaluation-panels/writing-alignment-preference-2026-11-18.json.gz
```
