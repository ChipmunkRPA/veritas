# Locating short prose insertions against published revisions

The existing single-file passage check found zero of seven fact-preserving
synthetic insertions. It completed on all 25 files and flagged none of the 18
ordinary earlier or revised excerpts. A statistical comparison with the earlier
copy also produced zero directional alerts on seven synthetic inserts and nine
archived inserts. Direct source comparison found all 16 inserted passages. A
boundary correction preserves the inserted sentence's own punctuation in Word
and plain text. [Aggregate receipt](evaluation-panels/writing-publisher-insertions-2026-11-19.json)

## Held examples

The nine held groups comprise eight English Programming Historian lessons and one
Data Carpentry course counted as a single work. We enumerated chronological
first-parent Markdown changes in the official lesson repository across the
2012–2016, 2017–2019, 2020–2021 and January–October 2022 windows. Six Carpentries
course repositories provided a separate 2017–2019 reserve. The selected course's
instructional licence was checked on the exact archived versions. Article
selection required consistent publication identity and lesson-specific licence.
The included instructional prose is available under CC BY 4.0. Code fragments,
footnotes, site notices and versions with uncertain publication identity were
excluded. [Publisher archive](https://github.com/programminghistorian/jekyll) ·
[Programming Historian licence](https://programminghistorian.org/en/author-guidelines) ·
[Data Carpentry licence](https://datacarpentry.github.io/R-ecology-lesson/LICENSE.html)

Each earlier excerpt has twelve sentences preserved exactly in the later
version. It supplies at least three unchanged sentences on either side of a
single insertion of one to three sentences. Insertions ran from 16 to 52
words. The archive supplies genuine before-and-after changes. The tools used
to author them are unknown. The held groups were frozen before either method
was scored and shared no normalized 12-word fragment with one another or 69
inputs in the prior measured panel. Checkpoint pretraining exposure is
unknown. The group count measures whole works. It does not establish nine
independent authors or a broadly representative writing sample.

Two local generator families received two fixed prompts per source. Seventeen
of 36 requests met the fixed geometry and text constraints. Review admitted
nine as equivalent, rejected five for changed meaning, and withheld three
as unclear. Keeping the earliest equivalent output per family and source
leaves seven generated variants across six works. All completed source and
generated excerpts round-tripped with the same codepoint coordinates through
50 TXT and DOCX files. The Word carriers were built for this evaluation from
the selected historical prose. [Qwen checkpoint licence](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) ·
[SmolLM2 checkpoint licence](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/LICENSE)

## Results

The active selector received each excerpt individually and used its regular
isolated and neighboring sentence readings. An exact localization had to
cover at least half the target's non-whitespace characters and reach 0.30
intersection over union. The paired experimental cue used the exact aligned
insertion, compared its log odds with the stronger of the equivalent number
of contiguous unchanged flanking sentences, and retained the smaller margin
of two readings. One reading used the text alone. The other used the same
outer context on both sides of each candidate. Positive margins in both views
were required for directional support in this experiment. The condition was
fixed before scoring without fitting a new point. [Detector checkpoint and terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

| Generator family | Eligible variants | Standalone exact locations | Paired directional support | Higher paired margin than its matched archive insertion |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 5 | 0 | 0 | 3 |
| SmolLM2 | 2 | 0 | 0 | 1 |

The one-core run evaluated 347 distinct classifier inputs in about 37
seconds. This small technical prose panel does not support deploying the
paired cue or changing the standalone AI assessment. It demonstrates a
failure to identify preserved-meaning short insertions within the tested
scope.

The earlier-copy feature uses the submitted files to locate actual changes.
It found all wording in all 16 pairs before and after the punctuation change.
The previous comparator assigned one repeated period to the wrong sentence in
three pairs. Two were archived revisions and one was a generated variant.
Complete source spans rose from 13 to 16 per direction across TXT to TXT,
DOCX to DOCX, TXT to DOCX, and DOCX to TXT. All 32 exports for this check
that ended in DOCX replayed their recorded paragraph-local ranges. The new
policy records the punctuation correction only when it has an exact,
unambiguous complete-sentence alignment. An earlier-copy comparison documents
text differences between the two files. It provides no AI assessment weight.

## Recount

The anonymous receipt contains counts and commitments. It excludes source
and generated prose, authors, case identifiers, and local paths. The reader
recounts case admission, matched ranking, ordinary control coverage and
exact source spans from the saved tallies. It requires no model and makes
no independent claim about the unpublished examples.

```sh
python3 -B -I -S backend/scripts/review_writing_publisher_insertions.py \
  docs/evaluation-panels/writing-publisher-insertions-2026-11-19.json
```
