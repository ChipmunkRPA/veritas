# Minimal writing edits in archived CDC reports

The passage check found **0 of 4** generated rewrites that preserved the
original claim. It assessed none of the 10–11-sentence excerpts under the
current file-length rule. A local probe lowered the minimum from 12 to 10
sentences and preserved the existing score cutoffs. It assessed every excerpt
and still located no rewrite. [Anonymized receipt](evaluation-panels/writing-mmwr-minimal-edits-2026-11-18.json.gz)

## Source and edit selection

A fixed title filter selected seven 2017 articles from the CDC's *Morbidity and
Mortality Weekly Report* before the prose was read. Each article supplied one
interior sentence with known text positions. The surrounding text stayed
identical between each original and rewrite. One official erratum supplied an
ordinary factual correction. The published issue PDF and current article HTML
agree on the earlier wording. The corrected test view replaces one sentence
with the wording in the erratum. The earlier view is identical to one of the
seven untouched originals, so it adds no independent control. The CDC makes
its series material available in the public domain. The seven sources shared
no exact normalized 12-word phrase with 624 text entries from seven earlier
panels. Checkpoint training exposure remains unknown. [CDC archive](https://www.cdc.gov/mmwr/index2017.html) ·
[Public-domain terms](https://www.cdc.gov/mmwr/about.html) ·
[Published correction](https://www.cdc.gov/mmwr/volumes/66/wr/mm6613a5.htm)

Two pinned Apache-2.0 generators made three fixed attempts per source and edit
size. A minimal edit had normalized word edit distance 0.10–0.25 and a
moderate edit 0.32–0.65. The first structurally eligible answer in each of 28
slots entered a factual review with the generator and edit-size labels masked.
Ten slots had an eligible answer. Four rewrites preserved the source claim,
five changed it and one was uncertain. Both preserved minimal edits came from
the first generator. Each generator contributed one preserved moderate edit.
This review concerns fidelity to the exact historical wording. [Pinned editor A and terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) ·
[Pinned editor B and terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674) ·
[Research on stylometry and AI editing](https://aclanthology.org/2026.emnlp-main.1725/)

## Local results

| Passage check | Views assessed | Preserved-claim edits located | All eligible edits located | Untouched files flagged |
| --- | ---: | ---: | ---: | ---: |
| Current, short excerpts | 0/19 | Unassessed | Unassessed | Unassessed |
| Local probe, short excerpts with 10-sentence minimum | 19/19 | 0/4 | 0/10 | 0/7 |
| Current, full article prose | 19/19 | 0/4 | 0/10 | 1/7 |

Full bodies used 16–84 sentences. The unchanged selector flagged six wrong
locations across three edited views. It flagged neither side of the ordinary
correction. The short probe also left both correction views without highlights.
Both used the unchanged classifier on two CPU threads with PyTorch 2.13.0.
[Pinned classifier](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

A useful match must cover half the target's non-whitespace characters and reach
0.30 intersection over union. The altered words occupied 6.5–21.7% of the
target sentence for all three structurally eligible minimal edits. An entire
sentence highlight could never meet the changed-word test in these cases.
The four preserved-claim edits and both ordinary correction views reproduced
their exact focal locations in DOCX and TXT.

The pilot covers a small set of English government reports. Lowering the file
length floor expanded coverage without recovering edits. The running app's
passage settings and authorship confidence stayed unchanged.

## Replay the numbers

The reader checks interval geometry, duplicate controls, generation intake,
comparable trial coverage, and the recorded Word and TXT witnesses. It recounts
sentence and changed-word overlap from compact non-whitespace coordinates. The
receipt contains anonymous source and generator numbers, position measurements
and input commitments. Source text, generated wording, and model inputs remain
local. This is arithmetic replay of the saved measurements.

```sh
python3 -B -S backend/scripts/review_writing_mmwr_minimal_edits.py \
  docs/evaluation-panels/writing-mmwr-minimal-edits-2026-11-18.json.gz
```
