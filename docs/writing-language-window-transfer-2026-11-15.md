# Recover English writing contexts from mixed-language documents

The current writing pipeline withholds statistical results when it
recognizes mixed languages. Two local prototypes attempt to recover a
three-sentence English context inside a single paragraph. The first
checks whether each sentence is independently identified as English.
The second also checks the two halves of each sentence where a half has
at least 45 letters. The prototypes never join across gaps. They reuse
the local language identifier and the pinned Apache-2.0 writing
checkpoint. Published work on code-switched language identification
and on authorship segmentation motivates measuring the two tasks
separately. The experiment uses no code or weights from either paper.
[GLUECoS](https://aclanthology.org/2020.acl-main.329/)
· [Authorship segmentation research](https://arxiv.org/html/2607.27183v1)
· [Writing checkpoint and license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small?hardware=rtx-5070)

## Controlled source comparison

The 112 original and AI-rewritten work pairs come from the previously
measured English news and report panel. The mixed-language deliveries
are new. They use 28 separate native sentences from each of four
public-domain works. Selection uses source shape and text digest before
scoring. Each pair receives the same foreign sentence, either appended
as a paragraph, inserted between sentences, or inserted as a clause
inside an English sentence. The intact English and short-book-title
cases serve as controls. A repeated foreign sentence checks that neither
prototype admits an entirely non-English input. Exact source ranges
identify the insertion. These are staged documents, with four authors
and four non-English languages. They do not represent all code-switching
or naturally bilingual documents.
[English source plan](writing-layer-transfer-2026-11-15.md)
· [Spanish](https://www.gutenberg.org/ebooks/15027)
· [French](https://www.gutenberg.org/ebooks/4650)
· [German](https://www.gutenberg.org/ebooks/22367)
· [Italian](https://www.gutenberg.org/ebooks/45334)
· [Source reuse terms](https://www.gutenberg.org/policy/license.html)

| Delivery | Files | Existing language gate allowed writing scores | Exact English context recovered by stricter prototype | Foreign text admitted by sentence-only / stricter prototype |
| --- | ---: | ---: | ---: | ---: |
| Clean English | 224 | 224 | 224 | 0 / 0 |
| Brief foreign title | 224 | 224 | 224 | 0 / 0 |
| Separate paragraph | 224 | 0 | 224 | 0 / 0 |
| Separate sentence in the same paragraph | 224 | 0 | 224 | 0 / 0 |
| Clause inside a sentence | 224 | 53 | 222 | 47 / 26 |
| Only non-English text | 112 | 0 | 0 | 0 / 0 |

An exact recovery uses the byte-identical three-sentence input from
the corresponding English excerpt. A file can contain both a recovered
context and an admitted context with foreign text. All 26 files that
admitted foreign text under the stricter prototype also passed the
existing gate as English. The stricter rule found at least one exact
context in 169 of the 171 within-sentence cases withheld by that gate.

## Authorship readout

The local classifier evaluated 768 unique exact English contexts.
The same source input digest has the same raw reading in each
delivery. This measures the existing classifier on the recovered
contexts. It assigns no score to an admitted context that differs from
every original English input. The table uses the existing internal
raw review point and reports only contexts centered on a sentence
changed in the rewrite or the corresponding original sentence.

| Delivery | Rewritten passage flagged | Matched original passage flagged | Only the rewritten passage flagged | Both versions flagged |
| --- | ---: | ---: | ---: | ---: |
| Separate paragraph | 41/112 | 32/112 | 11/112 | 30/112 |
| Separate sentence | 33/112 | 28/112 | 7/112 | 26/112 |
| Clause inside a sentence | 23/112 | 18/112 | 7/112 | 16/112 |

Recovery restores many unchanged inputs, but the writing score still
fails to distinguish most rewrites from their matched originals.
Both prototypes remain in evaluation. The 53 in-sentence switches
that the existing gate treated as English provide a separate target
for improving language coverage before adding any new authorship
finding.
[Measurement receipt](evaluation-panels/writing-language-window-transfer-2026-11-15.json.gz)

## Replay

The public review needs standard Python and contains no source text,
uploads, or weights. The receipt keeps source commitments, exact ranges
and native scalar readings. Python 3.12 and 3.14 give identical counts.
A second native run reproduced both the frozen input plan and measured
receipt byte for byte. Native replay needs the prior
[English input plan](writing-layer-transfer-2026-11-15.md#independent-replay)
and its checkpoint. Download each numbered Gutenberg plain-text source
at `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`. The runner
compares every intact book and the English plan with pinned hashes
before preparing test inputs. Keep reconstructed prose local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.writing_language_window_transfer
receipt=docs/evaluation-panels/writing-language-window-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$module" freeze "$english_plan" "$books" "$run/freeze.json.gz"
"$runner" -B -m "$module" capture "$run/freeze.json.gz" "$checkpoint" "$run/held.json.gz"
cmp "$receipt" "$run/held.json.gz"
```
