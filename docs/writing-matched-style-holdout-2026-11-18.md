# Matched style examples for short writing edits

The style comparison located **0 of 8** generated sentence rewrites. It
also ranked the changed sentence first in two archived editorial revisions.
The keyword-only comparison located one generated rewrite that changed a
fact and also ranked the corresponding sentence first in the earlier
version. The current passage selector located none of the eight. The style
comparison stays in research. [Anonymized numeric receipt](evaluation-panels/writing-matched-style-holdout-2026-11-18.json.gz)

## Matched sources and measurements

The panel uses excerpts derived from *The Turing Way*, copyright The Turing
Way Community, licensed under CC BY 4.0. The source study retained ordered
prose from the archived 2019–2020 parent and child files, removed Markdown
structure and link destinations, and marked one changed sentence per work.
Its earlier and later excerpts preserve the other 11 sentences. The style
panel excluded one entire work because a generated delivery exactly matched
the later archive wording. The remaining panel has 14 earlier/later pairs
and eight generated rewrites. Each rewrite changes only the marked sentence
of its earlier excerpt. Archived commits establish chronology and wording.
The provenance of individual sentences and checkpoint training exposure
remain uncertain. [Source study and attribution](writing-turing-editorial-history-2026-11-18.md) ·
[Pinned archive release](https://github.com/the-turing-way/the-turing-way/tree/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c) ·
[Release license](https://github.com/the-turing-way/the-turing-way/blob/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c/LICENSE.md) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Each candidate sentence supplies the same sorted content keywords and the
same 16 observed tokens to five readings. Two readings use examples selected
from the current file, one before and one after the focus. Each example is a
pair of sentences separated from the focus by one intervening sentence. Two
readings use outside examples matched to their corresponding local example.
Paired prompts differ by at most 12 tokens, and the difference in their
keyword distances from the focus is at most 0.20. Six reference works were
set aside by source chapter, recorded contributor group, commit, and
normalized 12-word overlap. The fifth reading uses the keywords alone.
Each call scores the current file with the frozen reference bank. [Style-conditioned likelihood study](https://aclanthology.org/2026.findings-emnlp.14/)

For each side, the style comparison subtracts the outside-example reading
from the local-example reading and measures the deficit below the file's
median. It ranks by the smaller of the two deficits and needs a unique
maximum with both deficits positive. The keyword-only comparison ranks by
its highest mean token loss. Both require at least four complete focus
sentences per file. The existing comparison replays the saved document-relative
passage selector with its report confirmation. A correct location covers at
least half the marked non-whitespace characters and reaches an intersection
over union of at least 0.30. This retrospective trial measures location ranks
and source coverage. [Pinned scoring checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323) ·
[Earlier baseline and generator pins](writing-turing-editorial-history-2026-11-18.md#local-passage-results)

## Generated rewrites and genuine editorial changes

The two experimental arms could assess four of seven Qwen files and the
single SmolLM2 file. Three of the Qwen targets and the SmolLM2 target had
complete focus readings in both the rewrite and the corresponding earlier
file. The Qwen editor shares a model family with the scorer. The two
families remain separate in the results.

| Reading | Qwen edit located | SmolLM2 edit located | Other locations on rewritten files |
| --- | ---: | ---: | ---: |
| Two-sided style | 0/7 | 0/1 | 4 |
| Keywords alone | 1/7 | 0/1 | 4 |
| Current passage selector | 0/7 | 0/1 | 1 |

A paired factual review compared the source sentence and rewrite before
opening any experimental scores, with the editor and focus eligibility
masked. Six rewrites preserved the source claim, and two changed a material
point. The exact target was eligible in both versions for three of the six
fact-preserving rewrites. All three readings located zero of the six. The
keyword-only hit came from the fact-changing group and also matched the
earlier target. The factual review assessed fidelity to the source's
assertions.

| Reading | Earlier and later both assessed | Both versions selected a passage | Historical target selected earlier | Historical target selected later | Historical target selected only after the edit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-sided style | 8/14 | 8 | 0 | 2 | 2 |
| Keywords alone | 8/14 | 8 | 3 | 3 | 0 |
| Current passage selector | 14/14 | 1 | 0 | 0 | 0 |

The results describe passage ranks in short archived English excerpts and a
small set of generated replacements. Estimating an operational false-alarm
rate would require a separate operating-point evaluation.

## Exact score reuse and independent replay

The closed local run evaluated all 705 commands over 141 candidate sentences
and 36 file views. It used one CPU thread with BF16 weights. The commands
contained 370 distinct combinations of complete input token IDs, prediction
positions, and observed token IDs. Every reading in all 215 repeated command
groups matched exactly in binary64. An evaluator with the same frozen
runtime and checkpoint can reuse 335 readings on this panel. The associated
input workload would fall from 118,716 to 62,880 tokens, a 47.0% reduction.
The original run measured every occurrence. Wall-clock savings from caching
remain unmeasured.

The portable receipt contains anonymous work and command numbers, selected
Unicode ranges, score values, and non-whitespace counts at exact interval
endpoints. The standard-library reader recomputes ranks, overlaps, the
paired natural and factual outcomes, and equality of repeated scores. It
replays the saved scalar evidence. Source extraction and model inference
are separate from this replay.

```sh
python3 -B -S backend/scripts/review_writing_style_receipt.py \
  docs/evaluation-panels/writing-matched-style-holdout-2026-11-18.json.gz
```
