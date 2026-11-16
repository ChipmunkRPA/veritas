# Check how formatting affects a writing response

Chipmunk Verify can reread a short English writing context after adjusting
selected invisible characters within words and listed Greek or Cyrillic
look-alikes in mostly Latin words. The evidence reports when the direction
changes or the adjusted readings disagree. It shows the original sentence,
character positions and native reference in the browser, Standard PDF and
JSON. The two readings carry zero AI-authorship weight.

The alternate input preserves visible punctuation, ordinary diacritics,
addresses, whole non-Latin words, emoji joiners and direction controls.
The method uses the existing Unicode mapping and two already pinned
Apache-2.0 checkpoints. The study of formatting normalization motivated
this test. We used project code and generated our own evaluation deliveries.
[November normalization study](https://aclanthology.org/2026.aacl-long.82/)
· [Unicode mixed-script mappings](document-invisible-characters-2026-11-14.md#look-alike-letters)
· [Primary checkpoint terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)
· [Second checkpoint terms](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985)

## Frozen held test

The roster uses 72 previously unused matched works, with 18 news and 18
report works from each of two benchmark families. The intake excludes known
source IDs and overlaps of 12 normalized words. GPT-5.4 examples come from
unused publisher training works and Gemini 2.5 Flash from publisher test
works. Each excerpt has six source-aligned sentences with one or two known
rewrites. The earlier text supplies labels for evaluation. Each inference
reads one delivery independently.
[Publisher construction and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

The unchanged delivery and three staged formatting variants use the same
sentence positions. There are 576 deliveries and 1,611 distinct
three-sentence context strings. Both checks ran on every affected context.
The adjusted views reproduce the clean context bytes exactly on this
test panel. On both sides of each pair, the formatted words are in the
same sentence slots and the truth positions come from the original
source alignment. Six separate controls cover unchanged multilingual
and conventional formatting.

A useful result marks at least one of the known edited sentences. The
replay also counts exact sentence sets, off-target context flags, word
overlap and unassessed foci. The table applies the same fixed raw-score
direction to both checks and counts original excerpts with any flag.

| Delivery, both checks | Useful rewrite locations /72 | Original excerpts flagged /72 |
| --- | ---: | ---: |
| Clean source | 11 | 18 |
| Invisible word breaks as submitted | 1 | 7 |
| Look-alike letters as submitted | 3 | 7 |
| Both kinds as submitted | 3 | 7 |
| After limited character adjustment, each formatting variant | 11 | 18 |

When both character types were present, the app-style comparison changed
toward AI-like on eight rewrites and eight matched originals. It shifted
away from AI-like on two rewrites. Both adjusted checks disagreed on the
affected sentence in 27 rewritten works and 24 originals. Across all
four interior contexts, the clean result selected the exact affected
sentence set in four rewrites and reached both word-level overlap criteria
in one. The formatting check exposes reliance on formatting. The held
data supports a sensitivity observation with zero authorship weight.
[Numeric replay receipt](evaluation-panels/writing-character-view-2026-11-15.json.gz)

## Local app and replay

Enable `VERITAS_TEXT_CHARACTER_ADJUSTED_REVIEW=true` with the local
candidate writing mode and second writing check. The option defaults off.
It rereads up to six verified three-sentence English contexts after the
regular check. Each marked sentence has at least one eligible character.
The source block must map to decoded TXT, a top-level Word body paragraph,
a selected plain or mapped HTML email body, or a selectable PDF page.

The isolated app replay exercised TXT, Word, HTML email and selectable PDF
after simulated USD 9.99 checkout. In the test excerpt the adjusted readings
disagreed. The PDF, JSON and signed package preserve the same character
locations and original excerpt. Signed replay re-extracts the source,
resegments the sentences and reconstructs both scoring input hashes. A
same-file TXT comparison with the option off yielded the same overall
assessment. [Local app receipt](evaluation-panels/writing-character-view-app-2026-11-15.json)

The standard-library replay checks the held roles and geometry, then
recounts location and ordinary flags from recorded reads. The native
replay uses the six pinned shards and parent plan from the earlier
[grammar study](writing-function-sequence-2026-11-15.md#replay).
The held receipt contains hashes, locations, scores and token counts.
Source articles, model inputs and rendered fixtures stay local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.writing_character_view_transfer
receipt=docs/evaluation-panels/writing-character-view-2026-11-15.json.gz
runner=backend/.venv/bin/python
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$module" freeze "$shards" "$parent_plan" "$run/frozen.json.gz" --reference "$receipt"
"$runner" -B -m "$module" score "$run/frozen.json.gz" "$primary" "$second" "$run/receipt.json.gz"
"$runner" -B -m scripts.writing_character_view_app "$run/app" --pdf-font "$pdf_font"
```

The fixture uses a local selectable PDF font with Latin, Greek and Cyrillic
coverage. The other three files also test invisible in-word controls. The
results apply to staged formatting changes and known English rewrite targets.
Measuring character prevalence and authorship accuracy in ordinary uploads
requires separate datasets.
