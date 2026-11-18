# Compare Rich Text with an earlier draft

In the local development app, **Compare with an earlier draft** accepts
Rich Text (`.rtf`) on either side. Users can compare two RTF files or match
RTF with plain text, Word, OpenDocument, or a supported selectable PDF.
Each file can be up to 1 MiB. The web view, PDF and JSON identify the
paragraph and the exact changed words on both sides. RTF positions count
Unicode characters in the extracted text, including within each paragraph.
The comparison checks wording and punctuation. It carries no AI-assessment
weight. Formatting changes alone do not create wording findings.

## Fresh revision tests

The held panel uses nine chronological parent-and-child changes from
*The Turing Way*, each with one changed sentence and 11 exact neighboring
sentences. They come from nine different chapter files, eight chapter groups
and six contributor groups. The source gate excludes paths from the previous
historical study and normalized 12-word passages from all six earlier input
plans. Both native files and the governing license are checked at their exact
Git commits. The first prospective intake also excluded repeated contributors
and chapter groups and admitted four pairs. A second intake allowed up to two
works per contributor or chapter and kept the original path, phrase, and
chronological requirements. No comparisons ran on the first intake. Three
admitted pairs are by returning contributor groups. The revised plan and nine
pairs were committed by SHA-256 before measuring the comparison output.
[Prior source study](writing-turing-editorial-history-2026-11-18.md)

The historical prose is derived from *The Turing Way*, copyright The Turing
Way Community, CC BY 4.0. Before/after text uses selected visible Markdown
paragraphs with markup removed. The Git records establish historical
revision order and manuscript correspondence. The same controlled file
test also used one admitted rewrite from each of two locally run Apache-2.0
editors. Sixteen of 18 generated attempts failed the fixed one-sentence,
edit-distance or location criteria and were excluded before comparison.
[Publisher archive](https://github.com/the-turing-way/the-turing-way/tree/e2394b4940f88c8e571dba6c496d0f7ea7a52d2c) ·
[Terms](https://creativecommons.org/licenses/by/4.0/) ·
[Editor revisions](writing-turing-editorial-history-2026-11-18.md#local-passage-results)

| Earlier → analyzed file | Historical edits with marked tokens on both sides | AI rewrites with marked tokens on both sides | Unchanged historical controls flagged |
| --- | ---: | ---: | ---: |
| RTF → RTF | 9/9 | 1/2 | 0/9 |
| Word → RTF | 9/9 | 1/2 | 0/9 |
| TXT → RTF | 9/9 | 1/2 | 0/9 |
| RTF → Word | 9/9 | 1/2 | 0/9 |

All marked tokens stayed inside the known edited sentences. The strict
criterion requires changed tokens in both source versions. The generated
deletion marks tokens in the earlier file and a character position inside
the changed sentence of the analyzed file. The other generated rewrite
marks its replacement in both versions. The four rows repeat the same
underlying works to check format transfer. These counts establish wording
comparison with the earlier version present. They give no estimate of
standalone AI-writing detection accuracy.
[Held numeric receipt](evaluation-panels/writing-rtf-draft-2026-11-18.json.gz)

## Application replay

Chrome compared a Word draft with a retained RTF analysis from the local
simulated USD 9.99 checkout fixture. The two highlighted words and paragraph
positions agreed with the API and PDF. The browser download matched the
JSON response. Independent replay checked both source hashes and exact
locations. The comparison left the existing assessment intact and created
no extra charge or analysis. The focused tests also exercised mixed
selectable PDF, Word, and RTF comparisons and rejected a changed position.
[Local browser receipt](evaluation-panels/writing-rtf-draft-app-2026-11-18.json)

The numeric receipt omits the publisher and generated text. Its stored
source hashes and exact ranges allow a model-free recount. To verify a
downloaded comparison you need the two actual files.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_rtf_revision_holdout review \
  docs/evaluation-panels/writing-rtf-draft-2026-11-18.json.gz

PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_document_revision_export \
  comparison.json earlier.docx analyzed.rtf
```

To rebuild the held panel, use the [pinned publisher archive and input
plans](writing-turing-editorial-history-2026-11-18.md#replay-locally). The
first prior input is the `source.json.gz` produced by the historical study.
The remaining five prior inputs are the plans referenced there. The two
editor replay commands from that study accept the newly reserved source
without changes. Use fresh output paths for each run.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
study=scripts.writing_rtf_revision_holdout
"$runner" -B -m "$study" freeze "$publisher" "$run/frozen.json.gz" \
  "$historical/source.json.gz" "$prior/plan/fit.json.gz" "$prior/plan/calibration.json.gz" \
  "$prior/plan/held.json.gz" "$prior/transfer-bound.json.gz" "$economic/held.json.gz"
"$runner" -B -m "$study" measure "$run/frozen.json.gz" "$run/result.json.gz" \
  --ai "$run/ai-plan.json.gz"
```
