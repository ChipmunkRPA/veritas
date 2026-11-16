# Compare changed passages when sentences split or merge

In an authenticated document analysis, open **Compare with an earlier draft**
and select **Include experimental paired writing check**. A changed passage
can span one or two sentences in either version if two unchanged sentences
uniquely bound the change. The comparison supports short English TXT, Word
body text and selectable PDF text, including mixed formats.

The results link the changed passage to the corresponding wording changes.
The browser and PDF show the exact wording and highlight the changed tokens.
Word retains paragraph positions. PDFs retain page positions and link to
highlighted source crops. The JSON export records the same spans, native
positions and two separate writing responses when available. Changes in
punctuation alone and ambiguous anchors remain outside the new comparison.
The writing responses are uncalibrated and carry zero weight in AI confidence.

## Local comparison

Five held OpAI-Bench works contained two adjacent rewritten sentences between
unchanged flanks. They include news and reports with two benchmark generators.
The split, merge, punctuation and repeated-anchor cases were authored for this
test. The search of the archived eLife versions yielded no qualifying split or
merge. Each source pair was rendered in three formats on each side with the
same wording, making nine ordered deliveries per work. Counts measure coverage
of the comparison, with common source text across deliveries.
[Benchmark construction and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)
· [Archived publisher versions](https://github.com/elifesciences/elife-article-xml/tree/58e688ec3d4523eb1f3b03bed6bb440bf91e4786/articles)

| Case | Works | Earlier check mapped | Expanded check mapped |
| --- | ---: | ---: | ---: |
| Two adjacent rewritten sentences | 5 | 5/45 | 45/45 |
| One sentence split into two | 1 | 0/9 | 9/9 |
| Two sentences merged into one | 1 | 0/9 | 9/9 |
| Punctuation-only split or repeated anchor | 2 | 0/18 | 0/18 |

Three local analyses exercised all nine format pairings through the app.
The worker started only after simulated USD 9.99 payment. Both writing checks
returned responses for every mapped passage. Neither fixture with changed
sentence counts nor the selected known rewrite reached a joint AI-like
increase. The run verified the changed-token ranges, Word positions, PDF
word boxes and linked crops. Changing a saved source position made the JSON
verifier reject the export. The standalone analyses retained their original
results. The broader [standalone writing study](writing-function-sequence-2026-11-15.md)
measures the separate question of which passages a single upload can identify.
[Mapping receipt](evaluation-panels/document-paired-blocks-mapping-2026-11-15.json)
· [App receipt](evaluation-panels/document-paired-blocks-app-2026-11-15.json)

## Replay

Enable `VERITAS_DOCUMENT_PAIRED_BLOCK_REVIEW=true` with document revision
comparison and independent paired review enabled. The new option defaults off.
The portable reviewer recounts coverage and checks that the paired inputs
match across formats. The full replay uses the fresh plan from the linked
standalone study and the existing local writing assets. `--elife` accepts the
prepared archive from `scripts.writing_elife_version_body_holdout`.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.replay_document_paired_blocks
panels=docs/evaluation-panels
receipt=$panels/writing-function-sequence-2026-11-15.json.gz
runner=backend/.venv/bin/python
python3 -B -S -m "$module" review "$panels/document-paired-blocks-mapping-2026-11-15.json"
"$runner" -B -m "$module" freeze "$run/held.json.gz" "$receipt" "$run/block-mapping" --elife "$elife"
"$runner" -B -m "$module" app "$run/block-mapping" "$run/block-app"
```

The public files contain numeric outcomes, source hashes and geometry
verification counts. Source excerpts and reports stay in the local replay.
