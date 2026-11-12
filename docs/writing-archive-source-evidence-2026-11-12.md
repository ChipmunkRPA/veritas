# Matching a single document to a public archive

The source comparison found the expected archive document and located all 21 generated sentence replacements and all 16 ordinary human sentence swaps in this indexed panel. The incumbent statistical windows located two edits in each group. The archive comparison also linked all 16 unchanged originals with zero differences and located one real publisher transcription correction. It reports exact text differences against a cited archive version. This evidence carries zero AI-origin or authorship confidence.

| Held outcome | Files | Archive source comparison | Current native windows |
|---|---:|---:|---:|
| Generated replacement usefully localized | 21 | 21 | 2 |
| Same-source human swap usefully localized | 16 | 16 | 2 |
| Unchanged original flagged as having a difference or highlighted | 16 | 0 | 13 |
| Actual earlier publisher correction located | 1 | 1 | Outside the comparator scope |
| Current publisher control with a difference | 1 | 0 | Outside the comparator scope |
| Missing-source body wrongly attributed | 9 | 0 | Outside the comparator scope |
| Missing-source email body wrongly attributed through its quote | 9 | 0 | Outside the comparator scope |
| Duplicate-reference ambiguity left unresolved | 4 | 4 | Outside the comparator scope |

## Source and case design

Four pinned Foreign Relations of the United States volumes supplied 118 indexed diplomatic documents and nine separate out-of-index works. Six consecutive body paragraphs from each of 16 in-index works were sampled in hash order. The 16 indexed test works and nine missing-source works had distinct recorded official senders. The source intake excluded embedded quotations, foreign material, and editorial notes. The same out-of-index bodies were tested as plain text and as email files with a quote from a different, registered document. The U.S. State Department publishes the master files under its public-domain and CC0 terms. [Publisher master files and license](https://github.com/HistoryAtState/frus) · [Official collection](https://history.state.gov/about/faq/what-is-frus)

Sixteen ordinary controls substitute a sentence from another paragraph of the same genuine document. Thirteen and eight rewrites from the two pinned Apache 2.0 generators qualified. Every replacement is bound to its known query code-point span. The broader six-paragraph historical revision intake found zero qualifying prior versions. A separate three-paragraph probe used the State Department's exact correction of a hyphenation typo in document d585. The audit matched the earlier query and current reference token coordinates. [Official document](https://history.state.gov/historicaldocuments/frus1964-68v24/d585) · [Publisher correction commit](https://github.com/HistoryAtState/frus/commit/41eae325d63e2f1513c8d56b56dde6a6a7e8b7bd)

## Source retrieval and exact differences

The local index searches normalized eight-word sequences and verifies their positions across at least three ordered body paragraphs. Common phrases, competing sources and marked email quotes cannot establish a reference on their own. After binding a source, the comparison records exact word and punctuation edits with query offsets, source paragraph identifiers and reference offsets. A single gap between two strong anchors can locate an inserted, removed or replaced paragraph. Fingerprints support retrieval and the exact tokens support the displayed differences. [Winnowing](https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf) · [PAN source retrieval and alignment evaluation](https://arxiv.org/abs/2510.06805)

The held 37 controlled replacements had zero unrelated difference paragraphs. In 20 additional geometry probes, the same frozen matcher passed four cases each of paragraph insertion, deletion, replacement, partial document lookup, and reversed-order abstention. The candidate measures the relation to a local library of known versions. It has no measured detection rate on unpublished writing or uncataloged material. The out-of-index sample contains nine source works and covers only this archive genre.

## Verification

The independent audit rebuilt the source votes and checked all 77 attributions, 705 archive paragraph bindings, 1,124 exact changed-token offsets and 1,102 native-window span bindings. The current classifier read 439 distinct inputs in 110 local CPU forwards. The audit rejected a modified changed-token offset. The [public JSON receipt](evaluation-panels/writing-archive-source-evidence-2026-11-12.json) includes the frozen source pins, case counts, source evidence coordinates, and audit commitments.

The [evaluation runner](../backend/scripts/writing_reference_archive_transfer.py) and [independent reviewer](../backend/scripts/review_writing_reference_archive_transfer.py) use the pinned local XML archives and the local checkpoint snapshots. The archival source excerpts, generated text and raw model responses remain in the local evaluation directory.
