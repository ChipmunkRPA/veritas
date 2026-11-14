# Match revised sentences independently in Word drafts

The optional earlier-draft writing check now works on a changed sentence
in a short English DOCX pair even when other edits cannot be aligned. The
same two unchanged neighboring sentences must identify a single place in
each draft. All three sentences must be inside one top-level Word body
paragraph on each side. The paragraph's position may change between
versions. Repeated or overlapping anchors and reordered neighborhoods are
excluded. The ordinary wording comparison continues to list the other edits.

The app highlights the exact changed words and gives the earlier and later
paragraph numbers and character ranges. The two saved writing responses apply
to the full three-sentence contexts. They carry zero weight in the overall AI
assessment. Some other body text, tables and headers can remain unchecked.
[Earlier plain-text method and checkpoint terms](writing-independent-anchor-transfer-2026-11-14.md)

## Word transfer

The previous news and academic abstract panels supply 216 draft pairs.
Pairs combine a seeded revision from one of two generators with genuine or
ordinary constructed changes. The prior panels establish the expected
locations, independent source labels, and analysis responses. Each pair was
encoded as Word twice. The second delivery inserts one unrelated paragraph
at the start of the analyzed draft. The replay verified identical writing
inputs at the newly mapped Word positions before reusing the recorded
responses. It verified the native Word coordinates from the two source
files. This tests transfer to Word and reuse of the prior readings. The
people and generated edits are the same across deliveries.
[Named sources, edit labels and license](writing-independent-anchor-transfer-2026-11-14.md#local-validation-on-new-source-works)

| Word version | Known model rewrites mapped | Useful locations | Mapped ordinary revisions flagged |
| --- | ---: | ---: | ---: |
| Existing body, news | 80/80 | 9/80 | 0/32 |
| Existing body, abstracts | 71/80 | 4/80 | 0/31 |
| Extra paragraph, news | 80/80 | 9/80 | 0/32 |
| Extra paragraph, abstracts | 71/80 | 4/80 | 0/32 |

In the extra-paragraph version, the previous Word path assessed none of
the 216 pairs. The new path mapped 215 sentence comparisons, including
151 generated edits and 64 ordinary revisions. The DOCX paths did not
identify an author. These controlled current-era benchmark polishes do
not establish results on all Word layouts or on a single upload.
[Per-file transport receipt](evaluation-panels/document-word-anchor-transfer-2026-11-14.json)

## Single-upload comparison

A separate experiment modeled ordinary observed token loss using the
expected cross-entropy and token count. On the two previously measured
mixed-writing panels, the conditional response relative to both neighboring
sentences reduced flags on ordinary excerpts from **58/66 to 44/66** and
from **32/34 to 22/34**. Useful locations also fell from **62/132 to 43/132**
and from **20/34 to 13/34**. The conditional rule stays offline and the
release keeps the current single-upload checks. These are native-context
research proxies rather than the saved document-level AI assessments.
[Numeric replay and comparison rules](evaluation-panels/writing-conditional-loss-transfer-2026-11-14.json)
· [Mixed-writing detection benchmark](https://arxiv.org/abs/2606.06481)

## Local app verification

Two simulated USD 9.99 checkouts ran the full document worker with the
same-Word format. The new comparison mapped the prior known rewrite and an
ordinary revision. It reported both source paragraph numbers and the exact
changed text. The ordinary pair produced no joint increase. The comparison
PDF and JSON agreed, the original-file replay checked all 24 context
sentence locations, and the saved overall assessments remained intact.
No job started before simulated payment.
[App receipt](evaluation-panels/document-word-anchor-app-2026-11-14.json)

Use **Compare with an earlier draft** on a completed Word analysis,
attach the earlier DOCX, and select **Include experimental paired writing
check**. Local development needs the existing draft comparison, separate
revision and independent review settings.

## Replay

The standard-library recount reads the checked-in numerical outcomes.
To regenerate the native Word files, prepare the two pinned benchmark
directories and the historical ordinary-source files using the
[plain-text panel instructions](writing-independent-anchor-transfer-2026-11-14.md#replay).
The native transfer reconstructs Word inputs, checks both extracted text
views and replays the previous responses against the Word source scopes.
The checkout smoke test independently scores two paired Word drafts.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B -S -m scripts.document_word_anchor_transfer \
  docs/evaluation-panels/document-word-anchor-transfer-2026-11-14.json
python3 -B -S -m scripts.writing_conditional_loss_transfer review \
  docs/evaluation-panels/writing-conditional-loss-transfer-2026-11-14.json
backend/.venv/bin/python -B -m scripts.document_word_anchor_transfer "$new_receipt" \
  --root "$ordinary_bank" --news "$news_bank" --abstracts "$abstract_bank"
backend/.venv/bin/python -B -m scripts.document_word_anchor_app \
  "$app_run" "$ordinary_bank" "$news_bank"
backend/.venv/bin/python -B -m scripts.review_document_revision_export comparison.json earlier.docx analyzed.docx
```
