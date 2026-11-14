# Keeping draft wording differences inside the changed sentence

Two short English TXT drafts can share unchanged sentences on either side
of a revision. A document-wide word diff can align repeated punctuation to
the wrong sentence. The earlier-draft comparison now uses unique, unchanged
neighbors as boundaries when that happens in eligible pairs. It can then
highlight the words and punctuation from the appropriate original sentence
in both versions. The browser, PDF and JSON use the same decoded Unicode
positions. Research on article revisions also studies sentence alignment and
word-level change extraction as separate steps.
[arXivEdits](https://arxiv.org/abs/2210.15067)

The fallback needs two to four changed sentences, two unchanged neighbors
within each paragraph, and the same sentence-to-paragraph layout in both
drafts. The document must match outside the mapped sentences, allowing
ordinary whitespace variation. Ambiguous repeated neighbors and moves of a
changed sentence between targets are excluded. A working existing statistical
comparison retains its current evidence. The refined wording differences
describe the supplied files and carry zero authorship confidence.

## Frozen comparison

The trial ran the existing pair of commercially permitted writing
checkpoints on both arms with their existing doubling-of-odds rule.
One arm used the document-wide wording diff. The second used exact
sentence anchors only when that diff could not map the revisions.
Both arms used the same model inputs wherever the first arm was already
assessable.
[Checkpoint terms and original comparison](writing-paired-center-transfer-2026-11-12.md)

We selected 32 GAO report works with no matching previously published work
hash or normalized 12-word sequence in the excluded benchmark reports.
Each supplied one known edit from Gemini 2.5 Flash and one from GPT-5.4
Nano. Admission required exact agreement between the published tagged
character labels and the changed sentence. We joined each three-sentence
report excerpt to a separate revision in archival prose, alternating which
paragraph came first. Ordinary controls include real historical journal
edits and an archived same-contributor edit. Most technical blog controls
replace a sentence with other genuinely authored prose from the same blog
post. The reports are new source works for the project comparison, and the
two-paragraph drafts are constructed test documents.
[OpAI-Bench paper](https://arxiv.org/abs/2606.06481)
· [Pinned benchmark](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)
· [Journal source record](writing-journal-body-transfer-2026-11-11.md)
· [Technical blog source method](../backend/scripts/writing_archive_prospective_intake.py)

| Cohort | Assessable, current → study | Exact AI-like flags, current → study |
| --- | ---: | ---: |
| New sources, 64 generated edits | 14 → 18 | 1 → 1 |
| Earlier comparison, 80 generated edits | 36 → 54 | 5 → 5 |
| Assessed ordinary source works, new-source comparison and ordinary pairs | 14 assessed by either arm | 0 additional |

Sentence anchors recovered assessable contexts but produced no additional
AI-like findings. The new-source comparison still missed 17 of 18
assessed rewrites and left 46 outside scope. The statistical fallback
stays in research. The app adopts the deterministic wording correction
when the ordinary comparison misplaced a change.
[Recorded geometry and readings](evaluation-panels/writing-sentence-anchor-transfer-2026-11-14.json)

The local checkout and comparison route returned exact positions
[89, 123) and [88, 89) for an authored punctuation case. JSON replay
and the PDF agreed, the optional statistical check reported the case
as outside scope, and zero analysis jobs started before simulated
payment.
[App verification receipt](evaluation-panels/writing-sentence-anchor-app-2026-11-14.json)

## Replay

The public receipt records hashes, positions, token ranges, and paired
measurements. It contains no source prose. The reader independently checks
the recorded geometry and response arithmetic in Python 3.11 or newer.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_sentence_anchor_transfer review \
  docs/evaluation-panels/writing-sentence-anchor-transfer-2026-11-14.json
```

To reproduce inference, supply the retained source rosters identified in
the receipt and the existing local pinned checkpoints. Earlier writing
ledgers take part in source exclusion. Use the writing ledger set recorded
in the receipt when reproducing the original roster. A different ledger
set can exclude additional sources.

```sh
cd backend
.venv/bin/python -B -m scripts.writing_sentence_anchor_transfer freeze "$freeze" --root "$source_data"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m scripts.writing_sentence_anchor_transfer \
  score "$freeze" --root "$source_data" --output "$results"
```

The draft comparison JSON verifier takes both original files and works
without the writing checkpoint weights.

```sh
cd backend
.venv/bin/python -B -m scripts.review_document_revision_export comparison.json earlier.txt later.txt
```
