# Short writing responses after typography normalization

This test asks whether an AI-like writing response stays useful when the
same sentence is read with its surface typography standardized. Recent
studies report that changes to superficial formatting can alter AI-text
detector decisions. They motivate paired counterfactual inputs.
Our test changes only its scoring view. Report locations always refer to
the submitted text. [Shortcut-controlled evaluation](https://aclanthology.org/2026.emnlp-main.2177/) ·
[Formatting counterfactuals](https://aclanthology.org/2026.docinsights-1.13/) ·
[Stylometry of AI writing and AI editing](https://aclanthology.org/2026.emnlp-main.1725/)

The unchanged primary comparator scans the first six native three-sentence
contexts above its established research point. The alternate checkpoint reads
each exact sentence once as submitted and once after Unicode normalization,
case folding, converting curly quotes and dashes, and replacing decimal
digits with zeros. Native segmentation occurs first, so all alternate
responses refer to the same original source range. At least two same-paragraph
neighbors are required, with three words in each neighbor and 12–65 words in
the focus. Native and normalized views use the same checkpoint and are
correlated. Their minimum measures response persistence under this transform.

The four alternate arms choose one focus each. They rank the native exact
sentence, normalized exact sentence, the lower of the two responses, and
the lower of the two responses after subtracting the stronger immediate
neighbor on the log-odds scale. Each arm's point is fixed by the second
highest genuine source maximum. It uses both dated published versions of
21 Wellcome and Gates articles, giving 42 distinct selection files.
Eighteen source groups had an eligible measured center. The other three
remain outside the threshold calculation.
[Authentic journal revision sources](writing-journal-body-transfer-2026-11-11.md)

The held groups contain eight dated forum threads and six USWDS pages.
They provide 28 genuine files, half original and half observed editorial
revisions, and 23 exact one-sentence rewrites across Granite and OLMo.
The article, forum and guidance banks had previous project exposure.
The held groups did not set the point and had never been measured under
this exact typography procedure. Both evaluation classifiers use pinned
Apache 2.0 checkpoints. No new model was trained.
[Forum source and generator receipts](writing-forum-revision-transfer-2026-11-13.md) ·
[USWDS sources and checkpoint terms](writing-uswds-revision-transfer-2026-11-13.md)

A useful location needs at least half of the edited non-whitespace
characters and at least 0.30 intersection over union in one returned
range. The check needs three additional source groups across both
generator families, at most one added genuine group and at least 90%
candidate assessment for promotion. The raw responses carry no authorship
confidence.

## Results

| Method | Forum and USWDS useful edits | Genuine files flagged | eLife useful edits | eLife human-source files flagged |
| --- | ---: | ---: | ---: | ---: |
| Existing primary contexts | 9/23 | 10/28 | 0/16 | 0/20 |
| Native isolated sentence | 1/23 | 0/28 | 1/16 | 0/20 |
| Normalized isolated sentence | 9/23 | 2/28 | 5/16 | 1/20 |
| Lower of native and normalized responses | 9/23 | 2/28 | 5/16 | 1/20 |
| Persistent contrast with adjacent sentences | 8/23 | 0/28 | 4/16 | 0/20 |
| Existing plus persistent contrast | 14/23 | 10/28 | 4/16 | 0/20 |

The persistent contrast added five exact locations across four forum and
USWDS works, including both generator families, and four locations across
three eLife works. It assessed every held file, produced zero wrong-only
flags, and contributed zero genuine flags in either panel. The normalized
absolute and minimum-response arms selected the same locations. A union
with the existing check found 16 of 23 forum and USWDS edits, with two
candidate genuine flags on one source group. The primary comparator is
the native window proxy, since other Full worker findings were outside
this study.

The eLife replication reused the frozen points with distinct source
groups. The original bank had 15 eLife articles. Ten qualified for a
three-sentence crop with both neighbors in the same paragraph. These
gave 16 generated edits and 20 human-source controls, half untouched and
half constructed from genuine prose elsewhere in the same article. The
crop is centered on the recorded edit or corresponding original sentence,
so it is a controlled short-input test. No eLife case affected operating
points or the first held panel. All eLife text had earlier project
exposure under other methods.
[eLife publisher terms, exact inputs and generator receipts](writing-elife-exact-sentence-confirmation-2026-11-12.md)

## Fresh diplomatic writing transfer

The pinned Foreign Relations of the United States XML archive supplied
17 documents from four volumes with distinct recorded official sender
keys. The qualifier excluded the 138 works already covered by the
archive studies and screened normalized 12-word overlap against the
selection and earlier held typography files. The source volumes had
prior project exposure. The selected documents and this method's
responses are fresh. Each source supplied one unchanged three-sentence
excerpt and one controlled substitution using a different genuine
sentence from the same document. These controls use authentic prose.
The substitution is a constructed edit.
[Archive and public-domain terms](writing-archive-source-evidence-2026-11-12.md)

The same two pinned generation families independently attempted two
rewrites per source. Sixteen Granite and eight OLMo replacements
qualified. The selected prose retains exact native sentence positions
and text outside the replacement. The frozen typography points remain
identical to the earlier panels.

| Method | Useful generated locations | Untouched or ordinary files flagged |
| --- | ---: | ---: |
| Existing primary contexts | 2/24 | 3/34 |
| Native isolated sentence | 0/24 | 0/34 |
| Normalized isolated sentence | 4/24 | 0/34 |
| Lower of native and normalized responses | 4/24 | 0/34 |
| Persistent contrast with adjacent sentences | 1/24 | 1/34 |
| Existing plus normalized isolated sentence | 5/24 | 3/34 |

All 58 fresh files were assessed. The four normalized or persistent
locations all came from Granite. Three added locations across three
works. The relative arm's single location was already detected by the
existing check, and the arm added one genuine flag. No candidate met
the two-generator advancement criterion on fresh sources. The methods
remain in research. The shorter held excerpts and known construction
geometry also leave natural customer edits outside this test.
[Fresh text-free result](evaluation-panels/writing-typography-frus-2026-11-13.json)

## Local replay

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.writing_typography_transfer freeze "$run" \
  --selection "$journal_cases" --forum "$forum_run" --uswds "$uswds_run" \
  --elife "$elife_plan"
backend/.venv/bin/python -B -m scripts.writing_typography_transfer selection "$run" \
  --primary "$primary_snapshot" --independent "$second_snapshot"
python3 -B -S -m scripts.writing_typography_transfer points "$run"
backend/.venv/bin/python -B -m scripts.writing_typography_transfer held "$run" \
  --primary "$primary_snapshot" --independent "$second_snapshot"
backend/.venv/bin/python -B -m scripts.writing_typography_transfer elife "$run" \
  --primary "$primary_snapshot" --independent "$second_snapshot"
python3 -B -S -m scripts.writing_typography_transfer report "$run" --output "$run/result.json"
python3 -B -S -m scripts.writing_typography_transfer review "$run/result.json"
```

The [text-free receipt](evaluation-panels/writing-typography-transfer-2026-11-13.json)
includes exact source positions, input hashes, model responses and decisions.
The standard-library reader recalculates points and held outcomes. Native
reconstruction also requires the previously retained published case
records, exact generated fixtures and pinned local checkpoints.

The separate fresh reader ties all 58 cases to the same earlier point and
recomputes each reported source decision. The public receipt contains
the official document links and pinned XML hashes. Reconstructing its
exact native inference also requires the captured archive fixture roster
and generated attempts retained locally.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_typography_frus \
  docs/evaluation-panels/writing-typography-frus-2026-11-13.json \
  docs/evaluation-panels/writing-typography-transfer-2026-11-13.json
```
