# Federal prose and rewritten passage controls

Use the GoHumanize team's [Open Humanizer dataset, version 0.4.0](https://doi.org/10.5281/zenodo.22843083),
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Bind the four assets at the [exact dataset revision](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset/tree/cc066df21227f5fdb73b513c330181f946ce82ef).
Use all 54 modern test pairs from 30 exact article URLs across the Federal
Reserve, NIST, and Department of Energy. The publisher supplies 39
gpt-4o-mini and 15 deepseek-chat rewrites. Require an unambiguous row ID,
the exact URL, the declared agency, and the publisher's public-domain
basis finding for every pair. The dataset card says these 30 article
URLs are absent from its own training split. This intake does not acquire
that split or independently establish overlap with detector training.

Preserve the publisher's human passage (`output`) and entire released
rewrite (`input`) verbatim. For each pair, attempt a short (8–16 words)
and longer (35–65 words) replacement. In both the human host and AI donor,
select contiguous complete sentences within the band, each with a
complete sentence before and after and a midpoint in the central half
of the passage. Count Unicode letters and numbers as lexical words, with
internal apostrophes and hyphens allowed. Apply deterministic punctuation
boundaries with basic abbreviation handling. This heuristic can miss
nonstandard sentence boundaries. Select the pair with the minimum sum of
normalized distance to each other and to the two document centers, breaking
ties by word-count difference and then source offsets. Copy the donor
characters exactly and retain the host prefix and suffix byte for byte.
Require different normalized lexical content.

For each selected replacement, also keep the same host range and replace
its central ordinary interword space with one line break. This is an
all-human formatting control with no AI span. Keep a separate unavailable
row for each arm when a pair lacks suitable spans. The plan has 324 rows.
Use zero-based, half-open Unicode code-point offsets into the unchanged
or constructed text for labels. Preserve article group membership across
all arms. Repeated passages and derived views share article ancestry.

Compare the 108 released base passages against the locally available,
hash-bound v3/v4 classifier contexts, v5 selected source answers, and v7
prepared texts. Use whole normalized equality and any exact normalized
12-word sequence as a possible overlap signal. Report aggregate candidate
counts and the scope of manifests checked. A match can reflect boilerplate.
Absence of a match leaves near duplicates, unseen prior exposures and
upstream model training overlap unresolved.

The controlled replacements may shift meaning or style at the joins.
Publisher origin labels and exact copied-span truth describe this construction.
The inputs do not establish natural AI-assisted writing behavior, email
performance, intellectual authorship or a detector's accuracy. Future
evaluation should preserve matched arms and group by article URL.

Use a directory containing the exact four release files. The preflight
verifies commitments and counts before any private cases are written.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_intake_v193 preflight /private/tmp/chipmunk-gohumanize-v4.RSZE20Zw
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_intake_v193 freeze /private/tmp/chipmunk-gohumanize-v4.RSZE20Zw
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_intake_v193 prepare /private/tmp/chipmunk-gohumanize-v4.RSZE20Zw
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_intake_v193 readout /private/tmp/chipmunk-gohumanize-v4.RSZE20Zw
```

The readout rebuilds the case journal and summary and checks exact bytes.
Source prose, private case rows, individual text hashes, publisher IDs,
article URLs and per-case match details stay in the private workspace.
Only aggregate counts and whole-artifact commitments are publishable from
this intake. This stage uses the Python standard library and runs zero models.
