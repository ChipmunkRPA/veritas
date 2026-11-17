# Short edits in published government reports

Comparing an upload with its earlier published version located more
of these short changes than examining writing style alone. The
improved local archive check recovered seven edited reports
that previously had competing possible sources. It compares the
wording and points to the precise parts that differ.

## Source-separated comparison

The study uses 32 calibration and 69 held reports. Each comes from the
pinned GovReport archive and contains an untouched version and an
annotated OpAI-Bench polish. The originals predate 2021 and their
non-whitespace wording matches the archive. The intake verifies
sentence labels and character ranges and requires unchanged sentences
to be identical. Identical generated sentences have no visible edit.
Each pair has one to four changed sentences. The source roster checks
prior published report IDs and text digests. The held files use
separate report groups, two AI generators and two publisher splits.
[GovReport terms](https://gov-report-data.github.io/#license) · [Pinned
benchmark and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9)

## Writing-style comparison

The experimental style check ranks one to three sentences at a time.
The first arm uses the center-context response relative to the document.
The second fits a separate narrow reading. The third adds how the scores
change with the fraction of the candidate included in nearby windows.
The two fits share 1,176 matched comparisons from 35 earlier sources.
They use the existing Apache-2.0 checkpoint. Separate original reports
fix the two operating points using both layouts. Choose the arm and
point before measuring the held files. This tests the research idea that
local response patterns can help locate a transition.
[Writing transition research](https://arxiv.org/html/2609.33787v1) ·
[Pinned checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

Published-layout outcomes require 30% intersection over union and
coverage of at least half of each known changed passage. Each proposed
range can match only one label. The counts combine the candidate with
the current writing-style ranges.

| Comparison | Known locations found / 212 | Ordinary originals flagged / 69 |
| --- | ---: | ---: |
| Current writing-style ranges | 0 | 3 |
| Add the frozen narrow context check | 3 | 10 |
| Add the wider fitted overlap check | 6 | 13 |

The frozen narrow context check ranked a useful location first on
14 edits before applying its point. All arms found zero labeled
locations in the 32 calibration reports. Even the wider comparison
added more false highlights than correctly located edits in the held
files. The experimental style checks stay outside local analysis.
[Short interval results](evaluation-panels/writing-short-intervals-2026-11-16.json.gz)

## Comparison with an earlier official report

The archive review may accept one main source when the only competing
matches are brief and inside confirmed changes. Exact matches to the
primary source must cover at least two-thirds of the uploaded passage
and reach close to both ends. Exact alignment must verify the cited
source and the competing matches must lie inside the bounded comparison,
with each competing phrase overlapping differing text. Strong competing
sources and unmatched material outside the bounds stay ambiguous.
This rule describes wording differences with zero AI-authorship weight.

| Comparison | Correctly matched edited reports / 69 | Known changes located / 212 | Untouched files with a difference / 69 |
| --- | ---: | ---: | ---: |
| Earlier archive check | 52 | 144 | 0 |
| With brief competing phrases | 59 | 161 | 0 |

The new rule adds 17 correct change locations. Both modes identify 61
of the 69 untouched sources and give no incorrect source attribution
on the paired panel. The archive can assist when its pinned collection
includes the earlier version. A regular analysis of other uploads still
uses the available writing-style checks. [Archive comparison results](evaluation-panels/writing-archive-competition-2026-11-16.json.gz)

All 36 controls made from two genuine source reports stayed ambiguous.
Twelve combine large passages, twelve insert 64 tokens from the second
source, and twelve append 16 tokens from it after the first report.
The earlier 32 report pairs separately show 53 to 74 correctly located
changes out of 106 and zero difference flags on their original files.

In local development, enable
`VERITAS_DOCUMENT_GAO_ARCHIVE_RESOLVE_COMPETITION=true` with the installed
GAO archive. Open a completed analysis and select **U.S. GAO report
highlights** under **Compare with public archive**. The comparison explains
why it could select a source and preserves the original locations in the
archive PDF and JSON.

Three simulated USD 9.99 checkouts verified the same recovered report in
TXT, Word and selectable PDF. Zero analyses queued before payment. The old
comparison reported three ambiguous files. The new comparison produced
six exact wording differences in two source sentences in each format.
The downloaded JSON replays against the original file and pinned archive.
The corresponding PDF contains the cited report and explanation.
[Local app replay](evaluation-panels/writing-archive-competition-app-2026-11-16.json)

## Replay locally

Use the numeric receipts to recount both comparisons. The style
readback verifies the context digests and physical text offsets for
276 held versions and 5,672 context windows. The receipts contain
locations, hashes and measured responses. Keep the raw prose local.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.writing_short_intervals verify "$panels/writing-short-intervals-2026-11-16.json.gz"
python3 -B -S -m scripts.writing_archive_competition verify "$panels/writing-archive-competition-2026-11-16.json.gz"
```

Use the pinned files recorded in the receipts to reconstruct the
private plan with `scripts.writing_short_sources --receipt`, using the
published short interval receipt to recover the exact exposure list. Separate the resulting
works by their `validation` and `test` roles. The archive runner accepts
both plans and the [verified GovReport pack](writing-gao-archive-comparison-2026-11-13.md).
The writing classifier runner `scripts.writing_multi_passage_panel score`
reads the two roles separately. The `fit` and `assess` steps in
`scripts.writing_short_intervals` fix the former before reading the latter.
