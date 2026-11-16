# Calibrating writing passage checks for different document types

A shared writing calibration flagged ten of the 20 untouched economic
reports in the earlier comparison. This study asks whether the app can
use a calibration for the apparent document type and reduce those alerts
while preserving the ability to locate known rewrites. The comparison
uses the same pinned Apache-2.0 writing checkpoint and the same isolated
sentence and overlapping three-sentence readings as the local app.
[Earlier within-document check](writing-local-baseline-transfer-2026-11-16.md)
· [Checkpoint publisher](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small)

## Method fixed before fresh scoring

The document profile uses sentence count, median sentence length, the
spread of sentence lengths and characters per word. A second version
also uses the lower end of the model readings in the file. Each profile
is compared with the nearest three source groups of each genre. It can
select a genre only inside that genre's fitted radius and with a clear
distance advantage over the other genre. A file that fits neither
closely enough falls back to the complete current check.

Each genre has its own word-count adjustments for isolated and contextual
readings. Those adjustments use medians and upper readings calculated
first within each genuine source group, then across groups. The interval
search is otherwise the same. We compared two points for each routing
method and included a reference run that was supplied the correct genre
in advance. The source-aware run measures what a correctly chosen genre
can contribute. It is unavailable for arbitrary uploads.

The two retained comparisons supplied 35 source groups for fitting and
nine separate groups for choosing the method. The shape-only profile at
the tighter research point was selected before running the new documents.
The freeze includes the candidate, all fitted parameters, training and
selection group digests, and the implementation hash. These are
engineering operating points and do not specify a public confidence level.
[Retained two-passage comparison](writing-multi-passage-transfer-2026-11-16.md)

## Fresh source groups and measurement

The held panel contains 11 historical author groups and seven Federal
Reserve report dates with both the untouched excerpt and the rewritten
version. Each edited document has one or two annotated replacements.
Selection takes one usable document per source group and favors the least
represented generator backend within each genre. Six publisher-reported
backends are represented. Some identities are requested or UI-reported
labels. The publisher calls the parquet file a training split. These
documents supplied only test inputs to the current study. Earlier source
groups, references and exact native text hashes were excluded. Exposure
during the checkpoint's original training is unknown.
[Pinned publisher card](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans/tree/8b31cdf2971ec4e959776e16744d0e5b3159ced8)

The genuine-only cases add 16 dated Boston district reports from
2017–2018 and 40 diplomatic excerpts from four Foreign Relations of the
United States volumes. The excerpts were previously indexed for archive
matching and supplied no readings to either profile's fit or selection.
The controls measure ordinary highlights and the behavior of the router
outside the two fitted genres. We keep titles, source terms and native
file hashes in the audit. The chosen publisher excerpts use vetted
Gutenberg and Board sources. The native texts stay local.
[Board archive for 2017](https://www.federalreserve.gov/monetarypolicy/beigebook2017.htm)
· [Board archive for 2018](https://www.federalreserve.gov/monetarypolicy/beigebook2018.htm)
· [Board source terms](https://www.federalreserve.gov/disclaimer.htm)
· [Foreign Relations archive license](https://github.com/HistoryAtState/frus/blob/216cdb7649a8e29504377d4f2858351dc46331a8/LICENSE.md)

Each file is tested in published and whitespace-neutral layouts. Overlap
is measured on the unchanged non-whitespace coordinate system. A correct
highlight must cover at least half of a labeled passage and have at least
0.30 intersection over union. Each highlight can match one label. The
study also counts suggestions outside labeled edits, false highlights on
untouched files and hits whose corresponding original passage was clear.

## Local results

All 184 file versions had complete readings on 7,124 distinct model
inputs. The table counts the original layout of each document. The 23
ordinary economic reports include the seven matched originals and 16
additional controls. The diplomatic controls have no rewritten versions.

| Source | Known passages located, current → selected | Untouched files flagged, current → selected | Unmatched highlights in edited files, current → selected |
| --- | ---: | ---: | ---: |
| Historical prose | 7 → 8 of 15 | 0 → 1 of 11 | 2 → 2 |
| Economic reports | 6 → 6 of 7 | 14 → 8 of 23 | 0 → 0 |
| Diplomatic correspondence | — | 1 → 5 of 40 | — |

In the whitespace-neutral layout, the selected method located nine of 15
historical edits, up from seven. It added one misplaced historical
highlight in that layout. Ordinary alerts and economic edit locations
were identical across layouts. Neither layout lost a location that the
current check found. Four of the six cleared economic alerts came from
the 16 additional controls. On the matched economic pairs, the new rule
located six edits, five of which had clear corresponding original
passages. The current check located six, three of which had clear
corresponding originals.

The document shape selected the historical calibration for 17 of the
40 diplomatic originals. The four additional false highlights came from
two of the four volumes. When supplied with the correct genre, the
tighter economic calibration kept six of seven edits and cleared all 23
ordinary reports. The alternative that uses both shape and model
readings flagged nine economic reports and one diplomatic original.
Its published-layout edit count stayed at 13 of 22, matching the
existing check. These secondary comparisons did not select the method.

The automatic change remains under evaluation. It improves some
historical locations and clears economic alerts, while its route for
unfamiliar documents adds false highlights. All 40 diplomatic controls
come from four related volumes. The economic and historical edit groups
are small, and this panel gives no accuracy estimate for unrestricted
customer documents. The app continues to show its current passage and
within-document evidence. The public receipt retains all choices for
fresh comparisons. [Per-delivery results](evaluation-panels/writing-profile-calibration-2026-11-16.json.gz)

The separate location counter reproduced all 42 method, genre and
layout summaries. The native readback verified 8,618 recorded context
windows and 388 proposed ranges against the exact extracted text and
non-whitespace character offsets. The fit replay reproduced the freeze
byte for byte using the two retained numeric panels.

## Replay

The public receipt includes scalar readings, source hashes, original
character locations and aggregate decisions. The first command replays
the frozen routing and passage selection in the standard library. The
second recounts the locations using the independent event matcher.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
receipt="$panels/writing-profile-calibration-2026-11-16.json.gz"
python3 -B -S -m scripts.writing_profile_calibration verify "$receipt"
python3 -B -S -m scripts.writing_profile_sources verify "$receipt"
```

For a native replay, save the pinned `data/mixed/train.parquet` as
`mixed-train.parquet` in `$publisher`. Save each of the 16 report pages
with its original `beigebookYYYYMM.htm` filename in the same directory.
Use the local classifier snapshot pinned at
`f1795c86806e6838d4afa33d0b1427f8430c9615` for `$snapshot`. The intake
and numeric receipt record the exact source and asset digests. The
`--plan` readback checks the text and the original offsets against the
local reconstructed inputs. Keep that plan on the local machine.
[Pinned source shard](https://huggingface.co/datasets/open-text-detector/heterogeneous-ai-spans/resolve/8b31cdf2971ec4e959776e16744d0e5b3159ced8/data/mixed/train.parquet)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
earlier="$panels/writing-multi-passage-transfer-2026-11-16.json.gz"
retained="$panels/writing-local-baseline-fresh-2026-11-16.json.gz"
python3 -B -S -m scripts.writing_profile_calibration fit "$earlier" "$retained" "$run/freeze.json"
"$runner" -B -m scripts.writing_profile_sources prepare "$publisher" \
  backend/app/evidence/data/frus-archive.json "$earlier" "$retained" "$run/held-plan.json.gz"
"$runner" -B -m scripts.writing_multi_passage_panel score "$run/held-plan.json.gz" \
  "$snapshot" "$run/held-readings.json.gz"
python3 -B -S -m scripts.writing_profile_calibration assess "$run/freeze.json" \
  "$run/held-readings.json.gz" "$run/result.json.gz"
python3 -B -S -m scripts.writing_profile_sources verify "$run/result.json.gz" --plan "$run/held-plan.json.gz"
```
