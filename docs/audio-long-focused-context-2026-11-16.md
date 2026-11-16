# Short speech edits in longer recordings

The local focused speech edit check works on a continuous six-second
recording. This test uses that same fitted check on six-second excerpts
of a nine- to fourteen-second recording. All measurements retain their
sample positions in the original upload. It tests whether agreement across
shifted excerpts can recover short locations while limiting reports on
ordinary speech. The repeated reads share audio and a model. The test
treats their agreement as a consistency check.
[Six-second app review](audio-focused-edit-local-2026-11-16.md)

Long-audio research compares ways to combine fixed-length responses. A
separate cross-condition study reports that shorter edited segments are
harder to locate under different acoustic conditions. We therefore compare
the same models and whole-file inputs under separate selection rules and
retain results from both loudness levels. The test uses the existing
commercially permitted weights and our own inference code.
[Long recordings and aggregation study](https://openaccess.thecvf.com/content/WACV2026W/SAFE-2026/papers/Gajewska_Audio_Deepfake_Detectors_vs._Real_Fraud_-_The_Fall_of_WACVW_2026_paper.pdf)
· [SPADE transfer study](https://arxiv.org/html/2609.25197v1#S4.SS4)
· [Encoder weight terms](audio-wavlm-v13-protocol.md#source-and-hypothesis)

## Fixed input and measurement

The comparison includes 18 published partial edits from six generators
and 18 genuine recordings from 36 different declared reader groups.
All 18 edits come from three new 32 MiB slices of the official Part B
archive. The genuine files come from bounded Part A ranges. Source IDs
and WAV and PCM digests differ from those in the eight linked source
ledgers. Thirty-four declared reader IDs were present in earlier
comparisons. These are fresh utterances for the checked history.
Earlier project and model-training exposure may recur. Ordinary files
come from the same publisher corpus and are matched by delivery
condition. Each edited file retains the publisher's original sample
labels, including all published intervals in a selected file.
[Licensed publisher audio and exact labels](https://zenodo.org/records/14214149)

Each recording keeps its native length between 9.050 and 13.770 seconds.
Test the original samples and add one real conversation at nominal
foreground-to-background ratios of −5 and −10 dB. Retain the exact
isolated background from each mix as a separate check. The six distinct
room ranges preserve an earlier pinned six-second prefix and extend it
eight seconds on the publisher clock. The foreground-to-background
ratio within each edit is recorded separately. This gives 180 deliveries
with 36 foreground sources and six repeated background sources.
[AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The comparator follows the local three-second schedule, including its
focused overlap rule through 11 seconds. Compare the addition of the
existing two-second check run within six-second excerpts, the fine-grained
check on those same excerpts, and the fine-grained check with agreement
across two excerpts. The excerpts start every two seconds and at the
exact last six seconds of the file. Agreement requires the two flagged
intervals to overlap by at least half of each interval. Where several
overlapping proposals mark the same location, retain the strongest
individual range on its original clock. The current reported ranges
stay in every comparison.

The score points stay fixed. A file is flagged when any interval is
reported. A useful range covers at least half of a
labeled edit and reaches 0.30 temporal intersection over union. The
review assigns reported ranges to labeled edits one to one. It also
counts alerts on ordinary files and isolated rooms, and genuine time
highlighted inside partially edited as well as ordinary recordings.

## Focused review results

The focused check assessed all 180 deliveries and examined 720 six-second
excerpts. The standalone results below measure the extra check by itself.

| Delivery | Useful labeled locations, single / repeated | Ordinary files flagged, single / repeated | Isolated backgrounds flagged, single / repeated |
| --- | ---: | ---: | ---: |
| Original | 0/115 · 0/115 | 0/18 · 0/18 | — |
| Competing speech, −5 dB | 1/115 · 0/115 | 3/18 · 0/18 | 5/36 · 0/36 |
| Competing speech, −10 dB | 1/115 · 0/115 | 2/18 · 2/18 | 5/36 · 0/36 |

The two useful results come from the same edit and source reader. Both
selected **0.700–1.200 s** against a published **0.594–1.004 s** interval.
The single-excerpt rule highlighted **2.97 s** of genuine time in the
−5 dB deliveries and **3.30 s** at −10 dB, including genuine portions
of edited audio. The repeated-excerpt rule highlighted zero genuine
seconds at −5 dB and one second at −10 dB. The repeated rule lost the
useful interval across both mixtures. These results support keeping
longer-file focused review in evaluation.
[Focused comparison receipt](evaluation-panels/audio-long-focused-context-pilot-2026-11-16.json.gz)

Every frozen delivery rebuilt byte for byte. A separate sample audit
verified the real room source, the exact 72 isolated-background
deliveries and the common gain within each mixture. The largest
deviation from the nominal whole-recording ratio was 0.000033 dB.
The portable location tool independently reproduced the event and
genuine-time counts for both focused rules.

## Completed paired comparison

The full comparison scored all 180 deliveries and reproduced all 720
focused-excerpt responses. The table tracks useful edit locations,
ordinary files with any flag, and isolated backgrounds with any flag.
Each row uses 18 ordinary and 18 edited recordings. The two noise
conditions each have 36 isolated-background deliveries. All ordinary,
edited and isolated inputs were assessed by the existing review.

| Delivery and review | Useful edits /115 | Ordinary flags /18 | Isolated background flags /36 |
| --- | ---: | ---: | ---: |
| Original, current | 46 | 0 | — |
| Original, add the two-second scan | 66 | 0 | — |
| Original, add the fine-grained scan | 46 | 0 | — |
| Original, add repeated fine-grained results | 46 | 0 | — |
| Competing speech, −5 dB, current | 22 | 8 | 0 |
| Competing speech, −5 dB, add the two-second scan | 36 | 12 | 1 |
| Competing speech, −5 dB, add the fine-grained scan | 23 | 10 | 5 |
| Competing speech, −5 dB, add repeated fine-grained results | 22 | 8 | 0 |
| Competing speech, −10 dB, current | 10 | 8 | 0 |
| Competing speech, −10 dB, add the two-second scan | 20 | 10 | 1 |
| Competing speech, −10 dB, add the fine-grained scan | 11 | 8 | 5 |
| Competing speech, −10 dB, add repeated fine-grained results | 10 | 8 | 0 |

The two-second scan preserved earlier useful locations and added 20
clean, 14 moderate and ten strongly masked locations. Each condition's
gains span all six generator families and respectively 11, 12 and nine
readers. The check also highlighted 7.10, 27.68 and 20.02 additional
seconds of genuine audio across those three sets. This includes genuine
portions of partly edited recordings. The repeated noise conditions
share sources and room excerpts. The fine-grained comparison supplied
only one distinct extra useful location across the two mixtures.

The standard-library exporter independently reproduced the one-to-one
event totals and exact genuine-sample counts for all four arms. The
two-second extension is still a longer-file candidate. Application
replay, a separate noise transfer, and confidence calibration remain
open. [Full paired comparison receipt](evaluation-panels/audio-long-focused-context-paired-2026-11-16.json.gz)

## Reproduce

Use the earlier [Part A and Part B range recipe](audio-noise-window-transfer-2026-11-12.md#replay)
for `$part_a`, `$part_b` and the Part A labels. For `$new_ranges`,
request 32 MiB starting at 4, 8 and 11 GiB from the official Part B
archive. Keep each response as `llama-<start-byte>.tarseg` with the
matching `llama-<start-byte>.headers` and verify the exact range
and the SHA-256 values in the receipt. The native source recipe uses
the original official audio locally. The portable tools require
standard-library Python and the numeric receipt.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-long-focused-context-pilot-2026-11-16.json.gz
python3 -B -S -m scripts.audio_focused_context_transfer review "$receipt"
python3 -B -S -m scripts.export_audio_focused_context "$receipt" "$exported"
paired=docs/evaluation-panels/audio-long-focused-context-paired-2026-11-16.json.gz
python3 -B -S -m scripts.audio_focused_context_transfer review "$paired"
python3 -B -S -m scripts.export_audio_focused_context "$paired" "$exported-paired"
python3 -B -S -m scripts.independent_review count-events \
  "$exported/manifest.json" "$exported/truth.json" "$exported/focused_excerpt.json" \
  "$event_counts" --match 0.30 0.50

runner=backend/.venv/bin/python
"$runner" -B -m scripts.audio_focused_context_sources capture "$rooms"
"$runner" -B -m scripts.audio_focused_context_sources prepare \
  "$part_b" "$new_ranges" "$part_a" "$a_labels" "$rooms" "$inputs"
"$runner" -B -m scripts.audio_focused_context_sources audit "$inputs" "$rooms"
"$runner" -B -m scripts.audio_focused_context_transfer score-focused \
  "$inputs" models/weights/audio-upgrade "$native"
```

The export records exact sample truth and which files were assessed.
Another detector can supply a response file on the same input hashes.
Run the same event-count command with that file to recount its
locations and misses. The local `score` command measures the longer-file
review and the two-second diagnostic alongside the focused arms.
