# Separate sentences with verified wording differences

The ordinary results view now shows complete submitted sentences that
contain a verified wording difference from an earlier published U.S. GAO
highlight. Each sentence has its own source position and links to the exact
earlier and submitted wording for each difference. The comparison uses
the pinned GovReport edition and the existing archive source selection.
Wording differences can result from ordinary editing, transcription or
AI assistance. The evidence carries zero weight in the overall AI
assessment. [Earlier wording check and scope](writing-published-wording-2026-11-16.md)
· [GovReport licence and attribution](https://gov-report-data.github.io/#license)

The sentence list applies to eligible plain text, mapped Word body text
and visible selectable single-column PDFs. Text has original character
positions, Word has paragraph and character positions, and PDF has page
numbers and unrotated page word boxes. The PDF report and JSON preserve
the same text, positions and difference links. A signed export verifies
the sentence ranges and links by rereading the submitted file and the
retained transcript.

## Measured localization

The two comparisons use the same exact word edits and earlier report
attributions. The earlier summary joins consecutive changed sentences in
one paragraph. The new view keeps each sentence with a wording difference
separate. An independent matcher requires at least half of the known
edit and 0.30 intersection over union on non-whitespace characters.
Each display range counts for at most one labeled edit.

| Panel | Report pairs | Correctly located sentences in grouped → separate ranges | Untouched files with a wording flag |
| --- | ---: | ---: | ---: |
| Earlier short-report panel | 69 | 161 → 175 of 212 | 0 of 69 |
| Reserved report panel | 48 | 112 → 128 of 149 | 0 of 48 |
| Fresh confirmation | 48 | 108 → 123 of 150 | 0 of 48 |

The fresh confirmation uses 24 pairs from each of two OpAI-Bench
generators. The intake excludes prior report IDs and text digests,
including the 48 reserved reports. Ordinary text matches the dated
reference transcript. The production code ran on all 330 deliveries and
verified original text and locations for all 295 source-matched records.
All qualified sources match the published reference identity. The gain
comes from locating already measured differences at separate sentence
positions. [Source benchmark and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench)
· [Numeric results and source commitments](evaluation-panels/writing-archive-sentences-held-2026-11-17.json.gz)

## Local checkout and exports

The TXT, Word and single-column PDF replay showed sentences 7, 17 and 18
at the same extracted character ranges. Each format linked to the six
exact wording differences. The matched ordinary report showed no changed
sentence. Five simulated USD 9.99 checkouts queued zero analyses before
payment. Chromium followed the evidence links, and four signed packages
verified against their submitted originals. [Local replay results](evaluation-panels/writing-archive-sentences-app-2026-11-17.json)

## Replay locally

The standard-library reader recounts matched ranges from the published
positions and checks the confirmation source bindings.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/writing-archive-sentences-held-2026-11-17.json.gz
python3 -B -S -m scripts.writing_archive_sentence_panel verify "$receipt"
```

Build the 69-pair earlier plan using the [short report
recipe](writing-short-intervals-2026-11-16.md#replay-locally) and the
48-pair reserved plan using the [entropy study source
intake](writing-entropy-prominence-2026-11-17.md#independent-replay).
Pass the reserved plan, pinned OpAI shard folder and local GovReport
pack to `scripts.writing_archive_sentence_panel reserve`. The `compare`
command reads that confirmation plan alongside the earlier and reserved
plans. It validates the exact source text and positions before recounting.
