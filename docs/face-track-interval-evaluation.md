# Face-track interval evidence

The video evidence view groups repeated high face responses on the same track
into a review interval. Each interval requires at least three scored samples
between two lower groups of two samples. A time gap over 0.8 seconds breaks the
track sequence. The range uses the exact first and last scored timestamps.
The view displays the nearest before and after samples and the strongest
sample during the interval. The same roles and source timestamps are in JSON
and the signed PDF. The earlier point evidence remains available for isolated
responses and cases that lack flanking context. The interval does not enter
the whole-video confidence calculation.

## Matched benchmark

The [RTFS-10K source release](https://huggingface.co/datasets/godmodes/rtfs-10k)
provides original clips and matched swaps from two generators. The acquisition
excluded all 48 prior YouTube source IDs and prior donor filenames. The
24 new source groups each have an untouched control and two manipulated
versions. The clips share a 24 fps time base for this test. A fixed source
hash chooses the insertion position and duration of 1.25, 1.5 or 2 seconds.
Exact frame digests verify which normalized source supplied every output
frame. The task measures localization of the inserted footage and does not
establish a pixel mask for the generator's edits.

The first eight source groups were used for development. The interval rule
was locked before scoring the remaining 16 groups. It requires a mean score
of 0.70 during the range, a lift of 0.25 over both neighboring groups, and
an individual sample score of at least 0.40. These are uncalibrated responses.
The baseline is the previously displayed highest frame response at or above
0.50 with its adjacent scored frames. Both methods use the same face scores.
For range comparisons, the baseline's context range runs between its two
displayed endpoint frames. An overlap of 0.30 means the intersection covers
30% or more of the combined known edit and reported range. Misses count as zero.

The source manifest SHA-256 is
`a604addc29bc9efa9b5f88e9d6b6a6fc2355ddede85be26da7c362e4ff547a49`.
The derived media manifest SHA-256 is
`88d21ead5044c55722fa70d71da9bc411d90c8a60d6aeee56bdb90b08e43e732`.

## Held-out result

All 48 held-out clips were scored, with 16 genuine controls and 32 swaps.
The locked rule and the full comparison have SHA-256 commitments
`e034c0d07b708d04e0d29618e36ff8d71c1c5aebaca003b8f73ab8ef62c812ff` and
`c9a471fea2108efdb6633c3caaf85dd7b95eee515ffa0133c74fbdac57cd5dd6`, respectively.

| Result | Current point and neighbors | Face-track interval |
| --- | ---: | ---: |
| Genuine clips with a review flag | 9/16 | 3/16 |
| Highest displayed response inside the edited range | 8/32 | 2/32 |
| First displayed range with overlap of at least 0.30 | 4/32 | 2/32 |
| Mean range overlap, including misses | 0.073 | 0.052 |

The interval method adds a narrower evidence option with fewer review flags
on originals and lower localization recall. The app keeps the existing
point checks and their normal evidence budget. Clip-wide swaps and edits at
the start or end of a clip may lack the surrounding samples the rule needs.
The evaluation uses short clips from one dataset and two swap generators.

## Example and reproducibility

The local report example uses a development clip and illustrates the evidence
format. The inserted swap spans 1.833333–3.083333 seconds, end exclusive. The
reported review range is 2.041667–2.833333 seconds. The before, during and
after samples are 1.791667, 2.583333 and 3.083333 seconds. The ordinary API,
JSON export and signed PDF agree, including the exact source crops and
frame hashes. An isolated test checkout required payment confirmation before
creating a job. The customer outputs omit detector names.

Run from `backend` with the two prior acquisition manifests combined into the
excluded list. Score to fresh receipt paths.

```sh
trial=data/independent-evaluation/video-v4
.venv/bin/python scripts/prepare_face_swap_upgrade_eval.py "$trial/face-sources" \
  --groups 24 --exclude-manifest "$trial/excluded.json"
.venv/bin/python scripts/prepare_face_interval_eval.py \
  "$trial/face-sources/manifest.json" "$trial/intervals"
PYTHONPATH=. VERITAS_DEVICE=mps .venv/bin/python scripts/score_face_interval_eval.py \
  "$trial/intervals/manifest.json" "$trial/development.jsonl" --split development
PYTHONPATH=. .venv/bin/python scripts/summarize_face_interval_eval.py \
  "$trial/intervals/manifest.json" "$trial/development.jsonl" "$trial/interval-rule.json"
PYTHONPATH=. VERITAS_DEVICE=mps .venv/bin/python scripts/score_face_interval_eval.py \
  "$trial/intervals/manifest.json" "$trial/test.jsonl" --split test
PYTHONPATH=. .venv/bin/python scripts/summarize_face_interval_eval.py \
  "$trial/intervals/manifest.json" "$trial/development.jsonl" "$trial/interval-rule.json" \
  --test "$trial/test.jsonl" --output "$trial/interval-summary.json"
```
