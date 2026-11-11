# Whole-clip face-swap pooling on separate source groups

The earlier face-swap comparison found distinct successes from two face-crop checks. A simple union could raise the false-alarm rate. This experiment tests a paired operating point and a separate rule that uses the mean of the two highest crop responses. [Earlier comparison](video-full-clip-face-swap-transfer.md)

## Frozen comparison

The four arms use the reference median, candidate median, candidate mean of the top two, and a paired maximum of the two medians. Each crop arm used six fixed native source frames and needed at least four measured responses. For the paired arm, each median is centered at the median of ten development originals and divided by its distance to the 90th-percentile operating point. The maximum of these two standardized responses has its own operating point fitted on the paired development originals. Each arm flagged one of the ten originals, using strict exceedance. The fit and code were locked before the new scores. These operating points are internal to this experiment.

The matched delivery panel used 12 v17 groups outside the earlier 66-clip study, ordered by SHA-256 of `paired-pooling-2026-11-11` plus source group. These recordings appeared in earlier appearance experiments. The native HD transfer panel used the first eight compatible `confidence_cal` groups in the saved v95 order. The native groups were qualified against the historical source graph. They came from the same dataset and generator families as the matched group. Every group had an original and two full-clip swaps. The two panels share no source or donor filename with each other or the earlier comparison. None of their responses set the operating points. The 24 separately reserved confirmation groups remain available for future studies. [RTFS acquisition and rights review](video-common-source-acquisition-v91.md) · [Native insert source protocol](video-native-inserts-v95-method.md)

The two arms consumed identical RGB pixels, frame indices, source timestamps and largest-face boxes across 360 samples per arm. The HD samples also matched the existing archived native commitments. Native clips include 720p and 1080p deliveries. Source cohorts and delivery settings changed together, so this comparison cannot isolate the effect of resolution or encoding.

## Local results

| Check | Matched swaps flagged /24 | Matched originals falsely flagged /12 | Native HD swaps flagged /16 | Native HD originals falsely flagged /8 |
| --- | ---: | ---: | ---: | ---: |
| Reference median | 9 | 1 | 3 | 0 |
| Candidate median | 2 | 0 | 8 | 0 |
| Candidate mean of top two | 2 | 0 | 7 | 0 |
| Paired maximum of medians | 9 | 1 | 5 | 0 |

All 60 clips were assessed in every arm. The paired rule added three native swap flags and lost one from the reference. Its matched results were identical to the reference. Overall it flagged 14/40 swaps and 1/20 originals, versus 12/40 and 1/20 for the reference. This small gain still left 26/40 swaps below the operating point. The candidate median was stronger on the native HD panel and weaker on the matched delivery panel. Choosing the candidate for HD and the reference for matched delivery after seeing these outcomes gives 17/40 and 1/20 on the same data. That routing result is exploratory. Keep the serving decision pending a prospective same-source delivery comparison and a larger independent panel. [Aggregate with per-family misses, transitions and resource receipts](evaluation-panels/video-whole-clip-pooling-transfer-2026-11-11.json)

The local workers used the existing commercial-admitted weights, Metal and float32. They completed in 94.512 seconds for the candidate and 96.221 seconds for the reference. Sampled process-tree peaks were approximately 1.68 GB and 1.35 GB. Both exited successfully without a resource stop.

## Independent arithmetic replay

The published scalar file contains six responses per arm on each of ten development originals and 60 evaluation clips, with anonymous group labels and no source media, identifiers, faces, crops, timestamps or embeddings. The offline script independently recomputes all four points and paired counts with the Python standard library.

```sh
python3 backend/scripts/replay_whole_clip_face_pooling.py \
  docs/evaluation-panels/video-whole-clip-pooling-scalars-2026-11-11.json \
  docs/evaluation-panels/video-whole-clip-pooling-transfer-2026-11-11.json
```

The replay verifies score aggregation. Re-running native inference requires authorized access to the source files, pinned weights, and matching decoder. [Anonymized scores](evaluation-panels/video-whole-clip-pooling-scalars-2026-11-11.json) · [Native receipt comparison](../backend/scripts/compare_whole_clip_face_scores.py)
