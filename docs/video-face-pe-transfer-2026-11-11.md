# Perception Encoder face-swap transfer

The candidate uses the pinned GenD Perception Encoder checkpoint and the same five-landmark alignment as the earlier CLIP arm. Each arm used the same six decoded RGB pictures, source timestamps and largest-face boxes. PE and CLIP have their own model preprocessing. The reference uses the existing face crop path. [Pinned model and inference preflight](video-face-delivery-transfer-2026-11-11.md#checkpoint-preflight-for-another-detector-arm)

## Frozen setup

Ten genuine development clips set the operating points. The 108 transfer clips were outside those ten source groups and had already been used in two earlier detector comparisons. The first transfer panel contributed 20 groups and 60 clips across two delivery cohorts. The second contributed eight other groups and 48 clips with two deliveries for each source clip. Each group contains a genuine original and two whole-clip swaps. The two comparison panels share no source groups with each other. [Earlier paired pooling comparison](video-whole-clip-pooling-transfer-2026-11-11.md) · [Same-source delivery comparison](video-face-delivery-transfer-2026-11-11.md)

The PE arm checked the exact input SHA, native frame and timestamp receipts, and largest face from both earlier arms before scoring each frame. Each detector needed at least four finite responses, then contributed the median of its available responses. Each combined rule scales a detector using the median and 90th percentile of its ten development originals. The maximum standardized response gets its own 90th-percentile operating point, derived on those same ten originals. All six rules produced one flag among ten development originals with strict exceedance. The PE operating points were sealed before the transfer run.

## Results

Each cell lists detected swaps followed by false original flags. Every case was assessed by every method.

| Rule | Cohort transfer, matched delivery | Cohort transfer, native HD | Paired delivery, matched | Paired delivery, native HD |
| --- | ---: | ---: | ---: | ---: |
| Reference | 9/24, 1/12 | 3/16, 0/8 | 6/16, 0/8 | 6/16, 0/8 |
| CLIP | 2/24, 0/12 | 8/16, 0/8 | 3/16, 0/8 | 2/16, 0/8 |
| PE | 7/24, 1/12 | 15/16, 0/8 | 8/16, 2/8 | 14/16, 0/8 |
| Reference + CLIP | 9/24, 1/12 | 5/16, 0/8 | 6/16, 0/8 | 5/16, 0/8 |
| Reference + PE | 12/24, 1/12 | 14/16, 0/8 | 9/16, 0/8 | 14/16, 0/8 |
| Reference + CLIP + PE | 12/24, 1/12 | 14/16, 0/8 | 9/16, 0/8 | 14/16, 0/8 |

Reference plus PE detected 49 of 72 swap deliveries, up from 23 with the reference. Each flagged one of 36 original deliveries, from one of 28 source groups. PE alone detected 44 swap deliveries and flagged three originals. Adding CLIP gave the same counts in all four cohorts. The 108 deliveries include paired versions of the same pictures, and these counts describe the observed local panels.

The PE development worker took 19.643 seconds and the transfer worker 145.126 seconds on local Metal with float32 and batch size one. Each peaked at approximately 1.66 GB of sampled process-tree memory. An independent readback matched all 118 clip bindings and 708 source frame observations per arm, then recomputed all thresholds and counts. The transfer supports a prospective check of reference plus PE on the separate v95 confirmation reserve. [Aggregate and paired transitions](evaluation-panels/video-face-pe-transfer-2026-11-11-result.json)

## Independent score replay

The fixture contains anonymous group and pair labels and the three clip medians per delivery. It contains no media, filenames, timestamps, crops, embeddings, or source identities. The Python standard-library replay recalculates the six operating points, delivery transitions, and per-family counts. Native inference verification additionally requires the pinned source media, checkpoints and decoder.

```sh
python3 backend/scripts/replay_face_pe_transfer.py \
  docs/evaluation-panels/video-face-pe-transfer-2026-11-11-medians.json \
  docs/evaluation-panels/video-face-pe-transfer-2026-11-11-result.json
```
