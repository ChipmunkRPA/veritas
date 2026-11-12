# Repeated picture sequences within one upload

Chipmunk Verify can now locate a copied sequence of moving pictures in two places in the same video. The check found all six authored exact copies on retained webcam sources, with no reported pair on the 18 unchanged, ordinary-cut, and static-hold controls. All six lossy copies missed. The result is exact picture evidence and carries zero AI-confidence weight. [Native comparison ledger](evaluation-panels/video-exact-sequence-transfer-2026-11-12.json)

## Coverage and evidence

The comparison uses the same bounded native scan as the consecutive-picture check. It compares the first video stream within the scanned prefix, up to 30 source seconds, 900 frames, or 1.5 GiB of cumulative decoded RGB. If the larger scan fails, a completed opening scan can still supply coverage. The same frame digests serve both checks. A pair needs two nonoverlapping ranges with equal decoded RGB, identical native frame spacing, at least eight distinct pictures, and at least half a second between each range's first and last timestamps. Unverified timestamps and gaps above 125 ms break a range. When several pairs meet the rule, the longest wins, followed by the earliest source positions. [Existing native scan and budget](video-extended-picture-hold-2026-11-12.md)

The normal view and PDF show both time ranges and each range's first and last pictures. The JSON binds each endpoint to its source frame index and timestamp and stores the digest of the whole matched sequence. An original-file replay of a signed evidence ZIP checks every decoded picture in the inspected scope, recomputes the match and verifies the four original endpoint images. The check records when there are too few usable frames to measure two ranges. Loops, repeated titles with animation, and ordinary editing can produce an exact match.

## Local test

Six retained VCD2 webcam recordings each supplied 180 native 320 × 180 pictures at 20 fps after an identical local delivery conversion. The source recordings previously appeared in the consecutive-picture study. These new deliveries copy frames 20–41 to 130–151, keep the original, insert an ordinary horizontal-flip cut, hold one picture at the target, change the copied frame spacing, or re-encode the authored copy with lossy H.264. Three source groups use lossless RGB H.264 and three use lossless VP9 for the first five views. All time bounds were set before the comparison. [Capture sources and terms](video-extended-picture-hold-2026-11-12.md#local-validation)

| View | Matching sequences | Views |
| --- | ---: | ---: |
| Exact moving copy | 6 | 6 |
| Unchanged recording | 0 | 6 |
| Ordinary cut | 0 | 6 |
| Single held picture | 0 | 6 |
| Copied pictures with different frame spacing | 0 | 6 |
| Lossy re-encoded copy | 0 | 6 |

The consecutive-picture check found the six static holds and none of the six exact moving copies. The separate sequence comparison found the full authored range in every lossless copy. Independent FFmpeg frame hashing and a separate starting-position search checked 6,480 native frames and reproduced all 36 outcomes. A second run with generated motion reproduced its six public cases without a source archive.

An isolated app run used simulated checkout. It showed both occurrences of a 24-picture copy at **5.000–6.150 s** and **20.000–21.150 s**, with 601 checked frames and 39 outside the scan. Chrome rendered all four source endpoints and sought to the second occurrence. The PDF, JSON, and signed replay used the same evidence. [Public synthetic fixture and app receipt](evaluation-panels/video-exact-sequence-transfer-2026-11-12.json)

## Replay

Run the six generated test cases without model weights or source media. To reproduce the webcam comparison, pass the six existing VCD2 delivery files from slots 01, 02, 04, 07, 08, and 11 using `--webcam FILE` in that order. The runner keeps aggregate results and source commitments and removes each temporary delivery after replay.

```sh
cd backend
.venv/bin/python -B -m scripts.video_exact_sequence_transfer /tmp/picture-sequence-replay
```
