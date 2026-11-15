# Reversed moving pictures within one video

Chipmunk Verify now looks for a sequence of decoded pictures that appears again in reverse in the same upload. A match shows the two ranges on the source clock and the first and last frame of each range. Reverse playback is a conventional editing operation. The match is an editing observation with zero weight in the AI estimate. [FFmpeg reverse filter](https://ffmpeg.org/ffmpeg-filters.html#reverse)

The check uses the existing native-frame scan. Its scope ends at the applicable limit of 30 source seconds, 900 frames, or 1.5 GiB of decoded pixels. It requires two disjoint ranges, at least eight distinct matching pictures, at least half a second between the first and last frame of each range, and matching frame intervals in reverse order. A gap greater than 125 ms or an unverified timestamp breaks a range. The browser links to the endpoints and their playback times. The PDF and JSON use the same frame indices, and the signed evidence package can replay the scan against the original video.

## Local comparison

Eight retained VCD2 camera recordings supplied 56 newly authored nine-second deliveries at 320 × 180 and 20 fps. Reverse inserts varied among 19, 23 and 29 frames and used different source positions. Lossless deliveries used RGB H.264 or VP9. The lossy deliveries used H.264 CRF 23. Separate FFmpeg frame hashes and an independent search reproduced all 56 reversed-sequence decisions across 10,080 decoded pictures. The ledger contains frame bounds, source commitments and outcomes. [Source qualification](video-vcd2-unseen-controls-v287-result.md) · [Dataset terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Evaluation ledger](evaluation-panels/video-reversed-sequence-2026-11-14.json)

| Delivery | Existing exact forward check | New exact reverse check |
| --- | ---: | ---: |
| Exact reverse copy | 0/8 | 8/8, all four endpoints exact |
| Unchanged, conventional cut, or single-picture hold | 0/24 | 0/24 |
| Exact forward copy | 8/8 | 0/8 |
| Reverse copy with different frame spacing | 0/8 | 0/8 |
| Lossy re-encoded reverse copy | 0/8 | 0/8 |

Two isolated local analyses used simulated Stripe confirmation and the actual video worker with unrelated inference weights disabled. No job started before confirmation. The edited sample matched at 1.000–2.050 s and 6.500–7.550 s. The packet API and JSON export matched, the PDF contained the exact evidence images, the exports exposed no model names, and signed replay against the original succeeded. The unmodified sample recorded a completed check without a match.

## Reproduce

Run the seven generated cases locally without weights. To replay the 56 camera-derived cases, add `--archive PATH` for the qualified VCD2 archive directory containing `video-vcd2-unseen-controls-v287`. The runner deletes temporary deliveries after recording their results.

```sh
cd backend
.venv/bin/python -B -m scripts.video_reverse_sequence_transfer /tmp/video-reverse-replay
```
