# Reversed pictures after lossy video compression

Chipmunk Verify can follow similar pictures in reverse order in the same
upload. It shows the two source ranges and their original endpoint frames
in the browser, Standard PDF and JSON. Each of the four endpoint images
comes from its own position in the submitted file. Reverse playback can
come from ordinary editing and carries zero weight in AI confidence.

The check reuses the forward-motion comparison's 40 × 30 luminance grid,
whole-picture similarity and motion requirements. It compares movement in
opposite temporal directions and requires at least eight matching motion
transitions across two disjoint sequences at the same frame spacing. The
scan uses the first nine seconds of upright 320 × 180 video at 20 fps or
640 × 360 video at 25 fps, with verified source timestamps and H.264 or VP9
4:2:0 delivery. A qualifying exact reversed sequence retains priority.
The reported endpoints delimit measured support. Compression, low motion,
or changes in speed can leave part of a reverse edit outside those bounds.
[Earlier forward-motion measurement](video-lossy-sequence-transfer-2026-11-12.md) ·
[Exact reverse comparison](video-reversed-sequence-2026-11-14.md)

## Held comparison

Eight retained camera recordings separate from the six development
recordings supplied the held panel. Each held recording was delivered at
both supported sizes, with H.264 on one size and VP9 on the other. Each
delivery contains one authored reversed segment, or one of five controls.
Originals, ordinary cuts to a mirrored source segment, picture holds,
and forward copies serve as controls. The last control varies picture
spacing inside the reversed segment. These are staged deliveries from
existing camera sources. A useful location requires at least 0.50 frame
intersection over union in each of the two ranges.
[Qualified source recordings](video-vcd2-unseen-controls-v287-result.md) ·
[Publisher dataset terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

| Held cases | Exact reverse check | New motion check |
| --- | ---: | ---: |
| Useful lossy reverse locations | 0/16 | 15/16 |
| All four endpoints exact | 0/16 | 4/16 |
| Matches on ordinary, hold, cut and forward-copy controls | 0/64 | 0/64 |
| Unassessed because of deliberately changed spacing | — | 16/16 |

The new check located 7/8 of the 320 × 180 deliveries and 8/8 of the
640 × 360 deliveries. A second native decode reproduced the 96 held decisions
and their measured clocks and image hashes across 19,440 pictures.
[Held measurements](evaluation-panels/video-reverse-motion-2026-11-15.json)

## App and evidence verification

Four local analyses used simulated prepaid checkout, including one positive
result in each codec, one ordinary original and one irregular-clock control.
No jobs began before simulated payment. The two positive reports displayed
**0.900–1.950 s and 5.850–6.900 s** for VP9, and **0.880–2.000 s and
5.800–6.920 s** for H.264. Both used packet `VPR04`. The evidence API and
JSON agreed. Each PDF and signed package reproduced four exact source images
and their contact sheet. Signed source replay used the actual encoded
upload. The isolated app test disabled unrelated video inference.
[Local app measurements](evaluation-panels/video-reverse-motion-app-2026-11-15.json)

## Replay

Use the source archive described in the [exact reverse test](video-reversed-sequence-2026-11-14.md#reproduce).
The first replay recounts the recorded bounds. Native replay decodes the
delivery bytes and reproduces the motion observations, timestamps and
endpoint hashes. The live-app test accepts the same retained deliveries.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.video_reverse_motion_transfer
receipt=docs/evaluation-panels/video-reverse-motion-2026-11-15.json
"$runner" -B -m "$module" review "$receipt"
"$runner" -B -m "$module" held "$run/deliveries" --archive "$archive" --keep
"$runner" -B -m "$module" replay "$run/replay" --archive "$run/deliveries" --receipt "$run/deliveries/receipt.json"
"$runner" -B -m "$module" app "$run/app" --archive "$run/deliveries" --receipt "$run/deliveries/receipt.json"
```
