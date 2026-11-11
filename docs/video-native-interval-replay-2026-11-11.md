# Native video interval replay and wider-window comparison

An offline verifier now checks whether three evidence pictures come from the stated source video and fall before, during and after a specified interval. It hashes the source bytes, probes the primary stream's full presentation clock, decodes the committed picture indices, and checks the native RGB digests. Optional PNG exhibits must each contain a single still image and match the corresponding decoded pixels. The interval includes its start and excludes its end. A picture at the first restored timestamp belongs to the after context.

## Run the verifier

Use Python 3.11 or later and FFmpeg with FFprobe on your path. The receipt and source are supplied separately. The optional expected receipt digest can be recorded when the evidence is created. The command exits with status 1 and a reason when a commitment fails.

```sh
python3 backend/scripts/verify_native_video_interval.py source.mp4 interval.json \
  --expect-receipt-sha256 "$EXPECTED_RECEIPT_SHA256" --exhibits ./native-exhibits
```

The receipt uses schema `chipmunk.native-video-interval/1` with these fields.

| Field | Meaning |
| --- | --- |
| `source` | `stream` is `v:0`. Record `sha256`, `bytes`, `width`, `height`, `time_base`, `frames` and `pts_sha256`. The clock digest hashes the full absolute integer PTS array serialized as ASCII JSON with compact separators. |
| `interval_pts` | Two integer ticks on that clock. The interval is half-open. |
| `anchors` | Three ordered records for `before`, `during` and `after`. Each has a `role`, zero-based picture `index`, absolute `pts` and `rgb_sha256` of native packed RGB24. |

The optional exhibit directory contains `before.png`, `during.png` and `after.png`. The output expresses times relative to the first displayed source picture as exact fractions. This replay covers unrotated sources with explicit increasing PTS. An RGB mismatch is a verification failure, including when decoder builds differ. The result confirms correspondence for three sampled pictures. The source or signer must be authenticated separately, and the replay supplies no judgment about the content's origin or the detector's accuracy.

## Local verification

Nine files passed. The genuine panel was two VCD2 recordings with matched unchanged, chroma-converted and recompressed H.264 deliveries, outside the camera source groups used in the earlier short-window study. A stream copy of one genuine delivery shifted every PTS by 7.125 seconds while preserving the verified relative anchors. The other two files were exact licensed publisher-derived inserts. Across the panel the verifier checked 27 native RGB anchors and three archived PNG exhibits. [VCD2 data license](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Matched delivery construction](video-vcd2-delivery-v206-method.md) · [Publisher insert construction](video-faceswapai-matched-v186-method.md)

The negative checks rejected a different source, a wrong receipt digest, a real picture assigned to the wrong interval role, a wrong RGB digest, missing context, and an exhibit from the wrong native picture, and an animated PNG presented as a still. [Aggregate and frozen commitments](evaluation-panels/video-native-interval-replay-2026-11-11.json)

## Retrospective duration comparison

The prior three-point core missed a 1.50-second swap in the separate synthetic photo sequence because its required flanks fell inside the edit. The wider-window replay tested 3, 5, 7, 9 and 11 core positions on the same 17 archived quarter-second score series. It held the paired review point, contrast, tracking checks and two measured flanks per side fixed. All material was already observed, and no new model inference was run. [Original short-window study](video-face-short-window-2026-11-11.md)

| Input | Three-point core | Variable duration | Current appearance |
| --- | ---: | ---: | ---: |
| Two insert durations from the separate synthetic photo source | 1/2 | 2/2 | 2/2 |
| Four renderings of the earlier publisher video swap | 0/4 | 0/4 | 4/4 |
| Synthetic original and ordinary controls flagged | 0/5 | 0/5 | 2/5 |
| Genuine camera control deliveries flagged | 0/6 | 0/6 | 0/6 |

The wider rule recovered corroborating samples at **2.500–3.750 seconds** inside the 1.50-second edit. Its before, during and after pictures at **2.250, 2.500 and 4.250 seconds** passed the native verifier. Widening the core left the four video versions missed. The candidate remains experimental. A moving face-swap comparison needs separate licensed paired footage with measured boundaries. FakeParts retains original terms for third-party videos, and HQSwap requires separate commercial-use coordination. [FakeParts data terms](https://huggingface.co/datasets/hi-paris/FakeParts#data-and-licence-notes) · [HQSwap data terms](https://github.com/dasec/HDA-HQSwap/blob/main/README.md)
