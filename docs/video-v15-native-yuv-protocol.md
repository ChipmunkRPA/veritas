# Fixed native colour samples: separate decoder comparison

The completed [v14 RGB experiment](video-v14-colour-rendering-audit.md) did not
reproduce its Mac-specific hypothesis on Linux. Preserve its source, failed
workflow and negative receipt. Separately encoding an identical RGB chart left
RGB-to-YUV conversion as a confound, since even explicit-renderer RGB hashes
differed across hosts. This new source experiment removes that conversion.

Generate12 self-authored128x64 planar YUV420P frames at8fps directly as integer
bytes. Eight fixed legal-range luma/chroma bars move by two luma samples per
frame, with corresponding chroma displacement. Encode losslessly as FFV1 with
explicit limited-range BT.709 tags, then recover **all147,456 native bytes**
exactly. Require exact dimensions, format, frame count, metadata and every
authored clock before comparing one fixed frame3 in OpenCV with the existing
single-decode FFmpeg default/explicit BT.601/explicit BT.709 branches.
Native input SHA-256:
`2025fc4531fbd91e0148b7b26b17181157c0e80ede8b1361d9c5a561f9a02cf4`.

The source-integrity check may fail. Once valid, renderer matches or differences
are recorded observations, not conditions for making this workflow green.
This is a separately specified experiment, not a replacement or relaxed version
of the old hypothesis. Compare native commitments and each renderer's RGB hash
between actual hosts before claiming pixel identity. If identical native YUV
still gives different explicit rendering, retain that discrepancy. Matching
explicit outputs but different defaults narrows the mismatch to the decoder/
colour-conversion path; it does not identify one causal library change.

Use the same two pinned array/decoder packages and Python3.12.6 in the public
Linux workflow. Record bundled codec/swscale builds and system FFmpeg version
and binary hash. Container hashes may differ; native sample identity may not.
No research/customer media, neural weights, neural forward, classifier score,
recalibration or original audit result is involved. This does not validate
real-clip detection, fix the rejected candidate's four native RGB failures,
provide human outside review, or alter the application or GCP infrastructure.

```sh
PYTHONPATH=backend python -m scripts.reproduce_video_yuv_v15 NEW_RECEIPT.json
```

## Observed local outcome; Linux pending

Two local executions recovered every native byte and authored timestamp
exactly, with identical native and renderer RGB hashes between runs. OpenCV
matches explicit BT.601; metadata/default matches explicit BT.709. The
[first local receipt](evaluation-panels/video-v15-local-native-yuv.json)
records the full commitments and actual decoder builds. Fourteen source/clock/
observation tests and seven existing replay-clock tests passed. No integrity
test requires an OpenCV mismatch. The separate Linux execution is pending;
cross-platform renderer identity has not yet been established.
