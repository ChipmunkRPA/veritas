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

## Observed local and Linux outcomes

Two local executions recovered every native byte and authored timestamp
exactly, with identical native and renderer RGB hashes between runs. OpenCV
matches explicit BT.601; metadata/default matches explicit BT.709. The
[first local receipt](evaluation-panels/video-v15-local-native-yuv.json)
records the full commitments and actual decoder builds. Fourteen source/clock/
observation tests and seven existing replay-clock tests passed. No integrity
test requires an OpenCV mismatch.

The [actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42357968808)
completed its source-integrity checks. All147,456 native bytes, their SHA-256
and every timestamp match the authored local source exactly. Linux OpenCV
matches its default/explicit BT.709 branch, rather than its BT.601 branch.
[Completed Linux receipt](evaluation-panels/linux-video-v15-native-yuv.json).

Despite identical native YUV input, **both explicit renderer RGB hashes still
differ across hosts**. The experiment removes source RGB-to-YUV encoding as
that discrepancy's cause, but does not identify a single library/version cause
or establish pixel-identical conversion. The actual system FFmpeg versions
are7.0.2 locally and6.1.1 on Linux; bundled OpenCV FFmpeg/swscale builds differ
as previously recorded. A package-version match is insufficient evidence for
an identical rendering contract. Future cross-platform calibration and native
evidence require the actual decoder/build and input/output commitments.

Linux workflow success means native source integrity and observations were
recorded, not that the earlier failed RGB hypothesis became true, the explicit
renderer hashes agree, a detector improved or an outside reviewer participated.
No original audit/gate, threshold, tolerance, score or serving method changed.
