# Video source rendering: explaining four failed RGB comparisons

7 November 2026. A post-hoc source-only audit explains the first RGB mismatch in
each of the four failed v13 native cases. The experimental OpenCV path produced
BT.601-compatible RGB, while FFmpeg's metadata-aware default produced BT.709 RGB.
All four clips declare BT.709 limited-range video. This is a rendering-path
difference, not evidence that their content was edited by AI.

The original [candidate quality rejection](video-demamba-v13-result.md), native
failure receipt and every score remain unchanged. This supplement runs no
neural inference and does not convert the failed original native requirement
into a pass or qualify the candidate for fresh confirmation.

## Evidence from the actual failed sources

The four cases are the Pexels010/074 originals and their matched versions from
the consumed DeepAction panel. The first selected frame in each reproduces its
old OpenCV hash and the separate FFmpeg default hash exactly. In a single decoded
FFmpeg graph, the default branch matches explicit BT.709. Changing only the
conversion to explicit limited-range BT.601 matches the old OpenCV RGB exactly.
Before that change, channel differences reached6–18 levels, with mean absolute
differences2.04–2.42. No output was accepted with a pixel tolerance.

[First-frame matrix experiment](evaluation-panels/video-v14-colour-matrix-diagnostic.json);
[earlier independent RGB/BGR diagnosis](evaluation-panels/video-v14-first-frame-rgb-diagnostic.json).

The follow-up covers **all32 originally selected frames**, eight per case.
Every original RGB hash, crop box, resized RGB hash, normalized float32 tensor
hash and complete eight-frame batch commitment matches exactly under the
explicit OpenCV-compatible BT.601 rendering. Source files remain byte-identical
to the original candidate journal. Stream and selected-frame metadata both
declare BT.709/tv. Separate source/filter integer PTS and rational time bases
agree, and all32 original recorded-second differences are exactly zero under
the unchanged1.1microsecond limit.
[Complete renderer supplement](evaluation-panels/video-v14-source-renderer-supplement.json).

This establishes what pixels the old candidate received and how to reproduce
them. It does not establish that BT.601 is the intended colour rendering of
these BT.709 sources. The four model outputs were not repeated by this source
supplement; their original neural audit remains incomplete. The other twelve
native cases and all six failed numerical quality requirements remain as
recorded. No original journal or native receipt is rewritten.

## Why the installed paths differ

The local OpenCV5.0.0 wheel reports FFmpeg avcodec61.19.101 and swscale8.3.100.
The [OpenCV5.0.0 conversion implementation](https://github.com/opencv/opencv/blob/5.0.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L1927)
uses its older initialized-context branch for that swscale version. The
[matching FFmpeg7.1 defaults](https://github.com/FFmpeg/FFmpeg/blob/n7.1/libswscale/swscale.h#L122)
identify the default coefficient table with BT.601, and its
[context initialization](https://github.com/FFmpeg/FFmpeg/blob/n7.1/libswscale/utils.c#L1337)
sets default tables when no table has been selected. This source inspection is
consistent with the observed conversion; the byte-exact controlled experiment
is the direct evidence for these clips and installed binaries. No global claim
about every OpenCV/FFmpeg build, codec or colour configuration follows.

The staged video-serving path and the v13 baseline already use the FFmpeg
decoder in `app/detectors/video/decode.py`. A separate legacy sampler in
`app/detectors/video/frames.py` still uses OpenCV. This research changes neither
path. Future comparisons must name the renderer and colour assumptions
explicitly, distinguish displayed source rendering from exact scored input,
and retain source clocks, crop geometry and complete tensor commitments.

## Public reproduction without models

The synthetic reproduction creates its own twelve-frame colour chart, explicitly
encodes BT.709 limited-range YUV and checks the two conversion paths. It needs
FFmpeg/ffprobe, NumPy and OpenCV, with no app, source clips, customer files or
neural weights. Two local runs produced identical source RGB and decoded RGB
hashes; generated Matroska file hashes differed. Container hashes are recorded
per run, not incorrectly required to reproduce across newly generated files.

```sh
python -m pip install -r backend/requirements-video-color-review.txt
PYTHONPATH=backend python -m scripts.reproduce_video_color_v14 /tmp/new-colour-review.json
```

[Local synthetic receipt](evaluation-panels/video-v14-local-synthetic-colour.json).
The public workflow adds an actual isolated Linux experiment. Its first
[attempt](https://github.com/ChipmunkRPA/veritas/actions/runs/42356352972)
stopped when the FFmpeg availability check failed; the
experiment never ran and no hypothesis outcome was produced. The workflow now
installs that missing system dependency when needed, with source experiment,
library pins and exact conditions unchanged.

The [next Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42356642465)
completed the experiment and **did not reproduce the Mac-specific hypothesis**.
Its OpenCV5.0.0 wheel bundles swscale9.5.101/avcodec62.28.101, versus the local
swscale8.3.100/avcodec61.19.101. On its generated source, OpenCV matches the
default/explicit BT.709 rendering, not BT.601. The workflow remains failed under
its original conditions; no result or requirement was revised to obtain a pass.
[Actual completed negative receipt](evaluation-panels/linux-video-v14-colour-experiment.json).

Both hosts used identical synthetic RGB input, but generated their encoded
sources separately. Explicit renderer RGB hashes differ across hosts too, so
this experiment did not isolate a single library/platform cause: native YUV
equivalence was not measured. A separate follow-up should feed fixed native
YUV samples, verify lossless sample recovery and record each renderer before
claiming cross-platform pixel identity. Do not change the completed experiment
or call the local result universal. Native decoder behaviour requires its own
recorded build and source-input commitments before cloud results can inherit
local calibration or evidence claims.

Synthetic decoder reproduction is not real-clip or detector validation, and it
is not participation by an outside human reviewer.

For the real-source supplement, first obtain the permitted original clips and
extract the unchanged `candidate.jsonl` member from the public
[lossless v13 packet](evaluation-panels/video-v13-development-review.json.gz).
No model files or private full execution lock are needed:

```sh
PYTHONPATH=backend python -m scripts.replay_video_color_v14 \
  docs/evaluation-panels/video-v13-demamba-freeze.json \
  /path/to/candidate.jsonl /path/to/original-video-v2-source-root \
  /tmp/new-renderer-supplement.json
```

The initial local preflight rejected a public-list/private-dictionary schema
assumption before any replay output; that log is retained. Corrected public
schema handling preceded the complete32-frame run. Seven clock tests pass,
including a synthetic display-rounding discrepancy that requires integer PTS,
equivalent tick bases, actual time mismatches, and missing/reordered/duplicate
outputs. This does not imply display rounding caused the real four failures.
No raw clips, pixels, model names in customer reports, serving calibration,
billing, overall confidence or GCP resources are changed by this work.
