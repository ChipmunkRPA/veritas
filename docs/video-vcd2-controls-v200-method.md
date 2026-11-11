# VCD2 paired webcam controls

Each qualified Microsoft VCD2 recording supplies three genuine-camera
controls. The three copies count as one recording component. The
published data has no subject key. The 838 recordings came from 799
participants, so the study cannot establish independent subjects. The
camera-only controls measure false alarms for the face-change check.
[VCD2 source roster and rights](video-vcd2-genuine-v195-method.md) ·
[Source qualification](video-vcd2-genuine-v197-method.md) ·
[Microsoft dataset](https://github.com/microsoft/VCD#vcd2--camera-native-talking-head-dataset)

The private v200 freeze retains all 12 planned slots and the qualification
attempt history. A successful slot uses the first qualified attempt.
Subsequent successful attempts must identify the same source bytes. An
unavailable slot needs explicit closure after acquisition has finished.
The script verifies the original v197 freeze and the archived source,
acquisition, and qualification hashes. Each construction attempt has a
new directory on Seagate.

## Picture and codec controls

| File | Pictures in the source-relative interval `[3,6)` |
| --- | --- |
| `unchanged.mp4` | Normalized source RGB |
| `chroma_conversion_only.mp4` | Identical source through limited-range BT.709 YUV 4:2:0 and lossless H.264 QP 0, then returned to RGB |
| `recompression_only.mp4` | Identical conversion and H.264 settings, with CRF 24 instead of QP 0 |

The intermediate encodes each start from the full source sequence to
retain the preceding prediction context. Only the decoded replacement
pictures inside the interval enter each final control. Every final file
uses the same silent full-range RGB H.264 QP 0 settings, native dimensions
and sample aspect ratio. The script reuses FFmpeg for deterministic color
conversion and verifies its output independently with the production RGB
decoder. Declared range and supported matrix tags take precedence. The
untagged HD YUV default is limited range and BT.709. Untagged JPEG-range
pixel formats use full range. Unspecified sample aspect ratio is treated
as square. The receipt identifies each assumption. A range or matrix
choice can affect absolute RGB values even though the paired comparisons
use the same source conversion.

The script preserves every observed relative presentation timestamp,
including alternating 33 and 34 ms intervals on the first qualified
recording. It subtracts the first source PTS once and retains that source
offset in the receipt. Interior MP4 sample durations follow the actual
next PTS. The final sample uses the qualified last packet duration or the
observed last PTS step if the source omits that duration. All packet PTS,
DTS and durations must match the expected final clock. The production
decoder independently verifies picture count, geometry, and PTS.

For each control, the script hashes every decoded RGB picture and verifies
the intended source or intermediate picture at that position. Changed
pictures must lie inside `[3,6)`. Per-frame mean absolute RGB error
describes the actual changes. The receipt also records which pictures
differ between the lossless chroma conversion and the CRF 24 control.
An intermediate can be discarded after the corresponding final file
passes its clock and picture verification.

## Local commands and storage

Run from the repository root after v197 acquisition finishes. Preflight
reads receipts and source metadata. Freeze and preparation write to
`/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/video-vcd2-controls-v200/`.
Prepare one slot at a time. Repeat with the next slot number. Use an
unused `--attempt` if a preparation failed. If an acquisition slot is
conclusively unavailable, include its number after `--unavailable` in
the preflight and freeze. Source files and exact frame receipts stay on
Seagate.

```sh
metadata=/private/tmp/chipmunk-vcd2.014eoQ
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_controls_v200 preflight --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_controls_v200 freeze --metadata-dir "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_controls_v200 prepare --metadata-dir "$metadata" --slot 1
```

Preflight reports the estimated RGB-equivalent storage as three times
the sum of width × height × frame count × 3 bytes. It also reports the
largest slot's workspace estimate, allowing one additional full RGB
equivalent for an intermediate. The published 3,386 frames give an
upper planning estimate of about 59 GiB of RGB-equivalent final pictures
if every picture is counted at 1080p. Actual H.264 output sizes depend
on the source and will be measured after construction. Preparation also
requires the existing 3 GiB Seagate reserve. These figures exclude
source files already stored by v197 and any intentionally preserved
failed construction outputs.
