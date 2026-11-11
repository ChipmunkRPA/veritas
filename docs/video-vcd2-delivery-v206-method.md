# VCD2 upload delivery controls

The raw RGB control for v200 slot 02 produced three files between 917 and
958 MiB. The normal video upload limit is 500 MiB. v206 prepares a fixed
upload delivery derivative for every role in each of the 12 recording slots.
v202 remains a separate study of the original v200 RGB controls, including
raw files that exceeded its upload limit. The delivery study addresses
false alarms on silent camera footage after one common delivery encode.
Its 36 planned files form 12 paired recording components. [v200 raw control
method](video-vcd2-controls-v200-method.md) · [v202 full-app
method](video-vcd2-full-app-v202-method.md)

## Fixed delivery encode

Every input uses the same software encoder, limited-range BT.709 YUV
4:2:0 conversion and x264 fixed quantizer 18. Every output picture is an
I-picture with a closed one-picture GOP. B-frames, rate-control lookahead,
adaptive quantization and macroblock-tree rate allocation are disabled.
Conversion and encoding run with one thread each. The file remains silent
and preserves width, height, sample aspect ratio, all presentation timestamps,
and the complete packet clock, including the final sample duration. The
BT.709 setting selects a YUV matrix. No color primaries or transfer function
are inferred.

The method checks the v200 file and receipt hashes and production-decoded
RGB for every input picture. It checks every output packet's PTS, DTS, and
duration, then checks the production frame table for exact native timestamps
and all-intra coding. It records every output picture's decoded RGB SHA-256.
The three delivered roles must have identical decoded pixels outside the
v200 source-relative ordinary-edit interval `[3,6)`. Any previously equal
same-index pictures must remain equal after delivery. All frame comparisons
use native dimensions and the actual variable picture clock.

The unchanged derivative is the delivery baseline. Differences from it
record the ordinary intervention that survives delivery. Each role also
records RGB mean absolute error against its own v200 input on every changed
picture, including changes caused by the shared delivery encode outside
`[3,6)`. A collapsed contrast is recorded explicitly when the encode
removes all differences from a particular comparison. A case still above
500 MiB is retained as `unsupported_upload_size` and is ineligible for a
normal-upload trial. The method does not lower quality or select a different
codec based on file size. A paired app comparison requires both input
roles to be eligible and the relevant contrast to remain present.

## Private receipt

The sealed input roster retains all 12 slots, v200 attempt status and the
selected receipt and case hashes. It includes the exact v200 published
freeze, the FFmpeg binary commitments and the v206 code and encode settings.
Successful derivatives have a new per-slot `qualification.json` with the
v206 freeze hash, v200 qualification hash, source and per-role input hashes,
native geometry, source-relative picture timestamps, final duration and
ordinary frame interval. Each role records `file`, `sha256`, `bytes`,
`status`, `rgb_sha256` and `verified_frames`. The paired difference list is
`changed_frames`. The role-to-own-input list is
`delivery_encode_changed_frames`. Both contain `index`, `pts` and
`rgb_mae`. The receipt retains `suppressed_pair_change_indices` per role
and `lossy_codec_changed_indices` and
`suppressed_lossy_codec_change_indices` for the recompression-versus-chroma
comparison.

Raw and derived media, exact hashes, and frame receipts stay on the private
Seagate archive. Any later aggregate should preserve ineligible slots and
distinguish the shared delivery distortion from the ordinary interventions.
This preparation makes zero model calls.

## Local preparation

The preflight can report pending v200 slots. Freeze after v200 preparation
finishes for the whole roster. If a qualified v200 source could not be
prepared, explicitly close its slot with `--unavailable N` before freezing.
The initial preparation writes
`video-vcd2-delivery-v206/slot-01-attempt-01/` under the private Seagate
archive. Use a new `--attempt` after a failed preparation.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_v206 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_v206 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_v206 prepare --slot 1
```

The preflight includes the v200 RGB-equivalent storage estimate as a
planning ceiling for the derived media. Preparation checks the equivalent
size of its three uncompressed RGB sequences and the existing 3 GiB
reserve before writing. Measured compressed sizes determine upload
eligibility.
