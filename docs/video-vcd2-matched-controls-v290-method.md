# Matched ordinary controls for eight additional camera recordings

Use all eight camera-original recordings qualified in v287 under the pinned
VCD2 CDLA-Permissive 2.0 release. Each recording yields one unchanged control,
one chroma-conversion control, and one recompression control. These three views
share one recording group. Distinct camera types leave participant overlap
unknown. [Source qualification](video-vcd2-unseen-controls-v287-result.md) ·
[Exact v200 construction](video-vcd2-controls-v200-method.md)

The v200 renderer applies limited-range BT.709 YUV 4:2:0 and
lossless H.264 to the chroma-conversion control. The recompression control
uses the same settings with CRF 24. Both replace only source-relative
pictures in `[3,6)`. All three outputs retain the same normalized geometry,
picture clock and lossless RGB delivery codec. The renderer verifies every
output picture and requires exact equality outside the interval. Each final
control is silent. The stage qualifies detector-negative inputs and makes
no detector calls.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
run_v290() { backend/.venv/bin/python -B -m scripts.video_vcd2_matched_controls_v290 "$@"; }
run_v290 preflight
run_v290 freeze
for slot in 1 2 3 4 5 6 7 8; do
  run_v290 prepare --slot "$slot"
done
run_v290 seal
```

Preflight checks receipt and clock metadata and reports the inherited
storage estimate. The freeze and 24 prepared controls live in the private
Seagate archive. The public freeze contains aggregate commitments. The
seal verifies all output bytes and compares the recorded control clocks to
the corresponding source clock. An incomplete slot prevents a complete
receipt.
