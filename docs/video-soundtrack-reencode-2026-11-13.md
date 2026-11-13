# Soundtrack timing across independently encoded video copies

The existing earlier-copy check can locate a moved soundtrack excerpt when the
two files have identical decoded pictures. This trial tests whether nearby
picture evidence can preserve that measurement after separate H.264 encodes.
It measures sound timing relative to the supplied earlier copy. It makes no
determination about the speaker's original lip synchronization or AI involvement.
[Existing exact-picture comparison](video-paired-soundtrack-transfer-2026-11-12.md)

## Frozen comparison

The earlier copy and submitted copy share 63 verified pictures at 25 fps,
480 × 360 upright geometry, and the same relative decoded audio clock. The
submitted picture stream is re-encoded at CRF 31 or 34, selected by the source
group's parity. Test variants retain the same presentation timestamps. PCM and
AAC are delivery pairs for each identical operation. The exact comparator
requires pixel equality and both alternate arms use the same existing
waveform comparison and signed displacement rules. Recent studies motivate
keeping local timing separate from content consistency, and multimedia
registration work describes using sound and picture together. The comparison
uses neither paper's software or weights. [AV-SyncBench](https://arxiv.org/abs/2607.00726)
· [AV-FABLE](https://arxiv.org/abs/2610.33413)
· [Video copy registration](https://www.sciencedirect.com/science/article/pii/S1877050915018827)

The first candidate checks average absolute difference between 80 × 60 luma
frames, with 65th and 90th percentile bounds of 2.5 and 4. The second also
checks 5 × 5 cells on that grid. It retains the maximum error across cells
after a median over each five consecutive frames, requiring the maximum to
stay below a separate bound. That bound uses the largest re-encode-only
response from four development actors plus 0.35, with a floor of 1.75.
The waveform comparison requires at least four shifted 160 ms windows,
distinct matches, and supporting same-time sound on both sides. Earlier
and submitted sound excerpts and three paired source pictures retain exact
sample or picture positions and hashes.

The prior qualification rosters supplied the first four expressive actors
for development. The transfer uses the remaining twelve planned actors and
four separate Wikitongues and NASA speakers. Two expressive actors had
already declined qualification before this study. These recordings were
used in the prior single-upload repeat study. The present same-clock pairs,
carrier processing, and known local edits are distinct from that study.
The source roster and applicable license terms remain pinned in those
documents. The existing MIT-licensed face localization asset is used only
to place the edited mouth controls. It does not score any file.
[Prior source qualification](video-lipsync-repeat-fresh-actors-2026-11-13.md)
· [Face localization asset terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each source contributes short signed sound moves of ±200 ms for 640 ms and
long signed moves of ±280 ms for 880 ms. Controls cover re-encode only,
local gain and low-pass filtering, a whole-track shift, a picture-only
shift, independent speech, a wrong earlier picture, and matching short
shifts of both sound and either the whole frame or mouth in both directions.
A useful report must cover the authored edit at 0.60 temporal intersection
over union and report its signed delay within 16 samples. For each transfer
bank and carrier, advancement requires at least half of all planned useful
locations, at most one ordinary flag, zero synchronized, wrong-source or
global-shift flags, and at least 75% coverage of timing-eligible controls.

## Held results

The development point was 6.27 in both carriers. PCM and AAC produced the
same decisions. The table counts each carrier separately. The expressive
bank planned 48 edited and 132 ordinary files per carrier. Two sources
lacked a qualified interval, leaving 40 edited and 110 ordinary files.

| Bank and method | Useful locations | Edited files assessed | Ordinary flags | Timing-eligible controls assessed |
| --- | ---: | ---: | ---: | ---: |
| Expressive, exact-picture check | 0/40 | 0/40 | 0/110 | 0/50 |
| Expressive, whole-picture check | 30/40 | 40/40 | 29/110 | 50/50 |
| Expressive, local picture grid | 30/40 | 40/40 | 0/110 | 50/50 |
| Conversational, exact-picture check | 0/16 | 0/16 | 0/44 | 0/20 |
| Conversational, whole-picture check | 12/16 | 16/16 | 9/44 | 20/20 |
| Conversational, local picture grid | 8/16 | 12/16 | 0/44 | 15/20 |

All 38 whole-picture alerts per carrier came from synchronized edits.
The local picture grid withheld every synchronized, wrong-source, and
picture-only control. It also withheld all five otherwise eligible
controls from one conversation when its simple re-encode exceeded the
local difference bound. It returned no misplaced edited ranges.
[Full replay ledger](evaluation-panels/video-soundtrack-reencode-2026-11-13.json)

One conversational delivery located a 200 ms later soundtrack excerpt
at **0.880–1.480 s**, matching **0.680–1.280 s** in the earlier copy.
The source intervals were **[14080, 23680)** and **[10880, 20480)** at
16 kHz. The three corresponding frame pairs occurred at **0.840, 1.160,
and 1.480 s**. Independent native readback verified 456 original frame
witnesses and 152 PCM excerpt witnesses across the 76 selected held
deliveries in the local grid arm.

## Local app scope

Enable `VERITAS_VIDEO_REFERENCE_SOUNDTRACK_REENCODE=true` in the local API
and use **Compare with an earlier video** on a completed, authorized video
analysis. The optional check requires 63 original 480 × 360 upright H.264
8-bit 4:2:0 pictures at exactly 25 fps on a matching relative clock, and
matching continuous sound clocks starting at the first picture. It shows
the signed shift and both playable excerpts and waveforms, with the
before, during and after picture from each copy. The PDF shows the same
picture pairs and sample ranges. The JSON records exact source ticks,
original picture and sample hashes, and the picture measurements that
qualified or excluded the timing check. The exact-picture pathway
remains available for copies with pixel-identical pictures. This
comparison supplies no AI confidence.

An isolated paid test used simulated Stripe confirmation at USD 9.99
before the Full analysis started. The earlier-copy comparison recovered
the held conversational +200 ms example and withheld the matched
mouth-and-sound control. Chrome displayed all six source pictures and
two playable sound excerpts. The downloaded PDF and JSON matched the
same time ranges and evidence from the API. The comparison required no
additional purchase.

## Replay

The two existing source plans provide public original URLs, pinned source
digests and acquisition instructions. Prepare those files first in separate
folders as described in the [source replay](video-lipsync-repeat-fresh-actors-2026-11-13.md#replay).
For the expressive panel only the pinned acquisition receipt and qualified
source files are required. For the conversational panel only the four
verified original WebM files are required. Set the three variables below to
the new output path and to those two prepared folders.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_soundtrack_reencode "$phase" "$run" "$actors" "$speech" || exit 1
done
PYTHONPATH=backend python3 -B -S -m scripts.review_video_soundtrack_reencode "$run/receipt.json"
```

To create or independently re-decode an app comparison from the supplied
originals, pass `--reencoded-soundtrack` when creating the export.
The verifier derives the method from the export's recorded schema.

```sh
backend/.venv/bin/python -B -m scripts.compare_video_references "$earlier" "$submitted" "$export" --reencoded-soundtrack
backend/.venv/bin/python -B -m scripts.compare_video_references "$earlier" "$submitted" "$export" --verify
```

The native trial separately reopens selected deliveries with FFmpeg and
FFprobe to verify sample slices, picture timestamps and exact original RGB
pixels. All original and re-encoded clips remain local. The portable reader
checks the frozen split, response points, timing evidence arithmetic and
control counts using only the published scalar receipts and hashes. It does
not rerun the waveform or picture measurements.

A second native run with the updated app reproduced every development and
held row, response point and aggregate byte for byte. The public scalar
reader matched on Python 3.11 and 3.12.
