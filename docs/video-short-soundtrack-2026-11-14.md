# Short soundtrack shifts between earlier and analyzed video

A third comparison pass can trace shorter timing changes when the
earlier and analyzed videos have corresponding pictures. It compares
80 ms soundtrack windows every 20 ms after the two existing 160 ms
passes yield no candidate. A range needs at least four distinctive
windows pointing to the same shift and two same-time matches on each
side. Both clips must have a verified 720p25 four-second picture clock,
continuous sound and agreement on the existing coarse and fine picture
grids. The time shift describes the audio relative to corresponding
pictures and carries zero AI confidence weight.
[Earlier-copy timing baseline](video-soundtrack-quiet-2026-11-14.md)
· [Related audio landmark matching research](https://research.adobe.com/publication/clustering-and-synchronizing-multi-camera-video-via-landmark-cross-correlation/)

## Held comparison

Five public Wikitongues recordings supplied two four-second excerpts
each, from 23 and 38 seconds on their source timelines. The roster was
fixed before inference. The clips had no matching source entry in the
project's prior receipts checked during intake. Each excerpt has two
authored edits, moving the soundtrack 200 ms earlier or later for 320
or 480 ms while leaving the picture content in place. Separate video
encodes model normal re-export. Each pair was delivered as PCM in MOV
and AAC in MP4. Useful localization requires signed lag within one
millisecond and temporal intersection over union of at least 0.50.

Controls include unchanged audio, gain, low-pass filtering, one
whole-track delay, alternate speech, near-mute, and a repeated audio
excerpt. Two other controls move the soundtrack along with the full
picture or just the detected mouth region. Both were rendered and
decoded before checking picture correspondence. The mouth boxes use
the project's pinned MIT-licensed face detector solely to author
these controls. The timing pass itself uses waveform matching and
picture differences.
[Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

| Held carrier | Earlier passes, useful /20 | With short pass, useful /20 | Added wrong edit ranges | Control flags /90 |
| --- | ---: | ---: | ---: | ---: |
| PCM | 14 | 20 | 0 | 0 |
| AAC | 14 | 20 | 0 | 0 |

Four recordings contributed the six additional locations in each
carrier. The two carriers share edit instances. All 20 synchronized
controls per carrier were withheld by picture correspondence. The
study measures localization of staged shifts between supplied copies.
It does not measure a single-upload lip-sync detector, fully generated
video, or general performance under compression, competing voices or
other frame rates.
[Held measurements](evaluation-panels/video-short-soundtrack-2026-11-14.json)

## App and independent replay

The local trial requires `VERITAS_VIDEO_REFERENCE_SHORT_SOUNDTRACK=true`
alongside `VERITAS_VIDEO_REFERENCE_SOUNDTRACK_REENCODE=true`. A positive
comparison retains both ranges, both waveforms, matched excerpts, and
before, during and after pictures from each source. The JSON marks
which pass ran so the verifier can also replay older exports.

One held AAC clip passed through simulated USD 9.99 checkout and the
normal comparison endpoint. Zero analysis jobs ran before payment.
The analyzer located a −200 ms movement in the submitted excerpt at
**1.320–1.600 s** with source pictures **32, 36 and 40**. The PDF showed
six pictures, and native replay checked them, both waveforms and both
playable sound excerpts against the supplied originals.
[App receipt](evaluation-panels/video-short-soundtrack-app-2026-11-14.json)

The ledger records the five [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
source pages, authors, byte hashes, and authored sample clocks. Save the
original video files using the names in the ledger. The repository
contains measurements and hashes only. The arithmetic review needs
Python 3.11 or newer. The native run also needs the existing local app
dependencies and FFmpeg. A second local native run recreated the
published receipt byte for byte.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panel=docs/evaluation-panels/video-short-soundtrack-2026-11-14.json
python3 -B -S -m scripts.video_short_soundtrack_transfer review "$panel"
mkdir "$native_run"
backend/.venv/bin/python -B -m scripts.video_short_soundtrack_transfer run "$native_run/receipt.json" --bank "$source_videos"
cmp "$panel" "$native_run/receipt.json"
backend/.venv/bin/python -B -m scripts.video_short_soundtrack_app "$native_run" "$panel" "$app_run"
```
