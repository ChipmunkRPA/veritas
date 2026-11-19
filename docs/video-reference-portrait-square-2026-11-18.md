# Portrait and square earlier-copy video comparison

The local-area earlier-copy check can assess matching upright 360 × 640 portrait
and 480 × 480 square H.264 4:2:0 pairs at 30 fps, with 65–91 pictures.
It reports a visual difference, a source-time interval, and earlier/analyzed
frames and crops from before, during and after. Conventional edits can produce
this evidence. Its overall AI-confidence weight is zero. The inherited robust
brightness fit, local response point and surrounding-frame guards were frozen.

In a completed local analysis, choose **Compare with an earlier video**.
The development worker needs `VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED=true`
and `VERITAS_VIDEO_REFERENCE_SHORT30_LOCAL_AREA=true`.

## Held-format evaluation

Eight CREMA-D source recordings supplied both profiles. Each source/profile
had three authored edits (large and small local tone changes, neighboring
texture copy), three ordinary controls (identical bytes, recompression, global
tone), and a different-source control: 112 pairs in total. Pictures were
center-cropped and resized before encoding. Edits were applied before delivery
encoding, with fixed 12–20-picture boundaries. A location requires temporal
IoU ≥ 0.50 and middle-picture overlap with the authored changed-pixel mask
≥ 0.10. This spatial criterion permits a coarse review area.

| Outcome | Portrait | Square |
| --- | ---: | ---: |
| Authored edits flagged | 24/24 | 24/24 |
| Edits meeting both location criteria | 23/24 | 24/24 |
| Large tone / small tone / texture locations | 8/8, 7/8, 8/8 | 8/8, 8/8, 8/8 |
| Ordinary controls flagged | 0/24 | 1/24 |
| Wrong-source pairs admitted | 0/8 | 0/8 |

Both profiles pass the frozen gate: at least 16 edit locations, at least three
per edit role, at most two ordinary flags, no wrong-source admissions, and
recorded replay of each finding's six native pictures and six crops. The
portrait small-tone `camera-06` finding misses both overlap criteria
(temporal 0.40 and spatial about 0.073). The square global-tone `camera-06` control
produces the ordinary false flag.

The historical local-area route was outside scope for all 112 pairs. The
results measure added format coverage for staged paired edits. The two
profiles and seven treatments repeat the same eight source and actor groups.
Sources had earlier project exposure in audio and video timing studies.
Selection excluded exact source names and hashes from inspected earlier
reference ledgers. Single-file detection and other pairing profiles require
separate evaluation.

CREMA-D supplies the recordings under DbCL 1.0 and its database under ODbL 1.0.
The receipt includes each original publisher URL and source SHA-256. [Publisher](https://github.com/CheyneyComputerScience/CREMA-D)
· [Pinned rights statement](https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt).

## Local app and independent checks

An isolated Chromium flow exercised two edit comparisons and two ordinary
recompressions. Four initial analyses used simulated payment. The comparisons
created zero additional analysis jobs, and the baseline analysis JSON remained
byte-identical. Browser PDF and JSON downloads matched API exports. Portrait
tone showed [13,25) at 13/30–5/6 seconds. Square texture showed [12,24) at
2/5–4/5 seconds. Both displayed before, during and after evidence and six verified
native crops. Both ordinary recompressions had no local range.

The independent standard-library recount passed all 112 rows and rejected nine
altered-record controls. With retained MP4s, its separate ffmpeg replay verified
128 input-file hashes, all 49 flagged pairs, 294 native picture commitments and
294 native crop commitments. The portable recount recomputes temporal overlap,
validates source and anchor geometry, and recounts the recorded spatial
overlap. A separate run rebuilt all 112 pairs from the pinned originals,
reproduced the encoded file hashes, and recomputed both overlap measures
from the authored pre-encoding masks. The exact replay used FFmpeg 7.0.2,
OpenCV 5.0.0, and NumPy 2.2.6.

```sh
python3 -I -S backend/scripts/review_video_reference_portrait_square.py
python3 -I -S backend/scripts/review_video_reference_portrait_square.py \
  --deliveries /path/to/retained/deliveries

PYTHONPATH=backend backend/.venv/bin/python -B \
  backend/scripts/replay_video_reference_portrait_square.py \
  docs/evaluation-panels/video-reference-portrait-square-2026-11-18.json.gz \
  /path/to/original/publisher/files /path/to/new/replay/output
```

The compressed [portable receipt](evaluation-panels/video-reference-portrait-square-2026-11-18.json.gz)
preserves frozen runner and implementation hashes. The reconstruction uses the
application's local dependencies and FFmpeg. The default recount uses Python's
standard library and makes no network calls.

## Earlier-copy comparison on short landscape 720p clips

Enable `VERITAS_VIDEO_REFERENCE_SHORT720_LOCAL_AREA=true` with the existing
earlier-video comparison in a development instance. This mode checks two
upright 1280 × 720 H.264 AVC1 MP4 files with 66–78 pictures and matching,
zero-origin picture instants at exactly 30 or 30,000/1,001 fps. Each file can
use a 30 kHz or 90 kHz time base. Matching playback times do not require equal
native tick values. A separate MP4 timing-table check verifies the full
picture duration and edit-list timing for each original.

Six distinct retained camera recordings supplied 48 new comparisons. Three
sources were native 720p. Two 1080p captures were resized down and one
960 × 544 capture was resized up. The original pictures stayed in the same
order. Half the deliveries used an exact 29.97 fps timeline. A fixed masked
local tone edit and a separate neighboring-texture replacement supplied the
known changes. All deliveries used H.264 software encoding, including direct
and second-encode controls. [VCD2 source terms](video-vcd2-unseen-controls-v287-method.md)
· [Wikitongues donor terms](https://wikitongues.org/wp-content/uploads/2024/09/Wikitongues-Recording-an-Oral-History-Sep-2024.pdf)

| Held outcome | Previous localized comparison | With short 720p comparison |
| --- | ---: | ---: |
| Edits meeting both location criteria | 0/12 | 10/12 |
| Exact authored start and end | 0/12 | 5/12 |
| Local tone / texture edits located | 0/6, 0/6 | 5/6, 5/6 |
| Ordinary pairs with a local range | 0/24 | 0/24 |
| Different recordings with a local range | 0/6 | 0/6 |
| Different picture clocks excluded | 6/6 | 6/6 |

The same pre-existing picture score and thresholds were used. The match
criteria were 0.50 temporal overlap and 0.10 changed-pixel overlap at the
selected middle frame. All ten reported ranges met both criteria. Both
misses came from one 29.97 fps source with insufficient background picture
support. Three ordinary pairs from that source were also unassessed. A native
replay checked all 60 reported frame/crop anchors, and a separate FFmpeg
replay independently checked one pair at each frame rate. Their ranges were
2/3–14/15 seconds and 7,007/7,500–1,001/750 seconds. The study measures paired
picture changes in authored fixtures. The comparison adds zero AI confidence.

Some MP4 writers use a 1 kHz movie clock. It can represent the duration of the
tested 66-picture 30 fps clips exactly. It rounds the duration of the tested
78-picture 29.97 fps clips. A byte-preserving remux confirmed that the check
withholds the local comparison on those rounded-duration files even when
their H.264 packets and picture timestamps match.

Three isolated, simulated USD $9.99 payments completed the original-video
analyses. Six subsequent comparisons preserved the underlying results and
created no extra jobs or charges. Five Chromium uploads exercised both rates,
ordinary recompression, an unrelated copy and a different cadence. The JSON
and PDF downloads matched the comparison API responses. All three original
analysis archives reverified against their uploads. [Anonymous held and app
verification receipt](evaluation-panels/video-reference-720p-exact-2026-11-19.json)
