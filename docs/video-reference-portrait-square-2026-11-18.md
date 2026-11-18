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
