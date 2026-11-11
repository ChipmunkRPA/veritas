# Supply the archived frame dimensions to evidence validation

The SQLite continuation opened the archived database and located the exact
appearance record. Evidence validation then stopped because its input
omitted the source width and height. This continuation passes the dimensions
from the verified native frame table. It checks the same frame count,
timestamps, dimensions, rotation and RGB commitments before measuring.

The earlier unavailable result, source archives and selected frames stay
bound to the comparison. The geometry and color calculation still uses the
same v222 method. Readout reports measurement availability for the original
three controls and exact frame match between each paired arm.
[Measurement method](video-face-confound-v222-method.md) · [Earlier readout](evaluation-panels/video-webcam-confound-unavailable-v227.json)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_source_geometry_v231 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_source_geometry_v231 readout
```
