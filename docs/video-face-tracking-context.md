# Face-tracking context for video review ranges

New video evidence records report median eye spacing before, during, and after a face-appearance review range. Each value is the distance between the two recorded eye landmarks as a percentage of face-box width. The calculation uses the exact native frames already recorded for the corresponding time group. Users can compare those measurements with the adjacent frames to see whether head angle or face visibility changed. [Webcam comparison](video-landmark-rerank-v259-result.md)

These measurements describe tracking geometry and carry no weight in the analysis score. The six webcam controls showed why the values belong beside the evidence. Skipping the unstable ranges exposed other false flags in all six uploads. The browser, API, JSON export, signed evidence package, and PDF use the same recorded measurements. The evidence verifier recomputes them from the saved landmark inputs. Earlier evidence versions remain readable as saved.

The local integration test builds a new evidence set from decoded frames with a controlled change in eye spacing, verifies the serialized values and report rendering, and rejects an altered measurement even when the packet is rehashed. It needs no detector weights or external media.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m pytest -q -o addopts='' \
  backend/tests/test_video_nested_evidence.py::test_appearance_evidence_records_the_same_eye_spacing_in_json_and_reports
```
