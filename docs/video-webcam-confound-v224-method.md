# Measure visible changes around a webcam appearance flag

Use the exact three archived analyses of the first matched webcam control.
Bind the v202 case plan, sealed audit and application records, and the
native picture clock and RGB commitments of each v200 source. Read the
checkpointed analysis database in memory. Verify that the saved appearance
window corresponds to the evidence packet in that analysis.
[Webcam triplet](video-vcd2-full-app-v202-first-triplet.md) · [Measurement method](video-face-confound-v222-method.md)

Decode only the selected before, during and after frames. Verify each
decoded picture hash and source timestamp against the archived window and
the source qualification. Apply the geometry and face/context color
measurements to the bound landmarks and boxes. Report missing support,
overlapping windows and mismatched records as unavailable. Compare the
three arms as frame-matched only when their selected source indices agree.

These measurements describe motion and color in one consumed webcam
recording. The non-face region uses the remaining frame as context and
can include moving objects. The readout keeps the face flag and its source
coordinates authoritative. Publish summary measurements and commitments.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_confound_v224 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_webcam_confound_v224 readout
```
