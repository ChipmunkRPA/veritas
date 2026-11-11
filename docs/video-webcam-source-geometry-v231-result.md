# Visible face changes coincided with an unchanged-webcam flag

The local replay verified the same 45 source frames in each of the three
webcam versions, including six before, 33 during and six after the flagged
interval. It matched the archived source clocks, frame dimensions and
RGB commitments. The two earlier unavailable readouts are retained.
[Method](video-webcam-source-geometry-v231-method.md) · [Verified measurements](evaluation-panels/video-webcam-source-geometry-complete-v231.json)

On the unchanged clip, the face box's median luma rose by 12.84 code values
from before the flag to during it. The rest of the frame fell by 1.56.
The normalized landmark change was 0.095 inter-eye distances from before
to during, compared with 0.022 from before to after. The two ordinary
codec treatments retained the same 12.84 luma increase and similar
landmark changes on the same selected frames.

The footage is a known unchanged control. This example shows how a real
change in facial appearance can accompany a false review interval. The
face box and background measurements describe the selected frames, and
the background is unregistered. A broader paired comparison is needed to
measure how often these patterns occur in camera controls and known swaps.
[Full-app webcam result](video-vcd2-full-app-v202-first-triplet.md)
