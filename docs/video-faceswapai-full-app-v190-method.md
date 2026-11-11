# Full local analysis of a timed face swap and matched controls

The v187 worker stopped before inference because importing the preparation
module initialized application settings before the isolated serving context
could install its own. This successor loads preparation on the control
side. A model-free preflight verifies fresh settings and the exact runtime
boundary. The original worker stop and freeze remain recorded.

Use the six verified v186 renderings in frozen order. The intended edit
covers native pictures 72 through 143, or 3 seconds up to 6 seconds.
All renderings have the same 240-frame native clock and no soundtrack.
Actual changed pictures are checked against the unchanged RGB reference.
[Matched source preparation](video-faceswapai-matched-v186-method.md)

Run the same full app path in a fresh offline worker for each rendering.
The v171 one-face adapter is the only change to the inherited registry.
The appearance review retains its consecutive-frame fallback in that
same run. This is a comparison of the saved review evidence alongside
the one-face frame classifier's raw response. The old batch-size arm
was measured separately on two genuine control clips and is outside
this six-rendering experiment. No detector or fusion setting in the
customer instance changes. [Prior admission result](video-frame-full-app-v175-result.md)

Verify the entire native timestamp table, the frame-score plan's sampled positions,
all sampled RGB bytes, and the absent audio track. For the frame
classifier, report coverage and the mean and maximum raw response in
each of the before, inside and after partitions. Preserve an absent
response as missing. These raw values are neither calibrated confidence
nor a new decision cutoff. Report the full app tendency separately.
For appearance or sequence intervals, verify all native inputs and
before, during and after images against the packet and source picture
hashes. For score-based intervals, verify the support timestamps and the
three displayed images. Record the review route's status and measured
window count. Its sparse tracking positions may depend on the source hash.
Count the inclusive first through last supported native frames
for overlap. Report union IoU against actual changed frames for each
review method on swap renderings. Retain any intervals on either
control so that recompression and unchanged motion are visible.

These six renderings share one synthetic source. The four color
interpretations are a fixed sensitivity check. They count as one
independent source, with no genuine-camera or lip-sync result. The
known substitution interval labels the edit in time. It supplies no
ground truth for affected face pixels inside the frame.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_full_app_v190 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_full_app_v190 preflight
for condition in unchanged recompression_only swap_tv_bt601 swap_pc_bt601 swap_tv_bt709 swap_pc_bt709
do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_full_app_v190 run "$condition"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_full_app_v190 readout
```

The existing v108 guard applies its 6 GiB initial RAM and sampled
process-tree RSS bounds, 900-second deadline, 3 GiB initial disk minimum
and 1.5 GiB running reserves. Each next case requires the preceding
completed guard receipt. A pre-launch admission denial leaves the stage
unattempted. Preserve any actual failure and its receipt. The app
database, source hashes, face crops and individual frame scores stay
local. Publish only the predeclared aggregate response and measured
interval timestamps.
