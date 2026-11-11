# One-face batching on the current camera panel

The reference app has a 4,608 MiB model-manager budget. Its existing video
frame adapter declares 6,400 MiB of activation allowance and can skip the
frame check despite successful decoding and face detection. Earlier local
comparisons used one face per forward with the same ensemble and an 800 MiB
activation allowance. They verified identical input timing and restored frame
scores on two genuine clips. [Nairobi control](video-frame-full-app-v171-result.md) ·
[USGS control](video-frame-full-app-v175-result.md)

Freeze all 24 v294 controls before candidate inference. Keep the current app
source, model assets, fusion settings, appearance check and existing runtime
limits. The isolated manifest substitutes the existing one-face adapter for the
video frame adapter. The worker takes one case at a time after the reference
campaign has completed. Preserve budget skips, genuine-file directions,
failures, frame counts and face-review intervals. A successful restored frame
check must score every detected-face sample, link to the committed raw output
and signal, and publish the exact frame series. Compare per-case native input,
condition profile, fuser identity and all other detector statuses. Keep raw
videos, face crops and per-case scores in the private archive.

The 24 ordinary controls share eight source groups. They measure frame check
availability, evidence integrity and false flags. Face-swap sensitivity needs
separate rights-qualified positives, including a distinct source and generator.
The earlier paired swap exposed low raw scores in the known changed interval,
so more frame coverage alone cannot establish that the route improves the
file-level verdict. [Prior swap diagnostic](video-faceswapai-full-app-v190-result.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
v302() { backend/.venv/bin/python -B -m scripts.video_vcd2_batch_one_v302 "$@"; }
v302 preflight
v302 freeze
v302 run slot-01-unchanged
v302 review slot-01-unchanged
v302 seal slot-01-unchanged
v302 release slot-01-unchanged
v302 readout
```

Repeat the four guarded case steps for the full predeclared roster and compare
both arms before considering default promotion. The reader binds the complete
reference seal set and verifies the bytes of each compared saved case record.
