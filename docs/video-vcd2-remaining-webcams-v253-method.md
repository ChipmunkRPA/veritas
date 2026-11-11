# Complete the two remaining webcam control groups with the verified cache fix

The [v202 study](evaluation-panels/video-vcd2-full-app-eight-triplets-v202.json) sealed 24 runs from eight genuine webcam recordings. [The v251 replay](video-vcd2-verify-cache-replay-v251-result.md) reproduced the exact formerly unavailable upload and verified both video routes after the cache fix. Two v202 source groups still have six prepared, unattempted uploads within the app's size limit. This successor evaluates those exact controls as two matched triplets. Each has an unchanged upload, a chroma conversion and a recompression. The applied edits are ordinary video processing. A face-swap review interval on any of these inputs is a false flag for this evaluation.

Bind the original v202 freeze, its full private plan and the commitment over the 24 historical seals. Verify the successful v251 replay and use its exact application source, installed model inventory, decoder binaries, resource guard and serving settings. Confirm the six original upload bytes and per-frame qualifications before freezing. Keep the new case workspaces and seals separate from the historical results.

Use the existing full-app worker and read-only audit. The audit checks the customer direction, route outcomes, sampled and rendered frame positions against the source picture clock, and any screenshot, crop or context video bound to a finding. Seal the evidence before releasing a workspace. Report audited coverage and measured appearance windows alongside the direction and face-review flag counts. The paired transition counts require measured appearance windows on both arms. Keep failed, unavailable and unmeasured outcomes visible. The public aggregate contains counts and whole-artifact commitments. The recordings, per-item hashes, exact evidence and guard logs stay private.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 preflight
backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 freeze
for slot in 11 12; do
  for role in unchanged chroma_conversion_only recompression_only; do
    case_name="slot-$slot-$role"
    backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 run "$case_name"
    backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 review "$case_name"
    backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 seal "$case_name"
    backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 release "$case_name"
  done
done
backend/.venv/bin/python -B -m scripts.video_vcd2_remaining_webcams_v253 readout
```

Treat the two new recordings as matched camera controls. The result measures false flags, coverage and operational reliability for these inputs. Estimating sensitivity to swaps or lip-sync edits requires a separate positive cohort.
