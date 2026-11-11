# Test ordinary video changes in the current customer analysis

Run all 24 controls sealed in [v290](video-vcd2-matched-controls-v290-method.md)
from the eight newly qualified VCD2 camera recordings. Each recording supplies
one unchanged control, one chroma-conversion control and one recompression
control. A face-swap review interval on any control is a false flag for this
evaluation. Count cases and paired recording groups. The cohort measures false
flags and coverage for these ordinary changes. Sensitivity to swaps and lip-sync
edits requires separate positive controls.

Bind the exact v290 completion receipt and all qualification records before
freezing. The preflight reports the 500 MiB upload limit, maximum size and
eligible and ineligible counts by resolution, and the number of complete
upload-eligible triplets. Any control over the limit remains in the denominator
with its own outcome. A separately frozen common-delivery encode for all 24
would be a different cohort.

Freeze current application source and the unmodified customer model manifest.
The isolated worker uses the full analysis profile, the locally installed
commercial model inventory, an owned fixture tenant and the existing process
and disk guards. Offline and disabled billing settings are verified before
running. The worker uses the upload and analysis functions directly in its
private workspace. It starts no payment, HTTP service or external detector
request. The report records the frame route's actual skips, including skips
for its memory budget, and appearance coverage separately.

The read-only review validates the exact uploaded bytes, native clock and
sampled RGB against v290, and verifies packet inputs, screenshots, crops,
annotations, context clips and their stored hashes. It also requires evidence
version 8 and independently reconstructs the before, during and after tracking
measurements from the rounded recorded eye landmarks and face-box widths.
The audit does not run the earlier experimental edge refinement.
Seal each review with the owned workspace before releasing that workspace.
Public readout contains aggregate counts and commitments. Recordings, per-case
hashes, screenshots and guard logs stay in the private archive.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
v294() { backend/.venv/bin/python -B -m scripts.video_vcd2_current_app_v294 "$@"; }
v294 preflight
v294 freeze
case_name=slot-01-unchanged
v294 run "$case_name"
v294 review "$case_name"
v294 seal "$case_name"
v294 release "$case_name"
v294 readout
```

After v290 is sealed, repeat the four per-case steps for each eligible case,
one case at a time. A guard or worker failure remains visible in its case
outcome. Readout remains partial while any eligible case is pending. Its paired
transition counts require measurable appearance coverage on both controls.
