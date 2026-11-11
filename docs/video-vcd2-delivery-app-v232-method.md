# Full-app comparison of delivered webcam controls

This isolated study fixes nine delivered v206 controls: unchanged, chroma
conversion and recompression for each of slots 01, 02 and 06. All are
face-swap negatives. Slot 01 also has three already sealed v202 raw-control
analyses, providing a same-role raw-to-delivery flag comparison. Keep raw and
delivered cohorts separate; slots 02 and 06 address former upload-size
exclusions and do not gain hypothetical raw app results.

Bind the published v206 completion and all 12 ordered private qualification
hashes, the frozen preparation method and v200 source receipts. Verify the
selected delivery files against their own receipt, native clock, per-frame RGB,
paired changed-frame indices and raw input commitments. Freeze and readout
verify the full raw archives against the published v202 archive-set hash.
Worker verification binds those receipts and audits without rereading the raw
archives. App upload, sampling,
render anchors, evidence, local settings, monitored resources, review and seal
reuse the v202 full-app contracts. Shared delivery changes relative to raw are
not confused with paired changes between delivered roles. Reuse the exact same
serving manifest and device; no setting or threshold is changed.

Execute one case at a time, independently guard, audit, seal and verify the
archive before explicitly releasing its local workspace. Preserve missing or
failed cases as unavailable. Summarize planned, scored and evidence coverage,
face-review flags, appearance range overlap and customer direction separately;
show paired role transitions within each delivered source and raw-to-delivered
flag transitions for slot 01. This small consumed webcam panel does not measure
swap sensitivity, population accuracy, or the causal reason for a flag.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 run slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 review slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 seal slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 release slot-01-unchanged
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_delivery_app_v232 readout
```
