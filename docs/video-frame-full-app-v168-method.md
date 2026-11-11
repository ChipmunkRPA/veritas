# Full-app admission test for one-face frame batching

The previous test compared the reference and one-face adapters on the same
15 verified crops. Their scores and evidence matched exactly. This study
checks whether the one-face adapter runs through the complete local app
and delivers its frame evidence within the existing model-manager budget.
[Standalone parity result](video-frame-batch-qualification-v160-result.md)

Both arms take the identical six-second 1080p genuine control from the
verified HD preparation. The primary input is the ordinary H.264
recompression. Its paired lossless export and the other source stay outside
this test. Freeze exact input bytes, all 150 native picture timestamps and
decoded RGB hashes, and the independently decoded 96,000 mono audio samples.
[HD preparation](video-genuine-hd-source-admission.md)

The reference uses the current production manifest. The candidate changes
one adapter entry and its version in a private manifest. Both use the
ordinary registry, model manager, full video pipeline, and the verified
commercial model assets. Each arm starts with separate owned APFS storage,
a fresh local database and its own sound-analysis cache. Billing, network
and service startup are disabled in the test processes.

Success requires complete results from both arms, matching input clocks
and decoded samples, a reference memory-budget skip, and a candidate
detector run that saves model-backed scores and the corresponding frame
series in the committed result. Keep the non-target route statuses and
fuser identity paired. Inspect the saved evidence set and confirm each
declared evidence asset is intact. Preserve any skipped, failed or
unavailable route and the actual monitor receipts.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v168 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v168 run reference
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v168 run batch-one
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_frame_full_app_v168 readout
```

Run the two guarded arms sequentially. The receipt reports process-memory
samples and route evidence on one genuine video. Face-swap sensitivity,
false alarms on independent sources and app default changes require a
separate evaluation.
