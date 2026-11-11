# Full-app replay of the video verification-cache failure

The v202 eight-triplet audit sealed 24 runs. In slot 10's chroma-conversion control, concurrent checks collided on the shared verification-cache temporary path. The frame and appearance routes both failed, the appearance route measured zero windows, and the customer result was unavailable. The v251 protocol replays that one ordinary-edit control to measure whether the affected app routes recover after the atomic cache fix.

Before the patch, the prospective freeze checks the published v202 result and freeze, reconstructs the commitment from all 24 adjacent archive receipts, and fully verifies the chosen tar and every file inside it. The retained metadata binds the old input plan, manifest, audit, guard profile and log to that archive. The current v200 prepared MP4 must have the same size and SHA-256 as the original uploaded bytes in the tar and match its native-frame qualification. The public freeze commits to the private input record and archive. Filenames from the source corpus, video, evidence, and detailed worker receipts stay local.

The run accepts exactly one changed app source file, `backend/app/engine/fetchers.py`, with the reviewed replacement SHA-256. It checks the historical runner and runtime packages, decoder binaries, model manifest, private model and fuser inventory, appearance-review artifact and resource limits. The inherited full-app worker independently verifies cached model and fuser bytes before startup and uses a fresh isolated workspace. The inherited audit verifies ingested content, source frame and evidence bindings. Separate replay receipts preserve the historical v202 runs.

Run the first two commands while the historical app source is present. Apply the reviewed cache fix after `freeze` and before `run`. Wait for each command to finish before the next. `seal` requires the external archive volume, and `release` verifies that seal before removing the replay workspace.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 review
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 seal
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 release
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_vcd2_verify_cache_replay_v251 readout
```

The sanitized readout compares the two route statuses, measured windows, frame and evidence coverage, customer direction, and the presence of the original collision. Recovery requires a scored replay with both previously failed routes `ok` and no collision. This single replay measures route reliability on the selected control. Detection accuracy and population false-alarm rates require a separate evaluation.
