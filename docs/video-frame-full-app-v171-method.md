# Corrected full-app frame admission comparison

The v168 reference worker initialized the full-profile app and model manager,
then stopped at the second environment confirmation before the video analysis
started. The original worker log, monitor journal, runtime record, and empty
result directory remain bound to the new trial. The input, manifests, detector
code, activation plans, app settings, and resource limits come from v168.
[Original comparison method](video-frame-full-app-v168-method.md)

A model-free reproduction found five values added during the real runtime
imports. OpenCV adds its installed library directory to `LD_LIBRARY_PATH`.
PyTorch Inductor adds its cache path under the owned temporary directory.
The scikit-learn initializer sets `KMP_DUPLICATE_LIB_OK=True` and
`KMP_INIT_AT_FORK=FALSE`. The OpenCLIP tokenizer sets
`TOKENIZERS_PARALLELISM=false`. The activation-plan lookup added nothing.

Keep the original first confirmation. At inference entry require every
original key and value, plus the five exact observed dependency values.
Use the pinned installed OpenCV path and PyTorch cache resolver to derive
the two local paths. Reject a cache path outside the owned temporary
directory. Then repeat the original offline, storage, credentials and
settings-singleton checks. The correction does not write to the environment.
The freeze binds the correction, dependency sources and preserved failure.

From `backend`, run each preflight in a fresh process. It uses the real
registry, SQLite initialization and model manager. It stops at the
corrected boundary with zero analysis rows and no loaded models.

```sh
.venv/bin/python -B -m scripts.video_frame_full_app_v171 preflight reference
.venv/bin/python -B -m scripts.video_frame_full_app_v171 preflight batch-one
```

Both model-free preflights reached the corrected boundary on the Mac's GPU.
The reference activation plan was 6,710,886,400 bytes and the one-face plan
was 838,860,800 bytes. Both kept the same 4,608 MiB manager budget. The
preflight created zero analysis rows and loaded zero models in each process.

Freeze the corrected context, then run the two monitored arms sequentially.
The reference arm must have a terminal monitor receipt before starting the
candidate. Readout verifies actual guard and app receipts before comparison.

```sh
.venv/bin/python -B -m scripts.video_frame_full_app_v171 freeze
.venv/bin/python -B -m scripts.video_frame_full_app_v171 run reference
.venv/bin/python -B -m scripts.video_frame_full_app_v171 run batch-one
.venv/bin/python -B -m scripts.video_frame_full_app_v171 readout
```

This trial measures app admission and evidence delivery on one genuine HD
clip. It cannot establish swap sensitivity, fresh-cohort accuracy or
readiness to change the default.
