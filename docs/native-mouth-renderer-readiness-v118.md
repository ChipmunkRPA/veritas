# Native mouth renderer: pending RAM admission

The [native picture bank](native-mouth-inputs-v116-result.md) is complete, and
the [counterfactual renderer procedure](native-mouth-renderer-v117-method.md)
is frozen in its [isolated runtime context](native-mouth-runtime-v118-method.md).
All 27 targeted software checks pass: 14 input, ten renderer and three runtime
checks. These are software/installed-file checks, not successful model execution.

The actual guarded `run` command verified the frozen context and then reported
`pending_headroom_no_stage_attempt`: available RAM was **3,051,831,296 bytes**
(2.84 GiB), below the unchanged 6-GiB initial requirement. No model-stage or
render-cache directory exists and no new neural call was made. This is a
read-only admission denial, not a failed or inconclusive detector result.
See the [actual readiness snapshot](evaluation-panels/native-mouth-renderer-readiness-v118.json).

The initial v117 freeze attempt's missing isolated dependency remains preserved
separately. The new context uses the already retained runtime; the application
environment was not modified. Source selection, image/state inputs, model weights,
seed, counterfactuals and resource limits were not changed to obtain a pass.

Once available RAM exceeds 6 GiB, resume the exact exclusive command:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.native_mouth_runtime_v118 run
```

Run outside the sandbox with the authorized Seagate drive mounted. Inspect the
actual terminal profile and completion receipt before proceeding. If the stage
has started or failed, preserve it; do not rerun or overwrite it. This completed
readiness snapshot remains historical after future execution.

Full native-clip construction, achieved synchronization, articulation detection,
fresh errors and customer-flow qualification remain open. Serving defaults and
rollback assets are unchanged. No Actions workflow, outside contact, real payment
or deployment was performed.
