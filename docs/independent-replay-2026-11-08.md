# Source-pinned independent infrastructure replay

This follow-up separates historical study reproduction from current application
regression checks. Automated runs on another machine are not outside human
review, fresh predictive validation or an improvement in detector accuracy.
GCP remains deferred. No customer files are used.

## Correct the historical audio checkout

[Run 42462627531](https://github.com/ChipmunkRPA/veritas/actions/runs/42462627531)
failed all three jobs before their numerical work: the portable reviewer refused
the changed `backend/app/engine/registry.py` on current main. The original study
binds 69 implementation files. Later app development legitimately changed some
of those files; replaying the historical capsule requires its matching source.

Before this change, all 69 committed file hashes were verified to match at
`2c5662af6f66cfae3eafef81e76531863571173c`, the revision used by the earlier
completed diagnostic run. The audio workflow now checks out a full immutable
study revision, records it, and retains every original source/object/feature
check. It narrows unrelated audio-script triggers to this replay's own entry
points. An updated reviewer must be provided as a new immutable study revision;
the default does not silently substitute today's app code.

The strict exact-refit check remains a separate required job with its original
equality test. Its previously documented floating-point mismatch is not waived,
renamed as a pass or hidden with `continue-on-error`. Arithmetic and diagnostics
must still complete independently. Failed earlier runs remain visible.

## New native visual replay

The manual visual workflow checks out
`c2ef887cc236490cdf0962f2754edf554d802a27`, containing the published authored
WAM contract and standalone acquisition helper. It uses a fresh standard
Ubuntu 24.04 runner, Python 3.12.6, PyTorch 2.13.0 CPU / torchvision 0.28.0,
and the experiment's pinned numerical/image packages. The CPU install follows
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
Source, dependency, explicit MIT checkpoint and frozen-code hashes remain
checked by the unchanged experiment.

Before execution, the comparison scope is:

- All 72 image and 96 frame cases, 18 generator / 173 detector forwards, the
  fixed pixel-logit/component rule and the source-time geometry must remain.
- Wrapper/direct parity and the repeated partial-case output must pass the
  original exact checks within this runtime. The model-free audit must reproduce
  this run's masks, packed messages, encoded frames and timestamps.
- Cross-platform model identity, case denominators and protocol must match.
  Linux CPU build, math libraries and FFmpeg encoder versions are recorded.
  Cross-platform masks/flags/overlaps are compared and differences retained;
  exact cross-platform floats or codec bytes are not assumed in advance.
- Native inference and audit each retain the 900-second / 6 GiB sampled-process
  memory / 1536 MiB available memory and disk guard. No implicit retry or point
  retuning. Failures are results, not omitted cases.
- Only allowlisted environment, acquisition, freeze, aggregate and guard/audit
  receipts are exported. Private procedural arrays, media and checkpoints are
  excluded. No app routes, AI confidence or early stopping are admitted.

The workflow has read-only repository permissions, pinned official setup and
artifact actions, no secrets, schedules, GPU or larger runner. Standard hosted
runner usage is [free for this public repository](https://docs.github.com/en/actions/concepts/billing-and-usage).
Execution outcomes must be inspected before claiming reproduction. The new
workflow itself is not evidence of a completed native run.
