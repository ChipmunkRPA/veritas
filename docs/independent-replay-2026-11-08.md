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

## Retained execution attempts

- [Audio run 42468303951](https://github.com/ChipmunkRPA/veritas/actions/runs/42468303951)
  reached the correct historical source. The diagnostic completed successfully;
  strict exact refitting still failed its original equality requirement. The
  arithmetic job stopped during Python installation after three HTTP 504
  responses, before its numerical review. A bounded retry of that setup-failed
  job is separate from changing or waiving its checks.
- [Visual run 42468335861](https://github.com/ChipmunkRPA/veritas/actions/runs/42468335861)
  installed the fixed CPU Python packages, then stopped while recording the
  environment because FFmpeg was absent from the hosted image. Public source
  acquisition, checkpoint loading and neural inference had not started. The
  runner setup now explicitly installs FFmpeg/FFprobe, records the actual codec
  version, and writes a setup-intent receipt before dependency installation.
  This is an environment correction; the immutable experiment revision and
  all its points, cases, guards and checks remain unchanged.

## Completed audio replay after source correction

The arithmetic-only retry in run 42468303951 completed successfully. Its
[receipt](evaluation-panels/linux-source-pinned-audio-v13-arithmetic-20261108.json)
reconstructed all **4,248 feature records** and **1,449 development outcomes**,
including the original byte-for-byte feature receipt and rejection of both
candidate arms. The separate
[diagnostic receipt](evaluation-panels/linux-source-pinned-audio-v13-diagnostics-20261108.json)
completed both fitting-only refits. It found maximum normalization-center
difference 2.67e-15, coefficient difference 8.21e-16 and development-probability
difference 2.22e-16. Neither arm changed a directional outcome or needed an
unavailable extra public probe. These are floating-point differences, not exact
equality. The strict equality job still failed on the standalone fit, so the
overall workflow remains failed. No original object or acceptance test changed.

Only the setup-failed arithmetic job was requested for retry; the completed
diagnostic/strict-fit outcomes were retained. The first attempt's HTTP 504
setup failure and the earlier changed-source failures remain in GitHub history.
No new audio-model inference, source recording scoring or speech calibration
was performed by this replay.

## Completed native visual replay and Mac comparison

[Corrected visual run 42468624233](https://github.com/ChipmunkRPA/veritas/actions/runs/42468624233)
completed native inference and its separate non-neural audit on Linux x86_64.
The exact public artifact SHA-256 was
`c2e6c0a3813fd79b229815a1b0ffdd6ccafeea1652ddc1daec1aad39e59011b2`.
Its ten files were checked against the allowlist before local extraction. The
checkpoint SHA-256, all 432 loaded tensors, 94,416,643 parameters, protected
source/policy identity and core package versions match the Mac study. The Linux
CPU wheel suffix and FFmpeg version are recorded rather than hidden.

Both platforms completed **168 cases / 191 neural forwards**. Each separate
audit replayed 11,010,048 presence pixels and 352,321,536 binary bit predictions,
redecoded six clips / 48 frames and verified source timestamps with zero neural
forwards. Linux native execution took 74.61 seconds with sampled process-tree
RSS 1,165,332,480 bytes; audit took 2.53 seconds with 199,680,000 bytes. Neither
guard triggered; both profiles report zero sampling errors. Sampled RSS is a
lower bound, not an assertion of the exact memory peak.

The [public comparison](evaluation-panels/public-visual-v1-cross-platform-replay.json)
retains seven groups with numerical differences. Only two groups changed any
flag/miss counts, both after H264 encoding:

| Same fixed authored condition | Mac | Linux | Denominator |
|---|---:|---:|---:|
| No-added-mark H264 frames flagged | 2 | 3 | 24 frames |
| Intended partial-mark H264 frames missed | 3 | 2 | 12 marked frames |
| Partial H264 flags outside intended marked interval | 6 | 4 | 12 outside frames |
| Partial H264 mean marked-region IoU | 0.2364 | 0.3138 | 12 marked frames |

All image flag/miss counts and raw-video flag/miss counts matched. Some native
and resized overlaps/pixel counts differ slightly. Linux used FFmpeg 6.1.1 and
CPU PyTorch builds; the Mac execution used its recorded local runtime. The
causes of cross-platform differences were not isolated. Neither platform is
declared better from this comparison. In particular, outside-interval H264
responses can reflect codec propagation or false-family responses; intended
insertion boundaries do not prove every changed pixel's cause.

Source, results, audits and resource receipt copies are public in
[Mac receipts](evaluation-panels/public-visual-v1-mac-receipts/publication.json)
and [Linux receipts](evaluation-panels/public-visual-v1-linux-receipts/publication.json).
Only source/output filesystem paths were redacted in the copies; manifests
retain original/public checksums and list every redaction. Canonical retained
records were not rewritten. No raw media, case pairings or prediction arrays
were published. Reproduce the aggregate comparison without a model download:

```sh
PYTHONPATH=backend python -m scripts.compare_public_visual_native_v1 \
  docs/evaluation-panels/public-visual-v1-mac-receipts \
  docs/evaluation-panels/public-visual-v1-linux-receipts \
  /tmp/new-public-visual-comparison.json
```

Thirteen selected model-free contract/comparison tests and static checks pass.
The comparison rejects changed model identities, cases, core package versions,
incomplete timestamp audits, triggered guards and a false serving-admission flag.
No test, decoder point or original study code was relaxed. Outside human
reviewers remain **zero**. No provider compatibility, real-media accuracy,
AI-confidence contribution or early-stop admission is established.

## Fresh natural-media preparation remains separate

Metadata-only preflight now covers 288 distinct Wikimedia file pages across
16 retained API responses. Initial date, format, attribution and per-file
Creative Commons license checks leave 78 image and 24 video candidates from
65 credited creators. These are publisher metadata claims, not authenticated
capture truth. No raw media was downloaded or scored for this panel. Visual
eligibility review, complete source-exclusion checks, downloaded-byte/decoded
geometry checks and a separate prospective scoring freeze are still required.
The real-media panel is not described as completed by the Linux authored replay.
