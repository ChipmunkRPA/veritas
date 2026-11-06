# Full frozen-representation audio experiment: preparation

6 November 2026. Prepared one supervised full-representation comparison after
the [conditional experiment](audio-conditional-v10-development-evaluation.md)
failed. The [prospective protocol](audio-full-representation-v11-protocol.md)
retains all sixteen gates and the current service baseline. No project-data
fitting or new neural forward has occurred at this preparation freeze.

The candidate retains every1024 frozen coordinate rather than selecting16
unsupervised directions. Fitting-only head-orthogonal diagonal normalization has
a total variance budget at most16. The prior weighted objective, target smoothing,
regularization, nonnegative raw slope and independent calibration are unchanged.
The completed, rejected PCA16 result remains a fixed representation control.
This tests information retention; it does not establish that discarded information
caused the prior failures or that the new representation improves detection.

## Bound inputs and verification

- All4,248 corrected consumed representations remain in their original roles:
  1,926 fitting,873 calibration,1,449 development inputs;214/97/161 originals.
- The original v9 features, projection/head and service journal, plus the completed
  v10 probe journal, have fixed original digests. Their424 exact cached filtered
  probes can be reused after per-input source/PCM/raw/feature checks.
- All1,080 reserved confirmation inputs remain fixed and unscored. Existing62
  unassessed feature inputs stay unassessed; neither missing nor failed checks
  can establish original-file authenticity.
- Same verified commercial-permission neural checkpoints and runtime; no new
  dependency or pretrained checkpoint is required. The58 original/new code and
  protocol files are pinned. Models, customer UI, confidence, billing/auth and
  GCP are unchanged.

Private lock SHA-256:
`bce7b7cb7cb142a28cc68c60b54a559b3eb6f9ea52efc67a0259a49c8ee5c384`.
The [safe public freeze](evaluation-panels/audio-v11-full-representation-freeze.json)
has its own SHA-256:
`9f23fe7ab11c4672ca13949956fbf57ed88a74308507015fb20db90f5a356c82`.
The public allowlist excludes the private source root.

All119 selected regression tests passed. New synthetic cases verify that a
a high-index input coordinate can affect a supervised response,
held roles cannot learn normalization/coefficients, metadata cannot alter a
prediction, missing speech remains unassessed, calibration retains missing
variants, and private paths/altered prospective claims are refused. These are
implementation checks, not detection accuracy. Lint and whitespace checks passed.

The model-free preflight checked the complete role matrix, unchanged commercial
checkpoint/runtime/service commitments and all58 code pins. Seven deliberate
changes to the actual public freeze were refused. With only the standard library:

```sh
python3 backend/scripts/export_audio_full_representation_v11.py review \
  docs/evaluation-panels/audio-v11-full-representation-freeze.json \
  /tmp/new-audio-v11-preflight.json --project .
```

## Execution order

Publish and remote-verify this freeze before any project fitting or new forward.
Then fit only fitting originals, save parameters before independent calibration,
and compare every development input. Preserve failed/abstained outcomes and
actual filtered-probe provenance. Complete numerical refit,99 predetermined
native cases and every new-probe replay before considering confirmation.

Any failed gate rejects the candidate without scoring reserved sources. A passing
development result alone does not enable serving, whole-recording confidence or
deployment. Outside human testing, image/video/writing improvements and eventual
cloud validation remain unfinished parts of the wider project.
