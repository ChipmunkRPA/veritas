# Corrected audio identity experiment — execution record

6 November 2026. The [identity repair](audio-source-identity-v9.md) has completed
its corrected development experiment. The initial freeze preceded extraction
and fitting. The running audio calibration and overall confidence are unchanged;
the separate [protocol](audio-identity-v9-execution-protocol.md) and implementation
pins remain immutable.

**Completed development:** the [full comparison](audio-identity-v9-development-evaluation.md)
failed five of sixteen fixed requirements despite fewer false AI calls and
better probability scores. All extraction, numerical/refit and 99 native
checks completed; no fresh scoring or serving change occurred. The prospective
freeze details below remain historical commitments.

## Bound inputs and method

The private execution lock SHA-256 is
`cf407808611a847702ad1c482af6705095d2dc011ead9ebed29e5afe3b3acb18`.
Its [public projection](evaluation-panels/audio-v9-execution-freeze.json) SHA-256 is
`b73e5933deb418129f570500a90672e062d159ed84d39ddf8bc8744d2f13ff8c`.
The projection removes private paths; its hash is distinct from the private
path-bearing lock hash. It retains the exact method, 51 implementation pins,
verified commercial checkpoint commitments, runtime, unchanged service bundle,
source/recipe commitments and predetermined 19 native cache inputs.

There are 4,248 consumed four-second inputs: 1,926 fitting, 873 calibration and
1,449 development. The unchanged statistical method and all sixteen development
advancement gates apply to the corrected 161-original development denominator.
The complete 1,080-input confirmation panel remains reserved and is refused by
the development runner. A development pass alone cannot enable a replacement.

The old stopped run supplied 1,882 complete terminal records, of which 1,846
match corrected inputs exactly. All original stored vectors, head arithmetic,
input-prefix bindings and verified checkpoint/runtime commitments passed the
freeze checks. Before any reuse the runner must pass 19 predetermined live
native replays, requiring exact VAD windows, status, raw response and embedding
digest. The other 2,402 corrected consumed inputs require new extraction.
Disagreement is recorded and halts execution; silent fallback is refused.

## Local verification before extraction

The selected source/identity/learner/calibration/guard/execution/independent-review
suite passed 132 tests. This includes new failure cases for source clocks,
background labels/roles, modified vectors/heads, failed cache entries, native
disagreements and export privacy. Ruff and whitespace checks passed.

The [standard-library metadata/code review](evaluation-panels/audio-v9-execution-freeze-review.json)
passed all declared 51 implementation pins. Eight deliberate alterations to the
actual public freeze were refused in the [tamper receipt](evaluation-panels/audio-v9-execution-tamper-review.json).
These are local engineering and reproducibility checks, not performance results
or outside reviewer participation.

From the repository root, without a model or private source manifest:

```sh
python3 backend/scripts/review_audio_identity_execution_v9.py review \
  docs/evaluation-panels/audio-v9-execution-freeze.json \
  docs/evaluation-panels/audio-v9-identity-plan.json.gz \
  docs/evaluation-panels/audio-v8-source-plan.json.gz \
  /tmp/new-audio-v9-execution-review.json --project .
```

The model runner is `backend/scripts/run_audio_identity_v9.py` with stages:
`freeze <corrected-root> <new-lock>`, `extract <lock> <new-receipt>`,
`train <lock> <new-fit-directory> --receipt <receipt>`,
`compare <lock> <new-responses> --receipt <receipt> --head <head>`, and
`review <lock> <new-summary> --receipt <receipt> --head <head> --responses <responses>`.
Use the Python environment with the pinned runtime and separately provisioned
commercial weights. Set model/data/HF cache locations explicitly and keep
offline mode enabled. Each resumption revalidates lineage and repeats the
predetermined native cache audit; failures remain visible rather than dropped.

## Remaining work

The prospective freeze was published and remotely verified at commit
`4e1c59cd4ce28a58c2c9e95f65fc7aa2def6ddc3` before new neural forwards or fitting.
All nineteen predetermined native cache replays then matched exactly, including
the unassessed case. The [complete native audit](evaluation-panels/audio-v9-native-cache-audit.json)
joins its IDs to the public source capsule and preserves the original JSONL
digest. Exact-input reuse and all consumed extraction subsequently completed.
The corrected candidate was fitted and rejected; no fresh scoring occurred.

All consumed extraction, fitting, separate calibration and development comparison
are now complete. The candidate is rejected under the unchanged rules.
The [additional review protocol](audio-identity-v9-review-protocol.md) fixes
99 native development cells and complete numerical/refit verification during
extraction, before fitting/comparison. It changes no acceptance rule or original
implementation pin. Both reviews passed on the completed result, including an
exact CPU refit and all 99 native cells. This does not override the failed
development requirements; do not advance this candidate to fresh confirmation.
Preserve serving defaults, rollback, payment and authentication.
Image, video, writing and actual independent human testing remain unfinished
parts of the wider program. GCP remains deferred.
