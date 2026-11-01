# Models used by Chipmunk Verify

`backend/app/models.yaml` is the current inventory: it pins each source, revision and file SHA-256. The supplied local archive
includes the fine-tuned text classifier `local.desklib_ft_v2` in `models/weights/desklib-ft-v2` (7 files, 1.7 GB).
Model files are excluded from the source repository and must be provisioned separately. This fine-tune of
`desklib/ai-text-detector-v1.01` is not available from a publisher; preserve the local copy. The 7 files matched their pins
on 11 October 2026.

The fetch command downloads the selected public weights and checks each file against its pin. An installed file is not
automatically an enabled check, and a successful integrity check is not a commercial licence approval or accuracy evaluation.

## Install the models

1. Set the model locations before you start the API and the worker:

       export VERITAS_MODELS_DIR="$PWD/models" VERITAS_HF_HOME="$PWD/models/hf"
       export HF_HOME="$PWD/models/hf" TORCH_HOME="$PWD/models/torch"

2. Download the public models permitted by the current policy and profile. Use the explicit model IDs for local candidate
   validation; do not use `--all` as a commercial deployment list.

       cd backend
       VERITAS_OFFLINE=false .venv/bin/python -m app.cli models fetch --profile full

3. Check all installed files, including any retained legacy weights. Verification does not enable a disabled model.

       .venv/bin/python -m app.cli models verify --installed

4. For Compose, provision the permitted public models in the `veritas-models` volume. The `models-init` service
   downloads the public models into that volume. The archived local fine-tune is optional for separately authorised
   research or historical reproduction; it is not required by the commercial runtime.

`docs/runbook.md` gives the full procedure ("Installing weights" and "Local models: the fine-tuned text classifier").
With the fine-tuned model disabled under the commercial policy, the base classifier `txt.desklib` reads documents.

## Licences

`backend/app/models.yaml` records publisher licences; `python -m app.cli models list` shows the effective policy.
The default commercial policy does not permit `non_commercial` or `unknown` weights. The former default exemption for
`local.desklib_ft_v2` has been removed: its training-data terms remain unreviewed and its licence is `NOASSERTION`.
Its local files are retained for reproducibility and separately authorised research.

See `docs/commercial-model-audit.md` for the current code, weight, dependency and training-source review. These are
different questions: a permissive weight licence does not establish rights in every training example or demonstrate
that a detector works on the intended inputs. Third-party attribution files must accompany distribution.

## Validation and retention decisions, 31 October 2026

The new local voice inspection and speech gate are installed, alongside sound-category inspection of mixed audio.
The candidate voice signal remains separate from the established calibrated voice check and has zero fusion weight.
On the retained 20.92575-second speech fixture, all nine old-check window coordinates and raw responses are unchanged
(maximum difference zero). Reference preprocessing and forward checks also match exactly on silence, noise and speech
controls. These are engineering checks, not evidence that the new detector outperforms the baseline.
Both audio and video now publish source-bound voice windows, background measurements and sound-category intervals.
See `docs/commercial-model-audit.md` and `backend/data/model-upgrade-review/audio/integration.json`.

GenD was downloaded and verified against its pinned SHA-256. On 32 publisher example crops, local preprocessing and
CPU forward output exactly match the reviewed upstream reference; CPU/MPS logit difference is at most 0.0000287.
Twenty fake examples crossed the zero-logit boundary, as did eight of twelve real examples. These are correlated
publisher examples with possible training overlap, not an independent accuracy benchmark. Re-cropping changed two
boundary decisions. GenD remains disabled by default, consistent with the failed historical admission records.
Local evidence: `backend/data/model-upgrade-review/gend/verification.json`; replay with
`backend/scripts/verify_gend_candidate.py` after preparing its pinned reference/example files.

The Qwen 1.5B pair stays in the full/deep document profiles for now. The local PDF, email and DOCX checks preserve
different, source-bound token observations across 18 evidence packets; they do not provide independent human/AI
labels for a removal ablation. Older published local results also describe a different ratio implementation and
cannot establish that the current check is redundant. No bigger language-model pair or physical deletion is justified
by this evidence. Local retention record: `backend/data/model-upgrade-review/text-retention.json`.

Earlier benchmark source files and raw scores referenced by `docs/evaluation.md` are not all present in this archive.
Treat their documented figures as historical records, not tests rerun on this computer. New candidate checks must not
inherit the old fusion calibration or claim a validated confidence percentage without a suitable independent evaluation.
