# Models of Veritas 2.1.1

`backend/app/models.yaml` pins 31 models: the source, the revision and the SHA-256 of each file. The supplied local archive
includes the fine-tuned text classifier `local.desklib_ft_v2` in `models/weights/desklib-ft-v2` (7 files, 1.7 GB).
Model files are excluded from the source repository and must be provisioned separately. This fine-tune of
`desklib/ai-text-detector-v1.01` is not available from a publisher; preserve the local copy. The 7 files matched their pins
on 11 October 2026.

The other 30 models are unchanged from their publishers. The fetch command downloads them and checks each file
against its pin.

## Install the models

1. Set the model locations before you start the API and the worker:

       export VERITAS_MODELS_DIR=$PWD/models HF_HOME=$PWD/models/hf TORCH_HOME=$PWD/models/torch

2. Download the public models. With the default licence policy, `--profile full` downloads the 13 models that the
   `full` profile needs. `--all` downloads all 30 public models.

       cd backend
       VERITAS_OFFLINE=false .venv/bin/python -m app.cli models fetch --profile full

3. Check all installed files. The output must include `VERIFIED  local.desklib_ft_v2`.

       .venv/bin/python -m app.cli models verify --installed

4. For Compose, copy `models/weights/desklib-ft-v2` into the `veritas-models` volume. The `models-init` service
   downloads the public models into that volume.

`docs/runbook.md` gives the full procedure ("Installing weights" and "Local models: the fine-tuned text classifier").
Without the fine-tuned model, the base classifier `txt.desklib` reads documents.

## Licences

`backend/app/models.yaml` gives the licence of each model, and `python -m app.cli models list` shows the policy. The
default policy (commercial + allowlist) does not load the models of class `non_commercial` and `unknown`, except
`local.desklib_ft_v2`, which the policy allows by name. That model has the licence `NOASSERTION`. Its base model is MIT,
and nobody reviewed the terms of its training data. Read those terms before you share this model.

## Not in this archive

The 30 public models, the tests, the QA records, the build loop's notes and the benchmark data. The source is in the
repository at tag `v2.1.1`. In `frontend/tsconfig.json`, the reference to the e2e test project is removed so that
`npm run build` works without the tests. That is the only change from the tag.
