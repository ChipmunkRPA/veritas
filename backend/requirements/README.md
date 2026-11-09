# Reviewed Python runtime dependencies

These contracts bind future **Linux amd64 / CPython 3.12** backend images to the Python library versions used by Veritas at source `74bcecc`. They prevent silent upgrades during installation. They do not establish Linux detector parity, image provenance, or bit-for-bit container reproducibility.

| File | Purpose |
|---|---|
| `tested-versions.txt` | 120 installed version constraints from the reachable production/PostgreSQL dependency graph on Darwin arm64, plus build tooling. Linux-only torch dependencies were excluded from this initial snapshot and resolved from the selected wheel metadata. |
| `cpu.in`, `cu126.in` | Official torch 2.13.0 / torchvision 0.28.0 CPython 3.12 Linux wheel URLs, with upstream SHA256 fragments and explicit accelerator suffixes. |
| `runtime-cpu.lock`, `runtime-cu126.lock` | Reviewed dependency graphs: 121 CPU / 140 CUDA packages, exact versions and published distribution hashes; initial resolver outputs plus the documented one-node extension. |
| `build.in`, `build.lock` | Pinned pip 26.2.1, setuptools 84.0.0, wheel 0.48.0 and packaging 26.3, installed before source builds. |

Both Linux graphs preserve all 120 tested package versions, except the intentional torch/torchvision accelerator suffixes. CPU adds greenlet 3.5.6, absent from the Darwin environment. CUDA additionally includes 19 dependencies resolved from the cu126 wheel. Those Linux-only packages are resolved and pinned, **not locally executed or model-validated**. Pytest is present because the installed c2pa-python package declares it as an unconditional dependency; it was not added as a Veritas development extra.

The Dockerfile installs hash-checked build tools, then the full runtime lock. Only Grad-CAM may build from source; every other package must have a compatible wheel. Grad-CAM 1.5.7 declares setuptools/wheel build requirements. `--no-build-isolation` prevents a second unpinned build environment. Installing the local project with `--no-deps --no-build-isolation` prevents its broad development constraints from replacing the locked runtime. The final `runtime_contract.py` check verifies every installed version, `pip check`, torch's compiled CUDA identity, and torchvision's native CPU NMS result. It does not require a GPU on the builder.

## Changes since the first resolution

- **2026-10-02, urllib3 2.7.0 → 2.8.0** (N-15): pip-audit of both locks reported PYSEC-2026-4175, -4176 and -4177 against 2.7.0, all fixed in 2.8.0. Both variants were resolved again with the command below, the existing lock as constraints and that one pin moved: the output differs from the earlier lock in the three `urllib3` lines only (120 CPU / 139 CUDA packages as before). `tested-versions.txt` and the development environment carry 2.8.0, and the local regression ran on it. The Linux graphs remain **not executed** here, as before.

Every change of a lock needs `make audit` afterwards: the suite compares the recorded audit (`docs/v2/qa/audit-run.json`) with the SHA-256 of each lock.

- **2026-11-08, add einops 0.8.2:** promoted the already installed MIT helper from development to runtime for the vendored regional graph. Official PyPI 0.8.2 metadata declares no dependencies; both locks add exactly this one package with its published wheel and source hashes. Existing pins and hashes are unchanged. This is a one-node lock extension, not a rerun of Linux dependency resolution or Linux execution. The installed Darwin snapshot includes the same version; local serving parity is recorded separately in the v47 study.

## Recheck or deliberately refresh

The locks were generated with uv 0.11.7. Use the project's Python 3.12 interpreter and a private cache; never install the Linux lock into the local Darwin development environment. The commands below perform dependency resolution, not Docker builds or deployment. Run from the repository root; choose `cpu` or `cu126` explicitly.

```sh
runtime_backend=cpu
uv pip compile backend/pyproject.toml "backend/requirements/${runtime_backend}.in" \
  --extra postgres --constraint "backend/requirements/runtime-${runtime_backend}.lock" \
  --build-constraints backend/requirements/build.in \
  --python backend/.venv/bin/python --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 --no-python-downloads \
  --generate-hashes --no-annotate --no-header \
  --cache-dir backend/data/runtime-lock-cache \
  --output-file "backend/data/rechecked-runtime-${runtime_backend}.lock"
```

Using the existing lock as constraints preserves Linux-only versions. The initial resolution instead used `tested-versions.txt`; reusing only that snapshot in future can upgrade Linux-only dependencies. Compare the complete generated output with the reviewed lock before replacing it. Regenerate `build.lock` from `build.in` with the same platform, Python and hash options.

For an intentional upgrade, update reviewed input pins, resolve both variants, inspect dependency and hash changes, and rerun local regression/model parity plus the future Linux image/inference checks. Do not regenerate the installed snapshot blindly or describe newly resolved versions as already tested. Every direct application/PostgreSQL dependency must remain covered; the deployment test enforces that contract.

A local smoke runs `backend/.venv/bin/python -B backend/scripts/image_smoke.py`; it uses the Darwin contract and remains explicitly local evidence. Future images write their backend to `requirements/image-backend`; the isolated smoke refuses missing Linux markers and any contract failure. See the [GCP runtime procedure](../../infra/gcp/README.md#isolated-runtime-smoke-contract).

Base image tags, OS packages, Python patch versions and externally supplied model weights are outside these Python locks. The release must retain an immutable built-image digest, installed system package inventory, model hashes, actual Linux smoke and CPU/CUDA numerical validation before deployment readiness is established.

Primary references: [official PyTorch 2.13 wheel matrix](https://pytorch.org/get-started/previous-versions/), [pip hash-checking requirements](https://pip.pypa.io/en/stable/topics/secure-installs/), [pip index selection](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages), and [uv PyTorch package variants](https://docs.astral.sh/uv/guides/integration/pytorch/).
