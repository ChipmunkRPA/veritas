# Veritas: local deepfake and AI-media detection with evidence reporting

Veritas examines images, videos, recordings and documents on your own machine and reports what it found as evidence: a fused
probability with its uncertainty, every signal behind it (observed value, reference, direction), exhibits (heatmaps, localisation
maps, face tracks, video and audio timeline lanes, sentence highlights), plain-language findings, and the limitations that applied.
Examiners work in cases with a hash-chained custody log, analyst notes, conclusions, sign-off and peer review, and issue signed
PDF reports and evidence packages that verify offline.

All models run locally, and production defaults are offline. Verdicts are probabilistic evidence, not proof: read
[`docs/limitations.md`](docs/limitations.md) before relying on one.

## What is inside

| Area | v2 |
|---|---|
| Images | Provenance: C2PA Content Credentials, generator metadata, EXIF consistency, JPEG quantisation tables, DWT-DCT watermark. Learned detectors in the calibrated image fuser (admitted on vb-1.1): DDA (DINOv2), DFDC-winner EfficientNet-B7 on YuNet faces, and two Hugging Face AI-image classifiers (ateeqq where no face is detected, dima806). Forensics: ELA, noise residual, JPEG double quantisation, copy-move (explanatory, capped). Built and off by default (rejected or not admitted): UnivFD (CLIP ViT-L/14 probe), Community Forensics ViT-S/384, GenD on faces, SPAI (deep profile only, no admission record), and the Mesorch and RADAR localisers, which show regions when enabled but do not move the decision |
| Video | Frame-accurate decode, shot cuts, face tracks (YuNet + DINOv2), per-frame detectors, face noise and temporal jitter / flicker, landmark lip-sync lane, codec lane, timeline lanes with segments. The calibrated video fuser reads Community Forensics and DDA on native frames, DFDC-B7 on faces, the codec lane and the voice detector; these detectors are fitted but not admitted (their admission records are pending, some failing). Shown with segments but weight 0 in the decision: GenD face-swap lane, D3 (X-CLIP) synthesis lane, lip-sync lanes (SyncNet research-only) |
| Audio | Standalone recordings up to 10 min: per-second speech / music / noise / silence gate, wav2vec2 anti-spoofing classifiers on windows of up to 4 s inside speech, voice lane |
| Documents | DeBERTa-v3 AI-text classifier (fine-tune v2), Binoculars (Qwen2.5-1.5B pair), stylometry, per-sentence fusion |
| Fusion | With a fitted artefact: provenance overrides, monotone stacker, per-context Platt calibration, split-conformal sets, exact contributions, abstention where performance is not validated. Without one: an uncalibrated weighted log-odds fallback with a capped verbal scale |
| Cases and custody | Cases with roles, browser pre-hash at intake, write-once evidence store, hash-chained custody log, annotations, conclusions, sign-off locks, peer review, re-analyse and re-run with per-signal deltas |
| Reports | WeasyPrint profiles summary, standard, forensic and newsroom (redacted); Ed25519 signatures; evidence packages; offline `verify` and `reproduce` |
| Service | FastAPI (`/api/v2`), SQLite or Postgres, DB job queue with leases, staged DAG runner with SSE progress, batches, administrator webhooks (off by default), licence policy over a pinned model manifest |
| UI | React "Folio" design (`frontend/design/`): cases dashboard, intake with custody fields and analysis profile, analysis workspace with Plain / Expert views, evidence viewer, video workspace, audio exhibit, export dialog, model catalogue, guided demo |

Architecture summary: [`docs/architecture.md`](docs/architecture.md); binding design and decisions: [`docs/v2/`](docs/v2/).

## Quick start (macOS / Linux, Python 3.12, Node 20+, ffmpeg)

**Archive limitation:** this source copy omits the test suites and most development scripts, including the background
launcher used by `make dev-api` and `make dev-web`. The commands and release records below describe the original complete
checkout; they cannot all be reproduced from this archive. Model weights are provisioned separately; see [MODELS.md](MODELS.md).
The supplied fine-tuned text model is not downloadable from the public model sources.

Install the dependencies, then start the API and frontend in separate terminals:

```bash
cd backend
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev,eval,postgres]" pip -c requirements/tested-versions.txt
VERITAS_OFFLINE=true VERITAS_MODELS_DIR=../models .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8160 --no-proxy-headers
```

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5230 --strictPort
```

Open `http://127.0.0.1:5230/`; stop each server with Ctrl-C. The application can start without all model files, but
analyses may be incomplete or not assessable. To install the public model weights, use the fetch and verification
procedure in [MODELS.md](MODELS.md); that procedure does not recover the separately supplied fine-tuned model.

ExifTool (metadata rules) and Pango (WeasyPrint reports) are recommended: `brew install exiftool pango`. Docker, air-gapped
installs, Postgres and GPU hosts: [`docs/runbook.md`](docs/runbook.md) (its "Build status" names the last commit whose images were built and what was run in them; the CUDA image has never been built).

## Guided demo

Click **Demo now** (sidebar, or `/demo`). The server analyses five bundled samples (`backend/demo/`) and the UI walks through the
results with coach-marks. Repeat demos reuse completed analyses matched by SHA-256.

## API

`/api/v2` is the contract (ARCHITECTURE §5); interactive docs at `/api/docs` in dev, and the schema is committed at
`frontend/src/lib/api/schema.json`.

```
POST /api/v2/cases · GET /api/v2/cases/{id} · POST /api/v2/cases/{id}/items?analyze=true&profile=full   intake with custody fields
GET  /api/v2/analyses/{id} · /events (SSE) · /signals · /findings · /exhibits · /deltas                evidence
POST /api/v2/analyses/{id}/reports · GET /api/v2/reports/{id}/file                                     reports
POST /api/v2/analyses/{id}/packages · GET /api/v2/packages/{id}/file                                   evidence packages
POST /api/v2/batches · GET /api/v2/audit · GET /api/v2/models · /api/v2/policy/thresholds · /api/v2/webhooks
GET  /api/health (liveness) · GET /api/ready (readiness; with an API key: the checks, the worker's device and fuser status per kind) · GET /api/metrics
```

The v1 routes (`POST /api/analyze/{kind}`, `/api/analyze-text`, `/api/analyses/*`) are still served in 2.1.1. Each answer carries `Deprecation: true` and `Sunset: Fri, 31 Dec 2027 23:59:59 GMT` (`VERITAS_V1_SUNSET`). Deleting an analysis is one of them (`DELETE /api/analyses/{id}`); v2 has no route for it.
**Authentication:** `VERITAS_API_KEY` bootstraps an administrator; clients send `X-API-Key` or use a session. In
`VERITAS_ENVIRONMENT=prod` the service refuses to start without a key, with a key shorter than 32 characters or with a
placeholder. Downloads use short-lived resource tokens. The first named administrator is made on the server
(`python -m app.cli principals create --username … --role administrator`), and browser sign-in from another machine needs
HTTPS (`docker-compose.tls.yml`): see "First start on a server" in [`docs/runbook.md`](docs/runbook.md). Its section
"Accounts" covers joiners, leavers, a forgotten password and a lost key (`principals set-password | disable | enable |
set-role | api-key`); a person replaces an issued password from the account menu ("Change password…").

## Evaluation

v2.1.1 goals are judged on Veritas-Bench `vb-1.1` (D-40): `make vbench-score`, `make calibrate`, `make vbench-report BENCH=vb-1.1`,
`make vbench-gate`. On `test`, with 95 % intervals: image AUC 0.9054 [0.8905, 0.9192] in the size-matched view (full view 0.9014
[0.8900, 0.9122]), video AUC 0.9613 [0.9374, 0.9806], audio AUC 0.8474 [0.8070, 0.8852]. `test` has been read by earlier report
runs and the v2.1.1 design was chosen after test-derived numbers were seen, so these are re-used-test, post-selection figures, not a
fresh lockbox. The gate passes 26 of 58 checks, fails 26, and 6 are missing (`docs/v2/bench/gate-v2.1.1.txt`); two in five test
images and half of test videos are left inconclusive (39.7 % and 50.2 % under the served decision rules). Every failing goal is listed with its measured value in
[`docs/v2/bench/shortfalls.md`](docs/v2/bench/shortfalls.md), and what the figures do and do not show is in
[`docs/limitations.md`](docs/limitations.md). Results per family, stratum and transform are in
[`docs/evaluation.md`](docs/evaluation.md); latency and memory per profile are in [`docs/v2/bench/perf.md`](docs/v2/bench/perf.md)
(J-4, v2.1 detector set; the G13 figures of the v2.1.1 set, 4.697 s and 11.877 s on a loaded host, are in
[`docs/v2/qa/j11-latency/`](docs/v2/qa/j11-latency/)).

The v0.3 numbers in `docs/evaluation.md` (public datasets, v0.3 pipelines) are kept for reference and are not v2 results.

## Retraining the text detector on newer LLM output

The shipped DeBERTa detector was trained on 2022–2024 LLM text. To adapt it to current models:

```bash
# 1. human side: Wikipedia intros, CNN/DailyMail, HC3 human answers  -> data/corpus/human.jsonl
# 2. machine side: local generations across 20 registers (free) and/or API generations
cd backend && .venv/bin/python scripts/build_ai_corpus.py --local 150            # Qwen2.5-3B-Instruct
ANTHROPIC_API_KEY=... .venv/bin/python scripts/build_ai_corpus.py --claude 300    # optional, frontier models
# 3. fine-tune from the shipped checkpoint (low LR, 2 epochs) and evaluate on the held-out set
.venv/bin/python scripts/finetune_text_detector.py --epochs 2                     # -> data/weights/desklib-ft-v2
```
The held-out set `backend/data/samples/text/` is never used for training; `docs/finetune_text.md` records the before/after numbers.

## Operations

- `make ci`: lint, model-free tests, typecheck, API types check, build, frontend unit tests, model-free browser suite (`e2e-fake`).
- `make test`: the model-free backend suite as four pytest processes over disjoint sets of test files (`SHARDS=<n>`; `SHARDS=1` is one
  serial pytest, which `make final-gate` uses). One summary is printed; each shard's log and `shards.json` are in `backend/.pytest_shards/`.
  Both forms run OpenMP code on one thread unless `OMP_NUM_THREADS` is set. A subset of the tests is run with `.venv/bin/pytest <paths>`
  from `backend/`.
- `make test-frontend` (`npm run test`): the frontend unit tests. Vitest writes a transform cache to a new temporary directory at
  every run and removes none; the script gives the run `frontend/.vitest-tmp` (not in version control), emptied before each run, so
  nothing gathers in the system's temporary directory. A run started another way (`npx vitest`, the watch mode) needs
  `TMPDIR=$PWD/.vitest-tmp` from `frontend/` for the same. One file: `npm run test -- src/lib/format.test.ts`.
- `make test-slow` (real models), `make e2e` / `make e2e-demo` (against the dev stack), `make egress` (offline check, idle host),
  `make audit` (advisories of both runtime locks and the npm lock; network), `make bench` (latency and memory, idle host).
- `make ci-release`: the release gate. `make ci` with the suite serial, the Postgres suite on a disposable local server,
  the tests of `infra/gcp`, `terraform test`, `make deploy-lint` and `make audit`; it fails at once when a tool is missing
  and builds no image (`docs/runbook.md`, "Release gate").
- `make purge` (`python -m app.cli retention purge`) applies the retention rule (cases past their retention date, and unsigned quick analyses older than
  `VERITAS_RETENTION_DAYS`; `DRY_RUN=1` lists each decision); `python -m app.cli store gc` frees objects no record references;
  `python -m app.cli audit verify` re-hashes the custody chain.
- `python -m app.cli models list` shows every model, its licence class and what the licence policy enables.
- The operator's commands, each `python -m app.cli …` from `backend/` (in the Compose stack
  `docker compose exec api python -m app.cli …`); `docs/runbook.md` has the section of each:
  - accounts: `principals create`, `list`, `set-password`, `set-role`, `disable`, `enable`, `api-key`;
  - backup: `backup create DIR`, `backup verify DIR`, `backup restore DIR`; `store verify` re-hashes every stored object;
  - keys: `keys adopt-derived`, once, at the upgrade of an instance that an earlier build ran in prod without a signing key;
  - what this server can run: `models plan` (each detector's memory need against the budget), `models footprint`;
  - the fusion models: `fusers status`, and the owner's `fusers review` and `fusers withdraw-review`;
  - PostgreSQL, as the owner: `db setup --app-role ROLE` (migrate, the application's role and its grants), `db grants`;
  - records: `custody export`, and for an exported package `verify PACKAGE.zip` and `reproduce PACKAGE.zip`, each with
    `--public-key` or `--expect-key-id`.
- `make settings-doc` writes `docs/settings.md` from the settings; `make sizing-doc` writes the memory tables of the
  runbook's "Sizing" from the capacity plan and its latency table from `docs/v2/bench/perf.json`. A test fails when
  either page is stale.
- Release record: [`CHANGELOG.md`](CHANGELOG.md) (the version is the constant of `backend/app/version.py`). Material of
  others, with its licences: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), written by `make notices-doc` from the
  licence files, the npm lock and the model manifest. The repository itself has no licence file yet.
- `make up` builds the images from the checkout's commit and starts the Compose stack (`docs/runbook.md`, "Deploy").
- Readiness of release 2.1.1 against its criteria, each with its evidence:
  [`docs/production-readiness.md`](docs/production-readiness.md). What the reviews found and the release does not fix:
  [`known-issues.md`](docs/v2/qa/prod-readiness-2026-09-29/known-issues.md). The reviews themselves are in
  `docs/v2/qa/prod-readiness-2026-09-29/`; the accepted dependency advisories in `docs/v2/qa/audit_exceptions.yaml`.
  The review of v0.3 is in `docs/history/`.

## Important limits

Documents support up to 120,000 extracted characters by default (`VERITAS_MAX_DOCUMENT_CHARS`). Larger documents fail before
scoring with a size explanation; split the original into smaller documents and analyse each separately. Previously truncated
extractions cannot be resumed into a new document result. Completed historical records remain available.

A `.txt` or `.md` file larger than 960,000 bytes (eight bytes for each character of that limit) is refused by its size, before
it is read. Every document is read, and its sentences are found, in separate processes that are stopped after 120 seconds
together (`VERITAS_DOCUMENT_DECODE_TIMEOUT_S`); the analysis then fails with "The document could not be read within the time
limit".

Detector outputs are statistical evidence, **not proof**. Newer generators may evade every detector, several lanes are not yet
calibrated, and SynthID is not detectable offline. See [`docs/limitations.md`](docs/limitations.md).
