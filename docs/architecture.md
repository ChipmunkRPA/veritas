# Architecture (v2 summary)

This page is the short map. The binding contract is [`docs/v2/ARCHITECTURE.md`](v2/ARCHITECTURE.md); decisions that override it
are in [`docs/v2/01-DECISIONS.md`](v2/01-DECISIONS.md), and requirements in [`docs/v2/PRD.md`](v2/PRD.md).

```
 browser (React, lib/api generated from OpenAPI)
   │  intake: browser SHA-256 pre-hash, acquisition fields, analysis profile
   ▼
 FastAPI  /api/v2/*  (v1 /api/* deprecated, through v2.1)
   │  auth (API keys, sessions, roles) · case scope · rate limits · body caps
   ├──► intake: stream + hash → content-addressed evidence store → custody event → job row
   │
 DB job queue (leases, heartbeat, poison guard)  ◄── worker (in-process, or `python -m app.engine.worker`)
   │
 staged DAG runner (engine/runner.py, engine/pipelines.py)
   ingest → probe → provenance → decode → faces / shots / tracks → condition → detect → localise | temporal | second_pass
          → fuse → synthesise → render → publish
   │
 evidence tables (signals, exhibits, assessments, findings, events) · write-once artefacts
   │
 reports (WeasyPrint, 4 profiles, Ed25519) · evidence packages · offline verify / reproduce
```

## Pipelines (ARCHITECTURE §3)

| Kind | Stages (`engine/pipelines.py`) |
|---|---|
| image | ingest, probe, provenance, decode, faces, condition, detect, localise, fuse, synthesise, render, publish |
| video | ingest, probe, provenance, decode, shots, tracks, faces, condition, detect, temporal, second_pass, fuse, synthesise, render, publish (`shots` / `tracks` in full and deep; `second_pass` in deep) |
| audio | ingest, probe, provenance, decode, condition, detect, temporal, fuse, synthesise, render, publish |
| document | ingest, decode, condition, detect, fuse, synthesise, render, publish |

- A failed detector never fails an analysis; only a critical stage (ingest, decode, publish, and probe for video and audio) can.
  Detectors that did not run contribute nothing and are listed under coverage.
- Stages publish progress events (SSE at `/api/v2/analyses/{id}/events`); persisted stages make a run resumable.
- Profiles `quick | full | deep` set which detectors and stages run and the sampling (`engine/profiles.py`); the instance default is
  full on accelerator hosts and quick on CPU hosts, and intake can choose per analysis.

## Detectors (ARCHITECTURE §2)

- One interface (`engine/detector.py`), a manifest (`app/models.yaml`: licence class, pinned revision and per-file sha256, profiles,
  hypotheses, admission state), a registry and a model manager that loads lazily and evicts least-recently-used models under the
  memory budget.
- The licence policy (`VERITAS_LICENCE_POLICY`, commercial preset by default, D-04) decides what may run. Missing weights end as
  `skipped: model not installed`, never as an error.
- Every detector has a model card (`/models/:id` in the UI); error rates say "not established" until the benchmark records them.

## Evidence model and fusion (ARCHITECTURE §4)

- **Signal v2** (observed value, reference, direction, calibrated or raw score, geometry and time), **exhibits** (heatmaps, masks,
  timelines, tables), **assessments** per hypothesis and lane, a **decision** and **findings** in plain language
  (`engine/synthesis/`).
- **Fusion with an artefact** (`engine/fusion/`): Stage 0 provenance overrides (trusted-list C2PA generative-AI declaration, trusted
  capture, self-declared markers with a floor); Stage 1 features; Stage 2 monotone logistic stacker; Stage 3 per-context Platt
  calibration with shrinkage; Stage 4 class-conditional split-conformal sets; exact additive contributions and a bootstrap 90 %
  interval; abstention in slices without validated performance.
- **Fallback** without an artefact for a kind: weighted log-odds of the manifest's legacy weights, labelled uncalibrated, with the
  verbal scale capped (D-31). Artefacts are fitted by `make calibrate` on the benchmark score cache.

## Storage, custody and review (ARCHITECTURE §4.7–4.8, §5.4)

- **Evidence store:** `data/objects/sha256/aa/bb/<sha>`, written once (0444); originals are re-hashed before reports, exports and
  downloads.
- **Database:** Alembic migrations; SQLite on one host, Postgres for several workers.
- **Cases:** implicit case per quick analysis; case roles owner, examiner, reviewer, viewer; non-members see not found.
- **Custody log:** append-only, hash-chained events (intake, analysis, review, reports, verification);
  `python -m app.cli audit verify`.
- **Review:** annotations and dispositions, conclusions, sign-off locks, peer review, withdraw, re-analyse and re-run with deltas.
  States: machine, in review, reviewed (examiner signed), approved (the peer review concurs, with or without changes), disputed
  (the peer review does not concur), superseded, withdrawn. Issuable: approved, or reviewed on a case that does not require a
  second review; a disputed revision is never issued, and the way forward is a new revision.
- **Review records:** conclusions and sign-offs are HMAC-signed. A record whose signature no longer matches its content is
  printed as "does not verify" in reports, and nothing is issued from the revision while one does not. Conclusion and sign-off
  custody events record how the actor authenticated (browser session or API key), and the sign-off appendix prints it.
- **Superseding a signed revision:** re-running or reanalysing a signed-off revision takes the rule of a withdrawal: a named
  examiner who signed it or wrote its conclusion, a case owner or an administrator, with a reason that the `rerun` custody event
  records. An API client cannot do it. Unsigned revisions are re-run by any case examiner.

## Reports and packages (ARCHITECTURE §6)

- WeasyPrint templates (`app/reports/templates/`), deterministic snapshots, profiles summary, standard, forensic and newsroom
  (redacted); draft or issue. A report is frozen with the `issuable` flag of its revision when it is requested: only an issuable
  report carries an id without DRAFT, the word "issued" and "Reviewed by …", and only it can be issued in a package. A report
  signed while a required second review is outstanding is a draft that says "Signed by <examiner>; second review outstanding";
  the peer reviewer's decision is printed beside the review line on every page and in the package README.
- A package carries `executive-summary.pdf` only when a summary report was frozen with the same review state, conclusion and
  sign-offs as its `report.pdf`. The standard and forensic reports state a custodian hash mismatch recorded at intake in the
  custody appendix.
- The v1 routes `GET /api/analyses/{id}/report.pdf` and `export.json` answer 409 for a withdrawn or superseded revision, and the
  v1 PDF prints the review status line under its title.
- Ed25519 signatures; evidence packages (`POST /api/v2/analyses/{id}/packages`) verify offline with
  `python -m app.cli verify <zip> --public-key <PEM>` (or `--expect-key-id <id>`), and `python -m app.cli reproduce <zip>` with
  the same option re-runs the packaged analysis and diffs the raw scores. The instance key is pinned by the reader: without it
  the verifier answers "SELF-CONSISTENT: signer not authenticated" (exit 3), since a package carries its own key
  (`app/reports/verify.py`; runbook "Verifying a package, and the instance key").

## Frontend (ARCHITECTURE §7)

React with generated API types (`make api-types`), TanStack Query, design tokens in `frontend/src/index.css` and the contract in
`frontend/design/`. Routes: cases dashboard `/`, case workspace `/cases/:id`, intake `/upload` and `/cases/:id/upload`, batches
`/batches/:id`, analysis workspace `/analyses/:id` (summary, findings, exhibit, how reached, evidence, provenance, methods,
appendix, revision, review), evidence viewer `/analyses/:id/evidence`, model catalogue `/models`, design gallery `/design`.

## Benchmark (ARCHITECTURE §9.5)

Veritas-Bench: `make vbench-fetch`, `vbench-split`, `vbench-perturb`, `vbench-core` (vb-1.0-core, D-39), `vbench-score`, `calibrate`,
`vbench-report`, `vbench-gate`. Goals and pins: `docs/v2/bench/goals.yaml`; results and shortfalls: `docs/evaluation.md`.

## Operations

Deployment, offline defaults, configuration, sizing and troubleshooting: [`docs/runbook.md`](runbook.md). Limitations:
[`docs/limitations.md`](limitations.md).
