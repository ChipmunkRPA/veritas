# Runbook

## Deploy

| Mode | Command | Notes |
|---|---|---|
| Dev | `make setup && make dev-api dev-web` (API :8160, UI :5230; PID files under `backend/data/jobs/`), stop with `make stop` | uv venv, Python 3.12, ffmpeg required; ExifTool ≥ 12.24 for the metadata rules (`brew install exiftool`, D-07; Pillow fallback without it); Pango for WeasyPrint reports (`brew install pango`) |
| Docker (CPU) | `VERITAS_API_KEY=… make up` → `models-init`, then `api` + `worker` + `web` (nginx) on `127.0.0.1:$VERITAS_WEB_PORT` (default 8080; plain HTTP, this host only: see "TLS"); a browser on that host signs in at `http://localhost:$VERITAS_WEB_PORT`, not at `127.0.0.1`, and from elsewhere over HTTPS | `models-init` fetches the pinned weights of the enabled detectors of `$VERITAS_MODELS_PROFILE` (default `full`) into the `veritas-models` volume once |
| Docker (NVIDIA) | `VERITAS_VARIANT=cuda VERITAS_API_KEY=… docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d` | CUDA 12.6 runtime image with cu126 torch wheels; only `worker` reserves the GPU (NVIDIA Container Toolkit on the host) |
| Postgres | `VERITAS_PG_PASSWORD=… VERITAS_PG_APP_PASSWORD=… VERITAS_API_KEY=… docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build -d` | SQLite on the shared `veritas-data` volume is the default (one host). The file adds the server (`postgres:16-alpine`, checked with `pg_isready`), the one-shot `db-setup` and the connection of `api` and `worker` as the role `veritas_app`, which owns nothing: see "Postgres: two roles". Both passwords are required. An earlier set-up (`--profile postgres` with the URL in an override file) no longer finds a `postgres` service: use the file, and remove the URL from the override. A bare `postgresql://` URL is served by the same driver (psycopg 3); another driver that is not installed is refused at start with the accepted form named. Each process tests a pooled connection before it uses it, so a request after a database restart reconnects. `VERITAS_DB_POOL_SIZE` (5) and `VERITAS_DB_MAX_OVERFLOW` (10) size the pool of each process; `VERITAS_DB_APPLICATION_NAME` names its sessions in `pg_stat_activity` (Compose: `veritas-api`, `veritas-worker`) |
| Air-gapped | on a connected host `VERITAS_BAKE_MODELS=full make up` (or `docker build --platform linux/amd64 --build-arg VERITAS_BUILD_COMMIT=$(git rev-parse HEAD) --build-arg BAKE_MODELS=full ./backend`), `docker save`, then `docker load` on the target | the build fetches and verifies the weights into `/models`; on the target `models-init` finds every file installed and only re-hashes |
| Single process, no container | build the frontend (`make build`) and start `uvicorn` from `backend/` ("A prod start without a container", below); the API serves `frontend/dist` at `/` when present | in-process worker (`VERITAS_RUN_WORKER=true`, the default). No image holds both the page and the API: in Docker they are the `web` and `api` containers |

### Supplemental local checks

These options use pinned local weights and affect new analyses. For local development,
set them in `backend/.env` and restart the API and worker while the queue is idle.
Check capacity with `python -m app.cli models plan` from the backend directory.

| Setting | Where it runs | What the evidence shows |
|---|---|---|
| `VERITAS_IMAGE_REGION_MODE=candidate` | Full or Deep, supported upright RGB PNGs. The region check runs on CPU. | Source crops, masks and the exact area flagged by either or both eligible region checks. The sensitive check found 45/59 known edited areas versus 10/59, with flags on 13/59 untouched inputs versus 3/59 on that panel. [Confirmation](image-native-confirmation-v46-result.md) · [App evidence](native-regional-evidence-replay-v51-result.md) |
| `VERITAS_VIDEO_WHOLE_FACE_REVIEW=true` | Quick, Full or Deep, supported videos with a verified native frame clock. | A face-swap cue across six sampled frames with representative original frames and crops. The paired check flagged 45/48 whole-clip swaps versus 26/48, with 0/24 original flags in both arms on that panel. A separate transfer missed the outside publisher's swap and flagged two variants of one camera recording. [Confirmation](video-face-pe-confirmation-2026-11-11.md) · [Outside transfer](video-face-pe-publisher-2026-11-11.md) |
| `VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW=true` | Eligible plain-text articles with the local candidate text mode selected. | Additional readings of up to two complete paragraphs without their surrounding context. The same classifier supplies both readings, and some genres produced frequent human-writing flags in evaluation. [Replication](writing-paragraph-replication-2026-11-11.md) |

These checks have zero overall confidence weight. The image evidence does not attribute
an edit to AI, and six video samples cannot establish the boundaries of a short swap.
The rollback values are `VERITAS_IMAGE_REGION_MODE=legacy`,
`VERITAS_VIDEO_WHOLE_FACE_REVIEW=false` and `VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW=false`.

**A prod start without a container.** The command line of the row "Single process, no container", from `backend/`,
with the flags of the image's own command and the loopback address (put a TLS proxy in front: "TLS"):

```
VERITAS_ENVIRONMENT=prod VERITAS_API_KEY=<the bootstrap key> VERITAS_DATA_DIR=/srv/veritas/data \
  .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8160 --workers 1 --no-access-log --no-proxy-headers \
  --timeout-graceful-shutdown 30
```

One process: it runs the jobs itself (`--workers 1`, as the image does; a second worker is an experiment, "Sizing"). `--no-access-log`
because download and stream links carry a token in the query; `--no-proxy-headers` because the client address comes
from `VERITAS_TRUST_PROXY` alone. Prod is offline by default: fetch the weights first
(`VERITAS_OFFLINE=false python -m app.cli models fetch --profile full`), and keep the key in the service's environment
file, not on a command line others can read. SIGTERM stops it; a job that still runs after 2 s is given back
("Stopping"). Run once as written (port 8169, the fake detectors): `/api/health` 200, `/api/ready` with the key
`ready`, the data directory created with mode 0700, JSON logs, stopped by SIGTERM.

**Docker Compose 2.24.4 or later.** The base file takes `env_file` entries with `required: false` (Compose 2.24.0, by
Docker's documentation) and the TLS file replaces the published ports of `web` with `!override` (2.24.4); the Postgres
file's `!reset` is older than both (the documentation names no version for it; it works in 2.23.3). An older Compose is
not always refused: 2.23.3 stops at `env_file` ("must be a
string"), but 2.24.0 to 2.24.3 read all three files without an error and ignore `!override`, so the TLS stack would
publish port 8080 beside 443 and 80. `docker compose version` names the installed one, and `make deploy-lint` compares
it first and says so when it is older (`backend/scripts/compose_version.py`). Run with the release binaries 2.23.3,
2.24.0, 2.24.3 and 2.24.4 on these files (`config` only; `docs/v2/qa/n17a-ops-rereview/compose-versions.log`); the
stack itself was run by the supervisor on Docker Desktop 29.4.0 alone.

**The commit of a build** (D-58). An image is built from a named commit. `make up` passes
`VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)`; without a full commit the build of the backend image fails at its dependency
check (`scripts/runtime_contract.py`, stage `identity`). Before any other command of this page that has `--build`, run
`export VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)` in that shell; from a release archive, give the commit the archive was
made from. The running instance states its build: `/api/ready` with a key (`version`, `build_commit`), the metric
`veritas_build_info`, every result manifest, and the label `org.opencontainers.image.revision` of both images. The
version is the constant of `backend/app/version.py`; no setting changes it.

**Services** (`docker-compose.yml`, ARCHITECTURE §8.4): one backend image (`backend/Dockerfile`, `ARG VARIANT=cpu|cuda`, non-root
uid and gid 10001, `VERITAS_ENVIRONMENT=prod`) runs three services.
- `models-init`: `python -m app.cli models fetch --profile full --no-record` with `VERITAS_OFFLINE=false`, the only service
  that downloads; it exits 0 when every file verifies. It mounts the models volume alone and is given neither the API key
  nor the data volume. The record of verified files is signed with a key in the data volume, so this service writes none
  (`--no-record`): it hashes every installed file at each `up`, and `api` and `worker` hash a file themselves before they
  first load it and keep the record. Its output says `FETCHED` for a model it downloaded and `INSTALLED` for one whose
  files were there and were only hashed (log events `weights.fetched` and `weights.verified`), so the log of a start
  tells a download from a check.
- `api`: `VERITAS_RUN_WORKER=false`, reached only through nginx. Its health check asks `/api/health` every 10 s.
- **Order and restarts.** `worker` and `web` start once `api` is healthy, which is after it has migrated the database.
  `api`, `worker`, `web` and `postgres` have `restart: unless-stopped`: they come back after a host reboot, a Docker
  restart or a kill for memory; `models-init` runs once. The worker writes a health file every 2 s
  (`--health-file /tmp/veritas-worker-health.json`) and its health check reads it (`python -m app.engine.worker_health`):
  healthy means its threads live and each claimed from the queue or renewed a lease within 45 s. `docker compose ps`
  shows both states. The container logs of every service are rotated: five files of 10 MB.
- **Confinement.** `models-init`, `api` and `worker` run with a read-only root filesystem, without capabilities and with
  `no-new-privileges`, as uid 10001. They write the two volumes and `/tmp`, which is memory (1 GB at most, counted in the
  service's limit: the worker's health file and the caches of fonts and plots). The code under `/srv` belongs to root.
  `web` is nginx as uid 101 on port 8080 of its container, without capabilities; its health check asks nginx for the shell
  and for `/api/health` through it.
- **Temporary files** of `api` and `worker` (upload spools, batch spools, report work files) go to `/data/tmp` on the data
  volume (`TMPDIR`), beside the working copies of running analyses; the process makes the directory at its start. Size the
  data volume for them ("Sizing"). What a killed process left there stays until it is removed by hand; only the working
  copies of analyses are swept (`VERITAS_TMP_SWEEP_MIN_AGE_S`).
- **Memory limits.** Each service has one, set in the environment of `docker compose` (or in `.env` beside the file):

  | Service | Variable | Default | Note |
  |---|---|---|---|
  | `worker` | `VERITAS_WORKER_MEM_LIMIT` | `28g` | the models load here; the model memory budget follows it ("Host" below, "Sizing") |
  | `api` | `VERITAS_API_MEM_LIMIT` | `8g` | loads no model |
  | `models-init` | `VERITAS_FETCH_MEM_LIMIT` | `4g` | not below 4 GiB for the full profile |
  | `web` | `VERITAS_WEB_MEM_LIMIT` | `512m` | |

  A service that exceeds its limit is killed by the kernel and started again (`docker inspect` shows `OOMKilled`). The
  limits are caps, not measurements: none of them has been observed under load in a container on the build host.
- **Host.** The reference host of these defaults is a CPU host with 32 GB of memory (D-74). The worker's limit of
  28 GiB gives it a model memory budget of 14,336 MB under `auto`, and the `full` profile, which the file selects,
  needs 12,977 MB ("Sizing", "Memory per detector and per profile"): every check of the profile can run.
  - **A smaller CPU host.** The limit is a cap and reserves nothing; a limit above the machine's memory changes
    nothing, because the budget is half of the smaller of the two. The worker names what it drops at its start
    (`capacity.detector_cannot_run`), and every result that such a check applies to says that the check could not run
    on this server. From the plan: a CPU host with 16 GiB of memory (budget 8,192 MB) cannot run `txt.binoculars`;
    one with 8 GiB of memory (budget 4,096 MB) cannot run `img.dfdc_b7`, `vid.frames.dfdc_b7`, `txt.desklib`,
    `txt.desklib_ft_v2`, `txt.binoculars`.
  - **An accelerator host** (`docker-compose.gpu.yml`) is sized from the table of "Sizing": the budget is 0.85 of the
    accelerator's memory, the `full` profile needs 7,089 MB there, and an accelerator with 8,340 MB or more gives it.
    The worker's limit still caps the process, not that budget.
  - **Disk.** The backend image (CPU variant) is 3.69 GB and the web image 78.5 MB, and the models volume of the
    `full` profile holds 11.5 GB (the supervisor's build and fetch in a container; "Sizing"). The data volume holds
    the database, every original, the exhibits, the reports and the temporary files (`/data/tmp`): size it for the
    files that are kept ("Sizing", "Data & retention").
  - **Before going live,** ask the host what it can run: `docker compose run --rm worker python -m app.cli models plan`
    prints each check of the default profile that loads a model, with its need under this host's budget, and exits
    with 1 when one cannot run. It loads no model.
  - **Where a command's memory counts.** `docker compose exec` runs the command inside the running container of the
    service, so it shares that service's memory limit with the service; `docker compose run --rm` starts a second
    container beside it, with a limit of its own. A command run with `exec` can therefore push the service over its
    limit, and one run with `run --rm` needs memory on the host beside the running stack.
- `worker`: `python -m app.engine.worker`; it leases jobs from the database, heartbeats every 10 s, and a lease whose worker
  stops is re-queued after 60 s. A worker that is still running when it finds its lease gone (a stalled host, a database
  outage) stops and writes nothing more for that analysis: no status, no stage or detector run, no `analysis_finished` event,
  and it leaves the working copy of the original for the worker that took the job over. That worker finishes the analysis,
  and the analysis counts one run per detector whatever the interrupted attempt left behind. Webhooks are delivered at least
  once: when a worker stopped between sending `analysis.completed` and completing its job, the next owner sends the event
  again with a new event id (same `analysis_id`).
- **Stopping.** `docker compose stop` (and `down`, and an upgrade) sends SIGTERM, and each service has 40 s
  (`stop_grace_period`) before Docker kills it.
  - The worker stops claiming and waits 30 s for the job it runs (`--stop-timeout`). A job that outlasts that is given
    back to the queue without counting the attempt, with one `worker.job_released` line, and its analysis reads `queued`
    again, not `running`; the next worker claims it at once and runs it from the beginning. Until then the stage list
    shows what the interrupted run had reached.
  - The API ends its open event streams at their next poll (the page reconnects by itself) and closes whatever is still
    open after 30 s (`--timeout-graceful-shutdown`).
  - In a single process (`VERITAS_RUN_WORKER=true`) the API's stop gives the running job 2 s and then releases it the
    same way.
  - A worker that is killed (out of memory, `kill -9`, a host that loses power) releases nothing: its lease runs out
    after 60 s (`VERITAS_JOB_LEASE_S`), another worker takes the job, and that does count as an attempt (see
    "Troubleshooting", "interrupted 3 times").

`web` is nginx: the SPA and `/api/*`; the SSE route `/api/v2/analyses/{id}/events` is proxied with `proxy_buffering off`, and
`/api/v2/batches` accepts 20 GB.

**Python runtime pins.** Backend Compose builds target Linux amd64. CPU/cu126 dependencies are exact and hash-locked in
[`backend/requirements`](../backend/requirements/README.md); the final build verifies package and native operator identity.
These checks do not validate Linux inference or make the mutable OS/base-image packages reproducible.

**Build status.** The gates of the build host do not start Docker: `make deploy-lint` validates the compose files with
`docker compose config`, `tests/test_deployment.py` pins the structure above, and `make ci-release` builds no image. The
images were built and run in the supervisor's checks, recorded in
`docs/v2/qa/prod-readiness-2026-09-29/supervisor-checks.md`:

- **Last built commit: a876a43a, the release commit** (checks 50 and 51, 2026-10-11).
  `docker compose build` exit 0 in 133 s, the three bases by digest, the web image on its base
  `nginxinc/nginx-unprivileged:1.30-alpine` as uid 101; the backend
  image is `linux/amd64` with user uid 10001, gid 10001, and the smoke of `infra/gcp/README.md` with
  `--require-image-runtime` passes seven of seven checks with `--user 10001:10001` and without it. On that commit:
  `up` from an empty data volume on SQLite (37 s, with the models volume of check 13), five analyses asked as `full`,
  20 reports and 5 packages, the journey of check 4 (22 of 22), stop and start, the TLS file (`nginx.tls.conf` in the
  web image on nginx 1.30), the Compose backup and restore of this runbook as written, steps 1 to 6 of "The sequence of
  an upgrade" from the data that 5f0cb16 wrote, and the Postgres file from empty volumes with its restore as written.
  Then step 7 of that sequence (check 51): the backup of step 3 restored with the image of a876a43a, and the images of
  5f0cb16 built again from its source, started on it and removed after the run; the images of a876a43a were built again
  from the build cache, so they are the images kept.
- **Before it:** the images of 5f0cb16 (check 13, after N-17a: every weight fetched from the network into an empty models
  volume, 13 models, 11.89 GB), of ff6c310 (check 10, 2026-10-02: both images for `linux/amd64` from bases pinned by
  digest, the smoke seven of seven with `--user 10001:10001`), and the stack of 4a0b93c on SQLite, with the TLS file and
  with the Postgres file (check 9).
- **Never built:** the CUDA variant (its base image was resolved by digest, not pulled). **Not done for any image:** a
  scan, an SBOM, a run on a GPU or in the cloud, a run on a Linux host (the stacks ran on Docker Desktop on macOS) and
  on a native `amd64` host (the backend image ran under emulation).

First start: the models load lazily per stage and are evicted under the memory budget (D-21). `GET /api/health` is the
liveness probe and `GET /api/ready` the readiness probe: what each asks, what it answers and to whom is in "Monitoring".
There is no warm-up: the first analysis of each kind after a start waits for its models to load, and readiness does not
wait for models. (`VERITAS_WARMUP_ON_START`, `make warmup` and `POST /api/models/warmup` loaded the model set of v0.3,
which a v2 instance does not install; they were removed. A variable that is still set is ignored.)

**Calibrated fusers.** The three served fusion artefacts (`fuser-image-2.1.1`, `fuser-video-2.1.1`, `fuser-audio-2.1.1`, with
their bootstrap arrays) ship with the source in `backend/app/engine/fusion/release/` and are inside the backend image; a hash
manifest lies beside them.
- **Where the loader looks.** `VERITAS_FUSER_DIR` when it is set, and only there. Otherwise `<data dir>/fusion` and the
  packaged directory together: per kind the newest version serves (or the version named in `VERITAS_FUSER_VERSIONS`). A fresh
  data volume therefore serves the packaged artefacts, and so does an instance that kept an older artefact in its data
  directory. Where both directories hold one version, the packaged file serves; if the other file has different bytes, the
  start log (`fusion.shadowed_artefact`) and `fusers status` name it. When the chosen artefact cannot serve, the uncalibrated
  fallback runs, never an older artefact.
- **What serves.** `python -m app.cli fusers status [--profile full]` prints, per kind, the version, the start of the SHA-256,
  the directory, the review state and whether the artefact or the uncalibrated fallback serves, with the reason. It exits 1 when
  the packaged files differ from their manifest. `GET /api/ready`, asked with an API key, carries the same per kind: `fusion`
  (`artefact` or `fallback`) and `fusers` (version, start of the SHA-256, `review`: `reviewed` or `pending`), as the worker
  that serves the queue reports them.
- **Profile.** The image and video fusers need detectors that `quick` does not run, and `VERITAS_DEFAULT_PROFILE=auto` is `quick`
  on CPU. The Compose file therefore sets `VERITAS_DEFAULT_PROFILE=full` for `api` and `worker` (override it in the
  environment). `quick` stays selectable per analysis and keeps the uncalibrated fallback with its caps.
- **A start on the fallback.** In `prod`, the API and the worker log one warning per kind that has an artefact file and still
  runs on the fallback (`fusion.fallback_at_start`, with the reason: weights not installed, a detector not enabled, a profile
  that does not run it). `/api/ready` answers 200 in that state; read its `fusion` field (with an API key).
- **Review.** The artefacts are pending human review until the owner records one: results and reports say "Fusion model pending
  human review". What is signed and the command are in `docs/v2/fuser-review-2.1.1.md`
  (`python -m app.cli fusers review --modality … --reviewer "<name>" --statement "<text>"`). The record is a file
  `fuser-<kind>-<version>.review.json` in the fuser directory; the artefact is never edited. Carry it to another deployment by
  copying that file into its fuser directory.
- **A review is in the custody log.** `fusers review` writes the record and a `fuser_review_recorded` event together: the
  kind, version and SHA-256 of the artefact, the reviewer, the review's date and the SHA-256 of the statement (its text
  without surrounding white space, as UTF-8; the text itself stays in the record). A second review of the same artefact is
  recorded as replacing the first. `python -m app.cli fusers withdraw-review --modality <kind> [--reason "<text>"]` removes
  the record and writes `fuser_review_withdrawn`. When the event cannot be written, either command exits 1 and the record is
  as it was. A record that was copied in, came with a release, or was replaced or deleted by hand is recorded by the first
  analysis fused after it (source `found on use`), before that result names the reviewer; `withdraw-review` records a
  deletion by hand at once. These events belong to the instance, not to a case: an administrator reads them with
  `GET /api/v2/audit?action=fuser_review_recorded` and in `GET /api/v2/audit/export`. A record inside the packaged release
  directory is not removed by the command.

## Offline operation (PL-01)

- **Default.** `VERITAS_OFFLINE` unset means true in `prod` and false in `dev`. Offline sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` before the hub libraries load, and every weight fetch raises before any network I/O (`models fetch`
  exits 2). A detector whose weights are missing ends as `skipped: model not installed`, and `/api/v2/models` shows
  `installed: false`.
- **Installing weights.** `VERITAS_OFFLINE=false python -m app.cli models fetch --profile full` (or `--enabled`, `--all`, or model
  ids; `--dry-run` prints the plan). Then `python -m app.cli models verify --installed`. HF snapshots go to `$HF_HOME/hub` at the
  pinned revision. URL, Drive and torch.hub files go to `$VERITAS_MODELS_DIR/weights` (images: `/models/weights`; default
  `$VERITAS_DATA_DIR/weights`). Every file is streamed with an incremental SHA-256 and published only when it matches its pin.
- **Local models: the fine-tuned text classifier.** `local.desklib_ft_v2` (7 files, 1.7 GB, hashes pinned in `models.yaml`) is
  produced by `scripts/finetune_text_detector.py`, is not tracked in git and is never downloaded. It exists on the build host
  only (`backend/data/weights/desklib-ft-v2`); where it is kept for deployments is the owner's choice. To install it:
  1. `python -m app.cli models fetch --profile full --dry-run` prints a `LOCAL` line with the target directory:
     `$VERITAS_MODELS_DIR/weights/desklib-ft-v2` (images: `/models/weights/desklib-ft-v2`, in the `veritas-models` volume;
     without `VERITAS_MODELS_DIR`: `$VERITAS_DATA_DIR/weights/desklib-ft-v2`).
  2. Copy the directory there. For Compose that is the `veritas-models` volume, which the API and the worker share; the files
     must be readable by the image's user (uid 10001).
  3. `python -m app.cli models verify --installed` re-hashes the seven files against their pins and prints
     `VERIFIED  local.desklib_ft_v2`; a wrong or partial copy is an `ERROR` line and exit 1.
  4. `GET /api/ready` with an API key then reads `text_classifier.detector: txt.desklib_ft_v2`, `fine_tune: installed`.
  Without it the base classifier (`txt.desklib`) serves documents: `/api/ready` reads `fine_tune: not_installed`, and every
  document result says that the base classifier scored it (`docs/limitations.md`, "Documents"). An install that kept the
  directory under the data directory while `VERITAS_MODELS_DIR` points elsewhere is still found there.
- **The only other outbound path** is webhooks (`VERITAS_WEBHOOKS_ENABLED`, off by default; API-02). The C2PA reader never fetches
  remote manifests or OCSP.
- **Egress check.** `make egress` (heavy; idle host) runs `backend/scripts/egress_check.py` in a fresh process with prod defaults. It
  analyses one image, video, audio file and document with the installed models under the egress monitor (patched socket plus
  psutil; non-loopback destinations are refused and recorded), and passes only with 0 recorded connections and every analysis
  `done`. The model-free variant runs in every `make test` (`tests/test_egress.py`), and the default suite blocks non-loopback
  sockets throughout.

## First start on a server

### Keys

The image runs with `VERITAS_ENVIRONMENT=prod`, and prod refuses to start with a key that can be guessed.

- `VERITAS_API_KEY` is the bootstrap administrator key: at least 32 characters and not a placeholder. Generate it:
  `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`. Keep it in a secret store; clients send it as `X-API-Key`.
- The signing key (resource tokens, the signatures of conclusions and sign-offs, webhook secrets) is never derived from the
  API key in prod. Unset, it is the file `data/keys/signing.key` on the data volume (0600): back it up with the data. To
  hold it in a secret store instead, set `VERITAS_SIGNING_KEY` (at least 32 characters, not the API key) before the first
  start. Changing it afterwards invalidates stored review signatures and open download links.
- **The key files appear at their first use, not at the start.** After a first start `data/keys` does not exist. Each
  file is generated when it is first needed: `api_key_pepper.key` with the first API key of a principal,
  `instance_ed25519.pem` when the instance's public key is first asked for or a report or package is first signed, and
  `signing.key` with the first link the page asks for (it asks for one when it opens an analysis), conclusion, sign-off
  or webhook. A backup taken before then says "none yet" for a key that is not there: take one once all three exist.
- A refused key is named in the error, never printed.
- Dev (`VERITAS_ENVIRONMENT=dev`) keeps short keys and the derived signing key.

### The data directory

Outside a container (the single-process mode, a worker on a host): a data directory that the service creates
(`VERITAS_DATA_DIR`) has mode 0700, so the database and every evidence file under it are readable by the service's user
alone. A directory that exists is left as it is. One that an earlier version created has mode 0755, with `veritas.db`
0644: on a host with other users, run `chmod 700 "$VERITAS_DATA_DIR"` once. The keys were always 0600 and the stored
objects 0444. The Compose volumes are not affected: one user (uid 10001) runs in the containers that mount them.

### TLS

Browser sign-in needs HTTPS: the session cookies are `Secure`, and a browser drops them on plain HTTP from every host but
`localhost`. API keys, evidence and reports are readable on the wire without it. `docker-compose.yml` therefore publishes
plain HTTP on `127.0.0.1` only.

**With the bundled nginx** (`docker-compose.tls.yml`, `frontend/nginx.tls.conf`):

```
VERITAS_TLS_CERT=/etc/veritas/tls/fullchain.pem VERITAS_TLS_KEY=/etc/veritas/tls/privkey.pem VERITAS_API_KEY=… \
  docker compose -f docker-compose.yml -f docker-compose.tls.yml up --build -d
```

- The certificate file holds the server certificate followed by its intermediates; the key file is its private key (PEM).
  nginx runs as uid 101 from its first process, so both files must be readable by that user and the key by no one else:
  `install -o 101 -g 101 -m 0400 privkey.pem /etc/veritas/tls/privkey.pem`. A key that only root can read stops nginx at
  its start (`docker compose logs web` shows "Permission denied"). Copy the files again after each renewal.
- nginx listens on 443 (`VERITAS_TLS_PORT`, `VERITAS_TLS_BIND`), redirects port 80 (`VERITAS_HTTP_PORT`) to
  `https://<host>/`, and sends `Strict-Transport-Security` for one year. Inside the container the two listeners are 8443
  and 8080. With HTTPS on another port than 443 the redirect is wrong: do not expose port 80 then.
- After a renewal: `docker compose -f docker-compose.yml -f docker-compose.tls.yml exec web nginx -s reload`.
- The client address the API uses for its limits is the one nginx saw (`VERITAS_TRUST_PROXY=true`, one hop; set in
  `docker-compose.yml`).

**With your own HTTPS proxy or load balancer** in front of the stack: leave `docker-compose.tls.yml` out, let the proxy
forward to the stack's HTTP port (`VERITAS_WEB_BIND=<address the proxy reaches>`, default `127.0.0.1`, and
`VERITAS_WEB_PORT`), and have it send HSTS. Two proxies now stand before the API, so tell it which entry of
`X-Forwarded-For` is the client:

- the proxy appends the client address to `X-Forwarded-For` (nginx: `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`),
  and `VERITAS_TRUSTED_PROXY_HOPS=2` in the environment of `api` (a `docker-compose.override.yml`); or
- the bundled nginx takes the client address from the proxy (`set_real_ip_from <proxy address>;` and
  `real_ip_header X-Forwarded-For;` in `frontend/nginx.conf`), and the hop count stays 1.

Without one of the two every user shares one address: ten sign-in attempts a minute, 300 requests a minute and 20 refused
keys a minute for all of them together. The bundled nginx must be reachable from the proxy only, or a client can write its
own address into the header. A tunnel that forwards TCP (no header) cannot be told apart from its users: all share its
address. The API itself never reads `X-Forwarded-For` unless `VERITAS_TRUST_PROXY=true` (uvicorn runs with
`--no-proxy-headers`).

**Limits on credentials.** A client address whose API keys or sessions were refused 20 times within a minute
(`VERITAS_CREDENTIAL_FAILURES_PER_MINUTE`) is answered 429 for every request that carries a key or a session, right or
wrong, until the count recovers (one attempt every three seconds). Sign-in is limited to 10 attempts a minute per client
address (`VERITAS_LOGIN_RATE_PER_MINUTE`).

**Refused before the body.** A request that will be refused is refused before its body is read, uploads included: the
request limit (429), the credentials (401, or 403 for a session without its CSRF token), the role the route needs (403)
and membership of the case in its path (404) are decided first, on every route but the sign-in and the sign-out. A client
without valid credentials therefore cannot make the API spool a file: nothing of a 500 MB upload or a 20 GB batch is
written to the temporary directory before the 401. A caller who is allowed is still bounded only by the size limits
of each kind and of a batch; there is no limit on concurrent uploads per principal and no byte quota, so a proxy in
front of an exposed instance should limit connections on `/api/analyze/`, `/api/v2/cases/…/items` and `/api/v2/batches`.

**Reading a document.** The text of a document is extracted in a child process of the worker (the same interpreter,
started with `-I`), and its sentences are found in a second one (together under two tenths of a second for an ordinary
document). Both are killed when `VERITAS_DOCUMENT_DECODE_TIMEOUT_S` (120 s, for reading and segmenting together) has
passed: a file that a parser or a pattern cannot finish, or a text whose sentences are not found in that time (one
paragraph of 120,000 characters made of two-character sentences), fails its own analysis ("The document could not be
read within the time limit of 120 s") and does not hold the worker, its heartbeat or, in a single process, the API. A
`.txt` or `.md` file above eight bytes for each character of `VERITAS_MAX_DOCUMENT_CHARS` (960,000 bytes) is refused by
its size before it is read ("Document is … bytes"); PDF, DOCX and HTML are bounded by the upload limit and by the time
limit. A document over `VERITAS_MAX_DOCUMENT_CHARS` is refused after it is read and is not segmented. A reader or a
segmenter that ends without an answer (a crash of a parser) fails the analysis as an internal error, and the worker log
has `document.extract_child_failed` or `document.segment_child_failed` with the exit code and the last of its error
output.

**Text a request may carry.** A NUL character (U+0000) is refused on both database engines, before any statement: in a
path or a query (`%00`) the address is not found (404, for every caller, with or without credentials), in a string of a
JSON body or a form field the request is invalid (422 "Text must not hold a NUL character"), in a cursor the cursor is
invalid (400), and a sign-in under such a name is a failed sign-in (401) that leaves no event. A file name loses the
character. A value the database itself refuses (on PostgreSQL, text or a number it cannot hold) is answered 422 "A value
in the request cannot be stored or compared" and logged as `api.data_error` with the route, the driver's error class and
its SQLSTATE, without the statement or the value; a run of them from one client is a probe, a single one from the
workspace is a defect to report. A note, a reason or a rationale is stored without characters that render as nothing (the
zero-width space, the word joiner, the byte order mark, the soft hyphen), and one in which nothing can be seen is refused
as empty ("A disputed status needs a note").

**Model cards and engineering records.** A model card is read by every signed-in role. Its "Engineering record" (the
build's own notes on the detector and its admission record, shown as a document) is on an administrator's page only, and
`GET /api/v2/models/{id}/records/admission:{id}` answers administrators only (403 for every other role). No other
document of the repository is served: the QA and review records under `docs/v2/qa/` have no address in the API. An image
built without the `docs/` tree has no admission records, and the cards say "not written".

**A session that has ended.** A browser keeps its session cookie after the session ended on the server (it expired after
`VERITAS_SESSION_TTL_S`, the account was disabled, or its password or role was changed). The cookie cannot be removed by
the page (it is `HttpOnly`), so the server removes it: a request under such a cookie is resolved as if it carried none (a
download link still opens, a single-user instance still answers as the local examiner, everything else is answered 401
"The session is not valid; sign in again"), and its response deletes both cookies. Each such request counts as a refused
credential (above); a browser sends at most the requests of one page load before it obeys the deletion.
`POST /api/v2/auth/logout` needs no credentials: it ends the session its cookie names, when that session is open and the
request carries its CSRF token, and deletes both cookies in every other case too (204). It answers 403 for an open
session without the token, and 429 or 500 when the session could not be looked up or ended; the cookies are then kept
and the person is still signed in.

**Sign-out and several tabs.** A sign-out that the server did not complete is said at the top of the workspace ("Sign-out
failed. You are still signed in as …", with the server's reason, "Try again" and "Dismiss"); the page is not reloaded and
the session stays open. A completed sign-out, and a sign-in, reach the other tabs of the same browser: each loads its
page again, so a tab never keeps the workspace of the person who signed out, and a form or a signer line never belongs
to another session than the cookie's. Text typed and not saved in such a tab is lost with the reload.

**A conclusion on an instance without sign-in.** A single-user instance (no API key, the loopback only) and a browser
that holds the bootstrap key or an `api_client` key are let in without a named account. They read everything and record
statuses and notes, but a conclusion, a sign-off, a withdrawal and the re-run of a signed revision carry a named person
and are refused to them by the API. The review panel therefore shows "Sign in" where the conclusion form would be, and
the re-run dialog of a signed revision says the same. "Sign in" opens `/sign-in`, which returns to the page it came from;
a stored key is replaced by the session. The account is one of "First administrator" below (on a single-user instance
without Docker: `cd backend && .venv/bin/python -m app.cli principals create --username … --role administrator`). An
administrator account concludes in every case; an account with another role must be a member of the case first
(`PUT /api/v2/cases/{id}/members/{principal}`, or the Members panel of the case). Signing out returns a single-user
instance to the local examiner.

### First administrator

The bootstrap key can do everything through the API, but the sign-in page needs a named principal with a password. Create
the first one on the server:

```
docker compose exec api python -m app.cli principals create --username jane.doe --role administrator --display-name "Jane Doe"
docker compose exec api python -m app.cli principals list
```

`create` prints a generated password once; it is stored only as a hash. The administrator signs in with it, in a browser
on the server's host at `http://localhost:<port>` (`VERITAS_WEB_PORT`; at `http://127.0.0.1:<port>` the browser keeps no
session, since the cookies are `Secure`: "TLS") and from any other host over HTTPS, and replaces it at once (the account menu at the foot of the sidebar, "Change password…") and creates the other principals
(`POST /api/v2/principals`, or the same command: roles `administrator`, `examiner`, `reviewer`, `viewer`, and `api_client`,
which gets an API key instead of a password). Both commands record nothing secret; the custody log names the principal as
created by the system from the command line. Without Docker: `cd backend && .venv/bin/python -m app.cli principals …` with
the service's environment.

### Accounts

Every command below changes one existing principal, named as `principals list` prints it, through the functions the API
uses (`PATCH /api/v2/principals/{id}`, `POST /api/v2/principals/{id}/api-keys`). Each records a `permission_changed` event
by the system, "command line", in the custody log; no event, log line or list holds a password or a key. An unknown name,
or a change that is already in place, changes nothing and exits 1.

```
docker compose exec api python -m app.cli principals set-password --username jane.doe
docker compose exec api python -m app.cli principals disable --username jane.doe
docker compose exec api python -m app.cli principals enable --username jane.doe
docker compose exec api python -m app.cli principals set-role --username jane.doe --role reviewer
docker compose exec api python -m app.cli principals api-key --username jane.doe
```

| Situation | What to do | What happens |
|---|---|---|
| A joiner | `principals create --username … --role … --display-name "…"`; hand the printed password over in private | The person signs in and replaces the password at once: account menu, "Change password…" (`POST /api/v2/me/password`). Until they do, whoever created the account knows its password, and a sign-off made by session is only as good as that. |
| A leaver | `principals disable` | The open sessions end; the password, the API key and open download links are refused from that moment. The row stays, so the custody log and signed reports keep the name. Nothing is deleted. |
| A returner | `principals enable`, then `set-password` | The password the account had works again, and so does its API key: replace the key with `api-key` unless it was kept safe. Ended sessions stay ended. |
| A forgotten password | `principals set-password` | A generated password is printed once and every session of the principal ends. The person replaces it after signing in. An `api_client` has no password: the command refuses. |
| A changed job | `principals set-role` | The sessions end, so the next sign-in starts with the new role; the API key acts with the new role at once. An `api_client` that becomes a person gets a generated password; a person cannot become an `api_client` (create a separate principal for a program). |
| A lost or exposed API key | `principals api-key` | A new key is printed once and replaces the old one, which is refused from that moment. A disabled principal gets none. The bootstrap key is not a principal's key: replace `VERITAS_API_KEY` in the secret store and restart. |

**A person's own password.** Anyone signed in with a password can replace it: the account menu, "Change password…". The
current password is asked again, the new one has at least 12 characters, the account's other sessions end and the one in
use stays. The custody log records the change (`permission_changed`, "password_changed", by the principal) without either
password. A wrong current password is a refused credential: it is recorded (`login_failed`, "password change") and counts
against the client address's 20 refused credentials a minute (above), after which every request with a key or a session
from that address is answered 429 until the count recovers. The bootstrap key and an `api_client` have no password to
change (409).

**Not covered.** No command or screen deletes a principal, and none stops the last administrator from being disabled or
demoted: the bootstrap key and `principals create` remain the way back in. A password change does not replace the
principal's API key; use `principals api-key` when the key may be known to someone else.

## Configuration

All settings live in `backend/app/core/config.py`, and [`docs/settings.md`](settings.md) lists every one with its
variable, type, default and description (generated from the code; a test fails when a setting has no description or the
file is stale). Override them with `VERITAS_*` environment variables or, in a directory install, `backend/.env`
(`.env.example`).
- **In Docker Compose** the image holds no `.env`. `api` and `worker` get a setting, in this order of precedence, from
  1. `docker-compose.yml` itself: the values it fixes (the two volumes, `VERITAS_RUN_WORKER`, `VERITAS_TRUST_PROXY`, the
     names of the database sessions; ○ in `docs/settings.md`), and the variables it takes from the environment of
     `docker compose`, or from the `.env` file beside it, with a default of its own: `VERITAS_API_KEY` (required),
     `VERITAS_DEVICE`, `VERITAS_DEFAULT_PROFILE`, `VERITAS_LOG_LEVEL`, `VERITAS_CORS_ORIGINS` (a JSON list) and
     `VERITAS_TRUSTED_PROXY_HOPS` (◐). Set these six there: a line for one of them in `veritas.env` is not read;
  2. the same environment, for the settings named in `docker-compose.settings.env` (●), when they are set there:
     `VERITAS_SIGNING_KEY`, `VERITAS_LICENCE_POLICY`, `VERITAS_RETENTION_DAYS`, `VERITAS_RATE_LIMIT_PER_MINUTE`,
     `VERITAS_LOGIN_RATE_PER_MINUTE`, `VERITAS_CREDENTIAL_FAILURES_PER_MINUTE`, `VERITAS_JOB_TIMEOUT_S`,
     `VERITAS_MAX_REQUEUE_ATTEMPTS`, `VERITAS_SESSION_TTL_S`, `VERITAS_MIN_FREE_BYTES`, `VERITAS_FUSER_VERSIONS` and
     `VERITAS_MODEL_MEMORY_BUDGET_MB`;
  3. the optional file `veritas.env` beside `docker-compose.yml`: one `NAME=value` line for any other setting, and for
     those of group 2 when the host does not set them. It is not in version control.

  `docker compose config` shows what each service will get. `models-init` reads neither file; it is given
  `VERITAS_LICENCE_POLICY` when that is set in the environment of `docker compose` (not when it is only in
  `veritas.env`), so set a licence policy there: the fetch then covers the detectors that policy enables.
- **Secrets:** `VERITAS_API_KEY` (bootstrap administrator key); `VERITAS_SIGNING_KEY` (HMAC resource tokens, review
  signatures, webhook secrets; default in prod: the generated `data/keys/signing.key`; in dev: derived from the API key);
  the report signing keys and HMAC keys under `data/keys/` (0600). Never commit `.env`. See "First start on a server".
- **Licence policy:** `VERITAS_LICENCE_POLICY` (D-04; default: the commercial preset plus `txt.desklib_ft_v2`).
  `python -m app.cli models list --policy …` prints what would be enabled and why.
- **Profiles:** `VERITAS_DEFAULT_PROFILE=auto` (full on accelerators, quick on CPU; the Compose file sets `full`, see
  "Calibrated fusers"). Intake can choose `quick`, `full` or `deep` per
  analysis (the "Analysis profile" select; `profile` on the upload routes, `?profile=` on case intake); without a choice the
  instance default applies.
- **Security settings** (`docs/v2/qa/security-review.md`):
  - **Single-user mode** (no API key, dev only): a request acts as the local examiner only with a loopback socket, a loopback
    `Host`, no forwarding header, and no cross-site `Origin`. It is off in prod and with `VERITAS_TRUST_PROXY=true`.
  - **Signing key:** in prod it is `VERITAS_SIGNING_KEY` or the generated `data/keys/signing.key`, never a derivation of the
    API key, so the API key can be rotated without touching stored review signatures. In dev it is derived from the API key
    when that is set.
  - **Body limits:** every request body is capped while it streams: upload kinds by `VERITAS_MAX_*_BYTES` (+1 MB), batches by
    `VERITAS_BATCH_MAX_BYTES` (+16 MB), all other routes by `VERITAS_MAX_JSON_BYTES` (2 MB). Put the temp directory on a
    volume with room for the largest upload (`TMPDIR`; Compose sets `/data/tmp`, and the process makes a directory that
    is named and missing).
  - **Event streams:** `VERITAS_SSE_MAX_PER_PRINCIPAL` (8, then 429) and `VERITAS_SSE_MAX_CONNECTIONS` (200, then 503).
  - **Webhooks:** deliveries go to public addresses only, unless `VERITAS_WEBHOOK_ALLOW_PRIVATE_NETWORKS=true` for an
    on-premises receiver.
  - **Headers:** every API response carries CSP, `nosniff` and `no-referrer`, and nginx adds them for the SPA. The policy
    is `default-src 'self'` with `object-src 'none'`, `base-uri 'none'`, `form-action 'self'` and `frame-ancestors 'none'`;
    the three copies (the API and the two nginx files) are one string, and a test compares them.
  - **Embedded thumbnails:** a thumbnail of a C2PA manifest is shown only when it is a JPEG, PNG or WebP by its first
    bytes, and what is stored is a re-encoding of its pixels without metadata. An exhibit an earlier build stored
    with another type (SVG, HTML) is served as a download in a sandbox, not as a page of this origin.
  - **Access logs:** resource tokens travel as `?token=`, so the image disables uvicorn's access log and nginx logs `$uri`
    without the query string.
  - **Logs:** the API, the worker and the command line write one JSON object per line in prod, in UTC, at
    `VERITAS_LOG_LEVEL` (default `INFO`): the service's own events (`event`, `level`, `timestamp`, and fields such as
    `analysis_id`), and the lines of uvicorn, the migrations and other libraries (`logger` names them). The API and the
    worker log to standard output; the command line logs to standard error, so what a command prints is its answer. An
    exception that ends a process is one line, `process.uncaught_exception`.
  - **Requests:** every response carries `X-Request-ID`. A plain token sent by a proxy in that header is kept (letters,
    digits, `.`, `_`, `-`; 64 characters at most), anything else is replaced. Each request leaves one line `request`
    with `request_id`, `method`, `route` (the template, such as `/api/v2/analyses/{aid}`), `status` and `duration_ms`,
    never the query string; the other lines of that request carry the same `request_id`. Requests answered with a
    5xx are logged at `error`; the probes (`/api/health`, `/api/ready`, `/api/metrics`) at `debug`, and a probe that
    answers 503 at `warning` (the line `ready.not_ready` beside it says what is not ready). uvicorn's own access log
    stays off in prod whatever its command line says.
  - **Quiet libraries:** the renderer's progress lines (`weasyprint.progress`, a line per layout step and page of
    every report) and the migrations' plugin lines (`alembic.runtime.plugins`) are logged from `warning` up only.
- **Monitoring:** `VERITAS_MIN_FREE_BYTES` (the free-space floor of the data volume, 1 GiB), `VERITAS_WORKER_HEARTBEAT_S`
  (10) and `VERITAS_WORKER_STALE_S` (45): see "Monitoring".
- **Dependency advisories:** `make audit` asks the advisory databases (network) about every package of both runtime locks
  (`backend/requirements/runtime-cpu.lock` and `runtime-cu126.lock`; `torch` and `torchvision` by name and version) and about
  the npm lock. It installs nothing. The reports and the receipt `audit-run.json` (the date, the tools, the SHA-256 of each
  audited lock) are written to `docs/v2/qa/`. It fails when a tool is missing, when a package was not audited, and on a
  pip advisory or an npm high or critical advisory that `docs/v2/qa/audit_exceptions.yaml` does not list. A lock that
  changes needs a new audit: the suite compares the receipt with the locks.
- **Release gate:** `make ci-release` (about 75 minutes) runs `make ci` with the backend suite as one serial pytest, the
  Postgres suite on a disposable local server (`backend/scripts/pg_suite.sh all`: the `pg` tests as an owner that is no
  superuser, then the default suite with one fresh database per test), every test of `infra/gcp/tests`, `terraform test`,
  `make deploy-lint` and `make audit`. It needs the development install (`make setup`), PostgreSQL 16 or 17 server
  binaries, `terraform`, and the `docker` client with Compose (no daemon); a missing tool ends it before anything runs.
  A failed step does not stop the later ones, and the summary names each step with its exit status
  (`backend/data/ci-release/summary.txt`, beside one log per step). It builds no image: the image build and the smoke in
  the image are a separate step of a release.

### Decision levels: the operator's lever on false positives

An image, video or audio result of a calibrated fusion model is called AI, or authentic, only when its conformal set at the
workspace's level `alpha_moderate` holds that one answer; otherwise it is inconclusive. `alpha_strong` is the level of the
strong wording. Both are 0.05 or 0.01, with `alpha_strong` never above `alpha_moderate`; the defaults are 0.01 and 0.05.
A workspace that must not call authentic media AI sets `alpha_moderate` to 0.01:

```sh
curl -X PUT http://127.0.0.1:8080/api/v2/policy/thresholds -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"alpha_strong": 0.01, "alpha_moderate": 0.01, "min_slice_auc": 0.70, "reason": "authentic media must not be called AI"}'
```

- **Who and how.** An administrator; there is no screen for it. `GET /api/v2/policy/thresholds` shows the version in force.
  A change is a new version with its reason and a `policy_changed` event in the custody log. Analyses made afterwards use
  it; an earlier analysis keeps the version it recorded, and a re-run of it uses that version again.
- **What it does not reach.** Documents, and `quick` image and video analyses, use the uncalibrated fallback, which has no
  such levels.
- **What the two levels gave on the benchmark** (`docs/evaluation.md`, "v2.1.1 on Veritas-Bench vb-1.1": authentic `test`
  items above the threshold of each level; a re-used test set, not a fresh one), and what 0.01 does to a decision (the
  thresholds of the released fusion models `fuser-*-2.1.1`):

  | Kind | Authentic items flagged at the 0.05 level | at the 0.01 level | With `alpha_moderate` 0.01 |
  |---|---|---|---|
  | Image | 7.5 % (162/2,152; Wilson 6.5 %–8.7 %), against a target of 5 % | 1.3 % (29/2,152; Wilson 0.9 %–1.9 %), against 1 % | by the context of the image (with a face, without one, low quality): "AI" needs an estimate above 0.961 to 0.998 instead of 0.628 to 0.758, "authentic" one below 0.029 to 0.062 instead of 0.122 to 0.496 |
  | Video | 2.5 % (4/162; Wilson 1.0 %–6.2 %) | no 0.01 level exists (74 authentic calibration items) | **no video result is called AI or authentic: every one is inconclusive** |
  | Audio | 2.3 % (6/262; Wilson 1.1 %–4.9 %) | not on that page | the two thresholds almost coincide (0.804 and 0.808): little changes |

- **The price.** Fewer authentic items are called AI, and fewer AI items are: everything between the two thresholds
  becomes inconclusive. How many more results are inconclusive at 0.01, and the detection rate at 0.01 over all image
  families, were not measured for the product's path (`docs/evaluation.md` gives the rate per family at the 0.01
  threshold). With both levels at 0.01 a decided result always carries the strong wording.
- **What the figures are.** Rates of the benchmark at its own thresholds. The product decides with thresholds of the same
  levels fitted per context (for images: with a face, without one, low quality), and then applies its abstention rules,
  so the share of authentic uploads called AI in a workspace is related to these rates and is not the same number.

## Sizing

- **Measured latency (`make bench` on an Apple M5 Pro with 48 GB, MPS, a host shared with other work).** The table is
  written from the benchmark's record, `docs/v2/bench/perf.json`, by `make sizing-doc`; `docs/v2/bench/perf.md` is the
  same run in full (stages, detectors, models).

  <!-- latency:begin (generated by backend/scripts/sizing_doc.py from docs/v2/bench/perf.json: run `make sizing-doc`, do not edit) -->

  Run of 2026-09-17 by `backend/scripts/bench.py` at commit `88274ce`, device `mps` (macOS-26.6.2-arm64-arm-64bit; Python 3.12.13; 18 logical CPUs, 51.5 GB of memory), 1-minute load average 6.15 before and 7.03 after. Upload to `done` as the client saw it, polled every 0.5 s: p95 over n timed runs after one warm-up per kind (at these counts the p95 is the slowest run).

  | Workload | `quick` | `full` | `deep` | Budget of `full` (PL-02) |
  |---|---|---|---|---|
  | Image 256 × 256 (`image_256px_face`) | 0.53 s (n = 5) | 0.55 s (n = 5) | 0.56 s (n = 5) | – |
  | Image 800 × 800 (`image_800px`) | 0.54 s (n = 5) | 1.06 s (n = 5) | 1.07 s (n = 5) | – |
  | Image 4000 × 3000 (`image_12mp`) | 0.54 s (n = 3) | 3.08 s (n = 3) | 3.12 s (n = 3) | ≤ 8 s |
  | Image 6000 × 4000 (`image_24mp_camera`) | 2.15 s (n = 3) | 4.12 s (n = 3) | 4.14 s (n = 3) | – |
  | Video 9 s, 1920 × 1080 (`video_9s_1080p`) | 2.62 s (n = 2) | 9.22 s (n = 2) | 9.75 s (n = 2) | – |
  | Video 60 s, 1920 × 1080 (`video_60s_1080p`) | 8.32 s (n = 1) | 50.79 s (n = 1) | 60.86 s (n = 1) | ≤ 120 s |
  | Audio 5 min (`audio_5min`) | 0.62 s (n = 1) | 5.18 s (n = 1) | 5.21 s (n = 1) | ≤ 60 s |
  | Document 1,000 words (`text_1000_words`) | 2.07 s (n = 3) | 3.11 s (n = 3) | 2.60 s (n = 3) | ≤ 15 s |
  | Peak resident memory of the API's process tree, warm-ups included | 3.32 GB | 7.40 GB | 4.18 GB | ≤ 36 GB |

  Pipelines of the run: `img-2.0-p6`, `vid-2.0-p7`, `aud-2.0-a2`, `doc-2.0-p7`. The slowest detectors of `video_60s_1080p` under `full`: `vid.synth.frames.dda` 18.6 s and `vid.tracks.yunet_dinov2` 15.5 s. `deep` ran as many detectors as `full` on every workload, so it measures no further work.

  <!-- latency:end -->

  These are the figures of that commit on that machine, under that load. The pipelines have changed since, and the
  whole set was not measured again on 2.1.1. A later run of four `full` workloads (commit `b4a3b2e`, 22 September 2026,
  1-minute load 14 to 15; `docs/v2/qa/j11-latency/README.md`) gave a p95 of 1.06 s for the 256 px image, 1.58 s for
  the 800 px image, 4.70 s for the 24 MP image and 11.88 s for the 9 s video. The detectors that read frames take
  longer on a longer video, and no recording above 60 s is in the run; neither are concurrent uploads, a cold start
  or a report. Re-measure on the target host with `make dev-api` then `make bench`
  (`PROFILES=full ONLY=image_12mp RUNS=5` narrows the run), and write the table again with `make sizing-doc`.
- **CPU-only hosts (measured under emulation; the supervisor's Compose run of bbd5f74,
  `docs/v2/qa/prod-readiness-2026-09-29/supervisor-checks.md`).** A Linux VM with 18 CPUs and 7.75 GiB of memory, the
  `linux/amd64` image under emulation on the build host, analyses requested as `full`: two images 5 s and 10 s, a 15 s
  audio file 10 s, a 1,000-word document 5 s, a video 81 s; the worker peaked at 4,417 MiB (sampled every 10 s). Five
  detectors could not run under that host's budget of 3,967 MB ("Memory per detector and per profile" below), so these
  are the times of analyses without them. Not measured: a native amd64 host, and a CPU host with the memory to run
  every detector.
- **Memory:** the model memory budget is `VERITAS_MODEL_MEMORY_BUDGET_MB=auto`: MPS `min(0.5·RAM, RAM−20 GB)`, CUDA `0.85·VRAM`,
  CPU `0.5·RAM`. Models above the budget are evicted least-recently-used. In a container, RAM is the smaller of the
  machine's memory and the container's memory limit (cgroup `memory.max`, or `memory.limit_in_bytes` on cgroup v1). In
  Compose the worker's limit is `VERITAS_WORKER_MEM_LIMIT` (28 GiB by default), so `auto` is 14,336 MB on a CPU host
  with 28 GiB of memory or more and half of the machine's memory on a smaller one ("Deploy", "Host"). On a larger
  host, raise the limit to raise the budget. A budget set by hand
  (`VERITAS_MODEL_MEMORY_BUDGET_MB`, passed on to `api` and `worker`) is taken as it is: keep it below the limit, or the
  kernel kills the worker where the model manager would have evicted a model. A CPU process whose budget is at or above
  its container's limit logs one warning with both numbers, `models.budget_above_memory_limit`, when it builds its model
  manager (at its first analysis at the latest).
- **What the budget can run.** `python -m app.cli models plan` (in Compose: `docker compose run --rm worker python -m
  app.cli models plan`) prints, for this server's device, default profile and budget, each detector that loads a model
  with its need (the weights that are loaded together while it runs, plus the activation allowance of its plan), whether
  it can run, and the smallest budget that runs the whole profile, with the memory at which `auto` reaches that budget.
  It exits with 1 when a detector cannot run. `--profile`, `--device` and `--budget-mb` ask for another server
  (`models plan --profile full --device cpu --budget-mb 3967` is a CPU host with 7.75 GiB of memory); `--json` gives
  the same as data. The answers are the model manager's own (the command takes each detector's lease on a manager that
  loads nothing), and no weights are read. A detector that cannot run is recorded as `skipped: memory budget` in every
  analysis it applies to: on a CPU host the `full` profile needs a budget of 12,977 MB (25,953 MB of memory under
  `auto`) for the two language models of `txt.binoculars`, and `quick` 6,911 MB (13,821 MB of memory) for
  `vid.frames.dfdc_b7`. The plan
  counts declared footprints and, on an accelerator, the loads this server has recorded (`models footprint`); it does
  not know other processes on an accelerator or memory outside the allowance (decoding, CPU buffers). On a CPU server
  a model is reserved at its declaration, before and after it is loaded: what the server says at its start that it can
  run stays so while it runs. The change of the process's resident memory across a load is recorded and shown by
  `models footprint` as an observation of the process (it is not the model's size: the same load was seen as 1,160 MB
  and as 2,302 MB) and reserves nothing; a figure well above the declaration is a reason to correct `app/models.yaml`.
- **The models volume and the fetch** (the supervisor's Compose run of bbd5f74): the models volume of the `full` profile
  holds 11.5 GB. The fetch (`models-init`) peaked at 2,459 MiB (sampled every 7 s) and was killed for memory at a limit
  of 2 GiB, during a 3.09 GB file; its limit is `VERITAS_FETCH_MEM_LIMIT` (4g by default). On a first start from the
  network at 5f0cb16 (the supervisor's check 13) it was sampled at 3,351 MiB at most, under the 4 GiB limit (84 % of
  it), and fetched 13 models, 11.89 GB. A fetch that was killed goes
  on with the files that are missing when it is run again.
- **Throughput:** one worker thread per accelerator (`VERITAS_WORKER_THREADS`). The supported and measured shape is one
  `worker` beside one `api`. Jobs are leased, so two workers never run the same job, and processes that start together
  migrate one after the other (see "Upgrades"); but more `worker` containers against Postgres and a shared
  `VERITAS_DATA_DIR` have **not been run under load**: no throughput figure exists, and the database connections and
  the memory of each further worker are unplanned. `infra/gcp/README.md` defers more worker slots, several machines and
  autoscaling for the same reason. Treat a second worker as an experiment to measure, not as a sizing answer.
- **Report requests and the size of the custody log.** A standard or forensic report request re-hashes the whole custody
  chain before it is queued, so it answers later as the log grows: 1.3 to 1.7 s for 50,000 events on the machine above, on
  SQLite and on PostgreSQL 17 (a shared host at a 1-minute load of 45 to 56). It holds no lock while it does so; only the
  events appended during that walk are hashed under the request's locks, and another writer waited 12 to 33 ms during such
  a request on SQLite and 20 to 70 ms on PostgreSQL. The snapshot of the report records the sequence number and hash of the
  last event that verified. A summary or a newsroom report verifies no chain.

### Memory per detector and per profile

Generated from the plan function (`models plan`, above) by `make sizing-doc`; a test fails when the block is not what
the script writes. The first table gives, for each detector that loads a model, its weights and its activation
allowance on CPU and its need on each device class. The second gives, per profile and device class, the smallest model
memory budget that runs every detector and the memory at which `VERITAS_MODEL_MEMORY_BUDGET_MB=auto` reaches it. The
lines after it say what a small host drops.

<!-- sizing:begin (generated by backend/scripts/sizing_doc.py: run `make sizing-doc`, do not edit) -->

| Detector | Profiles | Weights MB (CPU) | Allowance MB | Need MB, CPU | Need MB, MPS | Need MB, CUDA |
|---|---|---|---|---|---|---|
| `img.dfdc_b7` | quick, full, deep | 510 | 4,800 | 5,310 | 5,310 | 5,310 |
| `img.faces.yunet` | quick, full, deep | 1 | 20 | 21 | 21 (on CPU) | 21 (on CPU) |
| `vid.faces.yunet` | quick, full, deep | 1 | 20 | 21 | 21 (on CPU) | 21 (on CPU) |
| `vid.frames.dfdc_b7` | quick, full, deep | 510 | 6,400 | 6,910 | 6,910 | 6,910 |
| `aud.gustking_xlsr` | quick, full, deep | 1,204 | 1,200 | 2,404 | 2,404 | 2,404 |
| `txt.desklib` | quick, full, deep | 1,656 | 3,200 | 4,856 | 4,856 | 4,856 |
| `txt.desklib_ft_v2` | quick, full, deep | 1,656 | 3,200 | 4,856 | 4,856 | 4,856 |
| `img.dda` | full, deep | 1,173 | 1,750 | 2,923 | 2,923 | 2,923 |
| `img.hf.ateeqq` | full, deep | 354 | 150 | 504 | 504 | 504 |
| `img.hf.dima806` | full, deep | 327 | 720 | 1,047 | 1,047 | 1,047 |
| `vid.tracks.yunet_dinov2` | full, deep | 85 | 3,020 | 3,105 | 3,105 | 3,105 |
| `vid.synth.frames.commfor384` | full, deep | 83 | 600 | 683 | 683 | 683 |
| `vid.synth.frames.dda` | full, deep | 1,173 | 1,750 | 2,923 | 2,923 | 2,923 |
| `txt.binoculars` | full, deep | 11,776 | 1,200 | 12,976 | 7,088 (half precision) | 7,088 (half precision) |

| Profile | Device | Detectors that load a model | Smallest budget MB | Set by | Memory at which `auto` gives it, MB |
|---|---|---|---|---|---|
| `quick` | CPU | 7 of 19 | 6,911 | `vid.frames.dfdc_b7` | 13,821 (the machine's) |
| `quick` | MPS | 7 of 19 | 6,911 | `vid.frames.dfdc_b7` | 27,391 (the machine's) |
| `quick` | CUDA | 7 of 19 | 6,911 | `vid.frames.dfdc_b7` | 8,130 (the accelerator's) |
| `full` | CPU | 14 of 32 | 12,977 | `txt.binoculars` | 25,953 (the machine's) |
| `full` | MPS | 14 of 32 | 7,089 | `txt.binoculars` | 27,569 (the machine's) |
| `full` | CUDA | 14 of 32 | 7,089 | `txt.binoculars` | 8,340 (the accelerator's) |
| `deep` | CPU | 14 of 32 | 12,977 | `txt.binoculars` | 25,953 (the machine's) |
| `deep` | MPS | 14 of 32 | 7,089 | `txt.binoculars` | 27,569 (the machine's) |
| `deep` | CUDA | 14 of 32 | 7,089 | `txt.binoculars` | 8,340 (the accelerator's) |

A CPU host with 8 GiB of memory (budget 4,096 MB under `auto`) cannot run, of the `quick` profile: `img.dfdc_b7`, `vid.frames.dfdc_b7`, `txt.desklib`, `txt.desklib_ft_v2`.
A CPU host with 8 GiB of memory (budget 4,096 MB under `auto`) cannot run, of the `full` profile: `img.dfdc_b7`, `vid.frames.dfdc_b7`, `txt.desklib`, `txt.desklib_ft_v2`, `txt.binoculars`.
A CPU host with 16 GiB of memory (budget 8,192 MB under `auto`) cannot run, of the `full` profile: `txt.binoculars`.
A CPU worker under Compose's default memory limit of 28 GiB (`VERITAS_WORKER_MEM_LIMIT`; budget 14,336 MB under `auto` on a host with at least that much memory) can run every detector of the `full` profile, the default of the Compose file.

<!-- sizing:end -->

- The numbers are the declared footprints of `app/models.yaml` under the default licence policy, with no recorded
  loads. On an accelerator a load that measures larger raises the need on that server: `models footprint` shows it,
  and `models plan` there counts it. On a CPU server a recorded load raises nothing, and the CPU table holds as it is.
- `txt.desklib` is listed beside `txt.desklib_ft_v2`. On a server where the fine-tune is installed it is superseded and
  does not run.
- "The machine's" memory is the container's limit in a container. A budget set by hand
  (`VERITAS_MODEL_MEMORY_BUDGET_MB`) needs the memory for it, and the process needs memory beyond its models.
- The CUDA column is the plan's arithmetic for that device class; no CUDA host was run.

## Monitoring

### Probes

| Probe | Use it as | Answers | What it asks |
|---|---|---|---|
| `GET /api/health` | liveness (restart the container when it fails) | 200 `{"status": "ok"}` | nothing but the process: no database, no volume, no worker. A locked database or a full volume does not get a living API restarted |
| `GET /api/ready` | readiness (route traffic, alert) | 200 `{"status": "ready"}`; 503 `{"status": "not_ready"}` | the three checks below, each within 4 seconds |

- **The checks of `/api/ready`.**
  - `database`: on SQLite the write lock is taken and given back within 2 seconds (an instance that cannot write stores
    nothing, and a read does not see another connection's lock); on PostgreSQL `SELECT 1`.
  - `storage`: one small file is written in the data directory, and its volume has at least `VERITAS_MIN_FREE_BYTES` free
    (default 1 GiB; 0 switches the floor off).
  - `worker`: when the API runs no worker (`VERITAS_RUN_WORKER=false`, as in Compose), a worker's heartbeat is at most
    `VERITAS_WORKER_STALE_S` (45 s) old and says that all its threads reached the database lately. A worker writes it every
    `VERITAS_WORKER_HEARTBEAT_S` (10 s) into the table `worker_heartbeats` and removes it when it stops.
- **Who is told what.** Without credentials both probes answer with the `status` field and nothing else. `/api/ready`
  with an API key (or a session) adds `checks` (each with `ok`, and why it failed), `version`, `build_commit`, `queue_depth`
  and what the serving worker runs on: `device`, `profile`, `loaded` models, `fusion`, `fusers` and `text_classifier`
  (`?profile=quick|full|deep` for another profile). With a separate worker these are the worker's, read from its
  heartbeat, not those of the API process. A wrong key is told what no key is told, and counts as a refused credential.
- **`capacity`: what the serving worker cannot run.** A worker makes the plan of "Sizing" for its default profile when
  it starts (in the background: the first beats carry `null`), logs one warning per detector that cannot run under its
  model memory budget (`capacity.detector_cannot_run`, with `detector_id`, `need_mb`, `budget_mb` and the model
  manager's `reason`), and writes the summary into its heartbeat. `/api/ready` with credentials prints it as `capacity`:
  `profile`, `device`, `budget_mb`, `detectors` (how many the profile holds), `cannot_run` (each with `detector_id`,
  `need_mb`, `reason`), `largest_need_mb` and `smallest_budget_mb`. An empty `cannot_run` means that every detector of
  the profile fits. Such a server is ready: it takes work, and each result says which checks were skipped. A plan that
  could not be made is logged as `capacity.plan_failed` and leaves `capacity` at `null`.
- **Wiring.** The image's `HEALTHCHECK` asks `/api/health`. Point a load balancer or an uptime check at `/api/ready`, and
  act on three failures in a row, not on one: a long write (a purge, a report over a long custody log) can hold the SQLite
  write lock for longer than the check waits. A worker that waits for a healthy API must wait on `/api/health`, since
  `/api/ready` is 503 until that worker's first heartbeat. While the instance is not ready the API logs `ready.not_ready`
  with the reasons, once when they change and once a minute after that, and `ready.ready_again` when it is over.
- **The page.** The engine block of the sidebar reads `/api/ready`. An engine that answers and is not ready is said in a
  strip under the header, with the first reason a person can act on (storage, the analysis worker, the database). To an
  administrator the block also prints what `capacity` holds, in one sentence: "5 checks of the full profile cannot run on
  this server: memory budget 3,967 MB, largest need 12,976 MB." It names no check; `models plan` on the server does.

### Metrics

`GET /api/metrics` (administrator; an API key in `X-API-Key`) is the Prometheus text exposition. It is read from the
database and the data volume when it is asked for, so the API and every replica of it give the same answer, whichever
process ran the analyses. All series are gauges: rows are counted as they are at the scrape, and a deleted or purged
revision leaves the counts (take a change with `delta()`, not `rate()`).

| Series | Meaning |
|---|---|
| `veritas_build_info{version,commit}` | 1; the version and build commit of the answering process |
| `veritas_database_up` | 1 when the series below were read from the database; 0 when it could not be read (they are then left out) |
| `veritas_jobs{type,status}` | jobs of the queue; every type (`analysis`, `refusion`, `report`, `package`, `batch_expand`, `webhook_delivery`) and status (`queued`, `leased`, `done`, `failed`, `cancelled`) is printed, zeros included |
| `veritas_jobs_oldest_queued_age_seconds` | age of the oldest queued job; 0 when nothing is queued |
| `veritas_jobs_attempts_exhausted` | jobs failed because their run was interrupted `VERITAS_MAX_REQUEUE_ATTEMPTS` times (see "Troubleshooting") |
| `veritas_analyses{kind,status,fusion}` | revisions by kind, status and fusion mode (`artefact`, `fallback`, or `none` before a result) |
| `veritas_analysis_duration_seconds{kind,quantile}` | 0.5 and 0.95 quantile of `finished_at − started_at` over the last 500 finished analyses of a kind; `veritas_analysis_duration_samples{kind}` says over how many |
| `veritas_workers` | workers whose heartbeat is at most `VERITAS_WORKER_STALE_S` old and whose threads all reach the database |
| `veritas_worker_heartbeat_age_seconds` | age of the newest heartbeat; `+Inf` when no worker has one (stopped cleanly, or never started) |
| `veritas_worker_info{worker_id,hostname,device,profile,in_process,version}` | 1 per live worker; `in_process="true"` for the worker threads of an API process |
| `veritas_worker_fusion{kind,mode}` | 1 per kind: what the live worker that wrote last serves under its default profile |
| `veritas_data_free_bytes`, `veritas_data_total_bytes` | the volume of the data directory |
| `veritas_uptime_seconds`, `veritas_ready` | of the answering process: seconds since its start, and 1 once its start is over (migrations, the key check, the registry, the worker thread; no model is loaded for it) |

### Alert conditions

| Condition | Rule | Then |
|---|---|---|
| The instance cannot take or do work | `/api/ready` not 200 for 2 minutes | ask it with an API key: `checks` names the reason |
| No worker | `veritas_workers == 0` or `veritas_worker_heartbeat_age_seconds > 60`, for 2 minutes | `docker compose ps worker`, its log; `worker.on_hold` means it waits for free space |
| The queue does not drain | `veritas_jobs_oldest_queued_age_seconds > 900` | worker capacity ("Sizing"), or no worker |
| Analyses fail | `delta(veritas_analyses{status="failed"}[15m]) > 0`; `veritas_jobs_attempts_exhausted` rising | the analysis shows its reason; the worker log by `analysis_id` |
| Uncalibrated results | `veritas_worker_fusion{mode="fallback"} == 1` for a kind that has an artefact (image, video, audio under `full`) | "Calibrated fusers" above; `fusers status` |
| The volume fills | `veritas_data_free_bytes` below twice `VERITAS_MIN_FREE_BYTES`, or `predict_linear(veritas_data_free_bytes[6h], 86400) < 0` | "Storage" below |
| The database cannot be read | `veritas_database_up == 0` | the database, then `/api/ready` |
| Slow analyses | `veritas_analysis_duration_seconds{quantile="0.95"}` above the budget of the kind ("Sizing") | host load, device, profile |
| The API restarted | `veritas_uptime_seconds` dropped | the API log: `process.uncaught_exception` |

The thresholds are starting points, not measured service levels: no production load has been run against them.

### Storage

- **What grows.** `data/objects` (originals, exhibits, reports, packages), the database, and `data/batches` while a batch
  is taken in. An analysis stores several times its upload: the operations review measured 12.6 MB of objects for a
  1.16 MB video with the fake detectors (16 stills 3.3 MB, a report 2.3 MB, a package 6.8 MB). Size the volume from your
  own uploads and retention ("Data & retention"), and re-measure with `du -sh data/objects` after a day of real use.
- **The floor.** Below `VERITAS_MIN_FREE_BYTES` free on the data volume:
  - an upload (a quick analysis, a case item, a batch) is refused with 507 and the sentence "The server's evidence storage
    is full, so nothing was stored. Ask the administrator to free space, then try again."; nothing is stored;
  - `/api/ready` is 503 (`checks.storage`);
  - the worker claims no job and logs `worker.on_hold` once, when the hold begins, however the free space moves while it
    lasts (`/api/ready` gives the count of the moment). What is queued stays queued; no analysis is failed for it.
  A write that still meets a full volume (an upload larger than what is left) is answered 507 as well, and logged as
  `api.storage_full`.
- **Recovery.** Free space with the instance running, in this order: `make purge` (what the retention rule allows: it
  frees the bytes of what it deletes), then `python -m app.cli store gc` (objects that no record references), or grow the
  volume. Both free bytes while analyses, reports and packages wait in the queue: below the floor the worker claims
  none of it, and work that waits does not hold bytes back. An analysis that was interrupted and waits to run again keeps
  what its finished stages wrote (records name it). What it had written for the stage it was stopped in, no record names:
  `store gc` or a purge may unlink that, and the next run writes it again. What still blocks them is a job that a worker
  holds, that is, one that is running: both commands then unlink nothing and print "n job(s) that a worker holds may still
  index what they wrote". Wait for that job to finish; when its worker has died, its lease runs out by itself
  (`VERITAS_JOB_LEASE_S`, 60 seconds) and the job no longer blocks; then run `store gc`. A purge writes its own custody
  records; the floor leaves it the room to do so. The worker logs `worker.hold_ended` and takes the queue up again within
  seconds, and uploads are accepted; no restart is needed and no queued analysis has to be cancelled.
- **A worker whose loop fails** (the database is away, the volume is full below a floor of 0) waits longer after each
  failure, from twice its poll interval up to 30 seconds, logs each try as `worker.loop_error` with `consecutive` and
  `retry_in_s`, and `worker.loop_recovered` when it works again.

## Data & retention

- `data/objects` holds the content-addressed evidence store: originals, exhibits, report PDFs and evidence packages, each file
  named by its SHA-256; `data/batches` holds staged batch uploads. `data/uploads`, `data/artifacts` and `data/reports` are the
  per-analysis directories of the v1 routes (`uploads/<id>/<file>` is a hard link to the store object).
- **Retention rule.** A revision chain (an analysis with its re-runs, reports and packages) is purged whole or kept whole:
  - never purged: a chain with a queued or running revision, or a report or package still being produced; a case under legal
    hold; a case whose retention date lies in the future;
  - a case past its retention date: every chain goes, signed ones included;
  - no retention date: a named case is kept; a quick analysis goes when its newest revision is older than
    `VERITAS_RETENTION_DAYS` (default 30) and no revision is signed.
  An administrator sets the hold and the retention date on the case (`PATCH /api/v2/cases/{id}`).
- **Commands** (on a schedule of your choosing; nothing purges on its own):
  - `make purge DRY_RUN=1` (`python -m app.cli retention purge --dry-run`) prints every chain with KEEP or PURGE, the rule and the
    reason, and changes nothing. Read it before the first scheduled run.
  - `make purge` deletes the purgeable chains with their rows and per-analysis directories, removes the media record, case link
    and intake record of an item no analysis is left for, removes an implicit case with no item left, appends one
    `retention_purge` custody event per revision (ids and hashes only), and unlinks the store objects nothing references any more.
    The rows of a chain and its custody events are one database transaction: a purge that is stopped (a signal, a crash, a lost
    database connection) leaves each chain either whole or deleted with its events. What a stopped run can leave is the
    directories and bytes of the chain it had just deleted: the next `make purge` removes those directories and says so, and
    `store gc` frees the bytes. When another process keeps extending the custody log during a chain's append the chain is
    reported as FAILED, nothing of it is removed and the command exits 1; run it again.
    It also removes the media record of an upload that was never recorded. An intake that fails, or is stopped, after it
    made the media record of new bytes leaves that record with the index row of its original, and no analysis, case item
    or intake record names it; the upload itself was answered with an error and can be sent again. Such a record, first
    seen more than 15 minutes ago, goes with its index row, and its bytes are freed like every other object. No custody
    event is written for it (none names it); the command prints the number ("removed 1 media record(s) of uploads that
    were never recorded", and with `DRY_RUN=1` "would remove …"). A record that an upload names while the purge runs stays.
  - `python -m app.cli store gc [--dry-run]` sweeps the whole store for objects no record references. A package job whose row was
    finished, failed or deleted while its ZIP was being stored leaves such an object. Run it once after
    upgrading to 2.1.1 (earlier deletes left their objects behind) and after a purge that reported objects left in place. It does
    not examine an object written in the last 15 minutes: the record that names it may not be committed yet.
- **Bytes are freed only while no worker holds a job.** A job writes its objects before it indexes them, so while a worker
  holds one (an analysis, a report or a package that is running: a job of the queue under a lease that has not run out)
  the purge, the API delete and `store gc` unlink nothing and say so. Wait for the job to finish, or, when its worker has
  died, for its lease to run out (`VERITAS_JOB_LEASE_S`); then `store gc` frees what was left. Work that waits does not
  block: a queued analysis, report or package, or an analysis whose job was given back to the queue at a stop (it reads
  `queued` again) or whose worker died. A run that waits may lose objects that no record names yet to `store gc` or a
  purge, and writes them again when it runs: what a stage wrote is named by records once the stage is finished, so this
  is what was written for the stage the run was stopped in, and, on data that an earlier build left interrupted, what
  its finished stages wrote. A stage is reused only when every object it names is in the store, and no result is
  published while one is missing (the analysis then fails with the stage's name; run it again). A chain with a revision
  that is queued or running is still not purged itself.
- `DELETE /api/analyses/{id}` applies the same refusals (in progress, legal hold, retention date) and never deletes a signed
  revision; it frees the same bytes. It decides inside the transaction that deletes, and its `deletion_requested` event is
  part of that transaction.
- **A legal hold and a running purge.** A hold, or a retention date moved ahead, that is committed before the purge re-checks
  a chain keeps the chain. One that arrives while the purge is deleting that case's expired chains waits for it and is then
  stored: the custody log shows `retention_purge` before `case_updated`, never a hold followed by the purge of what it
  held. A report request, a re-run, a note, a conclusion or a sign-off that arrives in that moment waits and answers 404.
- Backup: see "Backup and restore" below. The custody chain verifies offline: `python -m app.cli audit verify`; the store,
  `python -m app.cli store verify`.
- **Verifying a case** (`GET /api/v2/cases/{id}/custody/verify`, "Verify chain and originals" on the case page) re-hashes
  the chain, every original of the case and every report against the hash its own `report_generated` event recorded: a v1
  report against the file that event names (`report.pdf`, or `report-<hash>.pdf` once the revision has a review status
  line), a v2 report against its stored PDF. After `DELETE /api/analyses/{id}` or a retention purge the reports of that
  analysis are listed with the status `removed` and the action and seq of the event that removed it; the case stays valid.
  A report that is missing or changed while its analysis is there makes the case invalid.
  The route is a read that every viewer of the case can repeat, so an original that verifies leaves no event in the custody
  log; one that does not is recorded as `hash_mismatch` ("custody_verify"), once for each finding and not on every call.
  The re-hash before an export, a report or an analysis is recorded as before.
  The v1 PDF of a result that is not a calibrated estimate prints what the result page prints (a score that is not
  calibrated, or that no estimate is shown), and its `report_generated` event records that number. A file that an
  earlier version made, which printed the score as "% p(AI)", is made again at its next request: the log then holds
  two events for the file, and verification reads the newer.
- **Failed sign-ins.** A wrong password for an account is a `login_failed` event with the account's name. A sign-in under
  a name that is no account appends no event and stores no text (people type passwords into that field); the log line
  `auth.login_failed known=false` counts it with the client address.
- **Case references, exhibit labels and the last owner.** The next case reference of a year, the next exhibit label of a
  case and a change of a case's members are each made one at a time (an advisory lock on Postgres, the write lock on
  SQLite), so concurrent case creations and concurrent uploads into one case all succeed, and two owners who remove or
  demote each other at the same moment leave one of them owner: the second request is refused.
- **Report ids** (`VR-<year>-<number>`) are given once: the number comes from a counter per year that a delete or a purge
  does not lower. On an instance that gave ids before this counter existed, the first report of a year starts above every
  id of that year that a report or the custody log names.
- **An action and its custody event are stored together.** A conclusion, a sign-off, a withdrawal, a case created or
  changed (legal hold, retention date, the second-review rule), a member change, a note or annotation, an intake, a new
  revision (with its job), a published result, an account or key change, a policy version and a webhook change are each
  one database transaction with their event: when the event cannot be written (a locked or unreachable database) the request
  answers 500 and nothing was changed, so it can be sent again. A sign-off written by an earlier build without its event
  gets it at the next start, marked `recorded_late`. Reads (`original_viewed`, `original_downloaded`, an export) record first
  and serve afterwards. Sign-ins and the worker's `analysis_started` / `analysis_finished` are appended on their own.
- **Reports and packages.** A report or package request stores its row with its job (and a redacted report with its
  `redaction` event) in one transaction, and the job's last write stores the file, the state `done` and the
  `report_generated` or `package_exported` event in one transaction. A job that finds its report or package already
  finished, failed or deleted writes nothing: a file that was downloadable keeps its SHA-256 and signature. When the event
  cannot be written the report or package is `failed` ("could not be recorded") and is requested again. A render or a
  package build that runs longer than `VERITAS_JOB_TIMEOUT_S` (900 s, the budget of an analysis) fails the report or package
  with "did not finish within the time limit" and frees the worker. At each start, a queued or running report or package
  without a job is queued again, or failed when its job had ended, and a finished one written by an earlier build without
  its event gets the event, marked `recorded_late` (log lines `reports.orphans_repaired`, `reports.custody_repaired`).
- **Verifying a package, and the instance key.** Reports and packages are signed with the instance's Ed25519 key
  (`data/keys/instance_ed25519.pem`, generated at its first use; back it up with the data). A package carries a copy of
  the public key, and that copy proves nothing: whoever alters a package can replace the key, sign every file again and
  rewrite the README. The recipient therefore checks a package against the key as the instance states it, received apart
  from the package:
  - the export dialog shows the key id (`ed25519:` and 16 hexadecimal characters) and saves the public key as a PEM file
    ("Save the key file"); `GET /api/v2/instance/public-key` returns both. Send the id or the file by another route than the
    package (a letter, a call, the case file), once per instance: the key stays the same until the key file is replaced;
  - `python -m app.cli verify <package.zip> --public-key <PEM file>` or `--expect-key-id <key id>`, with no server and no
    network. Exit code 0 and `VERIFIED`: every check passes and the package was signed with that key. Exit code 1 and
    `NOT VERIFIED`: a check fails, a different signing key among them. Exit code 3 and `SELF-CONSISTENT: signer not
    authenticated`: no key was given, so the files agree with each other and nothing says who signed them; a package altered
    and signed again gives the same answer. Exit code 2: the package or the key id cannot be read. `--json` carries the same
    in `result` (`verified`, `not_verified`, `self_consistent`); `ok` is true for `verified` only;
  - `python -m app.cli reproduce` takes the same two options and runs nothing without one (exit code 3);
  - a key id is the first 64 bits of the SHA-256 of the key. It stops a package signed with any other ordinary key; where
    the stake justifies an attacker's search for a key with the same id, compare the whole key with `--public-key`.

**The SQLite database is three files while the instance runs.** `init_db` puts the database into WAL mode at the first
start after this version (D-73): beside `veritas.db` the data directory then holds `veritas.db-wal` (the write-ahead log)
and `veritas.db-shm`. A read no longer blocks a write, and every connection waits up to 30 seconds for the write lock
before it gives up ("database is locked").

- **Running, or killed:** the three files together are the database. `veritas.db` on its own is stale or empty (after one
  analysis it can be 4 KB beside a log of 2 MB), and the three files copied one after the other can be a damaged copy.
- **Stopped cleanly** (SIGTERM, `docker compose stop`, the end of `uvicorn`): the API and the worker fold the log into
  `veritas.db` and close their connections when they stop, and the last of them to stop removes the `-wal` and `-shm`
  files: the one file is then the database. A process that is killed (SIGKILL, a stop that exceeds its grace period, a
  power loss) folds nothing in: the three files stay, and the next start reads the log and loses no committed write.
- **The log is bounded** to 64 MB on disk once a checkpoint has reset it. A read that stays open for a long time holds
  the reset back, and a busy writer grows the log meanwhile; it is cut back after the read ends.
- **What a backup of the database is:** `python -m app.cli backup create` (below), or `sqlite3 veritas.db ".backup
  <file>"`, or a file-system snapshot of the data directory, or all three files of an instance that is stopped. Never
  `veritas.db` alone from a running instance, and never the three files copied while it runs.
- WAL needs a local disk: SQLite on a network file system (NFS, SMB) is not supported, there the mode may not be set and
  the locks are not reliable. `synchronous` stays FULL. Postgres is unchanged.

## Backup and restore

**What an instance is.** Four things, and a restore needs all four from the same moment:

| Part | Where | In `backup create` |
|---|---|---|
| The database | `data/veritas.db` (SQLite), or Postgres | yes (SQLite); Postgres: `pg_dump`, below |
| The evidence | `data/objects` (the store), and the per-analysis files `data/uploads`, `data/artifacts`, `data/reports` | yes: what the database references |
| The keys | `data/keys/instance_ed25519.pem` (signs reports and packages), `data/keys/signing.key` (conclusions, sign-offs, webhook secrets, download links), `data/keys/api_key_pepper.key` (the principals' API keys) | yes, when they are files |
| The environment | `VERITAS_API_KEY`, and `VERITAS_SIGNING_KEY` / `VERITAS_API_KEY_PEPPER` where they are set in place of the two key files | no: keep them in the secret store; the manifest names their fingerprints |

`data/fusion` (fuser artefacts and review records) is in the backup too. Model weights are not: fetch them again
(`python -m app.cli models fetch`, and the fine-tuned text classifier as "Offline operation" says). Staged batch uploads
(`data/batches`) are not: a batch that was still expanding is submitted again.

**SQLite: make a backup.** The instance may keep running.

```
python -m app.cli backup create /backups/veritas-2026-10-01      # a new or empty directory
python -m app.cli backup verify /backups/veritas-2026-10-01
```

- The database is taken first, through SQLite's backup API (one consistent snapshot, the write-ahead log included); then
  every store object that snapshot references, re-hashed while it is copied; then the per-analysis files, the keys and
  `backup.json`, which lists every file with its size and SHA-256, the database revision, the custody head and the key
  ids and fingerprints (never a key). Never copy `veritas.db` alone from a running instance.
- Exit code 0 and `WRITTEN`: complete. Exit code 1 and `INCOMPLETE`: the backup was written and lacks objects the database
  names (`MISSING`: not in the store; `ALTERED`: the bytes no longer hash to their name); the store of the instance is
  damaged, see `store verify`. Exit code 2: refused (the directory holds files; Postgres).
- Do not run `retention purge` or `store gc` while a backup is made: an object they remove after the snapshot is reported
  as missing.
- The directory holds the keys in the clear (mode 0700). Store it where the keys may be, and apart from the instance.
- `backup verify` needs no running instance. It re-hashes every file, refuses unlisted files, runs the database's
  integrity check, checks that every indexed object is there under its own hash, walks the custody chain of the copy,
  checks that the instance signing key is the one the database registered and that every conclusion and sign-off of the
  copy verifies with the backed-up signing key. Exit code 0 and `VERIFIED`, or 1 with the failed checks. A signing key
  that is held in the environment is not in the backup: run `backup verify` where that variable is set, else the check
  "review signatures" fails and names the fingerprint it needs.
- `python -m app.cli store verify` re-hashes every object the running instance's database references (exit code 1 with
  `MISSING` or `ALTERED` lines). Run it after a restore and on a schedule.

**Restore, directory install.**

1. Stop the API and the worker. `backup restore` refuses a data directory on which either still runs ("An instance runs
   on …": they hold `data/instance.lock` while they run; a killed process holds nothing).
2. `python -m app.cli backup restore /backups/veritas-2026-10-01` restores into the configured data directory
   (`--data-dir <path>` names another one). It first checks the backup as `backup verify` does and restores nothing
   from one that fails (exit code 1).
   - Into an empty data directory this is all.
   - Over a data directory that holds files it needs `--replace`. The database that is there is then set aside with its
     log, as `veritas.db.replaced-<time>`, `…-wal` and `…-shm`, and is neither deleted nor left beside the restored file:
     a `veritas.db` put beside the `-wal` of the instance it replaces is not read as it is (the old log is laid over it:
     the database is malformed, or shows the earlier instance's records). Objects and per-analysis files of the earlier
     instance that the backup does not hold stay where they are; `store gc` removes unreferenced objects later.
   - Exit code 0 and `RESTORED`: every file is in place and the database passes its integrity check. Exit code 2:
     refused, nothing changed.
3. By hand, the same is: an empty data directory, then `cp -a /backups/veritas-2026-10-01/. "$VERITAS_DATA_DIR"/` (the
   backup has the layout of a data directory). Never copy a `veritas.db` into a directory that still holds a
   `veritas.db-wal` or `veritas.db-shm`: remove or move those two first.
4. Give the instance the same `VERITAS_API_KEY`, and the same `VERITAS_SIGNING_KEY` and `VERITAS_API_KEY_PEPPER` where
   the manifest says `environment` for them (the command names them).
5. Put the weights back (`models fetch`, or the models directory).
6. Start, then check: `python -m app.cli audit verify`, `python -m app.cli store verify`, and open a signed analysis:
   its review records read as valid, and a package exported before the backup still verifies against the instance key
   (`python -m app.cli verify <package> --expect-key-id <key id>`).

**Restore, Compose volumes.** The data is the volume `veritas-data` (mounted at `/data`), the weights `veritas-models`.

```
# once: the directory of the host that holds the backups, for the user of the image (uid and gid 10001)
sudo install -d -o 10001 -g 10001 -m 0700 /srv/veritas-backups

# back up (the stack may run); --no-deps: models-init is not started again, which would hash every weight file
docker compose run --rm --no-deps -v /srv/veritas-backups:/backups api python -m app.cli backup create /backups/2026-10-01
docker compose run --rm --no-deps -v /srv/veritas-backups:/backups:ro api python -m app.cli backup verify /backups/2026-10-01

# restore: stop the stack, then over the data volume (its database is set aside inside the volume, not deleted)
docker compose down
docker compose run --rm --no-deps -v /srv/veritas-backups:/backups:ro api python -m app.cli backup restore /backups/2026-10-01 --replace
docker compose up -d
docker compose exec api python -m app.cli audit verify
docker compose exec api python -m app.cli store verify
```

The container writes the backup as uid 10001: a `/srv/veritas-backups` that Docker or an administrator made is root's,
and `backup create` then fails on Linux ("Permission denied"). **These commands were not run on a Linux host.** The
supervisor ran them as written on the release commit on Docker Desktop (check 50): `backup create` with the stack
running, `WRITTEN`, 284 files; `backup verify`, seven PASS, `VERIFIED`; `down` and `backup restore … --replace`,
`RESTORED`, the database that was there set aside as `veritas.db.replaced-…`; `up`, `audit verify` VALID and `store
verify` with no object missing or altered. Docker Desktop maps the ownership of a bind mount and does not show the
failure above; run the three once on the server before relying on them.

**A start without the keys is refused.** In prod the API and the worker do not start, and no key is generated, when a
key is missing that records of the database were made with: "Start refused. The signing key is missing and a new one is
not generated: the database holds … conclusion(s), sign-off(s) and webhook(s) signed or sealed with it." The same for the
instance signing key (the database registered a key that signed reports) and for the API-key pepper (principals hold API
keys). Put the key file back from the backup, or set the variable the instance ran with. The one case in which there
never was such a file is an instance that a build before 2.1.1 ran without `VERITAS_SIGNING_KEY`: the refusal then names
`python -m app.cli keys adopt-derived` ("Upgrades", step 4). A first start, with no such
record, generates its keys as before. A key that is present and is not the earlier one starts: the log says
`startup.signing_key_mismatch` (review records then read "signature does not verify") or
`startup.instance_key_replaced` (reports signed earlier verify against the earlier key only).

**Postgres.** `backup create` refuses and prints this procedure:

1. Stop the writers: the API and the worker.
2. As the owner of the database: `pg_dump -Fc -f veritas.pgc <database>`.
3. Copy the data directory with `data/keys` (objects, uploads, artifacts, reports, fusion, keys).
4. Keep `VERITAS_API_KEY`, and `VERITAS_SIGNING_KEY` and `VERITAS_API_KEY_PEPPER` where they are set.

`pg_dump` and `pg_restore` are those of the server's major version (with the Compose file:
`docker compose exec postgres …`). A `pg_dump` of a later major version writes an archive that the server's own
`pg_restore` does not read, and a later `pg_restore` sends the server a setting it does not know.

Restore, into a new, empty database, which is made before anything is dropped:

1. `createdb -O <owner> <new database>` (the owner needs CREATEDB, or an administrator makes the database).
2. As the role that is to own the tables, never as a superuser that is not that role (the tables would be the
   superuser's, and the owner's step refuses them), and always with `--no-owner`:
   - where the roles the dump grants to exist (the same server: a repair, a back-out):
     `pg_restore --exit-on-error --no-owner -d <new database> veritas.pgc`. The dump's own grants come with it: the
     application role reads every table before any owner's step.
   - where they do not (another server, other role names), with one role as with two:
     `pg_restore --exit-on-error --no-owner --no-acl -d <new database> veritas.pgc`. A dump holds every grant that an
     operator made to a role of its own, and names a role as the owner of every object and of the schema.
3. The data directory.
4. With two roles, the owner's step ("Postgres: two roles"), with `--password-env` where the role is to be made:
   `python -m app.cli db setup --app-role <role> --database-url <owner url of the new database> [--password-env NAME]`.
   It sets the zone default and the schema's rights and renews the grants. With one role there is no owner's step.
5. Bring the new database into service: point the service at it (`VERITAS_DATABASE_URL`), or rename the two databases.
   The renaming needs four things: the owner has CREATEDB; it is done from a third database; both renames go in one
   transaction (`BEGIN; ALTER DATABASE <old> RENAME TO <aside>; ALTER DATABASE <new database> RENAME TO <old>;
   COMMIT;`), so that a second that fails undoes the first; and no session is inside either database. What belongs to a
   database stays with it and not with the name: the zone default of the restored database is the one the owner's step
   gave it.
6. Start and check as for SQLite (`audit verify`, `store verify`, a signed analysis, a package). Drop the old database
   when the restored instance has passed its checks.

With `docker-compose.postgres.yml` (the owner is the superuser `veritas`, the database `veritas`; written from that
file and not run by the build loop; the supervisor ran it as written on the release commit on Docker Desktop, check 50:
`pg_restore` exit 0, the case made after the dump gone, `audit verify` VALID, `veritas_app` still refused a `DELETE`
of the custody log): stop `api` and `worker`;
`docker compose exec postgres createdb -U veritas -O veritas veritas_restored`; copy the dump into the container and
`docker compose exec postgres pg_restore -U veritas --exit-on-error --no-owner -d veritas_restored /tmp/veritas.pgc`
(`veritas_app` exists on that server, so the dump's grants come with it); the data volume; rename the two databases
from the database `postgres` as above; `up`, which runs the owner's step on the database named `veritas` before the
API starts.

Three restores lose data. Two of them go into a database the application has already migrated (a start creates the
tables), the third into a database of a later revision:

- `pg_restore --data-only`: fails (exit code 1) and leaves empty every table that is loaded before the table it refers
  to. In the test of this build: `analysis_stages`, `case_media`, `case_members`, `intake_records`; the Postgres QA of
  2026-09-30 counted seven tables with more kinds of records.
- `pg_restore` without `--clean`: every object "already exists" (119 such errors in the test of this build) and the
  result is a mix of the old and the new rows.
- `pg_restore --clean --if-exists` into an existing database: `--clean` drops what the dump names and nothing else. In
  the test of this build a dump taken at revision 0018 and restored so into the same database at the head ended with
  exit code 0 at revision 0018 with 35 tables where 0018 has 32 (`fuser_reviews`, `report_counters` and
  `worker_heartbeats` stayed); the earlier build starts on such a database, and the next owner's step of this build
  stops at "relation "fuser_reviews" already exists". Earlier editions of this runbook gave this form; restore into a
  new database instead.

**Postgres: two roles.** The custody log is append-only only while the role the service connects as cannot drop its
triggers. So the owner role owns the tables and migrates, and the service connects as an application role that owns
nothing: it gets SELECT and INSERT on `audit_events`, SELECT on `alembic_version`, and SELECT, INSERT, UPDATE and DELETE
on the other tables, and USAGE and SELECT on the product's own sequences, by name.

- **The owner's step** is one command, run over an owner connection:
  `python -m app.cli db setup --app-role veritas_app --database-url <owner url> [--password-env NAME]`. It
  1. creates the role with the password in the environment variable `NAME`, or sets that password again when the role
     exists (without `--password-env` the role must exist and is left as it is);
  2. migrates to the head;
  3. sets the database's default time zone to UTC (`ALTER DATABASE … SET timezone TO 'UTC'`): the service sets the zone
     of each of its sessions itself, and this is what a session falls to behind a pooler that resets sessions;
  4. takes CREATE on the schema from PUBLIC and applies the grants;
  5. reads back that the application role may use the schema, may not create in it and reads a table of it.

  It prints one line per step, in the order migration, zone, role, grants. Run it again after every upgrade and after
  every restore: the grants are per table, and a table that a migration added answers "permission denied" until then.
  `db grants --app-role … [--apply]` prints or applies the grants alone; applied, it makes the refusals of the tables'
  owner and of the application role below and ends with the same reading. One step runs at a time on a database: a
  second one that is started meanwhile waits and then finds the head and the role.
  - **Who runs it.** The role that is to own the tables. The role that runs the step owns what it makes (the tables and
    the function `audit_events_append_only()` of the custody log's triggers), so every owner's step of a database runs
    as that one role. It needs the rights of the database's owner (where it is not the owner, have a superuser run
    `ALTER DATABASE <db> OWNER TO <role>`), CREATE on the schema, and every table of the product must belong to a role
    whose rights it has (after a restore: restore as the role that is to own the tables; tables and the function of
    another role are given to it by a superuser: `ALTER TABLE … OWNER TO …` for each table, `ALTER FUNCTION
    audit_events_append_only() OWNER TO …`). To make the application role it needs CREATEROLE. To set the password of a
    role that another administrator made it needs, from PostgreSQL 16, the ADMIN option on that role: without it, run
    the step without `--password-env`, and have a role that may alter it run `ALTER ROLE <role> PASSWORD …` (psql:
    `\password <role>`).
  - **What the application role must not be:** the role of the connection, a superuser or a member (with or without
    INHERIT) of a superuser role, a role with CREATEROLE, the owner of a table of the schema or of the function
    `audit_events_append_only()`, or a member (with or without INHERIT) of the role of the connection or of a role that
    owns the database, the schema, a table of the product or that function. Such a role could switch off the triggers
    that keep the custody log append-only (`DROP FUNCTION … CASCADE` drops both); a role with CREATEROLE needs no such right
    to serve, and on PostgreSQL 15 can make itself a member of the owner of the tables. For the
    function the refusal gives the statement for a superuser: `ALTER FUNCTION audit_events_append_only() OWNER TO
    <the owner of the tables>`.
  - **Refused before anything is changed** (exit code 2, one sentence on stderr): SQLite, a role name that is not a
    plain lower-case identifier, a role that does not exist when no password is given, an application role as above,
    a connection without the rights of the database's owner (the sentence names the database and its owner), without
    CREATE on the schema (the sentence gives the statement for the schema's owner), or with a table of the product
    that belongs to another role (the sentence names that role).
  - **The role comes first.** Where a later statement fails (the migration, for instance), the role exists and its
    password is the new one. The password never goes to the server in clear: the step sends the SCRAM-SHA-256 verifier
    that the driver makes of it, whatever the server's `password_encryption`, and the server stores that verifier.
  - **A statement that PostgreSQL refuses** ends the step with exit code 1 and `error:` with one line. For the role
    statement that line says that nothing else was changed and what to do: for a role that exists, run the command
    without `--password-env`, and have a role that may alter it change the password; for one that does not, have an
    administrator create it and run the command without `--password-env`. A role that another step made at the same
    moment (on another database of the server) is one that exists: the step goes on with it.
  - **A schema that the owner of the tables does not own** (`public` in a database that came from PostgreSQL 14 or
    earlier, or a schema an administrator made): there PostgreSQL only warns when the step takes CREATE from PUBLIC or
    grants USAGE. The reading back then ends the step with exit code 1 and the three statements for the schema's owner:
    `GRANT USAGE, CREATE ON SCHEMA <schema> TO <the owner of the tables>`, `REVOKE CREATE ON SCHEMA <schema> FROM
    PUBLIC`, `GRANT USAGE ON SCHEMA <schema> TO <the application role>`. Run them as that owner, then the step again.
- **With `docker-compose.postgres.yml`** the service `db-setup` runs that step at every `up`, after `postgres` answers
  `pg_isready` and before `api` starts: `veritas` (`VERITAS_PG_PASSWORD`) is the owner, `veritas_app`
  (`VERITAS_PG_APP_PASSWORD`) the role of `api` and `worker`. The passwords reach the processes as `PGPASSWORD`, not in
  `VERITAS_DATABASE_URL`, so they need no escaping; a changed `VERITAS_PG_APP_PASSWORD` takes effect at the next `up`. An
  upgrade is `docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build -d`.
- **A service that starts as the application role on a database behind the head** (a new version started before the
  owner's step, or on an empty database) stops with one sentence: the revision found, the revision needed, what Postgres
  refused, and the command above. Nothing is changed; run the step and start the service again. The API and the worker
  log that sentence as one event, `startup.refused` (level error, the sentence in `reason`), before the server's own
  traceback, which ends with the same sentence; a command of `python -m app.cli` prints `error:` and the sentence and
  ends with exit code 1.
- **By hand, without Compose:** create the database as the owner, run `db setup` with `--password-env`, and set
  `VERITAS_DATABASE_URL` of the API and the worker to the application role.

## Upgrades

### The sequence of an upgrade

For the Compose stack on one host; the commands are those of the sections named. A directory install does the same with
`python -m app.cli …` from `backend/` and its own way of stopping and starting the two processes.

1. **Read.** `CHANGELOG.md`, "What an operator has to know when upgrading a build of this repository", and note what
   runs now: `/api/ready` with a key gives `version` and `build_commit` ("The commit of a build").
2. **Stop the writers.** `docker compose stop`. Each service has 40 s; a job that was running is given back to the queue
   and starts again after the upgrade ("Stopping").
3. **Back up, with the backup command of the new build.** A build before 2.1.1 has no backup command. The command of
   2.1.1 backs up the data directory of an earlier build as it finds it: it leaves the database at the revision it has
   (in the tests on an empty database at revision 0016: source and copy at 0016, `backup verify` seven PASS; on the
   release commit by the supervisor on the data that `5f0cb16` wrote: 278 files, revision 0022, seven PASS, check 50)
   and the migration stays the start's, in step 4. So check out the new commit and build its
   image without starting it: `export VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)`, then `docker compose build api`
   (that command was not run by the loop: no Docker daemon). SQLite: once, `sudo install -d -o 10001 -g 10001 -m 0700
   /srv/veritas-backups` (the container writes as uid 10001), then
   `docker compose run --rm --no-deps -v /srv/veritas-backups:/backups api python -m app.cli backup create /backups/<date>`,
   then `backup verify` on the same directory ("Restore, Compose volumes"). Postgres: the `pg_dump` procedure. The keys of the environment are not in a
   backup ("Backup and restore").
4. **Migrate: start the new build.** With the new commit checked out (step 3), `make up` (or `export
   VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)` and the `docker compose … up --build -d` with the files the stack uses).
   No migration command is run by hand: the API migrates under a lock when it starts, the worker waits for it, and
   with the Postgres file the owner's step (`db setup`) runs before the API at every `up` (below). A migration that
   fails on SQLite leaves the database at the revision it had.
   - **An instance that a build before 2.1.1 ran in prod without `VERITAS_SIGNING_KEY`.** That build signed its
     conclusions, sign-offs and webhook secrets with a key derived from the API key. 2.1.1 never derives the key in
     prod, and its first start on such a data directory is refused after the migration: "Start refused. The signing
     key is missing … They verify under the key that an earlier build derived from VERITAS_API_KEY … Run
     `python -m app.cli keys adopt-derived` once". No key file ever existed, so there is none to restore. Run the
     command, then start again:
     `docker compose run --rm --no-deps api python -m app.cli keys adopt-derived`, then `make up`.
     - It checks every conclusion, sign-off and webhook of the database against the derived key, writes that key to
       `data/keys/signing.key` (0600), records the act in the custody log as the system's (`signing_key_adopted`: the
       file, the key's fingerprint, the numbers of records) and prints what it did. Exit code 0.
     - It writes nothing (exit code 1) when `VERITAS_SIGNING_KEY` is set or the key file is there, when the database
       holds no conclusion, sign-off or webhook (the start then generates the key), and when one record does not verify
       under the derived key: the sentence names how many and the first of them.
     - When the API key had to change for this build (prod refuses one shorter than 32 characters), the start's refusal
       is the general one ("Put data/keys/signing.key back …"), since the start cannot know the earlier key. Give that
       key in a variable of its own: `docker compose run --rm --no-deps -e VERITAS_EARLIER_API_KEY api python -m app.cli
       keys adopt-derived --api-key-env VERITAS_EARLIER_API_KEY`.
     - From then on the signing key is that file and no longer follows the API key. The backup of step 3 does not hold
       it: take a backup after the start ("Backup and restore"), which does. Then change `VERITAS_API_KEY`: a signing
       key derived from the API key lets a holder of a signed record test candidates for it. The review records stay
       valid.
     - Not run in a container or on the data of a real earlier build: the tests make the records with this build outside
       prod, which derives the key the same way.
5. **Fuser release.** The fusion artefacts of the release are inside the new image.
   `docker compose exec api python -m app.cli fusers status --profile full` prints, per kind, the version and SHA-256
   that serve and whether a review is recorded (`reviewed` or `pending`). Per kind the newest version of the data
   directory's `fusion` folder and the packaged ones serves; where both hold one version the packaged file serves, and a
   file with other bytes beside it is named by that command and in the start log. A review record is a file per kind
   and version in the fuser directory; recording one is the owner's step ("Calibrated fusers").
6. **Check the restart.** `/api/ready` with a key answers 200 and names the new `version` and `build_commit`; then
   `docker compose exec api python -m app.cli audit verify` and `… store verify`; open a signed analysis, and verify a
   package issued before the upgrade (`python -m app.cli verify`).
7. **Back out.** The service never downgrades a schema. To return to the earlier build: `docker compose down`, restore
   the backup of step 3 over the data volume (`backup restore … --replace`, "Backup and restore"), check out the
   earlier commit and `make up`. What was analysed after the upgrade is not in that backup.
   On PostgreSQL: the dump of step 3 into a new database, as the role that owns the tables,
   `pg_restore --exit-on-error --no-owner -d <new database> veritas.pgc` (the roles of the dump exist, so its grants
   come with it: an earlier build has `db grants --apply` and no owner's step, and this build's owner's step would
   migrate the restored database again); the data directory of step 3; the earlier build with `VERITAS_DATABASE_URL` set to that database.
   Never `--clean --if-exists` into the upgraded database ("Backup and restore", the restores that lose data). The
   Postgres reviewer ran this form with an earlier build (`postgres-qa.md`, the re-check at 878fa09, B3): ff77a6c with
   two roles, the restore with `--no-owner` (exit code 0, revision 0018, 32 tables, the application role reads 32 of
   32), the earlier build pointed at it: ready 200, and 491 of 514 GET responses equal to those before the upgrade, the
   23 others one analysis that was queued when the writers stopped and ran after the back-out, and the custody log's
   length. The restored database has no zone default, which a build that sets the
   zone of its own sessions (every build since the one that pins a session's zone at connect, N-4b) does not need.

Rehearsed as one sequence on the release commit for the Compose stack on SQLite, steps 1 to 7: the supervisor upgraded
the data that `5f0cb16` wrote on 4 October (schema 0022, 8 analyses, 6 cases, 22 reports) to `a876a43a`
(`docs/v2/qa/prod-readiness-2026-09-29/supervisor-checks.md`, check 50). Step 3 backed up 278 files, seven PASS; step 4
was healthy in 38 s after "Running upgrade 0022 -> 0023"; step 5 named the three packaged fusers 2.1.1, review pending;
in step 6 `/api/ready` named 2.1.1 and `a876a43a`, `audit verify` was VALID, `store verify` found no object missing,
every case's custody was valid, the signed analysis's records were valid, and the package that `5f0cb16` issued
verified with its key. Step 7 ran as written on the release (check 51): `docker compose down`, the backup of step 3
restored over the upgraded data volume with `backup restore … --replace` (`RESTORED`, 278 files, custody head 119), and
`5f0cb16` started on it at revision 0022; the analyses made after the upgrade were gone, `audit verify` was VALID,
`store verify` found 0 missing, every case's custody was valid, the signed analysis's records were valid, and the
package that `5f0cb16` issued verified with the instance's key. Not covered by it: a Linux host, and a back-out after an
upgrade that ran `keys adopt-derived`. Before it, the supervisor's check 1
upgraded a copy of the development instance in place. On PostgreSQL the Postgres QA ran steps 2 to 4 and 6 at
`a511ecf` as a directory install without Compose, on data that an earlier build (`ff77a6c`) had written with one role;
its back-out used the `--clean --if-exists` form that is gone from this runbook.

### What an upgrade does

- **Database schema:** Alembic (`backend/alembic/`). The API, the worker and the commands that write migrate to head when
  they start (`init_db()`); an unversioned v0.3 database is stamped at `0001` (baseline) first. Manual control:
  `cd backend && .venv/bin/alembic current | upgrade head` (URL from `VERITAS_DATABASE_URL` / `VERITAS_DATA_DIR`). Back up the
  database before upgrading.
- **One process migrates at a time.** A start that finds the database at the head changes nothing and takes no lock.
  One that finds an empty or an older database takes a lock and reads the revision again under it, so of processes that
  start together (the API and a worker, several workers, a command beside a start) the first one migrates and the others
  wait for it, find the head and change nothing. On SQLite the lock is the write lock, waited for up to 5 minutes
  instead of the 30 seconds of ordinary work, and the whole schema change is one transaction: a migration that fails
  leaves the database at the revision it had. On Postgres it is an advisory lock of the schema. The first start after an
  upgrade from a version before the write-ahead log waits for another process that is writing instead of failing.
- **Postgres with two roles:** the migration is the owner's step, `python -m app.cli db setup` ("Postgres: two roles");
  the service's own role cannot migrate and says so at its start. `docker-compose.postgres.yml` runs the step before the
  API at every `up`.
- **Commands that only read migrate nothing:** `audit verify` and `custody export` read the database as they find it,
  also without `--database-url`. On a database of an earlier version they either work or say that the custody log
  cannot be read as it is; they never bring it up to date. Start the service (or run `alembic upgrade head`) for that.
- **Foreign keys on SQLite:** every connection enforces them, as Postgres does; migrations run with the enforcement off
  (a table rebuild needs that) and end with `PRAGMA foreign_key_check`. A database written by an earlier version may hold rows
  whose parent row is gone: each start logs one warning, `db.foreign_key_violations`, with their number by table, changes
  none of them and goes on. Only a new or changed reference is refused.
- **Times are UTC on both engines.** Every Postgres session of the API, the worker and the command line is set to UTC when it
  connects (`SET TIME ZONE 'UTC'`), whatever the zone of the server or of the database; nothing has to be configured, and
  the stored instants do not change (every time column holds a zone). The API prints times in UTC with the suffix `Z`. A
  date a client sends (`retention_until` of a case, `date_acquired` of an item) needs a UTC offset and a year from 1970 to
  9000 after conversion to UTC, else the request is refused (422). A retention date that an earlier version stored from a
  value without an offset, on a server whose zone is not UTC, keeps the instant it was stored as: read the case and set the
  date again.
- **Migration 0018** makes `analyses.size_bytes` 64-bit and adds six indexes on referencing columns. On SQLite it rebuilds the
  `analyses` table (a copy of every row: allow for the time and the disk space of one more copy of that table); on Postgres the
  column change rewrites the table under an exclusive lock. Its downgrade is refused by Postgres while a stored size exceeds
  32 bits.
- **Migration 0021** adds an index on `analyses.supersedes_id`, which the lists of analyses and of cases need once revisions
  have failed or been cancelled: without it each such revision added a scan of the table to every list request on SQLite
  (about 3 s a request at 20,000 analyses with 1,000 of them, 0.04 s with the index). It is built at the first start, without
  a table rebuild; on Postgres `CREATE INDEX` holds a lock against writes to `analyses` while it runs.
- **Models:** change `backend/app/models.yaml`, run `python -m app.cli models pin <id>` (a human-reviewed diff of revision and
  per-file sha256), `models verify`, then the bench (`make vbench-score`, `make calibrate`, `make vbench-report`,
  `make vbench-gate`) before enabling a detector by default.
- **Fusion weights:** never change them without re-running calibration. The packaged artefacts are never edited (their SHA-256
  is cited by gates, records and published results); an artefact of a newer version placed in `<data dir>/fusion` is served in place
  of the packaged one of its kind, and a re-run of an earlier analysis still finds the artefact it names by its SHA-256 in
  either directory.
- **Dependencies:** `pip-audit` and `npm audit` in the security review; `transformers` pinned to 5.x; `weasyprint` pinned exactly
  (report determinism).

## Troubleshooting

| Symptom | Check |
|---|---|
| `/api/ready` answers 503 | ask it with an API key and read `checks`: `database` (locked or away), `storage` (the data directory cannot be written, or the volume is below `VERITAS_MIN_FREE_BYTES`), `worker` (no heartbeat, a stale one, or threads that do not reach the database). The API log has `ready.not_ready` with the same reasons. See "Monitoring" |
| The API or the worker does not start, and the log holds `startup.refused` | read its `reason`: the one sentence of the refusal, with the step that cures it. Four things refuse a start: a migration that the database role may not make ("Postgres: two roles"), an unversioned database of an earlier build whose `analyses` table does not match the first revision, a key that existing records were made with and that is missing ("Backup and restore", "Upgrades"), and a restore that is writing into the data directory. The traceback the server prints after it ends with the same sentence |
| Uploads answer 507 | the data volume is below `VERITAS_MIN_FREE_BYTES`, or a write met a full volume (`api.storage_full` in the log). Free space (`make purge`, then `store gc`, or a larger volume). Both free bytes while work waits in the queue; only a job that is running blocks them (wait for it, or for its lease to run out). Queued analyses then run without a restart. See "Monitoring", "Storage" |
| Analyses stuck `queued` | `veritas_workers 0` in `/api/metrics`, or `/api/ready` → `checks.worker`: no worker. A worker that logs `worker.on_hold` is waiting for free space. Else no worker: `docker compose ps worker`, or `VERITAS_RUN_WORKER=false` in a single process. A job that fails before its run (`worker.job_failed`, then `jobs.failed_outside_runner`) ends its analysis as `failed` with the reason; one left `queued` beside a failed job by an earlier version is ended as failed at the next start |
| Results read "uncalibrated" or are capped | `GET /api/ready` with an API key → `fusion`; `python -m app.cli fusers status` gives the reason (weights not installed, `quick` profile, `VERITAS_FUSER_DIR` pointing at an empty directory); the start log has `fusion.fallback_at_start` |
| An analysis failed with "Processing was interrupted 3 times; the input is not re-queued again." | the worker that ran it was killed, or lost its lease, `VERITAS_MAX_REQUEUE_ATTEMPTS` (3) times: a kill for memory (`docker inspect` shows `OOMKilled`; "Sizing"), a host that stopped, a database that was away for longer than the lease. A stop with SIGTERM (`docker compose stop`, an upgrade) does not count: the worker gives the job back. The limit is there so that an input which kills the worker is not tried for ever. Cure the cause, then reanalyse: `POST /api/v2/analyses/{id}/reanalyse` queues a new revision of the same original. `…/rerun` answers 409, because there is no published result to reproduce. `veritas_jobs_attempts_exhausted` counts these failures ("Monitoring") |
| `models-init` ended with an error or with exit code 137, and `api` and `worker` did not start | `docker compose logs models-init`: an `ERROR` line per model names the cause (no network, a file that changed at its source). Code 137 without such a line is a kill for memory: raise `VERITAS_FETCH_MEM_LIMIT` (4g by default). Then run `docker compose up -d` again: files that were fetched and verified are kept and are not fetched again, and the fetch goes on with the rest (its first line says so) |
| Detectors `skipped: memory budget` on every analysis | the model memory budget is too small for them: the worker's start log has one `capacity.detector_cannot_run` per detector, `/api/ready` with an API key lists them under `capacity`, and `python -m app.cli models plan` names each detector that cannot run under this server's budget, with its need and the smallest budget that runs the whole profile. Raise the memory (and, in Compose, `VERITAS_WORKER_MEM_LIMIT`) or set `VERITAS_MODEL_MEMORY_BUDGET_MB`; see "Sizing". A detector that is skipped only now and then fits the budget alone: the memory was held by another check that ran at that moment. The result and its reports say such a check "could not run on this server", count it in "What this means", and give as the next step to ask the administrator; analyse the file again once the server can run it |
| Detectors `skipped: model not installed` | weights missing under offline defaults: run `models-init` / `models fetch --profile full` with `VERITAS_OFFLINE=false`. The result and its reports say such a check "could not run on this server: its model files are not installed"; analyse the file again once they are |
| `failed` with "could not be decoded" | ffmpeg/ffprobe missing in the image, or corrupt upload (expected outcome) |
| Progress bar never moves behind a proxy | the proxy buffers SSE: `proxy_buffering off` for `/api/v2/analyses/*/events` |
| 500 with `database is locked` in the log (SQLite) | a write waited 30 s for another writer: look for a long transaction (`retention purge`, a report request on a long custody log), and that the data directory is on a local disk; `sqlite3 veritas.db 'PRAGMA journal_mode'` answers `wal` |
| A result lists a check as "failed: The record of this check's run could not be stored, so its result is not used" | the run of that detector could not be written after three attempts (`detector.persist_retry`, then `detector.persist_failed` in the log, with the error): the database was locked for longer than the waits, or the data directory was full or not writable. The result was built without that check. Cure the cause, then re-run the analysis; `detector.failed_run_not_stored` means that not even the failed run could be written, so the run list has no row for the check and coverage still counts it as failed |
| 429s from the UI | `VERITAS_RATE_LIMIT_PER_MINUTE` too low for the poll interval (default 300) |
| MPS/CUDA errors | set `VERITAS_DEVICE=cpu` to isolate |
| Memory killed | lower `VERITAS_MODEL_MEMORY_BUDGET_MB`, use the `quick` profile, lower `max_*_bytes` |
| Reports fail with 503 | Pango missing (WeasyPrint); the image installs `libpango-1.0-0` / `libpangoft2-1.0-0` |

## Incident response

1. Check `/api/health` (liveness) and `/api/ready` with an API key (readiness, with the failing check named); tail structured
   logs filtered by `analysis_id` or `request_id`.
2. A bad verdict is an evidence problem, not an outage: export the evidence package, review `signals[]` and the manifest, and file
   the sample for the next benchmark run.
3. Never suppress the limitations block in reports: it is part of the product's legal posture.
