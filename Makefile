.PHONY: setup backend frontend dev-api dev-web stop stop-all-dev smoke e2e e2e-demo e2e-fake test test-slow egress audit deploy-lint golden lint settings-doc sizing-doc notices-doc \
	typecheck build ci ci-release api-types api-types-check test-frontend eval-image bench eval-text purge up down \
	vbench-fetch vbench-audit vbench-split vbench-perturb vbench-score calibrate vbench-report vbench-gate vbench-smoke shortcut-gate

BG := bash backend/scripts/bg.sh
API_TYPES_TMP := frontend/.api-types-check
E2E_BASE_URL ?= http://localhost:5230
SHARDS ?= 4
VERITAS_MODELS_DIR ?= $(CURDIR)/models
VERITAS_HF_HOME ?= $(VERITAS_MODELS_DIR)/hf
# The commit an image is built from (N-15; D-58): the checkout's HEAD, or the value given where there is no checkout (a
# release archive). Every result manifest, the probes and the metrics of the image state it.
VERITAS_BUILD_COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)

setup:            ## install the backend (uv, Python 3.12) at the tested versions of backend/requirements/tested-versions.txt, the frontend from its lock file (npm ci) and the browser of the e2e gate; run again, it keeps the environment and brings it back to those versions
	cd backend && { test -x .venv/bin/python || uv venv --python 3.12; } \
		&& uv pip install --python .venv/bin/python -e ".[dev,eval,postgres]" pip -c requirements/tested-versions.txt
	cd frontend && npm ci && npx playwright install chromium
	@cd backend && .venv/bin/python scripts/runtime_contract.py --backend local >/dev/null \
		|| echo "The installed versions are not those of backend/requirements/tested-versions.txt (it describes macOS on Apple silicon): cd backend && .venv/bin/python scripts/runtime_contract.py --backend local names the difference"
	@cd backend && .venv/bin/python -c "from app.reports.engine import load_weasyprint; load_weasyprint()" >/dev/null 2>&1 \
		|| echo "Pango is missing, so v2 PDF reports cannot render: macOS brew install pango; Debian apt install libpango-1.0-0 libpangoft2-1.0-0"

backend:          ## run API with reload on :8160 (foreground)
	cd backend && VERITAS_MODELS_DIR="$(VERITAS_MODELS_DIR)" VERITAS_HF_HOME="$(VERITAS_HF_HOME)" .venv/bin/uvicorn app.main:app --reload --port 8160 --no-proxy-headers

frontend:         ## run Vite dev server on :5230 (foreground; proxies /api -> :8160)
	cd frontend && npm run dev

dev-api:          ## start the dev API on :8160 detached (PID file backend/data/jobs/dev-api.pid)
	@! lsof -nP -iTCP:8160 -sTCP:LISTEN >/dev/null || { echo "port 8160 is in use:"; lsof -nP -iTCP:8160 -sTCP:LISTEN; exit 1; }
	cd backend && VERITAS_MODELS_DIR="$(VERITAS_MODELS_DIR)" VERITAS_HF_HOME="$(VERITAS_HF_HOME)" bash scripts/bg.sh start dev-api -- .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8160 --no-proxy-headers

dev-web:          ## start the Vite dev server on :5230 detached (PID file backend/data/jobs/dev-web.pid)
	@! lsof -nP -iTCP:5230 -sTCP:LISTEN >/dev/null || { echo "port 5230 is in use:"; lsof -nP -iTCP:5230 -sTCP:LISTEN; exit 1; }
	cd frontend && bash ../backend/scripts/bg.sh start dev-web -- ./node_modules/.bin/vite --host 127.0.0.1 --port 5230 --strictPort

stop-all-dev:     ## stop the dev API/UI started by dev-api/dev-web (PID files only, never by pattern)
	@$(BG) stop dev-web
	@$(BG) stop dev-api

stop: stop-all-dev

smoke:            ## end-to-end smoke test against the running API (:8160)
	cd backend && .venv/bin/python scripts/smoke_e2e.py

e2e:              ## browser end-to-end flows against the running UI (:5230): uploads, verdicts, evidence, PDF/JSON, delete, theme
	cd frontend && node scripts/e2e.mjs "$(E2E_BASE_URL)"

e2e-demo:         ## browser test of the guided demo walkthrough (Demo now → 17 steps → Finish)
	cd frontend && node scripts/e2e_demo.mjs http://localhost:5230

test:             ## backend unit + API tests (model-free; non-loopback sockets blocked) as SHARDS pytest processes over disjoint sets of test files; SHARDS=1 is one serial pytest [SHARDS=4]
	cd backend && .venv/bin/python scripts/pytest_shards.py --shards $(SHARDS)

test-slow:        ## real-model integration tests (needs weights)
	cd backend && .venv/bin/pytest -m slow

egress:           ## PL-01 egress check: one analysis of every kind with the installed models under the socket monitor, prod defaults (heavy: idle host)
	cd backend && .venv/bin/pytest -o addopts="" -m slow tests/test_egress_slow.py -s

audit:            ## dependency advisories (Q-2; N-15): pip-audit of both runtime locks, every package by name and version, and npm audit → docs/v2/qa/pip-audit-runtime-{cpu,cu126}.json, npm-audit.json and the receipt audit-run.json [AUDIT_OUT=<directory, relative to backend/>]; fails on a missing tool, a package that was not audited, a pip advisory or an npm high / critical advisory outside docs/v2/qa/audit_exceptions.yaml (network: advisory databases)
	cd backend && .venv/bin/python scripts/audit_summary.py --run $(if $(AUDIT_OUT),--out $(AUDIT_OUT))

settings-doc:     ## write docs/settings.md from Settings.model_fields (tests/test_settings_doc.py fails when it is stale)
	cd backend && .venv/bin/python scripts/settings_doc.py

sizing-doc:       ## write the memory tables of docs/runbook.md "Sizing" from the capacity plan and its latency table from docs/v2/bench/perf.json (tests/test_sizing_doc.py and test_n16_sizing_latency.py fail when stale)
	cd backend && .venv/bin/python scripts/sizing_doc.py

notices-doc:      ## write THIRD_PARTY_NOTICES.md from the licence files of app/third_party and the report fonts, the font packages of the npm lock and the model manifest (tests/test_notices_doc.py fails when stale)
	cd backend && .venv/bin/python scripts/notices_doc.py

deploy-lint:      ## the Docker Compose version (2.24.4 or later), then interpolate and validate the compose files without the Docker daemon (images are built by `make up`)
	backend/.venv/bin/python backend/scripts/compose_version.py
	VERITAS_API_KEY=lint docker compose -f docker-compose.yml config -q
	VERITAS_API_KEY=lint VERITAS_PG_PASSWORD=lint VERITAS_PG_APP_PASSWORD=lint \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.postgres.yml config -q
	VERITAS_API_KEY=lint VERITAS_TLS_CERT=/etc/veritas/tls/fullchain.pem VERITAS_TLS_KEY=/etc/veritas/tls/privkey.pem \
		docker compose -f docker-compose.yml -f docker-compose.tls.yml config -q

golden:           ## write a detector's golden outputs (ARCHITECTURE §9.3): DETECTOR=<id> [DEVICES=cpu,mps] [FORCE=1]
	cd backend && .venv/bin/python scripts/v2/make_golden.py --detector $(DETECTOR) $(if $(DEVICES),--devices $(DEVICES)) $(if $(FORCE),--force)

test-frontend:    ## frontend unit tests (vitest + Testing Library, jsdom)
	cd frontend && npm run test

lint:
	cd backend && .venv/bin/ruff check app tests scripts alembic

typecheck:        ## frontend type check (tsc -b)
	cd frontend && npm run typecheck

api-types:        ## regenerate frontend/src/lib/api/schema.{json,d.ts} from the FastAPI app (commit both)
	cd backend && .venv/bin/python scripts/export_openapi.py ../frontend/src/lib/api/schema.json
	cd frontend && npx openapi-typescript src/lib/api/schema.json -o src/lib/api/schema.d.ts

api-types-check:  ## fail when the committed OpenAPI schema or generated TS types drift from the backend
	rm -rf $(API_TYPES_TMP) && mkdir -p $(API_TYPES_TMP)
	cd backend && .venv/bin/python scripts/export_openapi.py ../$(API_TYPES_TMP)/schema.json
	cd frontend && npx openapi-typescript .api-types-check/schema.json -o .api-types-check/schema.d.ts
	diff -q frontend/src/lib/api/schema.json $(API_TYPES_TMP)/schema.json || { echo "schema.json is stale: run make api-types"; exit 1; }
	diff -q frontend/src/lib/api/schema.d.ts $(API_TYPES_TMP)/schema.d.ts || { echo "schema.d.ts is stale: run make api-types"; exit 1; }
	rm -rf $(API_TYPES_TMP)

report-tokens:    ## export the light design tokens of frontend/src/index.css to backend/app/reports/templates/tokens.css (commit it)
	cd backend && .venv/bin/python scripts/report_tokens.py

report-fonts:     ## instance the bundled OFL report fonts from frontend/node_modules/@fontsource-variable into backend/app/reports/fonts (commit them)
	cd backend && .venv/bin/python scripts/report_fonts.py

e2e-design:       ## /design gallery (E-0): Vite on a probed port from 5239 (PID file), light/dark × 1440/900 + achromatopsia → frontend/test-results/design/
	bash frontend/scripts/e2e_design.sh

e2e-fake:         ## model-free browser suite (§9.6, D-19): fake-detector API + built SPA on a probed port from 8169, @playwright/test [SPEC=e2e/<file>.spec.ts]
	bash frontend/scripts/e2e_fake.sh $(SPEC)

build:            ## production frontend build (served by the API at / when present)
	cd frontend && npm run build

ci:               ## per-iteration gate (ARCHITECTURE §9.7): lint, model-free tests, typecheck, API types, build, frontend unit tests, e2e-fake
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) typecheck
	$(MAKE) api-types-check
	$(MAKE) build
	$(MAKE) test-frontend
	$(MAKE) e2e-fake

ci-release:       ## release gate (N-15; OPS-13): make ci with the backend suite serial, the Postgres suite on a disposable server (backend/scripts/pg_suite.sh all), every test of infra/gcp/tests and terraform test, deploy-lint, and make audit; a missing tool fails it before anything runs; builds no image; about 75 minutes; output in backend/data/ci-release/ [PG_SHARDS=4]
	bash backend/scripts/ci_release.sh

eval-image:       ## public-benchmark evaluation of image detectors -> docs/eval_image.md
	cd backend && .venv/bin/python scripts/eval_image.py --per-source 100

bench:            ## X-4 warm latency, stage / detector breakdown and memory per profile against the dev API (`make dev-api`; PROFILES=quick,full,deep ONLY=<labels> RUNS=<n> WORKER_PID=<pid> COVERAGE_CONTRACT=<json> PERF_OUT=<new-json>) → docs/v2/bench/perf.{json,md}
	cd backend && { [ -f data/bench/perf/workloads.json ] || .venv/bin/python scripts/bench.py --make-workloads; }
	cd backend && .venv/bin/python scripts/bench.py --base http://127.0.0.1:8160 --pid $$(bash scripts/bg.sh pid dev-api) \
	  --profiles $(if $(PROFILES),$(PROFILES),quick,full,deep) $(if $(ONLY),--only $(ONLY)) $(if $(RUNS),--runs $(RUNS)) \
	  $(if $(WORKER_PID),--worker-pid $(WORKER_PID)) $(if $(COVERAGE_CONTRACT),--coverage-contract "$(COVERAGE_CONTRACT)") $(if $(PERF_OUT),--out "$(PERF_OUT)")

vbench-fetch:     ## sample Veritas-Bench (ML §2.10): SUBSET=mini for vb-mini; DRY_RUN=1 writes docs/v2/bench/fetch_plan.md only; CACHED_ONLY=1 rebuilds manifest + report with no network request; MAX_SECONDS=n time-boxes the run (D-40); REPORT=path (relative to backend/)
	cd backend && .venv/bin/python scripts/vbench_fetch.py --spec ../docs/v2/bench/sources.yaml \
		$(if $(SUBSET),--subset $(SUBSET)) $(if $(DRY_RUN),--dry-run) $(if $(ONLY),--only $(ONLY)) $(if $(CACHED_ONLY),--cached-only) \
		$(if $(MAX_SECONDS),--max-seconds $(MAX_SECONDS)) $(if $(REPORT),--report $(REPORT))

vbench-audit:     ## agent label audit (D-06): contact sheets → backend/data/vbench/audit/ (SOURCES=a,b); RENDER=1 checks audit.yaml → audit.md
	cd backend && .venv/bin/python scripts/vbench_audit.py $(if $(SOURCES),--sources $(SOURCES)) $(if $(RENDER),--render)

vbench-split:     ## audit exclusions, fit/cal/test splits, LGO check, shortcut test (ML §2.6, P3) and D-45 bucketed gate 2a (exit 1 refuses) → manifest + docs/v2/bench/mini_summary.md; BENCH=vb-1.1 writes docs/v2/bench/vb-1.1_summary.md (SUMMARY=path overrides)
	cd backend && .venv/bin/python scripts/vbench_split.py $(if $(SUMMARY),--summary $(SUMMARY),$(if $(BENCH),--summary ../docs/v2/bench/$(BENCH)_summary.md))

vbench-perturb:   ## perturbation suite (ML §2.7) on the robustness subset → backend/data/vbench/perturbed/ (resumable; TRANSFORMS=a,b; MANIFEST= a scope parquet whose `robustness` column names the subset)
	cd backend && .venv/bin/python scripts/vbench_perturb.py --subset $(or $(SUBSET),robustness) \
		$(if $(MANIFEST),--manifest $(MANIFEST)) $(if $(TRANSFORMS),--transforms $(TRANSFORMS)) $(if $(LIMIT),--limit $(LIMIT))

vbench-score:     ## score through the production path → backend/data/vbench/scores/ (heavy: bg.sh --heavy, D-20); SCORER=v1-video VARIANT=native for the v0.3 baseline; MANIFEST=… (default vb-1.1 `data/vbench/manifest.parquet`, D-40; vb-1.0-core is `data/vbench/core/manifest.parquet`)
	cd backend && .venv/bin/python scripts/vbench_score.py --manifest $(or $(MANIFEST),data/vbench/manifest.parquet) \
		--variant $(or $(VARIANT),norm) --transform $(or $(TRANSFORM),none) \
		--subset $(or $(SUBSET),all) $(if $(LIMIT),--limit $(LIMIT)) $(if $(SCORER),--scorer $(SCORER))

calibrate:        ## fit the fuser artefact from the score cache (ML §4.8) → backend/data/fusion/fuser-<modality>-<version>.json + .boot.npy; MANIFEST=… (default vb-1.0-core)
	cd backend && .venv/bin/python scripts/calibrate.py --manifest $(or $(MANIFEST),data/vbench/manifest.parquet) --scores data/vbench/scores \
		--modality $(or $(MODALITY),image) --seed $(or $(SEED),7) --out data/fusion/ \
		$(if $(VERSION),--version $(VERSION)) $(if $(BOOTSTRAP),--bootstrap $(BOOTSTRAP)) $(if $(BENCH),--bench $(BENCH)) \
		$(if $(DETECTORS),--detectors $(DETECTORS)) $(if $(EXCLUDE_FOLDS),--exclude-folds $(EXCLUDE_FOLDS)) $(if $(FORCE),--force) \
		$(if $(GATES),--gates '$(GATES)') $(if $(CONTEXT),--context $(CONTEXT)) $(if $(SCOPE),--scope $(SCOPE)) \
		$(if $(ABLATE),--ablate $(ABLATE)) $(if $(ABLATE_BOOTSTRAP),--ablate-bootstrap $(ABLATE_BOOTSTRAP)) \
		$(if $(DROP_CONTEXT_TERMS),--drop-context-terms '$(DROP_CONTEXT_TERMS)') \
		$(if $(REFERENCE),--reference $(REFERENCE)) $(if $(REFERENCE_GATES),--reference-gates '$(REFERENCE_GATES)') \
		$(if $(REFERENCE_LABEL),--reference-label $(REFERENCE_LABEL))

vbench-report:    ## report (ML §2.8, §5.1) → docs/v2/bench/results.{md,json}, error_rates.json: ARTEFACT=path [LGO=dir] or SCORE_KEY=key; VIDEO_SCORE_KEY / AUDIO_SCORE_KEY; SPLIT=test; MANIFEST / BENCH (default vb-1.0-core); UPDATE_CARDS=1 writes card eval rows
	cd backend && .venv/bin/python scripts/vbench_report.py --manifest $(or $(MANIFEST),data/vbench/manifest.parquet) --scores data/vbench/scores \
		$(if $(ARTEFACT),--artefact $(ARTEFACT)) $(if $(SCORE_KEY),--score-key $(SCORE_KEY)) $(if $(LGO),--lgo-artefacts $(LGO)) \
		--eval-split $(or $(SPLIT),test) --bench $(or $(BENCH),vb-1.0-core) $(if $(LATENCY),--latency $(LATENCY)) $(if $(SCOPE),--scope $(SCOPE)) \
		$(if $(WRITE_ARTEFACT),--write-artefact) $(if $(VIDEO_SCORE_KEY),--video-score-key $(VIDEO_SCORE_KEY)) \
		$(if $(AUDIO_SCORE_KEY),--audio-score-key $(AUDIO_SCORE_KEY))

vbench-gate:      ## goals G1–G15 (ML §8) on docs/v2/bench/results.json; exit 1 on any FAIL or missing metric (BENCH=… for a non-goal bench)
	cd backend && .venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
		--manifest $(or $(MANIFEST),data/vbench/manifest.parquet) $(if $(BENCH),--bench $(BENCH))

shortcut-gate:    ## diagnostic run of the D-45 / D-52 shortcut gates (served artefacts, 2a–2c, structure; exit 1 refuses) → docs/v2/bench/shortcut_gate.json; ARTEFACT= (a diagnostic input, fails `served`), SCOPE= (default vb-1.1 scope)
	cd backend && .venv/bin/python scripts/shortcut_gate.py --manifest $(or $(MANIFEST),data/vbench/manifest.parquet) --scores data/vbench/scores \
		$(if $(ARTEFACT),--artefact $(ARTEFACT)) --scope $(or $(SCOPE),data/vbench/scope/vb-1.1.parquet)

final-gate:       ## Q-5 completion gate (D-02): make ci with the backend suite serial (includes e2e-fake) → shortcut gates as amended by D-52 (served artefacts; refusal) → goal table (FAIL / MISSING allowed, refusal not) → check_shortfalls (every non-PASS goal recorded, caps checked); takes no ARTEFACT, SCOPE or MANIFEST
	$(if $(ARTEFACT)$(SCOPE)$(MANIFEST),$(error final-gate takes no ARTEFACT, SCOPE or MANIFEST))
	$(MAKE) ci SHARDS=1
	cd backend && .venv/bin/python scripts/shortcut_gate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores \
		--scope data/vbench/scope/vb-1.1.parquet
	cd backend && { .venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
		--manifest data/vbench/manifest.parquet; code=$$?; [ $$code -le 1 ] || exit $$code; }
	cd backend && .venv/bin/python scripts/check_shortfalls.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
		--manifest data/vbench/manifest.parquet

vbench-smoke:     ## D-13: scoped serving-ensemble fit/cal smoke; new run under docs/v2/bench/smoke-v2/ (heavy: bg.sh --heavy)
	cd backend && .venv/bin/python scripts/vbench_smoke.py $(if $(SKIP_SCORE),--skip-score)

vbench-core:      ## B-4c (D-39): vb-1.0 → seeded stratified vb-1.0-core → backend/data/vbench/core/manifest.parquet + docs/v2/bench/vb-1.0-core_summary.md; pins goals.yaml (NO_PIN=1 leaves it)
	cd backend && .venv/bin/python scripts/vbench_core.py $(if $(NO_PIN),--no-pin)

vbench-scope:     ## H-2 (D-40): vb-1.1 scoring scope → backend/data/vbench/scope/vb-1.1.parquet (every image real, ≤ 5,000 image fakes, ≤ 350 video reals even over sources (D-43) + ≤ 300 fakes, all audio, 600 robustness images) + docs/v2/bench/vb-1.1_scope.md; MANIFEST= OUT= SUMMARY= (paths relative to backend/)
	cd backend && .venv/bin/python scripts/vbench_scope.py $(if $(MANIFEST),--manifest $(MANIFEST)) $(if $(OUT),--out $(OUT)) $(if $(SUMMARY),--summary $(SUMMARY))

probe-features:   ## I-13: class-blind views of the D-12 pool → backbone features → backend/data/vbench/features/ (BACKBONE=clip_l14|siglip2_so400m|pe_core_l14; FP16=1 after parity; heavy: bg.sh --heavy)
	cd backend && .venv/bin/python scripts/probe_features.py --backbone $(or $(BACKBONE),clip_l14) $(if $(FP16),--fp16) \
		$(if $(VIEWS),--views $(VIEWS)) $(if $(LIMIT),--limit $(LIMIT))

train-probe:      ## I-13: nested-LGO probe head on cached features → backend/data/weights/veritas-probe-v1[-lgo-F]/ (BACKBONE=clip_l14 LGO=F2|none)
	cd backend && .venv/bin/python scripts/train_probe.py --backbone $(or $(BACKBONE),clip_l14) --pool permissive \
		--lgo-exclude $(or $(LGO),none) $(if $(VIEWS),--views $(VIEWS)) $(if $(FORCE),--force)

eval-text:        ## HC3 evaluation + Binoculars calibration -> docs/eval_text.md
	cd backend && .venv/bin/python scripts/eval_text.py --n 150

purge:            ## retention (D-55): purge cases past their retention date and unsigned quick analyses older than VERITAS_RETENTION_DAYS, with their bytes; held, retained, signed, in-progress and named-case records stay [DRY_RUN=1 lists the decisions]
	cd backend && .venv/bin/python -m app.cli retention purge $(if $(DRY_RUN),--dry-run)

up:               ## docker compose stack (models-init, which fetches the weights and ends, then api + worker + nginx frontend), the images built from this commit [VERITAS_BUILD_COMMIT=<full commit> outside a checkout]
	@echo "$(VERITAS_BUILD_COMMIT)" | grep -Eqx '[0-9a-f]{40}' \
		|| { echo "make up: no commit to build from: this is not a git checkout, so set VERITAS_BUILD_COMMIT=<the full commit of this source>"; exit 1; }
	@git diff --quiet HEAD 2>/dev/null || echo "make up: the working tree differs from $(VERITAS_BUILD_COMMIT) (or is no checkout): the images will name that commit all the same"
	VERITAS_BUILD_COMMIT=$(VERITAS_BUILD_COMMIT) docker compose up --build -d

down:
	docker compose down
