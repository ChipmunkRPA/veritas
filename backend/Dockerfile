# syntax=docker/dockerfile:1
# Veritas API and worker image (ARCHITECTURE §8.4). One image serves `api`, `worker` and `models-init` (docker-compose.yml).
# Every build names its commit (N-15): C='--build-arg VERITAS_BUILD_COMMIT='$(git rev-parse HEAD); without it the build fails
# at the runtime contract. `make up` passes it.
#   docker build --platform linux/amd64 $C ./backend                                  CPU torch wheels (python:3.12-slim)
#   docker build --platform linux/amd64 $C --build-arg VARIANT=cuda ./backend         CUDA 12.6 runtime (Ubuntu 24.04, Python 3.12) + cu126 torch wheels
#   docker build --platform linux/amd64 $C --build-arg BAKE_MODELS=full ./backend     air-gapped image: the pinned weights of the enabled detectors of a
#                                                              profile (quick | full | deep) fetched and verified into /models
ARG VARIANT=cpu
# Base images are pinned by digest (N-15; OPS-21): the tag says what it is, the digest of the multi-platform index says which
# bytes, so the same commit builds on the same base every time. To move to a newer build of a tag, read its digest with
#   docker buildx imagetools inspect python:3.12-slim-bookworm        (the line "Digest:"; no daemon is needed)
# replace it here, build, and run scripts/image_smoke.py --require-image-runtime in the image. Read on 2026-10-02.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a54bc5c5146ae5ce68e6af5caa3df0fb690bb91544175c91f307e4356

# Runtime libraries: ffmpeg/ffprobe (decode), ExifTool >= 12.24 (metadata rules, D-07), libGL + GLib (OpenCV),
# Pango + HarfBuzz subset (WeasyPrint reports, R-1), DejaVu (fallback glyphs), curl (health check)
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS base-cpu
ENV TORCH_BACKEND=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libimage-exiftool-perl libgl1 libglib2.0-0 libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

FROM ${CUDA_IMAGE} AS base-cuda
ENV TORCH_BACKEND=cu126 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        ffmpeg libimage-exiftool-perl libgl1 libglib2.0-0 libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        fonts-dejavu-core curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

FROM base-${VARIANT} AS app
# prod: VERITAS_OFFLINE defaults to true (PL-01), docs off, JSON logs, an API key required
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VERITAS_ENVIRONMENT=prod VERITAS_DATA_DIR=/data VERITAS_MODELS_DIR=/models HF_HOME=/models/hf TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/tmp/veritas-cache MPLCONFIGDIR=/tmp/veritas-cache/matplotlib
WORKDIR /srv
COPY pyproject.toml ./
COPY requirements ./requirements
# Hash-locked Linux amd64 runtime; no resolver can replace the reviewed torch variant.
# grad-cam ships an sdist, so install pinned build tools first and disable isolated unpinned builds.
RUN python -m pip install --index-url https://pypi.org/simple --require-hashes --no-deps -r requirements/build.lock \
    && python -m pip install --index-url https://pypi.org/simple --require-hashes --no-build-isolation --only-binary=:all: --no-binary=grad-cam -r "requirements/runtime-${TORCH_BACKEND}.lock" \
    && printf '%s\n' "$TORCH_BACKEND" > requirements/image-backend
# ARCHITECTURE §6.6: the commit every result manifest records; the image has no .git
#   docker build --platform linux/amd64 --build-arg VERITAS_BUILD_COMMIT=$(git rev-parse HEAD) ./backend
# Declared after the locked install (N-14c, E-8): a build argument invalidates every step below its declaration, and above
# it the 2.7 GB dependency layer was installed again for each commit that was built.
ARG VERITAS_BUILD_COMMIT=
ENV VERITAS_BUILD_COMMIT=${VERITAS_BUILD_COMMIT}
LABEL org.opencontainers.image.revision=${VERITAS_BUILD_COMMIT}
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
# Of backend/scripts only what the image runs (N-15; OPS-21): the contract of the step below and the smoke that is run in
# the built image (infra/gcp/README.md). The other scripts are the development host's (benchmarks, generators, gates).
COPY scripts/runtime_contract.py scripts/image_smoke.py ./scripts/
COPY demo ./demo
# The local project is installed without dependency resolution after the complete locked graph. The contract also refuses
# an image without a full commit in VERITAS_BUILD_COMMIT (N-15; D-58): no result of it could name its build.
RUN python -m pip install --no-deps --no-build-isolation -e ".[postgres]" \
    && python scripts/runtime_contract.py --backend "$TORCH_BACKEND"
# non-root runtime (uid and gid 10001). The code under /srv stays root's: the runtime user writes /data, /models and /tmp
# only (SEC-17), so the modules are compiled here, once, and Compose mounts the root filesystem read-only. The group has
# the user's number (N-17a: `useradd -r` alone gave it the next system id, 999), so Compose, which runs the image as it
# is, the smoke's `--require-image-runtime` and the Google Cloud bundle's `--user 10001:10001` agree. The user is named
# by number: a file that a volume holds for group 999 from an earlier image stays its owner's (uid 10001).
RUN groupadd -r -g 10001 veritas && useradd -r -u 10001 -g 10001 veritas \
    && mkdir -p /data /models/hf /models/weights /models/torch && chown -R 10001:10001 /data /models \
    && python -m compileall -q app alembic
USER 10001:10001

FROM app AS runtime
ARG BAKE_MODELS=none
# The fetch is the image's only network use; the verify pass re-hashes what landed. These two commands serve nothing, so
# they pass the prod settings check as an open instance, for the length of the build only: no key is made up for them (a
# short placeholder key is refused in prod since D-59) and none is kept in the image.
RUN if [ "$BAKE_MODELS" != "none" ]; then \
        VERITAS_OFFLINE=false VERITAS_ALLOW_ANONYMOUS=true python -m app.cli models fetch --profile "$BAKE_MODELS" \
        && VERITAS_ALLOW_ANONYMOUS=true python -m app.cli models verify --installed; \
    fi
EXPOSE 8160
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8160/api/health || exit 1
# no uvicorn access log: resource tokens travel as ?token= (EventSource, download links); the app logs requests without them.
# no proxy headers: uvicorn would take the client address from X-Forwarded-For whenever its peer is loopback; the client
# address comes from VERITAS_TRUST_PROXY and VERITAS_TRUSTED_PROXY_HOPS alone (SEC-11)
# a graceful-shutdown timeout: on SIGTERM open event streams end at their next poll (app/core/lifecycle.py), and whatever is
# still open after 30 s is closed, so the process ends inside the 40 s Compose gives it (N-14; OPS-15)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8160", "--workers", "1", "--no-access-log", "--no-proxy-headers", "--timeout-graceful-shutdown", "30"]
