# Base images are pinned by digest (N-15; OPS-21), as in backend/Dockerfile: `docker buildx imagetools inspect <tag>` gives
# the digest of a newer build of the tag. Read on 2026-10-02; the nginx image on 2026-10-03.
FROM node:22-alpine@sha256:0a7108bf6c7bf5de370ffb1a3ed6be93d405b43ff159f681a8d18c0e2bc2e402 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# nginx as uid 101 from its first process (SEC-17): the image keeps its pid file and its temporary files under /tmp and
# cannot bind a port below 1024, so the server listens on 8080 (and on 8443 with nginx.tls.conf).
# nginx's stable line (N-17a; OPS-21): nginx.org no longer publishes a 1.27 release. On 2026-10-03 the tag held nginx
# 1.30.5 and `stable-alpine` was the same index. Move the pin along that line; a new line needs both nginx files run in it.
FROM nginxinc/nginx-unprivileged:1.30-alpine@sha256:ed04ec1ff34502c339ee5c3ae3f855442398edc1d05591e2b98981dcbbd20b1e
# GCP startup checks this declaration against the reviewed release commit.
ARG VERITAS_BUILD_COMMIT=
LABEL org.opencontainers.image.revision=${VERITAS_BUILD_COMMIT}
# Both copies are root's and read-only to the runtime user.
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
USER 101
EXPOSE 8080
# nginx answers on its plain listener (with nginx.tls.conf that answer is the redirect). Compose asks more: the shell and
# the API behind it (docker-compose.yml, docker-compose.tls.yml).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -fsS -o /dev/null http://127.0.0.1:8080/ || exit 1
