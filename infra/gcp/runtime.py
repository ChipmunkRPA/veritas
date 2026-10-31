"""GCE host launcher. Render/validate is offline; only explicit start/proxy reads metadata credentials."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path("/opt/veritas")
RUN = Path("/run/veritas")
DATA = "/srv/veritas/data"
MODELS = "/srv/veritas/models"
DIGEST = re.compile(r"^[a-z0-9.-]+(?:/[a-zA-Z0-9_.-]+)+@sha256:[a-f0-9]{64}$")


def validate(config: dict) -> None:
    # Truthy strings such as "false" must not enable a runtime or select a GPU.
    for flag in ("gpu_enabled", "runtime_enabled"):
        if not isinstance(config.get(flag), bool):
            raise ValueError(f"{flag} must be a JSON boolean")
    for key in ("backend_image", "worker_image", "web_image", "proxy_image"):
        value = config.get(key)
        if not isinstance(value, str) or not DIGEST.fullmatch(value):
            raise ValueError(f"{key} must be a registry image pinned by SHA-256")
        if not value.startswith(config["registry"] + "/"):
            raise ValueError(f"{key} must use the scoped Artifact Registry repository")
    if not re.fullmatch(r"[a-f0-9]{40}", config.get("build_commit", "")):
        raise ValueError("a complete build commit is required")
    if config.get("licence_policy") != "commercial":
        raise ValueError("this deployment requires the commercial licence policy")
    if not re.fullmatch(r"[a-z0-9-]+:[a-z0-9-]+:[a-z0-9-]+", config.get("sql_connection", "")):
        raise ValueError("invalid Cloud SQL connection name")
    if not re.fullmatch(r"[a-z0-9-]+@[a-z0-9-]+\.iam", config.get("sql_username", "")):
        raise ValueError("invalid Cloud SQL service-account database username")
    if not re.fullmatch(r"projects/[a-z0-9-]+/secrets/[a-zA-Z0-9_-]+/versions/[1-9][0-9]*", config.get("secret_resource", "")):
        raise ValueError("pin a numeric Secret Manager version")


def compose(config: dict) -> dict:
    validate(config)
    username = urllib.parse.quote(config["sql_username"], safe="")
    env = {
        "VERITAS_ENVIRONMENT": "prod", "VERITAS_OFFLINE": "true", "VERITAS_ALLOW_ANONYMOUS": "false",
        "VERITAS_SINGLE_USER": "false", "VERITAS_TRUST_PROXY": "true", "VERITAS_RUN_WORKER": "false",
        "VERITAS_DATA_DIR": "/data", "VERITAS_MODELS_DIR": "/models", "HF_HOME": "/models/hf",
        "TORCH_HOME": "/models/torch", "TMPDIR": "/data/tmp", "VERITAS_FUSER_DIR": "/data/fusion",
        "XDG_CACHE_HOME": "/tmp/veritas-cache", "MPLCONFIGDIR": "/tmp/veritas-cache/matplotlib",
        "VERITAS_TEXT_CLASSIFIER_FINETUNED_PATH": "/data/weights/desklib-ft-v2",
        "VERITAS_DATABASE_URL": f"postgresql+psycopg://{username}@sql-proxy:5432/veritas?sslmode=disable",
        "VERITAS_LICENCE_POLICY": "commercial", "VERITAS_DEFAULT_PROFILE": "full",
        "VERITAS_DETECTOR_SET": "manifest", "VERITAS_PIPELINE": "v2", "VERITAS_WEBHOOKS_ENABLED": "false",
        "VERITAS_WORKER_THREADS": "1", "VERITAS_DEVICE": "cpu",
        "VERITAS_CORS_ORIGINS": '["http://localhost:8080"]', "VERITAS_MODEL_MEMORY_BUDGET_MB": "24576",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
    }
    logs = {"driver": "json-file", "options": {"max-size": "10m", "max-file": "5"}}
    # Never let Docker turn a missing disk/secret source into an empty host directory.
    volumes = [{"type": "bind", "source": source, "target": target, "read_only": readonly,
                "bind": {"create_host_path": False}}
               for source, target, readonly in ((DATA, "/data", False), (MODELS, "/models", False),
                   (str(ROOT), "/opt/veritas", True), (str(RUN / "runtime.json"), "/run/secrets/runtime.json", True))]
    # Process failures retry; only systemd may start the stack after a daemon/VM restart.
    base = {
        "image": config["backend_image"], "init": True, "user": "10001:10001", "read_only": True,
        "entrypoint": ["python", "/opt/veritas/entrypoint.py"], "environment": env, "volumes": volumes,
        "tmpfs": ["/tmp:size=4g,mode=1777"], "networks": ["app"], "logging": logs,
        "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"], "stop_grace_period": "45s",
        "restart": "on-failure",
    }
    # Credentials and private-IP TLS are owned by the proxy. No database port is published on the host.
    proxy = {
        "image": config["proxy_image"], "command": ["--private-ip", "--auto-iam-authn", "--address=0.0.0.0", "--port=5432", config["sql_connection"]],
        "networks": ["app", "control"], "restart": "on-failure", "read_only": True,
        "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"], "logging": logs,
    }
    admin_proxy = {**proxy, "profiles": ["administration"], "ports": ["127.0.0.1:5433:5432"],
                   "command": ["--private-ip", "--address=0.0.0.0", "--port=5432", config["sql_connection"]]}
    preflight = {**base, "command": ["python", "/opt/veritas/preflight.py"], "restart": "no",
                 "healthcheck": {"disable": True}, "depends_on": {"sql-proxy": {"condition": "service_started"}}}
    api = {**base, "command": ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8160", "--workers", "1", "--no-access-log", "--no-proxy-headers",
                               "--timeout-graceful-shutdown", "30"],  # inside the 45 s stop grace period (N-14; OPS-15)
           "mem_limit": "8g", "cpus": 2,
           "depends_on": {"preflight": {"condition": "service_completed_successfully"}},
           # Liveness alone: the API answers once its start (migrations, key check, registry) is over. `/api/ready` also needs
           # the worker's heartbeat, and the worker waits for this check, so asking it here would never start either. The web
           # check below asks `/api/ready` through nginx, after the worker is healthy.
           "healthcheck": {"test": ["CMD-SHELL",
               "curl -fsS --connect-timeout 2 --max-time 4 http://localhost:8160/api/health >/dev/null"],
               "interval": "15s", "timeout": "10s", "retries": 8, "start_period": "60s"}}
    worker_health_file = "/tmp/veritas-worker-health.json"
    worker = {**base, "image": config["worker_image"],
              "command": ["python", "-m", "app.engine.worker", "--health-file", worker_health_file],
              "environment": {**env, "VERITAS_DEVICE": "cuda" if config["gpu_enabled"] else "cpu",
                              "VERITAS_MODEL_MEMORY_BUDGET_MB": "20000" if config["gpu_enabled"] else "24576"},
              "mem_limit": "48g", "cpus": 12,
              "healthcheck": {"test": ["CMD", "python", "-m", "app.engine.worker_health",
                                       "--file", worker_health_file, "--max-age", "45"],
                              "interval": "10s", "timeout": "5s", "retries": 3, "start_period": "30s"},
              "depends_on": {"api": {"condition": "service_healthy"}}}
    if config["gpu_enabled"]:
        worker["deploy"] = {"resources": {"reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}}
    # The frontend image runs nginx as an unprivileged user on port 8080 (frontend/Dockerfile).
    web = {"image": config["web_image"], "ports": ["8080:8080"], "networks": ["app", "ingress"],
           "restart": "on-failure", "logging": logs, "mem_limit": "512m", "cpus": 1,
           # The nginx Alpine image includes curl. Require both a complete transfer and HTTP 200:
           # redirects, proxy errors and a partial 200 response must not count as readiness.
           # Compose escapes $$ to a container-shell $, avoiding host-variable interpolation.
           "healthcheck": {"test": ["CMD-SHELL",
               "for path in /index.html /api/ready /api/health; do "
               "status=$$(curl -q -fsS --noproxy '*' --connect-timeout 2 --max-time 4 "
               "--output /dev/null --write-out '%{http_code}' \"http://127.0.0.1:8080$$path\") || exit 1; "
               "[ \"$$status\" = 200 ] || exit 1; done"],
               "interval": "15s", "timeout": "15s", "retries": 3, "start_period": "10s"},
           "depends_on": {"api": {"condition": "service_healthy"}, "worker": {"condition": "service_healthy"}}}
    return {"name": "veritas", "services": {"sql-proxy": proxy, "sql-admin-proxy": admin_proxy, "preflight": preflight, "api": api, "worker": worker, "web": web},
            "networks": {"app": {"internal": True}, "control": {}, "ingress": {}}}


def request_json(url: str, headers: dict) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def token() -> str:
    return request_json("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                        {"Metadata-Flavor": "Google"})["access_token"]


def run_compose(*args: str, timeout: float | None = None) -> None:
    subprocess.run(["docker", "compose", "--project-name", "veritas", "-f", str(RUN / "compose.json"), *args],
                   check=True, timeout=timeout)



def verify_web_revision(config: dict) -> None:
    """Bind the pulled frontend digest to the reviewed source before any service starts.

    This checks builder-declared metadata, not a signature or independent source attestation.
    Backend processes separately check their baked revision in entrypoint.py.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", config["web_image"]],
            check=True, capture_output=True, text=True, timeout=30,
        )
        labels = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("Unable to verify frontend image revision; startup refused") from None
    if not isinstance(labels, dict) or labels.get("org.opencontainers.image.revision") != config["build_commit"]:
        raise RuntimeError("The frontend image revision differs from the reviewed deployment or is missing")


def stop_project() -> None:
    """Drain only this Compose project, even when /run or its old configuration is gone."""
    result = subprocess.run(["docker", "ps", "--all", "--filter", "label=com.docker.compose.project=veritas",
                             "--format", '{{.ID}} {{.Label "com.docker.compose.service"}}'],
                            check=True, capture_output=True, text=True)
    containers = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[a-f0-9]{12,64}", fields[0]):
            raise RuntimeError("unrecognized owned-container inventory; refusing partial cleanup")
        containers.append((fields[0], fields[1]))
    if not containers:
        return
    ids = [cid for cid, _ in containers]
    # Disable legacy auto-restore policies before any stop; keep DB access while the worker drains.
    subprocess.run(["docker", "update", "--restart=no", *ids], check=True, stdout=subprocess.DEVNULL)
    priority = {"web": 0, "worker": 1, "api": 2, "preflight": 3, "sql-admin-proxy": 5, "sql-proxy": 6}
    for group in sorted({priority.get(service, 4) for _, service in containers}):
        batch = [cid for cid, service in containers if priority.get(service, 4) == group]
        subprocess.run(["docker", "stop", "--time", "45", *batch], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", *ids], check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "start", "stop", "admin-proxy"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if args.action == "stop":
        stop_project()
        return
    if args.action != "render":
        # Cleanup must succeed before config, credentials or image pulls, including disabled starts.
        stop_project()
    config = json.loads(args.config.read_text())
    document = compose(config)
    if args.action == "render":
        print(json.dumps(document, indent=2))
        return
    if args.action == "start" and not config["runtime_enabled"]:
        print("Runtime disabled: complete provisioning and release verification before enabling.")
        return
    if not (RUN / "provisioned").is_file():
        raise RuntimeError("host provisioning incomplete; rerun the metadata startup script")
    if not all(os.path.ismount(path) for path in (DATA, MODELS)):
        raise RuntimeError("both persistent disks must be mounted; refusing ephemeral storage")
    RUN.mkdir(mode=0o700, parents=True, exist_ok=True)
    (RUN / "compose.json").write_text(json.dumps(document, indent=2))
    access_token = token()
    registry_host = config["registry"].split("/", 1)[0]
    # Registry authentication remains on tmpfs, not on the persistent evidence or boot disk.
    os.environ["DOCKER_CONFIG"] = str(RUN / "docker")
    subprocess.run(["docker", "login", "--username", "oauth2accesstoken", "--password-stdin", registry_host],
                   input=access_token.encode(), check=True, stdout=subprocess.DEVNULL)
    try:
        if args.action == "admin-proxy":
            run_compose("pull", "sql-admin-proxy", "api")
            run_compose("up", "-d", "sql-admin-proxy")
            return
        response = request_json("https://secretmanager.googleapis.com/v1/" + config["secret_resource"] + ":access",
                                {"Authorization": "Bearer " + access_token})
        secret = json.loads(base64.b64decode(response["payload"]["data"], validate=True))
        from entrypoint import validate_secret
        validate_secret(secret)
        secret_path = RUN / "runtime.json"
        secret_path.write_text(json.dumps(secret))
        secret_path.chmod(0o400)
        os.chown(secret_path, 10001, 10001)
        run_compose("pull")
        verify_web_revision(config)
        run_compose("up", "-d", "--wait", "--wait-timeout", "300", "--force-recreate", "--remove-orphans", timeout=360)
    except Exception:
        # A refused preflight, health timeout or partial start must not leave a serving half-stack.
        stop_project()
        raise
    finally:
        subprocess.run(["docker", "logout", registry_host], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
