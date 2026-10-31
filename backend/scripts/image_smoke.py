"""Isolated, model-free runtime smoke; JSON stdout, nonzero on any failure.

Run in each future Linux image as uid/gid 10001 with --require-image-runtime.
A local Darwin run validates only this harness and the local dependencies.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve()
BACKEND = SCRIPT.parents[1]
CHECKS = ("configuration", "tools", "imports", "runtime", "cli", "migrations", "report")
IMAGE_BACKEND = BACKEND / "requirements" / "image-backend"


def isolated_env(workspace: Path) -> dict[str, str]:
    # Allow only runtime executable/library lookup; never inherit app settings, tokens or credentials.
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL") if key in os.environ}
    env.update({
        "TMPDIR": str(workspace / "tmp"), "XDG_CACHE_HOME": str(workspace / "cache"),
        "MPLCONFIGDIR": str(workspace / "matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "HF_HOME": str(workspace / "hf"), "TORCH_HOME": str(workspace / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "VERITAS_ENVIRONMENT": "dev", "VERITAS_OFFLINE": "true", "VERITAS_DEVICE": "cpu",
        "VERITAS_DATA_DIR": str(workspace / "data"), "VERITAS_MODELS_DIR": str(workspace / "models"),
        "VERITAS_HF_HOME": str(workspace / "hf"), "VERITAS_LICENCE_POLICY": "commercial",
        # The image's own commit where the variable exists (an image always declares it, empty when it was built without
        # one, and the dependency contract then refuses it: N-15); a made-up one on a host that has none
        "VERITAS_BUILD_COMMIT": os.environ.get("VERITAS_BUILD_COMMIT", "0" * 40), "VERITAS_REPORT_ENGINE": "weasyprint",
        "VERITAS_RUN_WORKER": "false", "VERITAS_WARMUP_ON_START": "false",
    })
    return env


def disable_settings_files() -> None:
    # Settings declares an absolute backend/.env. A clean cwd/env alone does not prevent reading it.
    # This override exists only in the disposable smoke child, before any application import.
    from pydantic_settings import BaseSettings

    original = BaseSettings.__init__

    def isolated_init(self, **values):
        original(self, **{**values, "_env_file": None, "_secrets_dir": ()})

    BaseSettings.__init__ = isolated_init


def deny_network(*args, **kwargs):
    raise RuntimeError("network access is forbidden in image smoke")


def check_configuration(workspace: Path) -> dict:
    from app.core.config import settings

    if (settings.api_key is not None or settings.signing_key is not None or settings.database_url is not None
            or settings.data_dir != workspace / "data" or settings.models_dir != workspace / "models"
            or settings.hf_home != workspace / "hf" or settings.licence_policy != "commercial" or not settings.offline):
        raise RuntimeError("smoke configuration is not isolated")
    return {"dotenv": "disabled before app import", "credentials": "none", "network": "denied",
            "configuration": "synthetic development settings; no server startup"}


def check_tools(workspace: Path) -> dict:
    versions = {}
    for name, flag in (("ffmpeg", "-version"), ("ffprobe", "-version"), ("exiftool", "-ver")):
        argv = [name, "-config", "", flag] if name == "exiftool" else [name, flag]
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=15, cwd=workspace)
        versions[name] = result.stdout.splitlines()[0]
    match = re.fullmatch(r"(\d+)\.(\d+)", versions["exiftool"].strip())
    if match is None or tuple(map(int, match.groups())) < (12, 24):
        raise RuntimeError("ExifTool 12.24 or later is required")
    return versions


def check_imports(workspace: Path) -> dict:
    import cv2
    import psycopg

    return {"opencv": cv2.__version__, "psycopg": psycopg.__version__, "libpq": psycopg.pq.version()}


def runtime_backend() -> str:
    if IMAGE_BACKEND.is_file():
        backend = IMAGE_BACKEND.read_text().strip()
        if backend not in {"cpu", "cu126"}:
            raise RuntimeError("invalid image backend marker")
        return backend
    if platform.system() == "Darwin":
        return "local"
    raise RuntimeError("Linux image backend marker is missing")


def check_runtime(workspace: Path) -> dict:
    from scripts.runtime_contract import check_runtime as validate_runtime

    result = validate_runtime(runtime_backend())
    if not result["ok"]:
        raise RuntimeError(f"dependency contract failed at {result['stage']}: {result.get('error', 'unknown failure')}")
    return result


def check_cli(workspace: Path) -> dict:
    from app.cli import main
    from app.engine import fetchers
    from app.engine.registry import get_registry

    fetchers.fetch_model = deny_network  # A dry-run regression must never turn into a model acquisition.
    try:
        main(["--help"])
    except SystemExit as exc:
        if exc.code != 0:
            raise RuntimeError("CLI help failed") from exc
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(["models", "fetch", "--profile", "full", "--dry-run"])
    planned = {line.split()[1].rstrip(":") for line in output.getvalue().splitlines() if line.startswith("PLAN")}
    registry = get_registry()
    required = {mid for det in registry.manifest.detectors if "full" in det.profiles and registry.decision(det.id).enabled
                for mid in registry.required_models(det)}
    if code != 0 or not required or planned != required:
        raise RuntimeError("commercial/full dry-run omitted required models")
    return {"help": "passed", "profile": "full", "licence_policy": "commercial", "planned_models": sorted(planned),
            "downloads": 0}


def check_migrations(workspace: Path) -> dict:
    from sqlalchemy import create_engine

    from app.storage.migrations import current_revision, head_revision, upgrade_to_head

    engine = create_engine(f"sqlite:///{workspace / 'smoke.db'}")
    try:
        expected = head_revision()
        first = upgrade_to_head(engine)
        second = upgrade_to_head(engine)
        if first != expected or second != expected or current_revision(engine) != expected:
            raise RuntimeError("temporary SQLite migration did not reach/stay at head")
        return {"dialect": "sqlite", "head": expected, "repeat_at_head": "passed", "postgresql_validated": False}
    finally:
        engine.dispose()


def check_report(workspace: Path) -> dict:
    import fitz

    from app.reports import engine, specimens

    snapshot = specimens.specimen("image", "summary")  # Checked-in synthetic snapshot; no case/sample/cache reads.
    rendered = engine.render_pdf(snapshot)
    if not rendered.pdf.startswith(b"%PDF-") or rendered.refused:
        raise RuntimeError("summary PDF failed or attempted a refused resource fetch")
    with fitz.open(stream=rendered.pdf, filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc)
        if snapshot.media.filename not in text or snapshot.report.display_id not in text:
            raise RuntimeError("summary PDF is missing expected synthetic exhibit/report text")
        fonts = {font[0]: font[3] for page in doc for font in page.get_fonts()}
        if not fonts or any(not any(family in name for family in ("Inter", "Newsreader", "JetBrains"))
                            or not doc.extract_font(xref)[3] for xref, name in fonts.items()):
            raise RuntimeError("summary PDF lacks embedded bundled fonts or uses a fallback font")
        return {"engine": rendered.engine_version, "pages": len(doc), "pdf_bytes": len(rendered.pdf),
                "embedded_fonts": sorted(set(fonts.values())), "synthetic_text_verified": True}


def child(check: str, workspace: Path) -> int:
    # Reapply isolation even for a manually invoked hidden child command.
    env = isolated_env(workspace)
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(BACKEND))
    socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = deny_network
    try:
        # Capture incidental library/CLI logs; stdout remains a single structured result.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            disable_settings_files()
            details = globals()[f"check_{check}"](workspace)
        result = {"name": check, "status": "passed", "details": details}
    except Exception as exc:
        result = {"name": check, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:600]}
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


def run_check(check: str, workspace: Path, timeout: float) -> dict:
    try:
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--check", check, "--workspace", str(workspace)],
                                env=isolated_env(workspace), cwd=workspace, capture_output=True, text=True, timeout=timeout)
        record = json.loads(result.stdout)
        if not isinstance(record, dict) or record.get("name") != check or record.get("status") not in ("passed", "failed"):
            raise ValueError("invalid child result")
        if result.returncode != 0 and record["status"] == "passed":
            raise ValueError("child exited unsuccessfully")
        return record
    except subprocess.TimeoutExpired:
        return {"name": check, "status": "failed", "error": f"timed out after {timeout:g} seconds"}
    except (OSError, ValueError) as exc:
        return {"name": check, "status": "failed", "error": f"child failed: {type(exc).__name__}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-image-runtime", action="store_true", help="require Linux amd64 and uid/gid 10001")
    parser.add_argument("--timeout", type=float, default=60, help="timeout in seconds per isolated check (default: 60)")
    parser.add_argument("--check", choices=CHECKS, help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 300:
        parser.error("--timeout must be greater than zero and at most 300 seconds")
    if args.check:
        if args.workspace is None:
            parser.error("internal check requires a workspace")
        return child(args.check, args.workspace.resolve())
    runtime = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
               "uid": os.getuid(), "gid": os.getgid()}
    image_runtime = (runtime["system"] == "Linux" and runtime["machine"] in ("x86_64", "AMD64")
                     and runtime["uid"] == runtime["gid"] == 10001)
    records = []
    if args.require_image_runtime and not image_runtime:
        records.append({"name": "image_runtime", "status": "failed", "error": "requires Linux amd64 uid/gid 10001"})
    removed = False
    try:
        with tempfile.TemporaryDirectory(prefix="veritas-image-smoke-") as temporary:
            workspace = Path(temporary).resolve()
            (workspace / "tmp").mkdir()
            records.extend(run_check(check, workspace, args.timeout) for check in CHECKS)
        removed = True
    except OSError as exc:
        records.append({"name": "workspace", "status": "failed", "error": f"temporary workspace: {type(exc).__name__}"})
    ok = all(record["status"] == "passed" for record in records)
    print(json.dumps({"schema_version": 1, "ok": ok, "runtime": runtime, "image_runtime_required": args.require_image_runtime,
                      "scope": "Isolated dependency smoke only; no image provenance, production startup, model inference, "
                               "PostgreSQL/Cloud SQL, GPU, cloud-volume permissions or deployment validation.",
                      "temporary_files_removed": removed, "checks": records}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
