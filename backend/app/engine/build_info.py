"""Reproducibility manifest facts (ARCHITECTURE §6.6; PRD CC-01, CC-03; BUILD_SPEC Q-4d).

- **Git commit.** `VERITAS_BUILD_COMMIT` baked at image build, else read from the checkout's `.git` files without running git
  (HEAD, a loose ref, `packed-refs`, a worktree `gitdir:` / `commondir`). An invalid value records nothing. Uncommitted changes in
  the working tree are not detected.
- **Configuration.** `config_sha256` is the canonical hash of the settings minus secrets, host paths and the commit itself: two
  hosts with the same tunables agree, and a changed threshold, weight or limit changes the hash.
- **Compute and libraries.** The devices and precisions of the detector runs; the installed versions of torch, transformers,
  open_clip, OpenCV (package metadata; nothing is imported) and ffmpeg (`ffmpeg -version`).
- **Sampling.** Video: sampling plan version, seed, frame counts, tracking rate and a hash of the frame indices (the indices
  themselves are compared by `rerun.deltas`). Audio: voice window size and limits. Image and document analyses sample nothing.
- **Intake link.** `input.intake_event_id` is the sequence number of the custody `intake` event of the analysis, or of the
  revision it supersedes; None when the analysis did not come through intake.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash
from app.core.config import BACKEND_DIR, settings
from app.engine import rerun
from app.storage.db import Analysis, AuditEvent

REPO_DIR = BACKEND_DIR.parent
SECRETS = frozenset({"api_key", "signing_key", "api_key_pepper", "database_url", "stripe_secret_key", "stripe_webhook_secret"})
HOST_PATHS = frozenset({"data_dir", "hf_home", "models_dir", "models_manifest", "fuser_dir", "report_library_dirs", "batch_local_roots",
                        "text_classifier_finetuned_path", "exiftool_path", "frontend_dir"})
NOT_CONFIG = frozenset({"build_commit", "billing_mode", "billing_app_origin"})
LIBRARIES: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "transformers": ("transformers",),
    "open_clip": ("open_clip_torch",),
    "opencv": ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"),
}
MAX_CHAIN = 1000  # revisions followed back to the intake event
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _commit(value: str) -> str | None:
    value = value.strip().lower()
    return value if _COMMIT.fullmatch(value) else None


def git_commit() -> str | None:
    """The commit this build runs: the baked value when set, else the checkout's HEAD."""
    if settings.build_commit:
        return _commit(settings.build_commit)
    return read_checkout(REPO_DIR)


def read_checkout(root: Path) -> str | None:
    """HEAD of the checkout at `root`, from its `.git` files; None when there is no checkout or HEAD cannot be resolved."""
    git = root / ".git"
    try:
        if git.is_file():  # a worktree or submodule: `gitdir: <path>`
            git = (root / git.read_text("utf-8").strip().removeprefix("gitdir:").strip()).resolve()
        head = (git / "HEAD").read_text("utf-8").strip()
        if not head.startswith("ref:"):
            return _commit(head)
        ref = head.removeprefix("ref:").strip()
        common = (git / (git / "commondir").read_text("utf-8").strip()).resolve() if (git / "commondir").is_file() else git
        for base in dict.fromkeys((git, common)):
            if (base / ref).is_file():
                return _commit((base / ref).read_text("utf-8"))
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text("utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return _commit(parts[0])
    except OSError:
        return None
    return None


def sanitised_settings() -> dict[str, Any]:
    """The settings a result depends on: every field except secrets, host paths and the build commit."""
    return {k: v for k, v in settings.model_dump(mode="json").items() if k not in SECRETS | HOST_PATHS | NOT_CONFIG}


def config_sha256() -> str:
    return canonical_hash(sanitised_settings())


def models_manifest_sha256() -> str | None:
    """SHA-256 of the model manifest file the engine reads (the cards behind each recorded admission); None when unreadable."""
    import hashlib

    from app.engine.manifest import DEFAULT_MANIFEST

    try:
        return hashlib.sha256(Path(settings.models_manifest or DEFAULT_MANIFEST).read_bytes()).hexdigest()
    except OSError:
        return None


def _package_version(names: Iterable[str]) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _ffmpeg_version() -> str | None:
    binary = shutil.which("ffmpeg")
    if binary is None:
        return None
    try:
        out = subprocess.run([binary, "-hide_banner", "-version"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    parts = out.split()
    return parts[2] if len(parts) > 2 and parts[:2] == ["ffmpeg", "version"] else None


@lru_cache(maxsize=1)
def library_versions() -> dict[str, str]:
    """Installed versions of the libraries a result depends on; a library that is not installed is left out."""
    found = {name: _package_version(dists) for name, dists in LIBRARIES.items()}
    found["ffmpeg"] = _ffmpeg_version()
    return {k: v for k, v in found.items() if v}


def compute(entries: Iterable[Mapping[str, Any]]) -> tuple[str | None, str | None]:
    """The devices and the precisions the detector runs used, each sorted and comma-separated."""
    rows = list(entries)
    devices = sorted({str(e["device"]) for e in rows if e.get("device")})
    precisions = sorted({str(e["dtype"]) for e in rows if e.get("dtype")})
    return ", ".join(devices) or None, ", ".join(precisions) or None


def sampling(kind: str, outputs: Mapping[str, Any]) -> dict[str, Any]:
    """What the analysis sampled from the media, compact enough for the manifest and the forensic appendix."""
    if kind == "video":
        probe, shots = outputs.get("probe"), outputs.get("shots")
        frames = rerun.sampled_frames(probe, shots)
        if frames is None:
            return {}
        plan = (shots or {}).get("plan") or (probe or {}).get("plan") or {}
        return {"plan_version": plan.get("version"), "seed": plan.get("seed"), "legacy_frames": len(frames["sampled"]),
                "tracked_frames": len(frames["tracked"]), "tracked_fps": (plan.get("sparse") or {}).get("rate_fps"),
                "frame_indices_sha256": canonical_hash(frames)}
    if kind == "audio":
        from app.detectors.video.audio import SR, VOICE_WINDOW_S
        from app.engine.stages.audio import AUDIO_MAX_SECONDS, AUDIO_MAX_WINDOWS

        return {"voice_window_s": VOICE_WINDOW_S, "max_windows": AUDIO_MAX_WINDOWS, "max_seconds": AUDIO_MAX_SECONDS, "sample_rate": SR}
    return {}


def intake_event_id(db: Session, analysis: Analysis) -> str | None:
    """The custody `intake` event that brought this file in: the analysis's own, else that of the revision it supersedes."""
    current: Analysis | None = analysis
    for _ in range(MAX_CHAIN):
        if current is None:
            return None
        seq = db.scalars(select(AuditEvent.seq).where(AuditEvent.action == "intake", AuditEvent.analysis_id == current.id)
                         .order_by(AuditEvent.seq).limit(1)).first()
        if seq is not None:
            return str(seq)
        current = db.get(Analysis, current.supersedes_id) if current.supersedes_id else None
    return None
