"""Re-run a packaged analysis and diff its raw scores (ARCHITECTURE §6.5 reproduction; PRD CC-03; BUILD_SPEC R-4).

`python -m app.cli reproduce <package.zip> (--public-key PEM | --expect-key-id ID) [--tolerance 0.001] [--json]`:
1. verifies the package (`app.reports.verify`); a package that does not verify is not reproduced, and neither is one whose
   signer is not authenticated (no instance key pinned: "SELF-CONSISTENT: signer not authenticated", exit 3; BUILD_SPEC N-11);
2. needs the original in `source/` (a package exported with `include_original`);
3. runs the v2 pipeline in a throwaway workspace under `data/reproduce/`: its own SQLite database, object store and uploads, with
   the instance's weights, verify records, keys and fuser artefacts linked in (`data/{weights,state,keys,fusion}`), under the
   packaged profile and fuser pin. Nothing is written to the instance database or custody log; the workspace is removed after;
4. compares each signal's status, `raw_score` and `calibrated_probability`, the fused probability and the decision code with the
   packaged result.
Same host and device: agreement to 3 decimals (tolerance 0.001). MPS against CPU: `--tolerance 0.02` (ARCHITECTURE §6.5).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from app.reports import verify as pv

DEFAULT_TOLERANCE = 1e-3
LINKED = ("weights", "state", "keys", "fusion")


class ReproduceRefused(ValueError):
    """The package cannot be reproduced (no original, or not a package)."""


class PackageNotVerified(RuntimeError):
    def __init__(self, verification: pv.Verification):
        self.verification = verification
        super().__init__("The package does not verify; it was not reproduced.")


class SignerNotAuthenticated(RuntimeError):
    """Every check passes against the key inside the package and no instance key was pinned: nothing is run."""

    def __init__(self, verification: pv.Verification):
        self.verification = verification
        super().__init__("The signer of the package is not authenticated; it was not reproduced.")


@dataclass(frozen=True)
class Delta:
    item: str
    field: str
    packaged: Any
    rerun: Any
    delta: float | None
    ok: bool


@dataclass
class Reproduction:
    analysis_id: str
    tolerance: float
    status: str
    error: str | None = None
    deltas: list[Delta] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "done" and all(d.ok for d in self.deltas)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "analysis_id": self.analysis_id, "tolerance": self.tolerance, "status": self.status, "error": self.error,
                "deltas": [d.__dict__ for d in self.deltas]}


def reproduce(path: Path, *, tolerance: float = DEFAULT_TOLERANCE, public_key_pem: bytes | None = None,
              expect_key_id: str | None = None) -> Reproduction:
    verification = pv.verify_package(path, public_key_pem=public_key_pem, expect_key_id=expect_key_id)
    if not verification.consistent:
        raise PackageNotVerified(verification)
    if not verification.ok:
        raise SignerNotAuthenticated(verification)
    bundle = verification.bundle
    source = bundle.get("source") or {}
    if not source.get("file"):
        raise ReproduceRefused("The package holds no original (source/); export it again with the original to reproduce it.")
    with zipfile.ZipFile(path) as zf:
        data = zf.read(source["file"])
    packaged, analysis, exhibit = bundle["result"], bundle.get("analysis") or {}, bundle.get("exhibit") or {}
    decision = packaged.get("decision") or {}
    status, rerun, error = _rerun(data, filename=PurePosixPath(source["file"]).name, kind=str(packaged.get("kind")),
                                  content_type=str(exhibit.get("content_type") or "application/octet-stream"),
                                  profile=analysis.get("profile"),
                                  fuser_pin=decision.get("fuser_sha256") if decision.get("fusion_mode") == "artefact" else None)
    out = Reproduction(analysis_id=str(analysis.get("id")), tolerance=tolerance, status=status, error=error)
    if rerun is not None:
        out.deltas = diff(packaged, rerun, tolerance)
    return out


def _same(item: str, name: str, a: Any, b: Any) -> Delta:
    return Delta(item, name, a, b, None, a == b)


def _close(item: str, name: str, a: Any, b: Any, tolerance: float) -> Delta:
    if a is None or b is None:
        return Delta(item, name, a, b, None, a is None and b is None)
    delta = abs(float(a) - float(b))
    return Delta(item, name, a, b, delta, delta <= tolerance + 1e-12)


def _probability(decision: dict[str, Any]) -> float | None:
    p = decision.get("probability_ai_calibrated")
    return decision.get("probability_ai_uncalibrated") if p is None else p


def diff(packaged: dict[str, Any], rerun: dict[str, Any], tolerance: float) -> list[Delta]:
    out: list[Delta] = []
    before = {s["id"]: s for s in packaged.get("signals") or []}
    after = {s["id"]: s for s in rerun.get("signals") or []}
    for sid in sorted(set(before) | set(after)):
        a, b = before.get(sid), after.get(sid)
        if a is None or b is None:
            out.append(Delta(sid, "signal", "present" if a else "absent", "present" if b else "absent", None, False))
            continue
        out.append(_same(sid, "status", a.get("status"), b.get("status")))
        out += [_close(sid, name, a.get(name), b.get(name), tolerance) for name in ("raw_score", "calibrated_probability")]
    pd, rd = packaged.get("decision") or {}, rerun.get("decision") or {}
    out.append(_same("decision", "code", pd.get("code"), rd.get("code")))
    out.append(_close("decision", "probability_ai", _probability(pd), _probability(rd), tolerance))
    return out


@contextmanager
def _workspace() -> Iterator[Path]:
    """Point settings and the database at a throwaway data directory; restore both afterwards."""
    from app.core.config import settings
    from app.storage import db

    instance = settings.data_dir
    saved = (settings.data_dir, settings.database_url, db._engine, db._SessionLocal)
    root = instance / "reproduce"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=root) as tmp:
        work = Path(tmp) / "data"
        work.mkdir()
        for name in LINKED:
            if (instance / name).is_dir():
                (work / name).symlink_to((instance / name).resolve(), target_is_directory=True)
        settings.data_dir, settings.database_url = work, None
        db._engine, db._SessionLocal = None, None
        try:
            db.init_db()
            yield work
        finally:
            if db._engine is not None:
                db._engine.dispose()
            settings.data_dir, settings.database_url = saved[0], saved[1]
            db._engine, db._SessionLocal = saved[2], saved[3]


def _rerun(data: bytes, *, filename: str, kind: str, content_type: str, profile: str | None,
           fuser_pin: str | None) -> tuple[str, dict[str, Any] | None, str | None]:
    from sqlalchemy import select

    from app.core.config import settings
    from app.engine.pipelines import run_analysis
    from app.storage import db
    from app.storage.files import safe_filename
    from app.storage.objects import default_store

    with _workspace():
        aid, name = str(uuid.uuid4()), safe_filename(filename)
        upload = settings.uploads_dir / aid
        upload.mkdir(parents=True)
        (upload / name).write_bytes(data)
        with db.get_session() as s:
            s.add(db.Analysis(id=aid, kind=kind, filename=name, content_type=content_type, size_bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest(), status="queued"))
            s.commit()
        outcome = run_analysis(aid, profile=profile or None, fuser_pin=fuser_pin)
        if outcome.status != "done":
            return outcome.status, None, outcome.error
        with db.get_session() as s:
            analysis = s.get(db.Analysis, aid)
            row = s.scalars(select(db.Artefact).where(db.Artefact.analysis_id == aid, db.Artefact.role == "stage_output",
                                                      db.Artefact.sha256 == analysis.result_hash)).first()
            if row is None:
                return "failed", None, "The re-run published no result document."
            return "done", json.loads(default_store().read_bytes(row.sha256)), None
