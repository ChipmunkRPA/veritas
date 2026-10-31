"""Evidence packages (ARCHITECTURE §4.7 `packages`, §6.5; PRD RP-05; D-11, D-27, D-30; BUILD_SPEC R-4).

- **Request** (`prepare`): one row per export of a finished, signed report, `draft` or `issue` (D-27, D-56: issue only for a
  report whose frozen snapshot is issuable), optionally with the original. The caller queues the `package` job in the same
  transaction (N-9): no queued package is left without a job.
- **Job** (`run`): re-hashes the original (CC-01), builds the ZIP within the job's time limit, signs `manifest.sha256` with the
  instance key and then, in one transaction, stores the ZIP as a `package` artefact, marks the row `done` and records
  `package_exported` with the ZIP's SHA-256, the manifest hash, its signature and the custody anchor. That final write is
  conditional (N-9; code review 14): a package that is already done keeps the bytes that were downloadable, and a row the
  queue failed stays failed. A package cannot hold its own export event: the log holds it.
- **Layout** (D-30, UX §6.4): `README.txt`, `manifest.sha256` + `.sig`, `bundle.json` (`veritas.evidence/2.0`), `report.pdf`,
  `executive-summary.pdf` (the newest finished summary report frozen with the same review state, conclusion and sign-offs as
  `report.pdf`, when one exists and `report.pdf` is not itself a summary; N-3),
  `artifacts/` (exhibit payloads, masks, previews, stills as stored), `source/` (optional), `instance_public_key.pem`,
  `reproduce.md`. Entries are written in path order with the export time as their timestamp.
- **bundle.json** holds the result the report was rendered from, the snapshot's case, exhibit identity, review and annotations
  (internal note text withheld, as in reports), the live custody extract of the analysis (with the case's events on its original,
  D-47) with the chain head as anchor, and the
  reports with their SHA-256 and signatures. Redacted (newsroom) reports are not packaged: a package carries the full result.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from app.audit import chain as audit
from app.cases import review as review_rules
from app.core import deadline
from app.core.canonical import canonical_json
from app.core.config import settings
from app.core.errors import friendly_error
from app.core.logging import get_logger
from app.reports import signing, snapshot, view
from app.reports.verify import SIGNER_NOT_AUTHENTICATED
from app.storage.artefacts import ArtefactNotFound, get_artefact, index_object
from app.storage.db import (
    Analysis,
    Artefact,
    AuditEvent,
    MediaObject,
    PackageRow,
    ReportRow,
    get_session,
    utcnow,
)
from app.storage.files import safe_filename
from app.storage.objects import CHUNK, ObjectNotFound, ObjectStore, default_store

log = get_logger(__name__)
OPEN = ("queued", "running")  # the states the final write of a job may change
SCHEMA = "veritas.evidence/2.0"
KINDS = ("draft", "issue")
EXHIBIT_FIELDS = (("payload_artefact_id", "payload"), ("reliability_mask_artefact_id", "reliability_mask"),
                  ("preview_artefact_id", "preview"))
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "application/json": ".json", "text/plain": ".txt"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class PackageError(RuntimeError):
    """The package cannot be built from what is stored; the message is shown on the package."""


@dataclass(frozen=True)
class Built:
    path: Path  # the ZIP as a work file under data/tmp/packages; the job stores it, then removes it
    size: int
    sha256: str
    manifest_sha256: str
    signature: str
    files: int
    anchor: dict[str, Any]


def _ts(dt: datetime) -> str:
    return (dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def filename(package: PackageRow, report: ReportRow) -> str:
    return f"veritas-evidence-{package.analysis_id[:8]}-r{report.revision:02d}{'-draft' if package.kind == 'draft' else ''}.zip"


def prepare(db: Session, analysis: Analysis, report: ReportRow, *, kind: str, include_original: bool, created_by: str) -> PackageRow:
    """Add the package row as `queued`, flushed and not committed. The caller queues the job in the same transaction and
    commits (`v2_reports.request_package`)."""
    row = PackageRow(id=str(uuid.uuid4()), analysis_id=analysis.id, report_id=report.id, case_id=analysis.case_id, kind=kind,
                     include_original=include_original, status="queued", created_by=created_by, created_at=utcnow())
    db.add(row)
    db.flush()
    log.info("packages.requested", package_id=row.id, analysis_id=analysis.id, report_id=report.id, kind=kind)
    return row


def create(db: Session, analysis: Analysis, report: ReportRow, *, kind: str, include_original: bool, created_by: str) -> PackageRow:
    """`prepare`, committed without a job: for a caller that runs the job itself (tools, tests)."""
    row = prepare(db, analysis, report, kind=kind, include_original=include_original, created_by=created_by)
    db.commit()
    return row


def _fail(package_id: str, error: str) -> str:
    with get_session() as db:
        row = db.get(PackageRow, package_id)
        if row is not None and row.status != "done":
            row.status, row.error, row.finished_at = "failed", error, utcnow()
            db.commit()
    log.warning("packages.failed", package_id=package_id, error=error)
    return "failed"


def mark_failed(package_id: str, error: str | None) -> None:
    """A lease expired and the poison guard failed or cancelled the job: reflect it on the package."""
    _fail(package_id, error or "The package job did not finish.")


def _outcome(package_id: str) -> str:
    """What a job that wrote nothing returns: the state another worker, or the queue, left on the row."""
    with get_session() as db:
        row = db.get(PackageRow, package_id)
        return "failed" if row is not None and row.status == "failed" else "done"


def run(package_id: str, *, store: ObjectStore | None = None, timeout_s: float | None = None) -> str:
    """The `package` job. Returns the job outcome (done or failed). `timeout_s` bounds the build (None: no limit)."""
    store = store or default_store()
    with get_session() as db:
        # conditional, as the final write: a package another worker finished meanwhile is not set back to running
        started_now = db.execute(update(PackageRow).where(PackageRow.id == package_id, PackageRow.status.in_(OPEN)).values(status="running")
                                 .execution_options(synchronize_session=False))
        db.commit()
        row = db.get(PackageRow, package_id) if started_now.rowcount == 1 else None
        if row is None or row.status != "running":  # done, failed or gone
            return "done"
        analysis_id, report_id, actor, kind, include_original = row.analysis_id, row.report_id, row.created_by, row.kind, row.include_original
    try:
        audit.verify_original("package", actor_id=actor, analysis_id=analysis_id)
    except audit.OriginalIntegrityError as e:
        return _fail(package_id, f"The original {e.problem}; the package was not produced.")
    except LookupError:
        return _fail(package_id, "The analysis no longer exists; the package was not produced.")
    def building() -> Built:  # in its own session: at the time limit the job goes on and this thread ends by itself
        with get_session() as db:
            return build(db, db.get(PackageRow, package_id), store=store, key=key, exported_at=utcnow())

    try:
        key = signing.instance_key()
        signing.ensure_registered(key)
        built = deadline.run_within(timeout_s, building, what="The package", late=lambda late: late.path.unlink(missing_ok=True))
    except (PackageError, signing.SigningKeyError) as e:
        return _fail(package_id, str(e))
    except (ArtefactNotFound, ObjectNotFound):
        return _fail(package_id, "A stored object of this analysis is missing; the package was not produced.")
    except deadline.JobTimeout as e:  # N-9: the build is left to end by itself; its work file is removed when it does
        log.error("packages.build_timeout", package_id=package_id, timeout_s=timeout_s)
        return _fail(package_id, f"{e}; the package was not produced. Request it again.")
    except Exception as e:
        log.error("packages.build_failed", package_id=package_id, error=str(e))
        return _fail(package_id, friendly_error(e))

    # N-10a (code re-verification R2-07): the ZIP is streamed into the store (D-48) before the unit, outside the append lock
    # and the database's write lock: a package of several hundred megabytes held both for as long as the disk took. The
    # object is content-addressed, so a job that then finds its row finished, failed or deleted leaves an object that no row
    # names, which `store gc` removes
    try:
        obj = store.put_file(built.path)
    except Exception as e:
        built.path.unlink(missing_ok=True)
        log.error("packages.store_failed", package_id=package_id, error=f"{type(e).__name__}: {e}"[:300])
        return _fail(package_id, "The finished package could not be stored, so it was not exported. Request the package again.")

    def unit() -> str | None:
        """The final write and its event, in one transaction (N-9). None when the row was not this job's to finish."""
        nonlocal obj
        with audit.writing() as db:
            # the transaction's first statement claims the row: done (its bytes were downloadable), failed or deleted, it is left
            claimed = db.execute(update(PackageRow).where(PackageRow.id == package_id, PackageRow.status.in_(OPEN))
                                 .values(status="done", error=None, finished_at=utcnow()).execution_options(synchronize_session=False))
            if claimed.rowcount != 1:
                db.rollback()
                return None
            if not store.exists(obj.sha256):  # a sweep took the object before a row named it (D-55): written once more
                obj = store.put_file(built.path)
            stored = index_object(db, obj.sha256, obj.size_bytes, media_type="application/zip", role="package", created_by="package",
                                  analysis_id=analysis_id)
            row = db.scalar(select(PackageRow).where(PackageRow.id == package_id).execution_options(populate_existing=True))
            row.artefact_id, row.sha256 = stored.id, stored.sha256
            row.manifest_sha256, row.signature, row.signing_key_id = built.manifest_sha256, built.signature, key.key_id
            row.file_count, row.size_bytes = built.files, built.size
            audit.commit_with(db, [{
                "action": "package_exported", "actor_id": actor, "analysis_id": analysis_id, "report_id": report_id,
                "object_ids": [stored.id], "after_hash": stored.sha256,
                "details": {"package_id": package_id, "kind": kind, "include_original": include_original, "files": built.files,
                            "manifest_sha256": built.manifest_sha256, "signature": built.signature, "signing_key_id": key.key_id,
                            "custody_anchor": built.anchor}}],
                before_commit=lambda seqs: setattr(row, "audit_event_seq", seqs[0]))
            return stored.sha256

    try:
        sha = audit.repeated(unit)
    except Exception as e:  # the row and its event are stored together: neither is, and the package is failed
        log.error("packages.record_failed", package_id=package_id, error=f"{type(e).__name__}: {e}"[:300])
        return _fail(package_id, "The finished package could not be recorded, so it was not exported. Request the package again.")
    finally:
        built.path.unlink(missing_ok=True)
    if sha is None:
        log.warning("packages.not_recorded", package_id=package_id, reason="the package was finished, failed or deleted meanwhile")
        return _outcome(package_id)
    log.info("packages.exported", package_id=package_id, files=built.files, sha256=sha)
    return "done"


# --- build ---------------------------------------------------------------------------------------------------------------------

def _read(store: ObjectStore, art: Artefact) -> bytes:
    data = store.read_bytes(art.sha256)
    if len(data) != art.bytes or hashlib.sha256(data).hexdigest() != art.sha256:
        raise PackageError("A stored object no longer matches its recorded hash; the package was not produced.")
    return data


def _verified(db: Session, store: ObjectStore, artefact_id: str, analysis_id: str) -> bytes:
    return _read(store, get_artefact(db, artefact_id, analysis_id=analysis_id))


def _open_original(db: Session, store: ObjectStore, analysis: Analysis) -> BinaryIO:
    """The original as a stream: it is copied into the ZIP in chunks and hashed on the way, never held whole (D-48)."""
    media = db.get(MediaObject, analysis.media_id) if analysis.media_id else None
    try:
        if media is not None and media.storage_artefact_id is not None:
            return store.open(media.sha256)
        return open(settings.uploads_dir / analysis.id / safe_filename(analysis.filename), "rb")
    except (ObjectNotFound, OSError):
        raise PackageError("The original is missing; the package was not produced.") from None


def _exhibit_artefacts(exhibit: dict[str, Any]) -> Iterator[tuple[str, str]]:
    for name, role in EXHIBIT_FIELDS:
        if exhibit.get(name):
            yield role, str(exhibit[name])
    for frame in exhibit.get("frames") or []:
        if frame.get("still_artefact_id"):
            yield f"still-{frame.get('frame_index')}", str(frame["still_artefact_id"])
    for i, series in enumerate(exhibit.get("series") or []):
        if series.get("payload_artefact_id"):
            yield f"series-{i}", str(series["payload_artefact_id"])


def _report_entry(row: ReportRow, file: str, *, issued: bool) -> dict[str, Any]:
    return {"id": row.id, "display_id": row.display_id, "revision": row.revision, "profile": row.profile, "file": file,
            "sha256": row.sha256, "signature": row.signature, "signing_key_id": row.signing_key_id,
            "supersedes_report_id": row.supersedes_report_id, "revision_reason": row.revision_reason,
            "review_state": row.review_state, "issued": issued, "page_count": row.page_count,
            "audit_event_seq": row.audit_event_seq}


def _review_identity(frozen: snapshot.ReportSnapshot) -> tuple[Any, ...]:
    """What two reports must share to describe the same review: the state, the issuable flag, the conclusion and the sign-offs,
    with the result of each signature check."""
    r = frozen.review
    return (r.state, r.issuable, (r.conclusion.id, r.conclusion.signature_valid) if r.conclusion is not None else None,
            tuple((s.id, s.signature_valid) for s in r.signoffs))


def _matching_summary(db: Session, store: ObjectStore, report: ReportRow, frozen: snapshot.ReportSnapshot) -> ReportRow | None:
    """The newest finished summary report whose snapshot holds the same review as `report` (N-3; code review 5): a summary
    requested before the conclusion, or before a sign-off, would contradict the report it rides with, so it is left out."""
    wanted = _review_identity(frozen)
    rows = db.scalars(select(ReportRow).where(
        ReportRow.analysis_id == report.analysis_id, ReportRow.profile == "summary", ReportRow.status == "done",
        ReportRow.redacted.is_(False), ReportRow.signature.is_not(None), ReportRow.artefact_id.is_not(None),
        ReportRow.review_state == report.review_state, ReportRow.issuable.is_(bool(report.issuable))).order_by(ReportRow.revision.desc()))
    for row in rows:
        if _review_identity(snapshot.load(_verified(db, store, row.snapshot_artefact_id, report.analysis_id))) == wanted:
            return row
    return None


def build(db: Session, package: PackageRow | None, *, store: ObjectStore, key: signing.InstanceKey, exported_at: datetime) -> Built:
    if package is None:
        raise PackageError("The package request no longer exists.")
    analysis, report = db.get(Analysis, package.analysis_id), db.get(ReportRow, package.report_id)
    if analysis is None or report is None or report.analysis_id != analysis.id:
        raise PackageError("The report of this package no longer exists; the package was not produced.")
    if report.status != "done" or report.artefact_id is None or not report.signature or report.redacted:
        raise PackageError("The report is not a finished, signed, unredacted report; the package was not produced.")
    if package.kind == "issue" and (analysis.withdrawn_at is not None or review_rules.superseded(db, analysis.id)):
        raise PackageError("The revision was withdrawn or superseded after the request, so the issue package was not produced.")
    if package.kind == "issue" and not report.issuable:  # D-56: the API refuses the request; a row written another way is refused here
        raise PackageError("The report was not requested in an issuable review state, so the issue package was not produced.")
    frozen = snapshot.load(_verified(db, store, report.snapshot_artefact_id, analysis.id))
    result = frozen.result
    issue = package.kind == "issue"

    files: dict[str, bytes] = {"report.pdf": _verified(db, store, report.artefact_id, analysis.id)}
    reports = [_report_entry(report, "report.pdf", issued=issue)]
    summary = _matching_summary(db, store, report, frozen) if report.profile != "summary" else None
    if summary is not None and summary.artefact_id is not None:
        files["executive-summary.pdf"] = _verified(db, store, summary.artefact_id, analysis.id)
        reports.append(_report_entry(summary, "executive-summary.pdf", issued=issue and bool(summary.issuable)))

    artifacts: list[dict[str, Any]] = []
    for exhibit in result.get("exhibits") or []:
        for role, artefact_id in _exhibit_artefacts(exhibit):
            entry = {"exhibit_id": exhibit.get("id"), "display_id": exhibit.get("display_id"), "kind": exhibit.get("kind"), "role": role}
            try:
                art = get_artefact(db, artefact_id, analysis_id=analysis.id)
            except ArtefactNotFound:
                artifacts.append({**entry, "file": None, "sha256": None, "note": "not in the artefact store (legacy exhibit)"})
                continue
            name = f"artifacts/{_UNSAFE.sub('_', str(exhibit.get('id')))}/{role}{EXTENSIONS.get(art.media_type, '.bin')}"
            files[name] = _read(store, art)
            artifacts.append({**entry, "file": name, "sha256": art.sha256})

    source = None
    if package.include_original:  # streamed into the ZIP below; its manifest hash is the recorded one, checked while copying
        source = {"file": f"source/{safe_filename(analysis.filename)}", "sha256": analysis.sha256}

    scope = AuditEvent.analysis_id == analysis.id
    if analysis.media_id is not None and analysis.case_id is not None:  # D-47: events on the original in this case (downloads, re-hashes)
        scope = or_(scope, and_(AuditEvent.analysis_id.is_(None), AuditEvent.media_id == analysis.media_id,
                                AuditEvent.case_id == analysis.case_id))
    custody = [audit.exported(e) for e in db.scalars(select(AuditEvent).where(scope).order_by(AuditEvent.seq))]
    tail = db.execute(select(AuditEvent.seq, AuditEvent.event_hash).order_by(AuditEvent.seq.desc()).limit(1)).first()
    anchor = {"head_seq": int(tail[0]) if tail else 0, "head_hash": str(tail[1]) if tail else audit.GENESIS, "at": _ts(exported_at)}
    review = frozen.review.model_dump(mode="json")
    bundle = {
        "schema": SCHEMA, "bundle_id": f"{analysis.id[:8]}-r{report.revision:02d}", "package_id": package.id, "kind": package.kind,
        "generated_at": _ts(exported_at),
        "engine": {"version": settings.version, "report_engine": report.engine_version, "template_set": report.template_set,
                   "manifest": result.get("manifest")},
        "case": frozen.case.model_dump(mode="json") if frozen.case is not None else None,
        "exhibit": {**frozen.media.model_dump(mode="json"), "label": frozen.case.exhibit_label if frozen.case is not None else None},
        "analysis": {**frozen.analysis.model_dump(mode="json"), "result_sha256": hashlib.sha256(canonical_json(result)).hexdigest()},
        "result": result,
        "review": review,
        "annotations": [{**a.model_dump(mode="json"), "body": a.body if a.visibility == "in_report" else None,
                         "body_withheld": a.visibility != "in_report" and bool(a.body)} for a in frozen.annotations],
        "custody": custody, "custody_anchor": anchor,
        "reports": reports, "artifacts": artifacts, "source": source,
        "signing": {"algorithm": signing.ALGORITHM, "key_id": key.key_id, "public_key_file": "instance_public_key.pem"},
    }
    name = filename(package, report)
    files["bundle.json"] = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    files["instance_public_key.pem"] = key.public_pem()
    files["reproduce.md"] = _reproduce_md(bundle, name).encode("utf-8")
    files["README.txt"] = _readme(bundle, report, name, sorted([*files, *([source["file"]] if source is not None else [])]),
                                  view.review_status(frozen).line).encode("utf-8")

    hashes = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    if source is not None:
        hashes[source["file"]] = analysis.sha256
    manifest = "".join(f"{hashes[n]}  {n}\n" for n in sorted(hashes)).encode("utf-8")
    signature = key.sign(manifest)
    files["manifest.sha256"], files["manifest.sha256.sig"] = manifest, (signature + "\n").encode("ascii")
    names = sorted([*files, *([source["file"]] if source is not None else [])])

    stamp = (exported_at if exported_at.tzinfo is not None else exported_at.replace(tzinfo=UTC)).astimezone(UTC).timetuple()[:6]
    workdir = settings.data_dir / "tmp" / "packages"
    workdir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=workdir, prefix=f"{package.id}-", suffix=".zip")
    path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for n in names:
                info = zipfile.ZipInfo(n, date_time=stamp)
                info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
                if n in files:
                    zf.writestr(info, files[n])
                    continue
                copied = hashlib.sha256()
                large = (analysis.size_bytes or 0) >= zipfile.ZIP64_LIMIT
                with _open_original(db, store, analysis) as src, zf.open(info, "w", force_zip64=large) as dst:
                    while chunk := src.read(CHUNK):
                        copied.update(chunk)
                        dst.write(chunk)
                if copied.hexdigest() != analysis.sha256:
                    raise PackageError("The original no longer matches its recorded hash; the package was not produced.")
        digest, size = hashlib.sha256(), 0
        with open(path, "rb") as f:
            while chunk := f.read(CHUNK):
                digest.update(chunk)
                size += len(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return Built(path, size, digest.hexdigest(), hashlib.sha256(manifest).hexdigest(), signature, len(names), anchor)


# N-11 (SEC-06): the package cannot vouch for its own key. No command printed in a package names the key id of that package.
PIN_LINES = (
    "  Where the instance key comes from: not from this package. Whoever alters a package can replace",
    "    instance_public_key.pem, sign every file again and rewrite this text, so the key id printed above is a claim to",
    "    compare, not a pin. Take the key or its id from the instance that issued the package, or from its operator: the",
    "    export dialog shows the key id and saves the key as a PEM file, and GET /api/v2/instance/public-key returns both.",
    f"  Without --public-key or --expect-key-id the verifier answers \"{SIGNER_NOT_AUTHENTICATED}\"",
    "    (exit code 3): the files agree with each other, and nothing says who signed them. Exit code 0 needs the key.",
)


def _readme(bundle: dict[str, Any], report: ReportRow, name: str, names: list[str], review_line: str) -> str:
    """`review_line` is the report's review status line (D-56): who signed, the peer reviewer's decision, or what is outstanding."""
    analysis = bundle["analysis"]
    state = report.review_state.replace("_", " ")
    kind = (f"ISSUED. Report {report.display_id}, review state {state}." if bundle["kind"] == "issue" else
            f"DRAFT. Not issued: the report was requested in the review state '{state}'.")
    artefacts = sum(1 for n in names if n.startswith("artifacts/"))
    rows = [("README.txt", "this file"),
            ("manifest.sha256", "SHA-256 of every other file, '<sha256>  <path>' sorted by path (not of itself or its signature)"),
            ("manifest.sha256.sig", "Ed25519 signature over manifest.sha256 (base64) by the instance key"),
            ("instance_public_key.pem", f"the instance public key {bundle['signing']['key_id']}"),
            ("bundle.json", f"{SCHEMA}: the result, case and exhibit identity, review, custody extract, reports and their signatures"),
            ("report.pdf", f"{report.profile} report {report.display_id}")]
    if "executive-summary.pdf" in names:
        rows.append(("executive-summary.pdf", "the summary report of the same review"))
    if artefacts:
        rows.append(("artifacts/", f"exhibit payloads, masks, previews and stills as stored ({artefacts} files)"))
    if bundle["source"] is not None:
        rows.append(("source/", "the original file"))
    rows.append(("reproduce.md", "how to re-run the analysis and compare its raw scores"))
    lines = ["Veritas evidence package", "", f"Package    {name}", f"Status     {kind}", f"Review     {review_line}",
             f"Analysis   {analysis['id']} (revision {analysis['revision']}, {analysis['kind']})",
             f"Report     {report.id} revision {report.revision} ({report.profile})", f"Exported   {bundle['generated_at']}",
             f"Signed by  {bundle['signing']['key_id']} (Ed25519)", "", "Files"]
    lines += [f"  {path:<24} {text}" for path, text in rows]
    lines += ["", "How to verify",
              "  Hashes only, without Veritas:  sha256sum -c manifest.sha256",
              "  Everything, with no server and no network, against the instance key:",
              f"    python -m app.cli verify {name} --public-key <instance key PEM>",
              f"    python -m app.cli verify {name} --expect-key-id <instance key id>",
              "    It checks every hash, that no file is missing or unlisted, the manifest and report signatures, the custody",
              "    extract (each event hash recomputed, the order of events), and that the signing key is the one you gave.",
              *PIN_LINES,
              "  This export is recorded in the instance's custody log as package_exported, with the SHA-256 of this ZIP.", ""]
    return "\n".join(lines)


def _number(value: Any) -> str:
    return "–" if value is None else f"{float(value):.6f}"


def _reproduce_md(bundle: dict[str, Any], name: str) -> str:
    result, analysis = bundle["result"], bundle["analysis"]
    manifest, decision = result.get("manifest") or {}, result.get("decision") or {}
    probability = decision.get("probability_ai_calibrated")
    if probability is None:
        probability = decision.get("probability_ai_uncalibrated")
    lines = [f"# Reproducing analysis {analysis['id']}", "",
             f"Kind {analysis['kind']} · profile {analysis.get('profile') or '–'} · pipeline {manifest.get('pipeline_version') or '–'} · "
             f"fusion {decision.get('fusion_mode') or '–'}" + (f" (fuser {decision['fuser_sha256']})" if decision.get("fuser_sha256") else ""), "",
             "## Command", "", f"    python -m app.cli reproduce {name} --public-key <instance key PEM> [--tolerance 0.001]",
             f"    python -m app.cli reproduce {name} --expect-key-id <instance key id> [--tolerance 0.001]", "",
             "The instance key or its id comes from the instance that issued the package, or from its operator (the export dialog;",
             "`GET /api/v2/instance/public-key`), and not from this package: see `README.txt`. Without `--public-key` or",
             f"`--expect-key-id` the command answers \"{SIGNER_NOT_AUTHENTICATED}\" (exit code 3) and runs nothing.", ""]
    lines += (["The command verifies this package against that key, runs the v2 pipeline on `source/` in a throwaway workspace (its own database and",
               "object store, the instance's pinned weights and fuser artefacts) and compares each signal's raw score and calibrated",
               "probability, the fused probability and the decision code with the values below. Nothing is written to the instance",
               "database or custody log. On the same host and device the values agree to 3 decimals; between MPS and CPU pass",
               "`--tolerance 0.02`."] if bundle["source"] is not None else
              ["This package holds no original (`source/`). Export it again with the original included to reproduce the analysis."])
    lines += ["", "## Detectors", "", "| Detector | Version | Weights | Licence |", "|---|---|---|---|"]
    lines += [f"| {d.get('id')} | {d.get('version') or '–'} | {d.get('weights_sha256') or d.get('revision') or 'model-free'} | {d.get('licence') or '–'} |"
              for d in sorted(manifest.get("detectors") or [], key=lambda d: str(d.get("id")))]
    lines += ["", "## Recorded raw scores", "", "| Signal | Status | Raw score | Calibrated probability |", "|---|---|---|---|"]
    lines += [f"| {s.get('id')} | {s.get('status')} | {_number(s.get('raw_score'))} | {_number(s.get('calibrated_probability'))} |"
              for s in sorted(result.get("signals") or [], key=lambda s: str(s.get("id")))]
    lines += ["", "## Fused estimate", "", f"Decision code `{decision.get('code')}`, probability {_number(probability)}.", ""]
    return "\n".join(lines)
