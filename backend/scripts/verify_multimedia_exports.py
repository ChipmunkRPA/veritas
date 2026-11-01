"""Live local PDF/JSON evidence correspondence, including acoustic context.

Run from backend. --baseline records immutable history; --analysis-id may be
repeated after new analyses finish. This script does not submit inference jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.public_projection import _ARCHITECTURE_LABEL, _FUSION_ID, _identities
from app.evidence import service
from app.reports import engine, multimedia_evidence, signing, snapshot, verify
from app.storage.db import Analysis, Artefact, ReportRow, get_session
from app.storage.objects import default_store


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f"Refusing to overwrite different output: {path}"
    else:
        path.write_bytes(data)


def dump(path, data):
    save(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def history():
    with get_session() as db:
        return {
            "analyses": [{k: getattr(a, k) for k in ("id", "status", "result_hash", "sha256", "revision")} |
                         {"result_json_sha256": sha(a.result_json.encode()) if a.result_json else None}
                         for a in db.query(Analysis).all() if a.status in {"done", "failed", "cancelled"}],
            "reports": [{k: getattr(r, k) for k in ("id", "analysis_id", "status", "template_set", "sha256", "snapshot_hash", "signature")}
                        for r in db.query(ReportRow).all() if r.status == "done"],
            "evidence": [{"id": a.id, "sha256": a.sha256} for a in db.query(Artefact).all()
                         if a.role == "exhibit_payload" and "evidence-packets/" in (a.created_by or "")],
        }


def scan(text):
    # Source quotations must remain verbatim. These controlled QA sources contain
    # no implementation names, so scanning their entire public exports is safe.
    text = text.replace("FUSED ERROR RATES IN TESTING", "COMBINED ERROR RATES IN TESTING")
    ids, names = _identities()
    for value in set(ids) | set(names) | {"wav2vec", "Gustking", "Tesseract", "SyncNet", "AV-HuBERT"}:
        flags = 0 if value == "FUSED" else re.I
        assert re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, flags) is None, f"Private implementation name: {value}"
    for pattern in (_ARCHITECTURE_LABEL, _FUSION_ID):
        match = pattern.search(text)
        assert match is None, f"Private architecture or fusion identifier: {match.group() if match else ''}"


def pixels(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.size, sha(image.convert("RGB").tobytes())


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response


def wait_job(client, path, record):
    deadline = time.monotonic() + 600
    while record["status"] not in {"done", "failed", "cancelled"}:
        assert time.monotonic() < deadline, f"Timed out: {path}"
        time.sleep(1)
        record = request(client, "GET", path).json()
    assert record["status"] == "done", record
    return record


def run(client, aid, folder, store, key):
    folder.mkdir(parents=True, exist_ok=True)
    evidence = request(client, "GET", f"/api/v2/analyses/{aid}/evidence-packets").json()
    detail = request(client, "GET", f"/api/v2/analyses/{aid}")
    scan(detail.text)
    save(folder / "analysis.json", detail.content)
    exported = request(client, "GET", f"/api/analyses/{aid}/export.json")
    assert exported.json()["evidence"] == evidence and evidence["status"] == "ready"
    scan(exported.text)
    dump(folder / "evidence.json", evidence)
    save(folder / "export.json", exported.content)
    with get_session() as db:
        row = db.get(Analysis, aid)
        result = json.loads(store.read_bytes(row.result_hash))
        multimedia_evidence._validate(evidence, kind=row.kind, analysis_id=aid, revision=row.revision,
                                      result=result, source_sha256=row.sha256)
    verified = multimedia_evidence.verify_assets(evidence, store, result=result)
    assert set(verified) == set(service.asset_refs(evidence))
    for digest, asset in service.asset_refs(evidence).items():
        response = request(client, "GET", asset["url"])
        assert sha(response.content) == digest and response.content == verified[digest]
        ext = {"image/png": "png", "audio/wav": "wav", "video/mp4": "mp4"}[asset["media_type"]]
        save(folder / "assets" / f"{digest}.{ext}", response.content)
        if ext in {"wav", "mp4"}:
            partial = request(client, "GET", asset["url"], headers={"Range": "bytes=0-63"})
            assert partial.status_code == 206 and partial.content == response.content[:64]
    saved_report = folder / "report-request.json"
    if saved_report.exists():
        report = json.loads(saved_report.read_bytes())
    else:
        report = request(client, "POST", f"/api/v2/analyses/{aid}/reports", json={"profile": "standard", "reuse": True}).json()
        dump(saved_report, report)
    report = wait_job(client, f'/api/v2/reports/{report["id"]}', report)
    assert report["downloadable"] and report["template_set"] == engine.TEMPLATE_SET
    dump(folder / "report.json", report)
    pdf_bytes = request(client, "GET", f'/api/v2/reports/{report["id"]}/file').content
    assert sha(pdf_bytes) == report["sha256"] and signing.verify(signing.load_public_key(key), pdf_bytes, report["signature"])
    save(folder / "report.pdf", pdf_bytes)
    with get_session() as db:
        record = db.get(ReportRow, report["id"])
        frozen = snapshot.load(store.read_bytes(record.snapshot_hash))
    assert frozen.evidence == evidence
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    texts = [page.get_text() for page in pdf]
    scan("\n".join(texts))
    save(folder / "report-text.txt", "\n".join(texts).encode())
    manifests = [json.loads(pdf.embfile_get(name)) for name in pdf.embfile_names() if name.endswith(".json")]
    manifest = next(item for item in manifests if "evidence" in item)
    assert manifest["evidence"] == multimedia_evidence.manifest(frozen)
    assert {a["sha256"] for a in manifest["evidence"]["artifacts"]} == set(verified)
    scan(json.dumps(manifest))
    dump(folder / "pdf-manifest.json", manifest)
    embedded = {pixels(pdf.extract_image(xref)["image"]) for xref in
                {i[0] for page in pdf for i in page.get_images(full=True)}}
    for digest, asset in service.asset_refs(evidence).items():
        if asset["media_type"] == "image/png":
            assert pixels(verified[digest]) in embedded, f"PDF missing exact evidence pixels: {digest}"
    acoustic = evidence.get("acoustic_context") or evidence.get("audio", {}).get("acoustic_context")
    if acoustic:
        assert "Background and acoustic context" in "\n".join(texts)
        for event in acoustic["events"]:
            assert event["id"] in "\n".join(texts)
            assert " ".join(event["observation"].split()) in " ".join("\n".join(texts).split())
    (folder / "pages").mkdir(exist_ok=True)
    subprocess.run(["pdftoppm", "-scale-to", "1250", "-png", str(folder / "report.pdf"), str(folder / "pages/page")], check=True)
    saved_package = folder / "package-request.json"
    if saved_package.exists():
        package = json.loads(saved_package.read_bytes())
    else:
        package = request(client, "POST", f"/api/v2/analyses/{aid}/packages",
                          json={"report_id": report["id"], "kind": "draft", "include_original": False}).json()
        dump(saved_package, package)
    package = wait_job(client, f'/api/v2/packages/{package["id"]}', package)
    dump(folder / "package.json", package)
    archive_bytes = request(client, "GET", f'/api/v2/packages/{package["id"]}/file').content
    assert sha(archive_bytes) == package["sha256"]
    path = folder / "evidence.zip"
    save(path, archive_bytes)
    checked = verify.verify_package(path, public_key_pem=key)
    assert checked.ok, checked.as_dict()
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence
        assert archive.read("report.pdf") == pdf_bytes
        bundle = json.loads(archive.read("bundle.json"))
        assert not bundle["source"]
        for asset in bundle["artifacts"]:
            assert archive.read(asset["file"]) == verified[asset["sha256"]]
        for name in archive.namelist():
            if name.endswith((".json", ".txt", ".md")):
                scan(archive.read(name).decode())
    summary = {"analysis_id": aid, "evidence_sha256": evidence["sha256"], "version": evidence["version"],
               "report_id": report["id"], "pages": len(pdf), "assets": len(verified),
               "acoustic_events": len(acoustic["events"]) if acoustic else 0,
               "source_media_and_coordinates_verified": True, "api_json_snapshot_equal": True,
               "all_pngs_exact_in_pdf": True, "public_names_scan_passed": True,
               "signed_pdf_and_offline_package_verified": True}
    dump(folder / "verification.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--analysis-id", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        baseline = history()
        dump(args.output / "history-before.json", baseline)
        print(json.dumps({k: len(v) for k, v in baseline.items()}))
        return
    store = default_store()
    with httpx.Client(base_url="http://127.0.0.1:8160", timeout=120) as client:
        key = request(client, "GET", "/api/v2/instance/public-key").json()["public_key_pem"].encode()
        save(args.output / "trusted-instance.pem", key)
        rows = [run(client, aid, args.output / aid, store, key) for aid in args.analysis_id]
    before = json.loads((args.output / "history-before.json").read_bytes())
    current = history()
    for kind, previous in before.items():
        now = {row["id"]: row for row in current[kind]}
        assert all(now[row["id"]] == row for row in previous), f"Historical {kind} changed"
    dump(args.output / "verification.json", {"results": rows, "historical_records_unchanged": {k: len(v) for k,v in before.items()}})


if __name__ == "__main__":
    main()
