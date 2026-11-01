"""Run a locked, externally labelled sample list through the ordinary local API.

Ground-truth labels never enter the submitted filename or analysis request.
This is an end-to-end smoke evaluation, separate from the audio calibration study.
Usage: python scripts/run_web_evaluation.py MANIFEST --out DIRECTORY
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import time
from collections import defaultdict
from pathlib import Path

import httpx


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_items(manifest: Path) -> tuple[list[dict], dict]:
    data = json.loads(manifest.read_text())
    items = data if isinstance(data, list) else data["items"]
    ids, hashes = set(), {}
    locked = []
    for source in items:
        item = dict(source)
        if item["id"] in ids or item["label"] not in {"ai", "real"}:
            raise ValueError("Duplicate sample ID or unsupported truth label")
        ids.add(item["id"])
        if item["kind"] not in {"image", "video", "audio", "document"}:
            raise ValueError("Unsupported media kind")
        path = Path(item["path"])
        if not path.is_absolute():
            # Sample preparation writes paths relative to the backend workspace.
            path = Path(__file__).resolve().parents[1] / path
        path = path.resolve()
        if digest(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"Source changed: {item['id']}")
        if item["sha256"] in hashes:
            raise ValueError(f"Duplicate source bytes: {item['id']} / {hashes[item['sha256']]}")
        hashes[item["sha256"]] = item["id"]
        for field in ("source_url", "source_group"):
            if not item.get(field):
                raise ValueError(f"Missing {field}: {item['id']}")
        item["path"] = str(path)
        item["upload_name"] = "sample-" + item["sha256"][:16] + path.suffix.lower()
        locked.append(item)
    lock = {"manifest_sha256": digest(manifest.read_bytes()), "items": locked,
            "profile": "full", "selection_policy": "All prespecified items, including failures and abstentions",
            "purpose": "Externally labelled smoke test; not a representative accuracy or calibration benchmark"}
    return locked, lock


def outcome(item: dict, detail: dict) -> dict:
    assessment = detail.get("customer_assessment") or {}
    decision = detail.get("decision") or {}
    record_led = (assessment.get("applies") is False or bool(detail.get("stored_record"))
                  or decision.get("driving_hypothesis") == "H3_provenance_integrity")
    direction = assessment.get("direction", "unavailable")
    if record_led:
        predicted = "record_led"
    elif direction in {"ai", "original"}:
        predicted = "ai" if direction == "ai" else "real"
    else:
        predicted = "inconclusive"
    if detail.get("status") != "done":
        predicted = "failed"
    return {"sample_id": item["id"], "kind": item["kind"], "label": item["label"],
            "source_group": item["source_group"], "source_url": item["source_url"],
            "generator": item.get("generator"), "analysis_id": detail["id"],
            "status": detail.get("status"), "prediction": predicted,
            "correct_direction": predicted == item["label"],
            "confidence_pct": assessment.get("confidence_pct"), "scope": assessment.get("scope"),
            "headline": assessment.get("headline"), "decision_code": decision.get("code"),
            "driving_hypothesis": decision.get("driving_hypothesis"),
            "record_led": record_led, "raw_combined_ai_estimate": decision.get("probability_ai_calibrated"),
            "error": detail.get("error"),
            "local_url": f"http://127.0.0.1:5230/analyses/{detail['id']}?view=plain"}


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"{row['kind']}/{row['label']}"].append(row)
    summary = {}
    for key, entries in sorted(groups.items()):
        summary[key] = {"n": len(entries), "correct_direction": sum(r["correct_direction"] for r in entries),
                        "wrong_direction": sum(r["prediction"] in {"ai", "real"} and not r["correct_direction"] for r in entries),
                        **{name: sum(r["prediction"] == name for r in entries)
                           for name in ("inconclusive", "record_led", "failed")}}
    return {"purpose": "Small external sample check, not general accuracy; all planned outcomes retained",
            "video_scope": "A visual-only tendency evaluates the picture, not unassessed audio",
            "groups": summary, "items": rows}


def run(manifest: Path, folder: Path, base_url: str, selected: set[str] | None, timeout: float) -> None:
    items, lock = load_items(manifest)
    folder.mkdir(parents=True, exist_ok=True)
    lockfile = folder / "manifest-lock.json"
    if lockfile.exists() and json.loads(lockfile.read_text()) != lock:
        raise ValueError("Manifest or configuration changed after this run was locked; use a new output directory")
    if not lockfile.exists():
        save_json(lockfile, lock)
    statefile = folder / "run-state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {"case": None, "submitted": {}}
    with httpx.Client(base_url=base_url, timeout=120) as client:
        def request(method: str, url: str, **kwargs):
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

        if state["case"] is None:
            state["case"] = request("POST", "/api/v2/cases", json={
                "title": "Independent web-sample evaluation", "classification": "internal",
                "description": "Prespecified public samples. Ground truth and source provenance are retained in the local evaluation manifest; this case holds detector outputs."})
            save_json(statefile, state)
        for item in items:
            if selected and item["id"] not in selected:
                continue
            entry = folder / "items" / item["id"]
            entry.mkdir(parents=True, exist_ok=True)
            if (entry / "outcome.json").exists():
                continue
            if item["id"] not in state["submitted"]:
                with Path(item["path"]).open("rb") as stream:
                    response = request("POST", f"/api/v2/cases/{state['case']['id']}/items",
                                       params={"analyze": "true", "profile": "full"},
                                       data={"kind": item["kind"], "client_sha256": item["sha256"],
                                             "acquisition_method": "url_download"},
                                       files={"file": (item["upload_name"], stream,
                                                      mimetypes.guess_type(item["path"])[0] or "application/octet-stream")})
                state["submitted"][item["id"]] = response["analysis"]["id"]
                save_json(statefile, state)
                save_json(entry / "intake.json", response)
            aid = state["submitted"][item["id"]]
            deadline = time.monotonic() + timeout
            previous = None
            while True:
                detail = request("GET", f"/api/v2/analyses/{aid}")
                stage = (detail["status"], detail.get("current_stage"))
                if stage != previous:
                    print(json.dumps({"item": item["id"], "id": aid, "state": stage}), flush=True)
                    previous = stage
                if detail["status"] in {"done", "failed", "cancelled"}:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Analysis remains active; resume with the same directory: {aid}")
                time.sleep(3)
            save_json(entry / "analysis.json", detail)
            if detail["status"] == "done":
                exported = request("GET", f"/api/analyses/{aid}/export.json")
                assert exported["customer_assessment"] == detail["customer_assessment"]
                save_json(entry / "export.json", exported)
            save_json(entry / "outcome.json", outcome(item, detail))
            rows = [json.loads(path.read_text()) for path in sorted((folder / "items").glob("*/outcome.json"))]
            save_json(folder / "results.json", summarize(rows))
            with (folder / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8160")
    parser.add_argument("--item", action="append")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.out.resolve(), args.base_url, set(args.item) if args.item else None, args.timeout)
