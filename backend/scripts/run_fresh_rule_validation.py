"""Run the frozen production pipelines in isolated storage, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "data/independent-evaluation/rule80-v1"
PROFILE = "full"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    suffixes = {".py", ".json", ".yaml", ".npy", ".j2", ".css"}
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "app").rglob("*"))
            if p.is_file() and p.suffix in suffixes}


def save_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def freeze():
    from app.engine.build_info import config_sha256, sanitised_settings
    from app.engine.customer_assessment import RULE, THRESHOLD
    from app.engine.detector import Modality
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    registry = get_registry()
    detectors = {kind: registry.enabled(modality, PROFILE) for kind, modality in (
        ("image", Modality.IMAGE), ("video", Modality.VIDEO), ("audio", Modality.AUDIO), ("document", Modality.TEXT))}
    ids = sorted({mid for selected in detectors.values() for did in selected
                  for mid in registry.required_models(registry.manifest.detector(did))})
    models = {}
    for mid in ids:
        model = registry.manifest.model(mid)
        if model is None or model.licence.class_ != "commercial":
            raise ValueError("An enabled model lacks commercial checkpoint permission")
        models[mid] = {"license": model.licence.model_dump(mode="json"),
                       "files": {str(p.resolve()): digest(p) if p.is_file() else None
                                 for p in ensure_verified(model)}}
    with sqlite3.connect(f"file:{ROOT / 'data/veritas.db'}?mode=ro", uri=True) as db:
        row = db.execute("select id,version,thresholds,sha256 from policy_thresholds order by version desc limit 1").fetchone()
    policy = {"id": row[0], "version": row[1], "thresholds": json.loads(row[2]), "sha256": row[3]}
    lock = {"schema": "chipmunk-rule80-validation/1", "created_at": datetime.now(UTC).isoformat(),
            "customer_rule": RULE, "customer_threshold": THRESHOLD, "profile": PROFILE,
            "settings": sanitised_settings(), "config_sha256": config_sha256(),
            "source_sha256": source_hashes(), "detectors": detectors, "models": models, "policy": policy,
            "runner_sha256": digest(Path(__file__)), "preparation_sha256": digest(Path(__file__).with_name("prepare_fresh_rule_validation.py")),
            "definitions": {
                "false_alarm": "An AI tendency on a genuine input, divided by all planned genuine inputs.",
                "false_original": "An original tendency on an AI input, divided by all planned AI inputs.",
                "missed_ai": "AI inputs without an AI tendency, including inconclusive, unavailable and failed runs.",
                "inconclusive": "Completed runs with no AI/original tendency, excluding provenance-led results.",
                "failure": "Unavailable source or failed pipeline. Kept in the planned denominator.",
                "provenance_led": "Reported separately from the statistical 80% rule."},
            "freshness": "Source recordings, image IDs and article groups excluded against retained local manifests. Detector/backbone training overlap is unknown. Audio voices overlap previous studies.",
            "sampling": {"image": "12 camera/generation cases per class plus 12 original/splice pairs",
                         "video": "12 native camera/generation pairs and 12 original/swap pairs",
                         "audio": "24 fresh recordings per class; four synthesis families; clean, noise 0 dB and music 5 dB",
                         "document": "24 genuine/generated article pairs across two source datasets"}}
    save_new(TRIAL / "frozen-rule.json", lock)
    print("Frozen rule", RULE, "threshold", THRESHOLD, "commercial model entries", len(models), flush=True)


def run(kind):
    # Imports occur after isolated paths have been set, before any database is opened.
    work = TRIAL / f"runs-{kind}"
    work.mkdir(exist_ok=True)
    os.environ["VERITAS_DATA_DIR"] = str(work)
    os.environ["VERITAS_DATABASE_URL"] = "sqlite:///" + str(work / "veritas.db")
    keys = work / "keys"
    if not keys.exists():
        keys.symlink_to(ROOT / "data/keys", target_is_directory=True)

    import torch

    from app.core.config import settings
    from app.engine.build_info import config_sha256
    from app.engine.customer_assessment import customer_assessment
    from app.engine.models import get_model_manager
    from app.engine.pipelines import run_analysis
    from app.engine.registry import get_registry
    from app.policy.thresholds import Thresholds
    from app.policy.thresholds import digest as policy_digest
    from app.storage.db import Analysis, PolicyThresholdsRow, get_session, init_db
    from app.storage.objects import default_store

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise ValueError("This frozen local run requires the Mac accelerator")
    lock = json.loads((TRIAL / "frozen-rule.json").read_text())
    if source_hashes() != lock["source_sha256"] or config_sha256() != lock["config_sha256"]:
        raise ValueError("Source or scoring configuration changed after the freeze")
    if digest(Path(__file__)) != lock["runner_sha256"]:
        raise ValueError("The evaluation runner changed after the freeze")
    manifest = TRIAL / f"{kind}-manifest.json"
    inputs = json.loads(manifest.read_text())["items"]
    receipt = TRIAL / f"{kind}-receipt.json"
    record = {"manifest_sha256": digest(manifest), "rule_sha256": digest(TRIAL / "frozen-rule.json"),
              "count": len(inputs), "kind": kind, "started_at": datetime.now(UTC).isoformat()}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if any(previous[key] != record[key] for key in ("manifest_sha256", "rule_sha256", "count", "kind")):
            raise ValueError("Resume inputs differ from the initial receipt")
    else:
        save_new(receipt, record)
    init_db()
    policy = lock["policy"]
    assert policy_digest(Thresholds(**policy["thresholds"])) == policy["sha256"]
    with get_session() as db:
        if db.get(PolicyThresholdsRow, policy["id"]) is None:
            db.add(PolicyThresholdsRow(id=policy["id"], version=policy["version"], thresholds=policy["thresholds"],
                                      sha256=policy["sha256"], reason="Frozen local validation policy", created_by="system",
                                      created_at=datetime.now(UTC)))
            db.commit()
    registry = get_registry()
    models = get_model_manager()
    assert registry.enabled("text" if kind == "document" else kind, PROFILE) == lock["detectors"][kind]
    output = TRIAL / f"{kind}-outcomes.jsonl"
    completed = {row["id"] for row in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    store = default_store()
    for item in inputs:
        if item["id"] in completed:
            continue
        started = time.perf_counter()
        row = {**item, "rule_sha256": record["rule_sha256"], "manifest_sha256": record["manifest_sha256"]}
        if item["status"] != "ready":
            row.update(status="source_unavailable", prediction="failed")
        else:
            source = TRIAL / item["path"]
            if digest(source) != item["sha256"]:
                raise ValueError("A frozen sample changed")
            aid = str(uuid.uuid4())
            name = item["sha256"][:16] + source.suffix
            upload = settings.uploads_dir / aid
            upload.mkdir(parents=True)
            (upload / name).write_bytes(source.read_bytes())
            with get_session() as db:
                db.add(Analysis(id=aid, kind=kind, filename=name, size_bytes=source.stat().st_size,
                                content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                                sha256=item["sha256"], status="queued"))
                db.commit()
            try:
                outcome = run_analysis(aid, profile=PROFILE, registry=registry, models=models, policy_thresholds_id=policy["id"])
                with get_session() as db:
                    analysis = db.get(Analysis, aid)
                    result_hash = analysis.result_hash
                if outcome.status != "done" or not result_hash:
                    raise RuntimeError(f"{outcome.status}: {outcome.error}")
                result = json.loads(store.read_bytes(result_hash))
                assessment = customer_assessment({**result, "status": "done", "kind": kind})
                direction = assessment["direction"]
                prediction = direction if direction in {"ai", "original", "record"} else "inconclusive"
                if prediction in {"ai", "original"}:
                    assert assessment["confidence_pct"] >= 80
                row.update(status="done", prediction=prediction, analysis_id=aid, result_sha256=result_hash,
                           assessment=assessment, decision_code=result["decision"]["code"],
                           fuser_sha256=result["decision"].get("fuser_sha256"), condition_profile=result.get("condition_profile"),
                           checks=[{"id": e["detector_id"], "status": e["status"], "reason": e.get("reason")}
                                   for stage in ("provenance", "tracks", "faces", "detect", "localise")
                                   for e in outcome.outputs.get(stage, {}).get("detectors", [])])
            except Exception as error:
                row.update(status="failed", prediction="failed", analysis_id=aid, error=f"{type(error).__name__}: {error}")
        row["seconds"] = round(time.perf_counter() - started, 3)
        with output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(item["id"], row["status"], row["prediction"], row["seconds"], flush=True)
    if source_hashes() != lock["source_sha256"] or config_sha256() != lock["config_sha256"]:
        raise ValueError("The scoring implementation changed during the run")
    print(kind, "finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("freeze", "image", "video", "audio", "document"))
    args = parser.parse_args()
    if args.kind == "freeze":
        freeze()
    else:
        run(args.kind)


if __name__ == "__main__":
    main()
