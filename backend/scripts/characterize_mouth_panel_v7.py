"""Characterize the unchanged current review check on consumed-source new-family edits."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from app.detectors.video.face_sequence import REVIEW_POINT
from scripts import compare_mouth_sequence_v4 as current
from scripts.liveportrait_mouth_benchmark import digest


def implementation():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "backend/scripts/characterize_mouth_panel_v7.py",
        "backend/scripts/compare_mouth_sequence_v4.py",
        "backend/app/detectors/video/face_sequence.py",
        "backend/app/detectors/video/adapters/face_sequence.py",
        "backend/app/detectors/video/adapters/tracks.py",
        "backend/app/detectors/video/tracking.py",
        "backend/app/detectors/video/adapters/lipsync_xcorr.py",
        "backend/app/detectors/video/xcorr.py",
        "backend/app/models.yaml",
    ]
    return {p: digest(root / p) for p in paths}


def freeze(manifest, output):
    data = json.loads(manifest.read_text())
    cases = data["items"]
    conditions = {"muse-resample", "local-blur-2", "local-blur-4", "muse-ai-aligned", "muse-ai-earlier"}
    groups = {row["group"] for row in cases}
    if len(cases) != 40 or len({row["id"] for row in cases}) != 40 or len(groups) != 8:
        raise ValueError("All forty distinct planned cases and eight source groups are required")
    if any(
        {row["condition"] for row in cases if row["group"] == group} != conditions for group in groups
    ) or any(row["label"] != int(row["condition"].startswith("muse-ai-")) for row in cases):
        raise ValueError("A planned condition or source label changed")
    body = {
        "schema": "chipmunk.mouth-extension-characterization-lock/7",
        "manifest_sha256": digest(manifest),
        "implementation_sha256": implementation(),
        "current_review_point": REVIEW_POINT,
        "planned_cases": len(data["items"]),
        "role": "Unchanged current-check characterization, not fresh confirmation",
        "deployment_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"lock_sha256": digest(output), "planned": len(data["items"])}))


def score(manifest, lock_path, output):
    import torch

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    lock = json.loads(lock_path.read_text())
    if (
        lock["manifest_sha256"] != digest(manifest)
        or lock["implementation_sha256"] != implementation()
        or lock["current_review_point"] != REVIEW_POINT
    ):
        raise ValueError("Frozen inputs or current check changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models, weights = {}, {}
    for mid, loader in (("yunet.2023mar", load_yunet), ("dinov2.small", load_dinov2_small)):
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Commercial checkpoint declaration required")
        weights[mid] = [{"name": p.name, "sha256": digest(p)} for p in ensure_verified(spec, force=True)]
        models[mid] = loader(mid, "cpu", "float32")
    header = {
        "schema": "chipmunk.mouth-extension-characterization-scores/7",
        "manifest_sha256": digest(manifest),
        "lock_sha256": digest(lock_path),
        "weights": weights,
        "device": "cpu",
        "torch_version": torch.__version__,
    }
    cases = json.loads(manifest.read_text())["items"]
    complete = []
    if output.exists():
        old, *complete = [json.loads(line) for line in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r["id"] for r in cases[: len(complete)]]:
            raise ValueError("Cannot resume different source outcomes")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for case in cases[len(complete) :]:
            started = time.monotonic()
            row = {**case, "status": "not_applicable", "base_windows": []}
            try:
                if case["status"] == "ready":
                    path = manifest.parent / case["path"]
                    if digest(path) != case["sha256"]:
                        raise ValueError("Prepared file changed")
                    result = current.score_case(path, case, models)
                    row.update(
                        {
                            k: result[k]
                            for k in (
                                "status",
                                "reason",
                                "base_windows",
                                "sampling_centers",
                                "time_base",
                                "baseline_capture_parity",
                                "timing",
                            )
                        }
                    )
                    row["roi_bindings"] = {k: v for k, v in result["roi_bindings"].items() if k == "base"}
                else:
                    row["reason"] = case.get("reason", "Prepared input unavailable")
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "seconds": row["runtime_s"],
                        "reason": row.get("reason"),
                    }
                ),
                flush=True,
            )


def review(manifest, receipt, output):
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    data = json.loads(manifest.read_text())
    if header["manifest_sha256"] != digest(manifest) or [r["id"] for r in rows] != [
        r["id"] for r in data["items"]
    ]:
        raise ValueError("Every planned source outcome is required")
    evaluated = current.evaluate(rows, "base", REVIEW_POINT, manifest.parent)
    result = {
        "schema": "chipmunk.mouth-extension-characterization-summary/7",
        "manifest_sha256": digest(manifest),
        "score_receipt_sha256": digest(receipt),
        "current_review_point": REVIEW_POINT,
        "summary": current.summary(evaluated),
        "conditions": {
            c: current.summary([r for r in evaluated if r["condition"] == c])
            for c in sorted({r["condition"] for r in evaluated})
        },
        "terminal_statuses": dict(Counter(r["status"] for r in rows)),
        "production_capture_parity_cases": sum(bool(r.get("baseline_capture_parity")) for r in rows),
        "timing_statuses": dict(Counter(r.get("timing", {}).get("status", "unassessed") for r in rows)),
        "timing_flagged_track_readings": sum(
            r.get("timing", {}).get("metrics", {}).get("flagged_tracks", 0) for r in rows
        ),
        "cases": evaluated,
        "deployment_eligible": False,
        "limitations": [
            "Eight consumed source actors and existing sentence families; a new generation family is not fresh source confirmation.",
            "Review-cue localization and ordinary review flags are not whole-file authenticity decisions or probabilities.",
            "Timing readings are context; synchronization does not establish an original file.",
            "No replacement candidate is selected or re-fitted; no outside independent review has occurred.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "conditions", "limitations")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "score", "review"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path, nargs="?")
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.manifest, a.input)
    elif not a.output:
        p.error("score/review requires output")
    elif a.stage == "score":
        score(a.manifest, a.input, a.output)
    else:
        review(a.manifest, a.input, a.output)
