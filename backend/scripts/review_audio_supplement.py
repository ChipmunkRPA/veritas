"""Recount a frozen audio supplement using submitted scores and acoustic routes."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.bench.independent import SCHEMA as MANIFEST_SCHEMA  # noqa: E402
from app.bench.independent import audit_receipt, digest  # noqa: E402

CONDITIONS = ("clean", "mostly_single_interferer", "overlapping_interferers", "unknown")
NOISY = frozenset(CONDITIONS[1:3])
DIRECTIONS = ("ai", "original", "inconclusive", "unassessed")
IDENTITY_FIELDS = (
    "system", "implementation_sha256", "method_kind", "weight_sha256", "score_kind", "response_units",
    "calibration_sha256", "incumbent_sha256", "router_sha256",
)
FREEZE_SCHEMA = "chipmunk.audio-supplement-freeze/1"
REPORT_SCHEMA = "chipmunk.audio-supplement-review/1"
POLICY = "incumbent_inconclusive_and_routed_response_strictly_above_point"


def _measurements(manifest: dict, receipt: dict, split: str) -> list[dict]:
    if any(case.get("modality") != "audio" or case.get("condition") not in CONDITIONS
           for case in manifest.get("cases", [])):
        raise ValueError("This review needs an audio-only manifest with declared acoustic conditions")
    for field in ("incumbent_sha256", "router_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Incumbent and acoustic router SHA-256 identities are required")
    rows = audit_receipt(manifest, receipt, split)
    groups = {}
    for case in manifest["cases"]:
        for key in ("input_sha256", "source_sha256"):
            prior = groups.setdefault(case[key], case["group"])
            if prior != case["group"]:
                raise ValueError("An exact input or parent source appears in multiple source groups")
    results = []
    for case, prediction in zip(rows, receipt["predictions"], strict=True):
        incumbent = prediction.get("incumbent_direction")
        observed = prediction.get("observed_condition")
        if incumbent not in DIRECTIONS or observed not in CONDITIONS:
            raise ValueError("Every case needs a recorded incumbent direction and acoustic route")
        results.append({**case, "incumbent_direction": incumbent, "observed_condition": observed})
    return results


def _validate_points(points: dict, score_kind: str) -> None:
    if (not isinstance(points, dict) or not points or not points.keys() <= NOISY
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or (score_kind == "calibrated" and not 0 <= value <= 1) for value in points.values())):
        raise ValueError("Provide finite operating points for one or both measured noisy routes")


def _identity(receipt: dict) -> dict:
    return {key: receipt.get(key) for key in IDENTITY_FIELDS}


def freeze(manifest: dict, development: dict, points: dict) -> dict:
    """Bind chosen development points before the confirmation run."""
    rows = _measurements(manifest, development, "development")
    _validate_points(points, development["score_kind"])
    return {
        "schema": FREEZE_SCHEMA,
        "policy": POLICY,
        "manifest_sha256": digest(manifest),
        "development_receipt_sha256": digest(development),
        "identity": _identity(development),
        "points": dict(points),
        "development_support": {
            route: {
                "genuine_groups": len({row["group"] for row in rows if row["label"] == 0
                                       and row["observed_condition"] == route and _can_evaluate(row, points)}),
                "synthetic_groups": len({row["group"] for row in rows if row["label"] == 1
                                         and row["observed_condition"] == route and _can_evaluate(row, points)}),
            }
            for route in sorted(points)
        },
    }


def _can_evaluate(row: dict, points: dict) -> bool:
    return (row["incumbent_direction"] == "inconclusive"
            and row["observed_condition"] in points and row["status"] == "ok")


def _class_summary(rows: list[dict], points: dict) -> dict:
    directions = Counter(row["incumbent_direction"] for row in rows)
    routed = [row for row in rows if row["observed_condition"] in points]
    eligible = [row for row in routed if _can_evaluate(row, points)]
    added = [row for row in eligible if row["response"] > points[row["observed_condition"]]]
    supplemented = directions.copy()
    supplemented["ai"] += len(added)
    supplemented["inconclusive"] -= len(added)
    routes = Counter(row["observed_condition"] for row in rows)
    added_routes = Counter(row["observed_condition"] for row in added)
    return {
        "planned": len(rows),
        "incumbent_assessed": len(rows) - directions["unassessed"],
        "supplement_scored": sum(row["status"] == "ok" for row in rows),
        "routed": len(routed),
        "eligible": len(eligible),
        "observed_routes": {condition: routes[condition] for condition in CONDITIONS},
        "incumbent": {direction: directions[direction] for direction in DIRECTIONS},
        "supplemented": {direction: supplemented[direction] for direction in DIRECTIONS},
        "added_flags": len(added),
        "added_flags_by_route": {route: added_routes[route] for route in sorted(points)},
        "added_flag_groups": len({row["group"] for row in added}),
    }


def _summary(rows: list[dict], points: dict) -> dict:
    routes = Counter(row["observed_condition"] for row in rows)
    return {
        "planned": len(rows),
        "observed_routes": {condition: routes[condition] for condition in CONDITIONS},
        "genuine": _class_summary([row for row in rows if row["label"] == 0], points),
        "synthetic": _class_summary([row for row in rows if row["label"] == 1], points),
    }


def compare(manifest: dict, confirmation: dict, lock: dict) -> dict:
    rows = _measurements(manifest, confirmation, "confirmation")
    lock_sha = digest(lock)
    if (lock.get("schema") != FREEZE_SCHEMA or lock.get("policy") != POLICY
            or lock.get("manifest_sha256") != digest(manifest)
            or confirmation.get("freeze_sha256") != lock_sha):
        raise ValueError("Confirmation must reference the exact frozen benchmark")
    if lock.get("identity") != _identity(confirmation):
        raise ValueError("Incumbent, router, supplement and calibration must match the frozen identities")
    points = lock.get("points")
    _validate_points(points, confirmation["score_kind"])
    return {
        "schema": REPORT_SCHEMA,
        "policy": POLICY,
        "manifest_sha256": lock["manifest_sha256"],
        "freeze_sha256": lock_sha,
        "confirmation_receipt_sha256": digest(confirmation),
        "active_routes": sorted(points),
        "overall": _summary(rows, points),
        "reference_conditions": {
            condition: _summary([row for row in rows if row["condition"] == condition], points)
            for condition in CONDITIONS
        },
    }


# Hand-authored scores exercise the replay without an encoder, corpus, or media.
_DEVELOPMENT = [
    ("cal-human", 0, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .6),
    ("cal-synthetic", 1, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .9),
    ("cal-overlap", 0, "overlapping_interferers", "overlapping_interferers", "inconclusive", .8),
]
_CONFIRMATION = [
    ("test-synthetic", 1, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .9),
    ("test-tie", 1, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .6),
    ("test-human", 0, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .8),
    ("test-human", 0, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", .7),
    ("test-misrouted", 0, "overlapping_interferers", "mostly_single_interferer", "inconclusive", .9),
    ("test-overlap", 1, "overlapping_interferers", "overlapping_interferers", "inconclusive", .9),
    ("test-original", 1, "mostly_single_interferer", "mostly_single_interferer", "original", .9),
    ("test-unassessed", 0, "overlapping_interferers", "mostly_single_interferer", "unassessed", .9),
    ("test-unknown", 1, "mostly_single_interferer", "unknown", "inconclusive", .9),
    ("test-no-score", 1, "mostly_single_interferer", "mostly_single_interferer", "inconclusive", None),
    ("test-clean", 0, "clean", "clean", "original", .9),
    ("test-prior-ai", 1, "clean", "clean", "ai", .2),
]


def _demo() -> dict:
    cases = []
    predictions = {}
    for split, rows in (("development", _DEVELOPMENT), ("confirmation", _CONFIRMATION)):
        predictions[split] = []
        for i, (group, label, reference, observed, incumbent, score) in enumerate(rows, 1):
            case_id = f"{split}-{i}"
            input_sha = digest({"authored_input": case_id})
            cases.append({
                "id": case_id, "split": split, "group": group, "label": label, "modality": "audio",
                "family": "authored-score-example", "condition": reference,
                "source_sha256": digest({"authored_source": group}), "input_sha256": input_sha,
                "license": {"id": "authored example", "evaluation_permitted": True, "redistribution_permitted": True},
            })
            prediction = {
                "id": case_id, "input_sha256": input_sha, "status": "ok" if score is not None else "not_applicable",
                "response": score, "incumbent_direction": incumbent, "observed_condition": observed,
            }
            if score is None:
                prediction["reason"] = "Authored missing response"
            predictions[split].append(prediction)
    manifest = {"schema": MANIFEST_SCHEMA, "cases": cases}
    identity = {
        "manifest_sha256": digest(manifest), "system": "Authored supplement example",
        "implementation_sha256": digest({"authored": "supplement"}), "method_kind": "model_free",
        "weight_sha256": [], "score_kind": "raw", "response_units": "authored response units",
        "incumbent_sha256": digest({"authored": "incumbent"}), "router_sha256": digest({"authored": "router"}),
    }
    development = {**identity, "split": "development", "predictions": predictions["development"]}
    points = {"mostly_single_interferer": .6}
    lock = freeze(manifest, development, points)
    confirmation = {**identity, "split": "confirmation", "predictions": predictions["confirmation"],
                    "freeze_sha256": digest(lock)}
    return {"manifest": manifest, "development": development, "confirmation": confirmation, "points": points,
            "frozen": lock, "summary": compare(manifest, confirmation, lock)}


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo")
    demo.add_argument("output", type=Path)
    lock = subcommands.add_parser("freeze")
    review = subcommands.add_parser("compare")
    for command, fields in ((lock, ("manifest", "development", "points", "output")),
                            (review, ("manifest", "confirmation", "frozen", "output"))):
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            args.output.mkdir()
            for name, value in _demo().items():
                _write(args.output / f"{name}.json", value)
        elif args.command == "freeze":
            frozen = freeze(_read(args.manifest), _read(args.development), _read(args.points))
            _write(args.output, frozen)
            print(digest(frozen))
        else:
            _write(args.output, compare(_read(args.manifest), _read(args.confirmation), _read(args.frozen)))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
