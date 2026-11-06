"""Portable evaluation receipts: split integrity, complete denominators and paired results.

This module does not load detectors, infer authorship, or establish dataset rights.
It validates declared metadata and summarizes submitted responses. A valid receipt
is an engineering record, not independent review or evidence of general accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import groupby
from statistics import mean

SCHEMA = "chipmunk.independent-evaluation/1"
MEDIA = {"image", "video", "audio", "text"}
SPLITS = {"development", "confirmation"}
STATUSES = {"ok", "inconclusive", "not_applicable", "failed"}


def digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def audit_manifest(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("Expected a nonempty independent-evaluation manifest")
    ids, ancestry = set(), {}
    strata = Counter()
    for case in manifest["cases"]:
        ident = case.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError("Case IDs must be nonempty and unique")
        ids.add(ident)
        if case.get("modality") not in MEDIA or case.get("split") not in SPLITS:
            raise ValueError("Unsupported modality or split")
        if type(case.get("label")) is not int or case["label"] not in (0, 1):
            raise ValueError("Ground-truth labels must be integer 0/1")
        for key in ("group", "family", "condition"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Missing declared {key}")
        if not _hash(case.get("input_sha256")) or not _hash(case.get("source_sha256")):
            raise ValueError("Input and parent-source SHA-256 are required")
        terms = case.get("license", {})
        if not isinstance(terms, dict) or not terms.get("id") or terms.get("evaluation_permitted") is not True:
            raise ValueError("Each case needs declared evaluation permission and license identity")
        if type(terms.get("redistribution_permitted")) is not bool:
            raise ValueError("Redistribution permission must be explicit")
        lineage = case.get("lineage", [])
        if not isinstance(lineage, list) or any(not isinstance(x, str) or not x.strip() for x in lineage):
            raise ValueError("Lineage must contain known nonempty source/speaker/topic identifiers")
        # A group must remain in one split across modalities. Hashes and explicit
        # lineage catch mislabeled groups and transformed versions of one source.
        keys = [("group", case["group"]), ("hash", case["input_sha256"]),
                ("hash", case["source_sha256"]), *(("lineage", x) for x in lineage)]
        for key in keys:
            prior = ancestry.setdefault(key, case["split"])
            if prior != case["split"]:
                raise ValueError("Development/confirmation source or lineage leakage")
        strata[(case["modality"], case["split"], case["condition"], case["label"])] += 1
    return {"manifest_sha256": digest(manifest), "cases": len(ids),
            "groups": len({c["group"] for c in manifest["cases"]}),
            "strata": [{"modality": m, "split": s, "condition": c, "label": y, "count": n}
                       for (m, s, c, y), n in sorted(strata.items())]}


def audit_receipt(manifest: dict, receipt: dict, split: str) -> list[dict]:
    audit = audit_manifest(manifest)
    if split not in SPLITS or receipt.get("manifest_sha256") != audit["manifest_sha256"]:
        raise ValueError("Receipt does not match the declared manifest/split")
    if receipt.get("split") != split or receipt.get("score_kind") not in {"raw", "calibrated"}:
        raise ValueError("Receipt needs its exact split and response interpretation")
    if not receipt.get("system") or not _hash(receipt.get("implementation_sha256")):
        raise ValueError("System identity and implementation digest are required")
    if not isinstance(receipt.get("response_units"), str) or not receipt["response_units"].strip():
        raise ValueError("Response units must distinguish raw measurements from calibrated estimates")
    weights = receipt.get("weight_sha256")
    if not isinstance(weights, list) or not weights or any(not _hash(x) for x in weights):
        raise ValueError("Pinned model weight digests are required")
    if receipt["score_kind"] == "calibrated" and not _hash(receipt.get("calibration_sha256")):
        raise ValueError("Calibrated responses need their own calibration digest")
    expected = {c["id"]: c for c in manifest["cases"] if c["split"] == split}
    rows = receipt.get("predictions")
    if not expected or not isinstance(rows, list):
        raise ValueError("Missing planned predictions")
    if len(rows) != len(expected) or {r.get("id") for r in rows} != expected.keys():
        raise ValueError("Every planned case must appear exactly once, including failures")
    result = []
    for row in rows:
        case = expected[row["id"]]
        if row.get("input_sha256") != case["input_sha256"] or row.get("status") not in STATUSES:
            raise ValueError("Changed input bytes or unsupported result status")
        response = row.get("response")
        valid = (type(response) in (int, float) and math.isfinite(response)
                 and (receipt["score_kind"] == "raw" or 0 <= response <= 1))
        if row["status"] == "ok" and not valid:
            raise ValueError("Scored responses must be finite; calibrated estimates must lie in [0, 1]")
        if row["status"] != "ok" and response is not None:
            raise ValueError("Unassessed and failed cases cannot carry authenticity responses")
        if row["status"] != "ok" and not row.get("reason"):
            raise ValueError("Unassessed and failed cases need reasons")
        result.append({**case, "status": row["status"], "response": response})
    return result


def freeze(manifest: dict, development: dict, *, thresholds: dict[str, float]) -> dict:
    """Fix internal benchmark flag points before reading confirmation responses.

    These flag points are for matched scientific comparisons, not public product
    confidence claims. This function does not fit or select thresholds.
    """
    rows = audit_receipt(manifest, development, "development")
    modalities = {c["modality"] for c in manifest["cases"]}
    if set(thresholds) != modalities or any(type(x) not in (int, float) or not math.isfinite(x)
                                           or (development["score_kind"] == "calibrated" and not 0 <= x <= 1)
                                           for x in thresholds.values()):
        raise ValueError("Every planned modality needs a finite frozen operating point")
    return deepcopy({"schema": SCHEMA, "manifest_sha256": digest(manifest),
            "development_receipt_sha256": digest(development),
            "development_cases": len(rows), "thresholds": thresholds,
            **{k: development.get(k) for k in ("system", "implementation_sha256", "weight_sha256",
                                               "score_kind", "calibration_sha256", "response_units")}})


def _auc(rows: list[dict]) -> float | None:
    positives = sum(r["label"] for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    wins, before = 0., 0
    for _, tied in groupby(sorted(rows, key=lambda r: r["response"]), key=lambda r: r["response"]):
        values = list(tied)
        p = sum(r["label"] for r in values)
        n = len(values) - p
        wins += p * (before + .5 * n)
        before += n
    return wins / (positives * negatives)


def summarize(manifest: dict, confirmation: dict, lock: dict) -> dict:
    rows = audit_receipt(manifest, confirmation, "confirmation")
    if lock.get("manifest_sha256") != digest(manifest) or lock.get("schema") != SCHEMA:
        raise ValueError("Frozen plan differs from the source manifest")
    for key in ("system", "implementation_sha256", "weight_sha256", "score_kind", "calibration_sha256", "response_units"):
        if confirmation.get(key) != lock.get(key):
            raise ValueError(f"Confirmation changed frozen system field: {key}")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["modality"], row["condition"])].append(row)
    summaries = []
    for (modality, condition), cases in sorted(groups.items()):
        threshold = lock.get("thresholds", {}).get(modality)
        if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                or (confirmation["score_kind"] == "calibrated" and not 0 <= threshold <= 1)):
            raise ValueError("Invalid frozen benchmark operating point")
        scored = [r for r in cases if r["status"] == "ok"]
        real = [r for r in cases if not r["label"]]
        edited = [r for r in cases if r["label"]]
        def flags(row, point=threshold):
            return row["status"] == "ok" and row["response"] >= point
        statuses = dict(Counter(r["status"] for r in cases))
        # Missing responses count as AI not alerted, but never as original verdicts.
        summaries.append({"modality": modality, "condition": condition, "planned": len(cases),
                          "source_groups": len({r["group"] for r in cases}), "scored": len(scored),
                          "status_counts": statuses, "originals": len(real), "ai_inputs": len(edited),
                          "false_alerts": sum(flags(r) for r in real),
                          "ai_alerts": sum(flags(r) for r in edited),
                          "ai_without_alert": sum(not flags(r) for r in edited),
                          "inconclusive": statuses.get("inconclusive", 0),
                          "unassessed": sum(r["status"] != "ok" for r in cases),
                          "assessable_auc": _auc(scored),
                          "assessable_brier": (mean((r["response"] - r["label"]) ** 2 for r in scored)
                                                if scored and confirmation["score_kind"] == "calibrated" else None)})
    return {"schema": SCHEMA, "manifest_sha256": digest(manifest), "lock_sha256": digest(lock),
            "receipt_sha256": digest(confirmation), "system": confirmation["system"], "strata": summaries,
            "limits": "Aggregate declared-source results; AUC/Brier cover scored cases only. No proof of authorship, "
                      "external review, unseen training sources, dataset rights, or production accuracy. "
                      "Variants are grouped by source and must not be treated as independent samples."}


def compare(manifest: dict, baseline: dict, baseline_lock: dict, candidate: dict,
            candidate_lock: dict, *, iterations: int = 2000, seed: int = 20261105) -> dict:
    """Matched all-source alert differences, with source-group bootstrap intervals."""
    if type(iterations) is not int or iterations < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    old = summarize(manifest, baseline, baseline_lock)
    new = summarize(manifest, candidate, candidate_lock)
    prior = {r["id"]: r for r in audit_receipt(manifest, baseline, "confirmation")}
    latter = audit_receipt(manifest, candidate, "confirmation")
    strata = defaultdict(list)
    for row in latter:
        strata[(row["modality"], row["condition"], row["label"])].append(row)
    randomizer = random.Random(seed)
    differences = []
    for (modality, condition, label), rows in sorted(strata.items()):
        grouped = defaultdict(list)
        for row in rows:
            before = prior[row["id"]]
            old_flag = before["status"] == "ok" and before["response"] >= baseline_lock["thresholds"][modality]
            new_flag = row["status"] == "ok" and row["response"] >= candidate_lock["thresholds"][modality]
            grouped[row["group"]].append(int(new_flag) - int(old_flag))
        # Equal source-group weight; repeated variants do not create extra groups.
        values = [mean(group) for group in grouped.values()]
        interval = None
        if len(values) >= 2:
            draws = sorted(mean(randomizer.choices(values, k=len(values))) for _ in range(iterations))
            interval = [draws[math.floor(.025 * (iterations-1))], draws[math.ceil(.975 * (iterations-1))]]
        differences.append({"modality": modality, "condition": condition,
                            "metric": "ai_alert_rate" if label else "false_alert_rate",
                            "planned": len(rows), "source_groups": len(values),
                            "candidate_minus_baseline": mean(values), "group_bootstrap_95_percent": interval})
    return {"baseline": old, "candidate": new, "paired_differences": differences,
            "bootstrap": {"iterations": iterations, "seed": seed, "unit": "declared source group",
                          "weight": "equal source groups within modality, condition, and true class"},
            "interpretation": "AI without an alert includes unassessed cases; it does not mean an original verdict. "
                              "Intervals describe this source panel and do not resolve unknown training overlap or domain shift."}
