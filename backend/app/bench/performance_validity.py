"""Fail-closed acceptance of observed performance; no model or hardware imports.

A contract must be declared before inference. It names every detector expected for
one profile/workload, permitted statuses, and actual devices for required learned
runs. Observations without that contract remain useful diagnostics, never accepted
latency evidence. This checks execution coverage, not detection accuracy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

CONTRACT_SCHEMA = "veritas.perf-coverage/1"
RUN_SCHEMA = "veritas.perf-run-validity/1"
VALIDITY_SCHEMA = "veritas.perf-validity/1"
LATENCY_SCHEMA = "veritas.perf-latency/2"
_PROFILES = {"quick", "full", "deep"}
_DEVICE = re.compile(r"(?:cpu|mps|cuda(?::(?:0|[1-9][0-9]*))?)\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _finite_nonnegative(value: Any) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def parse_contract(doc: Any) -> dict[str, Any]:
    """Validate and copy the predeclared JSON contract; malformed policy is refused.

    Shape: {schema, profiles: {profile: {workload: {
      expected: {detector_id: [ok | not_applicable, ...]},
      required_learned: {detector_id: [cpu | mps | cuda[:index], ...]}
    }}}}. Required learned detectors must permit only ``ok``. The explicit
    designation is an assertion by the protocol author, not inferred from output.
    """
    if not isinstance(doc, Mapping) or set(doc) != {"schema", "profiles"} or doc.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("performance coverage contract has an unsupported schema or fields")
    profiles = doc["profiles"]
    if not isinstance(profiles, Mapping) or not profiles or set(profiles) - _PROFILES:
        raise ValueError("performance coverage contract requires known profiles")
    result: dict[str, Any] = {"schema": CONTRACT_SCHEMA, "profiles": {}}
    for profile, workloads in profiles.items():
        if not isinstance(workloads, Mapping) or not workloads:
            raise ValueError("performance coverage profile requires workloads")
        normal_workloads = result["profiles"][profile] = {}
        for workload, entry in workloads.items():
            if not _name(workload) or not isinstance(entry, Mapping) or set(entry) != {"expected", "required_learned"}:
                raise ValueError("performance coverage workload has invalid fields")
            expected, learned = entry["expected"], entry["required_learned"]
            if not isinstance(expected, Mapping) or not expected or not all(_name(k) for k in expected):
                raise ValueError("performance coverage requires explicit detector IDs")
            for statuses in expected.values():
                if (not isinstance(statuses, list) or not statuses or not all(isinstance(s, str) for s in statuses)
                        or len(set(statuses)) != len(statuses) or set(statuses) - {"ok", "not_applicable"}):
                    raise ValueError("performance coverage allows only unique ok/not_applicable statuses")
            if not isinstance(learned, Mapping) or not learned or set(learned) - set(expected):
                raise ValueError("performance coverage requires explicit learned detector IDs")
            for detector, devices in learned.items():
                if expected[detector] != ["ok"]:
                    raise ValueError("required learned detectors must require ok")
                if (not isinstance(devices, list) or not devices
                        or not all(isinstance(d, str) and _DEVICE.fullmatch(d) for d in devices)
                        or len(set(devices)) != len(devices)):
                    raise ValueError("required learned detectors need explicit valid devices")
            normal_workloads[workload] = {
                "expected": {k: sorted(v) for k, v in sorted(expected.items())},
                "required_learned": {k: sorted(v) for k, v in sorted(learned.items())},
            }
    return result


def _digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_run(contract: Mapping[str, Any] | None, *, profile: str, workload: str,
                 analysis: Mapping[str, Any], detector_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one terminal observation, including a warmup; retain failure reasons.

    ``analysis`` is the v2 detail with authoritative terminal ``status`` and ``id``.
    ``detector_runs`` contains every page, not only successful detector rows.
    This never retrieves raw detector outputs or changes recorded observations.
    """
    normalized = parse_contract(contract) if contract is not None else None
    digest = _digest(normalized) if normalized is not None else None
    entry = (normalized["profiles"].get(profile) or {}).get(workload) if normalized else None
    errors = []
    detail = analysis if isinstance(analysis, Mapping) else {}
    aid = detail.get("id")
    if not _name(aid):
        errors.append("analysis_id_missing")
    if detail.get("status") != "done":
        errors.append("analysis_not_done")
    if profile not in _PROFILES or detail.get("profile") != profile:
        errors.append("requested_profile_mismatch")
    if detail.get("effective_profile") != profile:
        errors.append("effective_profile_mismatch")
    if entry is None:
        errors.append("coverage_contract_missing")
    rows = list(detector_runs) if isinstance(detector_runs, (list, tuple)) else []
    if not isinstance(detector_runs, (list, tuple)) or any(not isinstance(r, Mapping) for r in rows):
        errors.append("detector_rows_invalid")
        rows = [r for r in rows if isinstance(r, Mapping)]
    ids = [r.get("detector_id") for r in rows]
    if not all(_name(i) for i in ids):
        errors.append("detector_id_missing")
    good_ids = [i for i in ids if _name(i)]
    if len(good_ids) != len(set(good_ids)):
        errors.append("detector_id_duplicate")
    statuses = [r.get("status") for r in rows]
    observed_statuses = Counter(s for s in statuses if isinstance(s, str))
    coverage = detail.get("coverage")
    if (not isinstance(coverage, Mapping) or type(coverage.get("detectors_total")) is not int
            or coverage.get("detectors_total") != len(rows) or not isinstance(coverage.get("by_status"), Mapping)
            or any(type(v) is not int or v <= 0 for v in coverage["by_status"].values())
            or dict(coverage["by_status"]) != dict(observed_statuses) or sum(observed_statuses.values()) != len(rows)):
        errors.append("coverage_counts_mismatch")
    devices: dict[str, Any] = {}
    if entry is not None:
        expected, learned = entry["expected"], entry["required_learned"]
        if set(good_ids) != set(expected):
            errors.append("detector_ids_mismatch")
        for row in rows:
            detector = row.get("detector_id")
            if not isinstance(detector, str) or detector not in expected:
                continue
            if row.get("status") not in expected[detector]:
                errors.append("detector_status_disallowed")
            if detector in learned:
                device = row.get("device")
                devices[detector] = device if isinstance(device, str) else None
                if device not in learned[detector]:
                    errors.append("required_learned_device_mismatch")
    return {"schema": RUN_SCHEMA, "status": "invalid" if errors else "valid", "profile": profile,
            "workload": workload, "contract_sha256": digest, "analysis_id": aid if _name(aid) else None,
            "expected_detectors": len(entry["expected"]) if entry else None, "observed_detectors": len(rows),
            "required_learned_devices": devices, "errors": sorted(set(errors))}


def _valid_receipt(run: Any) -> bool:
    return (isinstance(run, Mapping) and run.get("schema") == RUN_SCHEMA and run.get("status") == "valid"
            and run.get("errors") == [] and _name(run.get("analysis_id")) and _name(run.get("workload"))
            and isinstance(run.get("profile"), str) and run["profile"] in _PROFILES and isinstance(run.get("contract_sha256"), str)
            and _SHA.fullmatch(run["contract_sha256"]) is not None
            and _positive_int(run.get("expected_detectors")) and type(run.get("observed_detectors")) is int
            and run["expected_detectors"] == run["observed_detectors"]
            and isinstance(run.get("required_learned_devices"), Mapping) and bool(run["required_learned_devices"])
            and all(_name(k) and isinstance(v, str) and _DEVICE.fullmatch(v)
                    for k, v in run["required_learned_devices"].items()))


def summarise_validity(*, profile: str, workload: str, requested_runs: int,
                       runs: Sequence[Mapping[str, Any]], warmups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind every timed run and warmup receipt to one predeclared contract/profile.

    The caller supplies the warmup(s) for this workload's kind. No warmup, a failed
    warmup, reused analysis IDs, or a missing requested run invalidates acceptance.
    """
    errors = []
    observed, warm = list(runs), list(warmups)
    all_runs = observed + warm
    valid = sum(_valid_receipt(r) for r in observed)
    warm_valid = sum(_valid_receipt(r) for r in warm)
    if profile not in _PROFILES or not _name(workload):
        errors.append("workload_binding_invalid")
    if not _positive_int(requested_runs) or len(observed) != requested_runs:
        errors.append("requested_runs_mismatch")
    if valid != len(observed):
        errors.append("invalid_timed_runs")
    if not warm or warm_valid != len(warm):
        errors.append("warmup_not_validated")
    if any(not isinstance(r, Mapping) or r.get("profile") != profile for r in all_runs):
        errors.append("profile_binding_mismatch")
    if any(not isinstance(r, Mapping) or r.get("workload") != workload for r in observed):
        errors.append("workload_binding_mismatch")
    kind = {"image": "image", "video": "video", "audio": "audio", "text": "document", "document": "document"}.get(
        workload.split("_", 1)[0] if isinstance(workload, str) else "")
    expected_warmup = f"warmup:{kind}" if kind else None
    if expected_warmup is None or any(not isinstance(r, Mapping) or r.get("workload") != expected_warmup for r in warm):
        errors.append("warmup_binding_mismatch")
    hashes = {r.get("contract_sha256") for r in all_runs if isinstance(r, Mapping) and isinstance(r.get("contract_sha256"), str)}
    digest = next(iter(hashes)) if len(hashes) == 1 else None
    if digest is None or not _SHA.fullmatch(digest) or any(
            not isinstance(r, Mapping) or r.get("contract_sha256") != digest for r in all_runs):
        errors.append("contract_binding_mismatch")
    ids = [r.get("analysis_id") for r in all_runs if isinstance(r, Mapping) and _name(r.get("analysis_id"))]
    if len(ids) != len(all_runs) or len(ids) != len(set(ids)):
        errors.append("analysis_identity_mismatch")
    return {"schema": VALIDITY_SCHEMA, "status": "invalid" if errors else "valid", "profile": profile,
            "workload": workload, "contract_sha256": digest, "requested_runs": requested_runs,
            "observed_runs": len(observed), "valid_runs": valid, "failed_runs": len(observed) - valid,
            "warmup_runs": len(warm), "warmup_valid_runs": warm_valid, "errors": errors,
            "runs": copy.deepcopy(observed), "warmups": copy.deepcopy(warm)}


def latency_rejection(record: Any, *, expected_profile: str = "full") -> str | None:
    """Return a reason to refuse acceptance; legacy numeric-only records are unverified.

    Recompute validity tallies from retained per-run receipts so a stripped or
    internally inconsistent projection cannot silently pass G13.
    """
    if not isinstance(record, Mapping) or record.get("schema") != LATENCY_SCHEMA:
        return "latency evidence lacks the verified workload schema"
    validity = record.get("validity")
    if not isinstance(validity, Mapping) or validity.get("schema") != VALIDITY_SCHEMA:
        return "latency evidence lacks execution validity"
    if (record.get("profile") != expected_profile or validity.get("profile") != expected_profile
            or not _name(record.get("workload")) or validity.get("workload") != record.get("workload")):
        return "latency evidence profile/workload binding differs"
    if not isinstance(validity.get("runs"), list) or not isinstance(validity.get("warmups"), list):
        return "latency evidence lacks per-run validity"
    recomputed = summarise_validity(profile=expected_profile, workload=record["workload"],
                                   requested_runs=validity.get("requested_runs"), runs=validity["runs"], warmups=validity["warmups"])
    if recomputed != dict(validity) or recomputed["status"] != "valid":
        return "latency evidence is incomplete or inconsistent"
    if type(record.get("n")) is not int or record["n"] != validity["requested_runs"] or record.get("failed"):
        return "latency evidence omits requested or failed runs"
    if not _finite_nonnegative(record.get("p95")):
        return "latency p95 is missing, nonfinite or negative"
    return None
