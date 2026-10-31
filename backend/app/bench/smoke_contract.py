"""Pinned serving architecture for a reproducibility smoke, never a new admission.

Only metadata, cached-file presence and the small fitted fuser are read. Detector
weights are checked by the existing registry loader without loading any model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from app.engine.detector import Profile
from app.engine.fusion import derived
from app.engine.fusion import registry as fusion_registry
from app.engine.fusion.artefact import SCHEMA, Fuser
from app.engine.fusion.bootstrap import bootstrap_columns
from app.engine.fusion.features import METADATA_TERMS, SIZE_FACE_TERMS, FeatureSpace, gate_literals
from app.engine.fusion.policy import default_policy

_HASH = re.compile(r"[0-9a-f]{64}")
_DETECTOR = re.compile(r"[a-z0-9][a-z0-9._-]*")
_WEIGHT = re.compile(r"[0-9a-f]{12}|noweights")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _nonfinite(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _read(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value, hashlib.sha256(data).hexdigest()


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value) or len(set(value)) != len(value):
        raise ValueError(f"{name} must be a list of distinct nonempty strings")
    return tuple(value)


def _gates(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError("gates must be an object")
    out = {}
    for key, raw in sorted(value.items()):
        literals = _strings(raw, f"gate {key}")
        gate_literals(literals)
        if any(term.lstrip("!") in METADATA_TERMS for term in literals):
            raise ValueError("image smoke refuses metadata gates")
        out[key] = list(literals)
    return out


def _layout(doc: Mapping) -> dict:
    if doc.get("schema") != SCHEMA or doc.get("modality") != "image":
        raise ValueError("smoke requires a veritas.fuser/1 image artefact")
    if not isinstance(doc.get("reviewed_by"), str) or not doc["reviewed_by"].strip():
        raise ValueError("serving artefact must have reviewed_by set")
    bench = doc.get("bench")
    if not isinstance(bench, dict) or not isinstance(bench.get("version"), str) or not bench["version"]:
        raise ValueError("artefact needs benchmark provenance")
    if bench.get("context_source") != "scored":
        raise ValueError("unsupported context: image smoke requires scored context")
    detectors = doc.get("detectors")
    if not isinstance(detectors, list) or not detectors or any(not isinstance(d, dict) for d in detectors):
        raise ValueError("artefact needs explicit detector records")
    members = []
    for det in detectors:
        detector_id, pin = det.get("id"), det.get("weights_sha12")
        if not isinstance(detector_id, str) or not _DETECTOR.fullmatch(detector_id):
            raise ValueError("invalid detector id")
        if not isinstance(pin, str) or not _WEIGHT.fullmatch(pin) or det.get("score_key") != f"{detector_id}@{pin}":
            raise ValueError(f"detector {detector_id}: score key and weights pin differ")
        transform = det.get("transform")
        if transform != derived.input_transform(detector_id, pin):
            raise ValueError(f"detector {detector_id}: unsupported cache transform")
        if det.get("missing") not in (None, "size", "face"):
            raise ValueError(f"detector {detector_id}: unsupported missingness contract")
        members.append({"id": detector_id, "score_key": det["score_key"], "weights_sha12": pin,
                        "transform": transform, "missing": det.get("missing")})
    ids = tuple(d["id"] for d in members)
    gates = _gates(doc.get("gates"))
    context_terms = _strings(doc.get("context_terms"), "context_terms")
    if set(context_terms) & (set(SIZE_FACE_TERMS) | set(METADATA_TERMS)):
        raise ValueError("image policy excludes additive size, face and metadata context terms")
    groups = _strings(doc.get("groups"), "groups")
    space = FeatureSpace(ids, context_terms, groups, gates=tuple((d, tuple(v)) for d, v in gates.items()))
    features = _strings(doc.get("features"), "features")
    if tuple(space.names[1:]) != features:
        raise ValueError("artefact features differ from detector/context layout")
    size = doc.get("size_policy")
    expected = default_policy("image").as_json()
    if not isinstance(size, dict) or any(type(size.get(k)) is not bool or size[k] != v for k, v in expected.items()):
        raise ValueError("unsupported image size policy")
    fixed = _strings(size.get("fixed_missing"), "fixed_missing")
    if fixed != tuple(d["id"] for d in members if d["missing"] == "size"):
        raise ValueError("size-driven missingness must retain its fixed missing features")
    if any(d["missing"] == "face" and "has_face" not in gates.get(d["id"], ()) for d in members):
        raise ValueError("face-driven missingness requires a has_face gate")
    coef = doc.get("coef")
    if not isinstance(coef, dict) or set(coef) != set(space.names):
        raise ValueError("coefficient names differ from the feature contract")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in coef.values()):
        raise ValueError("coefficients must be finite numbers")
    if any(coef[f"miss:{d}"] != 0 for d in fixed):
        raise ValueError("fixed missing features must have zero coefficient")
    boot = doc.get("bootstrap")
    if not isinstance(boot, dict):
        raise ValueError("artefact requires bootstrap metadata")
    if not isinstance(boot.get("file"), str) or Path(boot["file"]).name != boot["file"] or not _HASH.fullmatch(str(boot.get("sha256"))):
        raise ValueError("artefact requires a pinned bootstrap matrix beside it")
    if doc.get("skipped_detectors"):
        raise ValueError("smoke refuses an artefact that skipped required detectors")
    return {"detectors": members, "gates": gates, "context_source": "scored", "context_terms": list(context_terms),
            "groups": list(groups), "features": list(features), "policy": expected, "fixed_missing": list(fixed)}


def _admission(doc: dict, source: dict, layout: dict, manifest_sha: str) -> None:
    if doc.get("schema") != "veritas.admission-records/1":
        raise ValueError("unsupported admission record schema")
    if doc.get("manifest_sha256") != manifest_sha or doc.get("bench") != source["bench"].get("version"):
        raise ValueError("admission source manifest or benchmark version differs")
    elimination = doc.get("elimination")
    if not isinstance(elimination, dict):
        raise ValueError("admission requires final ensemble metadata")
    joint = elimination.get("joint_check")
    if elimination.get("blocked") != [] or not isinstance(joint, dict) or joint.get("passed") is not True:
        raise ValueError("admission final ensemble has no passing joint check or is blocked")
    end_set = elimination.get("end_set")
    if not isinstance(end_set, list) or any(not isinstance(row, dict) for row in end_set):
        raise ValueError("admission has no final end_set")
    ids = [row.get("detector_id") for row in end_set]
    expected_ids = [d["id"] for d in layout["detectors"]]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("admission end_set differs from serving detector members")
    admitted_gates = {}
    for row in end_set:
        literals = _strings(row.get("gate"), "admission end_set gate")
        if literals:
            admitted_gates[row["detector_id"]] = list(literals)
    if _gates(admitted_gates) != layout["gates"]:
        raise ValueError("admission gates differ from serving gates")
    records = doc.get("records")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("admission has no detector records")
    names = [r.get("detector_id") for r in records]
    if any(not isinstance(i, str) for i in names) or len(set(names)) != len(names):
        raise ValueError("admission detector records are not unique")
    by_id = {r["detector_id"]: r for r in records}
    for detector_id in expected_ids:
        row = by_id.get(detector_id) or {}
        allowed = "admitted_context_gated" if detector_id in admitted_gates else "admitted"
        if row.get("decision") != allowed or row.get("lane"):
            raise ValueError(f"detector {detector_id}: admission does not approve this union-fuser membership")
        if any(not isinstance(row.get(k), dict) or row[k].get("pass") is not True for k in ("A1", "A2", "A6")):
            raise ValueError(f"detector {detector_id}: prerequisite admission evidence is not passed")


@dataclass(frozen=True, slots=True)
class SmokeContract:
    keys: tuple[str, ...]
    gates: Mapping[str, tuple[str, ...]]
    context_source: str
    context_terms: tuple[str, ...]
    policy: Mapping[str, bool]
    source_manifest_sha256: str
    scope_sha256: str
    serving_artefact_sha256: str
    admission_sha256: str
    profile: str
    _record_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "gates", MappingProxyType({k: tuple(v) for k, v in self.gates.items()}))
        object.__setattr__(self, "context_terms", tuple(self.context_terms))
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))

    def as_record(self) -> dict:
        """Detached serializable metadata; mutating it cannot change this contract."""
        return json.loads(self._record_json)

    def require_cache(self, scores_dir: Path) -> tuple[Path, ...]:
        """Require exact caches and a context marker; row exclusions remain calibrate's job."""
        scores_dir = Path(scores_dir)
        paths = tuple(scores_dir / f"{key}.parquet" for key in self.keys)
        absent = [p.name for p in paths if not p.is_file() or p.stat().st_size == 0]
        if absent:
            raise ValueError(f"required serving detector caches missing: {', '.join(absent)}")
        markers = [p for p in scores_dir.glob("v2.image_fused@*.parquet") if p.is_file() and p.stat().st_size > 0]
        if not markers:
            raise ValueError("scored context requires an image fused marker cache")
        return paths

    def validate_output(self, path: Path) -> dict:
        """Check the refit's architecture, not equality of fitted coefficients or quality."""
        path = Path(path)
        doc, sha = _read(path)
        layout = _layout(doc)
        if layout != self.as_record()["layout"]:
            raise ValueError("fitted smoke artefact changed the serving feature contract")
        bench = doc.get("bench") or {}
        if bench.get("scope_sha256") != self.scope_sha256 or not _HASH.fullmatch(str(bench.get("manifest_sha256"))):
            raise ValueError("fitted smoke artefact has mismatched scope or missing subset manifest pin")
        boot = doc["bootstrap"]
        if (type(boot.get("B")) is not int or boot["B"] <= 0 or type(boot.get("seed")) is not int
                or boot["seed"] != 7 or boot.get("grouped_by") != "group_key"):
            raise ValueError("fitted bootstrap requires positive B, seed 7 and group_key grouping")
        fuser = Fuser.from_bytes(path.read_bytes(), path.parent)  # actual bootstrap hash and shape, no inference
        if boot.get("columns") != list(bootstrap_columns(fuser.space)):
            raise ValueError("fitted bootstrap columns differ from the feature contract")
        if fuser.boot is None or fuser.boot.dtype != np.dtype("float32") or not np.isfinite(fuser.boot).all():
            raise ValueError("fitted bootstrap must be a finite float32 matrix")
        if _sha(path) != sha:
            raise ValueError("fitted smoke artefact changed during validation")
        return {"sha256": sha, "contract_valid": True, "serving_artefact_sha256": self.serving_artefact_sha256}


def load_contract(artefact_path: Path, manifest_path: Path, scope_path: Path, admission_path: Path, *,
                  registry=None, profile: Profile | str | None = None) -> SmokeContract:
    """Resolve the explicit artifact through the serving loader, with all metadata pins checked."""
    paths = tuple(Path(p).resolve(strict=True) for p in (artefact_path, manifest_path, scope_path, admission_path))
    artefact_path, manifest_path, scope_path, admission_path = paths
    source, source_sha = _read(artefact_path)
    admission, admission_sha = _read(admission_path)
    manifest_sha, scope_sha = _sha(manifest_path), _sha(scope_path)
    layout = _layout(source)
    bench = source.get("bench") or {}
    if bench.get("manifest_sha256") != manifest_sha or bench.get("scope_sha256") != scope_sha:
        raise ValueError("serving artefact source manifest/scope provenance differs from supplied bytes")
    _admission(admission, source, layout, manifest_sha)
    if registry is None:
        from app.engine.registry import get_registry

        registry = get_registry()
    if profile is None:
        from app.engine.pipelines import default_profile

        profile = default_profile()
    profile = Profile(profile).value
    selection = fusion_registry.select_for_registry(registry, "image", profile, pin=source_sha)
    if (selection.mode != "artefact" or selection.fuser is None or selection.sha256 != source_sha
            or selection.path is None or selection.path.resolve() != artefact_path):
        raise ValueError(f"serving artefact is not compatible with the enabled installed registry: {selection.reason}")
    pins = (source_sha, manifest_sha, scope_sha, admission_sha)
    if any(_sha(path) != pin for path, pin in zip(paths, pins, strict=True)):
        raise ValueError("smoke contract inputs changed during validation")
    record = {
        "schema": "veritas.vbench-smoke-contract/1", "source_manifest_sha256": manifest_sha, "scope_sha256": scope_sha,
        "serving_artefact_sha256": source_sha, "admission_sha256": admission_sha,
        "serving_version": source.get("version"), "reviewed_by": source["reviewed_by"], "profile": profile, "layout": layout,
        "removed_context_terms": (source.get("size_policy") or {}).get("removed_context_terms", {}),
        "admission_binding": "supplied admission record: manifest, bench, final members/gates and recorded pass; no direct artefact or scope hash",
        "purpose": "reused fit/cal reproducibility smoke; no independent validation or new admission",
    }
    return SmokeContract(tuple(d["score_key"] for d in layout["detectors"]),
                         MappingProxyType({k: tuple(v) for k, v in layout["gates"].items()}), "scored", tuple(layout["context_terms"]),
                         MappingProxyType(dict(layout["policy"])), manifest_sha, scope_sha, source_sha, admission_sha, profile,
                         json.dumps(record, sort_keys=True, allow_nan=False))
