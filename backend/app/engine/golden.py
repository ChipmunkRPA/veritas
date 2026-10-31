"""Detector goldens (ARCHITECTURE §9.3) and a single-detector harness on the production loaders.

- **File.** `tests/golden/<detector_id>@<version>.<device_class>.json`: per fixture, the status, unit scores and aggregate
  (raw, 6 dp), exhibit shapes and model revisions. Timings are not stored.
- **Tolerances.** |Δraw| ≤ 1e-3 against the same-device golden; ≤ 1e-2 against the CPU golden on an accelerator.
- **Regeneration.** Explicit only: `make golden DETECTOR=<id>` (`scripts/v2/make_golden.py`) refuses to replace a golden of
  the same adapter version unless `--force` is given, and prints the differences first.
- **Harness.** The manifest adapter and `load.adapter_loader`, with every pinned file verified first (nothing downloads),
  on one preferred device; decoding follows `StoredMedia` (EXIF orientation, 8-bit RGB); exhibits go to a scratch store.
"""
from __future__ import annotations

import importlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.config import BACKEND_DIR

GOLDEN_DIR = BACKEND_DIR / "tests" / "golden"
FIXTURE_DIR = BACKEND_DIR / "tests" / "fixtures"
FIXTURES: dict[str, tuple[str, ...]] = {"img.commfor384": ("metadata/P1000827.jpg",), "img.dda": ("metadata/P1000827.jpg",),
                                        "img.spai": ("metadata/P1000827.jpg",), "loc.mesorch": ("metadata/P1000827.jpg",),
                                        "loc.radar": ("metadata/P1000827.jpg",),
                                        "face.gend_clip_l14": ("faces/kuriyama_2023.jpg", "metadata/P1000827.jpg")}
FACE_DETECTOR = "img.faces.yunet"
SAME_DEVICE_TOL = 1e-3
CROSS_DEVICE_TOL = 1e-2
SCHEMA = "veritas.golden/1"


def _import(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def golden_path(detector_id: str, version: str, device_class: str) -> Path:
    return GOLDEN_DIR / f"{detector_id}@{version}.{device_class}.json"


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as src:
        return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(src).convert("RGB")))


def _shape(exhibit: Any) -> dict[str, Any]:
    shape: dict[str, Any] = {"kind": exhibit.kind}
    for name in ("regions", "rows", "series", "segments"):
        if isinstance(getattr(exhibit, name, None), list):
            shape[f"n_{name}"] = len(getattr(exhibit, name))
    if getattr(exhibit, "grid", None) is not None:
        shape["grid"] = list(exhibit.grid)
    return shape


def _score(unit: Any) -> dict[str, Any]:
    return {"unit_ref": unit.unit_ref, "raw": round(float(unit.raw), 6), "raw_kind": unit.raw_kind}


class DetectorHarness:
    """One detector, one preferred device, models loaded once and reused across inputs."""

    def __init__(self, detector_id: str, device: str, work_dir: Path):
        from app.engine.exhibits import ExhibitWriter
        from app.engine.fetchers import ensure_verified
        from app.engine.models import FootprintStore, ManagedModel, MiB, ModelManager
        from app.engine.registry import build_registry
        from app.storage.objects import ObjectStore

        manifest = build_registry(policy="research").manifest
        entry = manifest.detector(detector_id)
        if entry is None or not entry.adapter:
            raise KeyError(f"{detector_id} has no adapter in the manifest")
        self.entry, self.detector = entry, _import(entry.adapter)()

        def loader_for(model_id: str) -> Any:
            model = manifest.model(model_id)
            assert model is not None and model.load.adapter_loader
            ensure_verified(model)
            return _import(model.load.adapter_loader)

        models = {m: ManagedModel.from_manifest(manifest.model(m)) for m in entry.models}
        self.manager = ModelManager(budget_bytes=8192 * MiB, models=models, loader_for=loader_for,
                                    store=FootprintStore(work_dir / "footprints.json"), default_device=device)
        self.writer = ExhibitWriter("golden", ObjectStore(work_dir / "objects"))
        self.device = device
        # detectors that need faces get them from YuNet on the CPU, as the faces stage does
        needs_faces = "faces" in self.detector.spec.requirements.needs
        self.faces = DetectorHarness(FACE_DETECTOR, "cpu", work_dir / "faces") if needs_faces else None

    def _execute(self, media: Any, outputs: Any) -> tuple[Any, list[Any]]:
        from app.engine.boundary import InMemoryRunStore, run_detector
        from app.engine.detector import Profile, RunContext

        before = len(self.writer.exhibits)
        ctx = RunContext(analysis_id="golden", media=media, profile=Profile.FULL, device=None, dtype=None, lease=None,
                         outputs=outputs, deadline=None, exhibits=self.writer.bind(self.entry.id), log=None)
        result = run_detector(self.detector, ctx, store=InMemoryRunStore(), models=self.manager)
        return result, self.writer.exhibits[before:]

    def run(self, data: bytes) -> dict[str, Any]:
        from app.detectors.fakes import InMemoryMedia
        from app.engine.detector import StageOutputs
        from app.storage.files import sha256_bytes

        try:
            pixels = _decode(data)
        except (OSError, SyntaxError):
            pixels = None
        media = InMemoryMedia(sha256=sha256_bytes(data), kind="image", data=data, pixels=pixels)
        outputs = StageOutputs()
        if self.faces is not None:
            from app.engine.stages.image import faces_from_exhibits

            _, face_exhibits = self.faces._execute(media, StageOutputs())
            outputs = StageOutputs(faces=faces_from_exhibits([e.model_dump(mode="json") for e in face_exhibits]))
        result, exhibits = self._execute(media, outputs)
        return {"status": result.status, "reason": result.reason, "device": result.device,
                "unit_scores": [_score(u) for u in result.unit_scores],
                "aggregate": None if result.aggregate is None else _score(result.aggregate),
                "exhibits": [_shape(e) for e in exhibits],
                "model_revisions": dict(sorted(result.model_revisions.items()))}


def generate(detector_id: str, device: str, work_dir: Path) -> dict[str, Any]:
    harness = DetectorHarness(detector_id, device, work_dir)
    fixtures = {name: harness.run((FIXTURE_DIR / name).read_bytes()) for name in FIXTURES[detector_id]}
    return {"schema": SCHEMA, "detector_id": detector_id, "version": harness.entry.version, "device_class": device,
            "fixtures": fixtures}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dumps(doc: dict[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _units(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return [*fixture["unit_scores"], *([fixture["aggregate"]] if fixture["aggregate"] else [])]


def max_raw_delta(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Largest |Δraw| over every unit score and aggregate; +inf when the structure differs."""
    worst = 0.0
    for name, exp in expected["fixtures"].items():
        act = actual["fixtures"].get(name)
        if act is None or (act["status"], act["exhibits"]) != (exp["status"], exp["exhibits"]):
            return math.inf
        exp_units, act_units = _units(exp), _units(act)
        if [(u["unit_ref"], u["raw_kind"]) for u in exp_units] != [(u["unit_ref"], u["raw_kind"]) for u in act_units]:
            return math.inf
        worst = max([worst, *(abs(a["raw"] - e["raw"]) for a, e in zip(act_units, exp_units, strict=True))])
    return worst if set(actual["fixtures"]) == set(expected["fixtures"]) else math.inf


def diff_lines(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    lines = []
    for name in sorted(set(expected["fixtures"]) | set(actual["fixtures"])):
        exp, act = expected["fixtures"].get(name), actual["fixtures"].get(name)
        if exp is None or act is None:
            lines.append(f"{name}: {'added' if exp is None else 'removed'}")
            continue
        for key in ("status", "device", "exhibits", "model_revisions"):
            if exp.get(key) != act.get(key):
                lines.append(f"{name} {key}: {exp.get(key)!r} → {act.get(key)!r}")
        exp_units = {u["unit_ref"]: u["raw"] for u in _units(exp)}
        for unit in _units(act):
            before = exp_units.pop(unit["unit_ref"], None)
            if before is None:
                lines.append(f"{name} {unit['unit_ref']}: added {unit['raw']:+.6f}")
            elif before != unit["raw"]:
                lines.append(f"{name} {unit['unit_ref']}: {before:+.6f} → {unit['raw']:+.6f} (Δ {unit['raw'] - before:+.2e})")
        lines.extend(f"{name} {ref}: removed" for ref in exp_units)
    return lines
