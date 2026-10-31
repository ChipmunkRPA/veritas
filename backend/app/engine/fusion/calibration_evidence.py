"""Pinned historical reliability tables; unavailable evidence never affects inference.

The published fuser digests are linked to evaluated digests by the retained
metrics-publication record. This does not reconstruct the earlier fuser bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.schemas.evidence import ReliabilityDiagram

BUNDLE = Path(__file__).with_name("data") / "reliability-vb-1.1.json"
BUNDLE_SHA256 = "529d5665d76c017ad355bc25f98582f91661d43aec687a2726c33f80992e946f"
REPORT_PATH = "docs/v2/bench/results.json"
REPORT_SHA256 = "2c702c6c61aa89966bcbe82929386be94144c6e7fa02e4fa22ac78d15accd2bd"
PUBLICATION_PATH = "docs/v2/bench/admissions/fuser-image-2.1.1.md"
PUBLICATION_SHA256 = "626b0fa193b5251d016c60ddf62165b5ff8b30a3523532a21282154dd8d061f4"
PUBLISHED = {
    "image": "746ada66f73d77c3ca948eb86d38e3f422ac3e6843ccfbab64f807b065ad153a",
    "audio": "44b63c7c8029103cf1b049c0e3548e6c2a39a027d95e39119145b83bc71990d3",
    "video": "fbbccc434f1af6a9e1fb64ccfddbbc1dc70a01f1f4fab9457b49e699f7ffcb45",
}
EVALUATED = {
    "image": "c4303453c7bba3e491bf2498497510d365c8fa2aebe64a9c02475e157bfdb82f",
    "audio": "35fb3e96d21506a5ec258476703b31925edc69908232bd0cf08e97e550f9e6cb",
    "video": "1b15680fa87a7cb8a4a86e81e3bd2f758ec1a5b60fe5fe397d15b6016f3c5107",
}
LIMITATION = (
    "Historical benchmark, not independent source validation or current end-to-end accuracy. "
    "The evaluated-to-published artefact relationship is documented metrics publication; earlier artefact bytes "
    "were not independently reconstructed. Scores are unclamped statistical fuser estimates before documentary "
    "overrides and publication caps. ECE has a grouped-bootstrap interval; per-bin intervals are not recorded. "
    "Persisted numeric fields may be rounded to six decimal places; the pinned source report and bundle preserve full precision."
)


def source_precision_valid(diagram: ReliabilityDiagram) -> bool:
    """Original package/source aggregates have not passed through canonical 6dp storage."""
    ece = sum(b.weight * abs(b.mean_y - b.mean_s) for b in diagram.bins)
    return (math.isclose(sum(b.weight for b in diagram.bins), 1.0, abs_tol=1e-12, rel_tol=0)
            and math.isclose(ece, diagram.ece, abs_tol=1e-12, rel_tol=0))


def _matches(diagram: ReliabilityDiagram, doc: Mapping[str, Any]) -> bool:
    bench = doc.get("bench") or {}
    test = (doc.get("metrics") or {}).get("test") or {}
    modality = diagram.population.split("/")[0]
    return all((
        doc.get("modality") == modality,
        doc.get("version") == "2.1.1",
        bench.get("version") == diagram.bench_version,
        bench.get("manifest_sha256") == diagram.manifest_sha256,
        bench.get("scope_sha256") == diagram.scope_sha256,
        test.get("bench_version") == diagram.bench_version,
        test.get("manifest_sha256") == diagram.manifest_sha256,
        test.get("generated_at") == diagram.report_generated_at,
        test.get("n0") == diagram.n_real,
        test.get("n1") == diagram.n_fake,
        test.get("ece") == diagram.ece,
        ((test.get("view") or {}).get("primary") == diagram.view if modality == "image"
         else not test.get("view")),
    ))


def diagram_for_artefact(sha256: str, doc: Mapping[str, Any]) -> ReliabilityDiagram | None:
    """Exact selected-byte identity only; every call returns independent validated data."""
    if sha256 not in PUBLISHED.values():
        return None
    try:
        with BUNDLE.open("rb") as handle:
            raw = handle.read(128 * 1024 + 1)
        if len(raw) > 128 * 1024:
            return None
        if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
            return None
        bundle = json.loads(raw)
        if set(bundle) != {"version", "records"} or bundle["version"] != "veritas.calibration-evidence/1":
            return None
        records = [ReliabilityDiagram.model_validate(r) for r in bundle["records"]]
        if len(records) != 3 or {r.published_artefact_sha256 for r in records} != set(PUBLISHED.values()):
            return None
        for r in records:
            modality = r.population.split("/")[0]
            if (not source_precision_valid(r) or r.published_artefact_sha256 != PUBLISHED[modality]
                    or r.evaluated_artefact_sha256 != EVALUATED[modality]
                    or r.report_sha256 != REPORT_SHA256 or r.publication_record_sha256 != PUBLICATION_SHA256):
                return None
        match = next(r for r in records if r.published_artefact_sha256 == sha256)
        return match if _matches(match, doc) else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
