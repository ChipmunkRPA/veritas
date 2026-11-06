"""Locally calibrated screening of eligible English plain-text email bodies."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit
from sqlalchemy.exc import IntegrityError

from app.bench.metrics import wilson
from app.detectors.text.extract import Document, _ocr_language
from app.engine.customer_assessment import EMAIL_REFERENCE_SLICE, THRESHOLD
from app.engine.fusion.runtime import annotate
from app.engine.stages import common
from app.schemas.evidence import Decision
from app.schemas.legacy import FALLBACK_CAVEAT
from app.storage.artefacts import register_artefact
from app.storage.db import FuserRow

DETECTOR = "txt.four_class"
MODEL = "hf.wasitaigenerated.text_small"
RETIRED_ARTICLE_PIN = "d12ff32c1177b660867f4a9bfaccaed9081d5e538778491f3c9263700ab26597"
RETIRED_ARTICLE_REASON = "The pilot article calibration was withdrawn after testing on a second article source."
EMAIL_PATH = Path(__file__).resolve().parents[2] / "detectors/text/calibrations/local-email-v1.json"
EMAIL_SHA256 = "a03d90c9d3d342000e742505743e30d4847017775b57c5b50fb9373c09f8db2b"
H1 = "H1_whole_content_synthesis"
EMAIL_NOTE = (
    "This screening covers the selected plain-text message body. Strong responses can indicate AI generation or "
    "rewriting. A lower response leaves authorship inconclusive because some rewrites are missed. "
    "Sender identity, attachments, and other messages in a thread require separate verification."
)


@lru_cache(maxsize=1)
def load() -> tuple[bytes, dict]:
    raw = EMAIL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EMAIL_SHA256:
        raise ValueError("The local document calibration does not match its pinned hash")
    return raw, json.loads(raw)


def estimate(raw_score: float, artifact: dict) -> tuple[float, tuple[float, float], float]:
    if not math.isfinite(raw_score) or not 0 <= raw_score <= 1:
        raise ValueError("The document check returned an invalid response")
    score = min(1 - 1e-6, max(1e-6, raw_score))
    x = math.log(score) - math.log1p(-score)
    intercept, slope = artifact["estimate"]
    p = float(expit(intercept + slope * x))
    bootstrap = np.asarray(artifact["bootstraps"], dtype=float)
    interval = np.quantile(expit(bootstrap[:, 0] + bootstrap[:, 1] * x), (0.05, 0.95))
    return p, (float(interval[0]), float(interval[1])), x


def _matches(signal: dict, calibration: dict) -> bool:
    return (
        signal.get("detector_id") == DETECTOR
        and signal.get("detector_version") == calibration["detector_version"]
        and (signal.get("model_revisions") or {}).get(MODEL) == calibration["checkpoint_revision"]
        and signal.get("raw_score") is not None
    )


def email_reference_scope(doc: Document) -> bool:
    """The reference uses one readable plain-text body from one message."""
    return (
        doc.meta.get("method") == "email"
        and (doc.meta.get("email") or {}).get("body_content_type") == "text/plain"
        and 50 <= len(doc.text.split()) <= 220
        and bool(doc.source_map)
        and all(part.get("quote_role") == "body" for part in doc.source_map)
        and _ocr_language(doc.text).get("code") == "en"
    )


def _commit(ctx, data: bytes, calibration: dict, pin: str) -> dict:
    with ctx.sessions() as db:
        row = register_artefact(db, ctx.store, data, media_type="application/json", role="stage_output",
                                created_by="document-confidence", analysis_id=ctx.analysis_id)
        if db.get(FuserRow, pin) is None:
            try:
                with db.begin_nested():
                    db.add(FuserRow(sha256=pin, modality="document", version=calibration["version"],
                                    bench_version=calibration["bench_version"]))
            except IntegrityError:
                if db.get(FuserRow, pin) is None:
                    raise
        db.commit()
        return {"id": row.id, "sha256": row.sha256, "size_bytes": row.bytes, "media_type": row.media_type,
                "role": row.role, "created_by": row.created_by}


def apply(ctx, doc: Document, fallback: dict) -> dict:
    pin = EMAIL_SHA256
    primary = None
    if email_reference_scope(doc):
        data, calibration = load()
        primary = next((s for s in fallback["signals"] if _matches(s, calibration)), None)
    if primary is None:
        if ctx.state.get("fuser_pin") == pin:
            raise ValueError("The saved analysis needs its original calibrated text response and extraction.")
        return fallback

    p, interval, x = estimate(primary["raw_score"], calibration)
    intercept, slope = calibration["estimate"]
    start = float(expit(intercept))
    if p >= THRESHOLD:
        code, band = "ai_supported", "likely_ai"
        headline = "Leaning towards AI-written or AI-rewritten email text"
    else:
        code, band = "inconclusive", "uncertain"
        headline = "Inconclusive email authorship"
        p, interval = None, None
    note = EMAIL_NOTE
    reason = f"{headline}. {note}"
    contributions = [] if p is None else [
        {"kind": "neutral_start", "label": "Reference starting estimate", "logit": intercept, "pp": 100 * start},
        {"kind": "detector", "label": "Text authorship check", "signal_id": primary["id"],
         "logit": slope * x, "pp": 100 * (p - start), "counted": True, "baseline": "neutral_point"},
    ]
    decision = Decision(code=code, verbal=headline, reason=reason, band=band, confidence="low" if band == "uncertain" else "medium",
                        fusion_mode="artefact", fuser_sha256=pin, driving_hypothesis=H1,
                        probability_ai_calibrated=p, ci90=interval, contributions=contributions).model_dump(mode="json")
    fields = {} if p is None else {
        primary["id"]: {"calibrated_probability": p, "calibration_ref": pin, "effective_weight": 1.0,
                        "contribution_logit": slope * x, "contribution_pp": 100 * (p - start)},
    }
    signals = annotate(fallback["signals"], fields, artefact_detectors=(DETECTOR,) if fields else (), outside={})
    profile = {"slice_id": EMAIL_REFERENCE_SLICE, "language": "en",
               "within_validated_envelope": True, "validation_support": "direct", "limiting_conditions": [note]}
    caveats = [note, *[c for c in fallback["caveats"] if c != FALLBACK_CAVEAT and "confidence is capped at low" not in c]]
    verdict = common.legacy_verdict({**fallback["v1"]["verdict"], "caveats": caveats}, decision, profile)
    verdict.update(reasoning=[reason], summary=headline)
    tested = calibration["transfer"]
    n_ai, n_human = tested["ai"], tested["human"]
    ai_correct, ai_false = tested["ai_correct"], tested["ai_false"]
    metadata = {
        "subject": f"fuser-document-{calibration['version']}", "method": "platt", "artefact_sha256": pin,
        "bench_version": calibration["bench_version"],
        "bench_manifest_sha256": calibration["transfer_manifest_sha256"],
        "brier": tested["brier"], "error_rates": [
            {"context": "Held-out plain-text business email",
             "threshold_kind": "fixed", "threshold": THRESHOLD,
             "tpr": ai_correct / n_ai, "fpr": ai_false / n_human, "tpr_ci95": wilson(ai_correct, n_ai),
             "fpr_ci95": wilson(ai_false, n_human), "n_pos": n_ai, "n_neg": n_human,
             "bench_version": calibration["bench_version"]},
        ],
    }
    return {
        **fallback, "decision": decision, "signals": signals, "caveats": caveats,
        "condition_profile": profile, "calibration": [metadata],
        "assessments": [{"hypothesis": H1, "applicable": True, "probability": p, "ci90": interval,
                         "contributions": contributions, "note": note}],
        "fusion": {"mode": "artefact", "version": calibration["version"], "sha256": pin},
        "v1": {**fallback["v1"], "verdict": verdict},
        "artefacts": [*fallback.get("artefacts", []), _commit(ctx, data, calibration, pin)],
    }
