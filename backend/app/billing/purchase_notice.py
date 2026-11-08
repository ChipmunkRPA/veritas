"""Versioned pre-purchase disclosure. No refund or rights-waiver terms are implied."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from app.core.canonical import canonical_hash

VERSION = "purchase-risk/v2"
NOTICE = {
    "version": VERSION,
    "title": "Before you pay",
    "points": [
        "Automated deepfake screening can be unreliable, and performance varies across generators, editing, compression, background noise and writing context. Use the result cautiously as a starting point for further review.",
        "Automated checks can miss AI-generated or deepfake material.",
        "Genuine material can be incorrectly flagged as AI-generated or altered.",
        "Your uploaded file or document may produce an inconclusive result, no usable confidence estimate, or no reliable conclusion.",
        "Payment covers the automated analysis and available evidence; it does not guarantee detection or a conclusive answer.",
        "The standard SaaS report is preliminary and must not be relied on as court evidence. A private expert report requires a separate signed engagement.",
        "For consequential decisions, contact sales@chipmunkrpa.com for a separately scoped private investigation of source files, provenance and corroborating evidence. Expert review does not guarantee a conclusion or court admissibility.",
    ],
    "acknowledgement_label": "I understand and accept these analysis limitations before paying US$9.99.",
    "document_notice": (
        "Article checks currently provide inspection evidence without a calibrated whole-article authorship conclusion. "
        "Email-body screening is a separate, limited check."
    ),
    "stripe_message": (
        "Automated deepfake screening can be unreliable. Use results cautiously. "
        "Automated analysis can miss deepfakes, flag genuine material, or return no conclusion. "
        "Payment covers analysis and available evidence, not guaranteed detection or a conclusive answer. "
        "This preliminary SaaS report must not be relied on as court evidence; private expert work requires a separate engagement."
    ),
}
SHA256 = canonical_hash(NOTICE)
# Historical orders retain the disclosure they actually accepted. New checkout
# requests must accept VERSION; changing wording must not strand a paid order.
_V1_NOTICE = json.loads(Path(__file__).with_name("purchase_notice_v1.json").read_text())
_RECORDED_NOTICES = {canonical_hash(_V1_NOTICE): _V1_NOTICE, SHA256: NOTICE}


def public_notice() -> dict:
    return {**copy.deepcopy(NOTICE), "sha256": SHA256}


def validates(accepted: bool, version: str | None, sha256: str | None) -> bool:
    return accepted is True and version == VERSION and sha256 == SHA256


def receipt(owner_id: str, accepted_at: str, *, stripe_notice: bool) -> dict:
    return {"accepted": True, "accepted_by": owner_id, "accepted_at": accepted_at,
            "notice": copy.deepcopy(NOTICE), "notice_sha256": SHA256, "stripe_notice": stripe_notice}


def recorded(receipt: dict | None, owner_id: str) -> bool:
    return (isinstance(receipt, dict) and receipt.get("accepted") is True and receipt.get("accepted_by") == owner_id
            and isinstance(receipt.get("accepted_at"), str) and bool(receipt["accepted_at"])
            and isinstance(receipt.get("notice_sha256"), str)
            and receipt.get("notice_sha256") in _RECORDED_NOTICES
            and receipt.get("notice") == _RECORDED_NOTICES[receipt["notice_sha256"]])
