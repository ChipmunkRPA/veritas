"""Real report rendering of authored source/scoring fixtures, not model accuracy."""

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_document_packets import packet_factory as packet_factory
from test_multimedia_evidence_reports import evidence_case as evidence_case

from app.core.canonical import canonical_hash
from app.reports import engine, multimedia_evidence
from app.reports.snapshot import SnapshotCoverage


@pytest.mark.parametrize("evidence_case", ["document"], indirect=True)
def test_context_comparison_in_actual_full_and_redacted_pdf(evidence_case, monkeypatch):
    frozen, store, _, analysis, _, _ = evidence_case
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    result = copy.deepcopy(frozen.result)
    result["input"].update(
        filename="AUTHORED-SOFTWARE-QA.pdf",
        size_bytes=analysis.size_bytes,
        mime_sniffed=analysis.content_type,
    )
    result["decision"].update(
        code="inconclusive",
        verbal="Authored software QA",
        reason="Authored response fixtures; no neural inference.",
        probability_ai_calibrated=None,
        probability_ai_uncalibrated=None,
        fuser_sha256=None,
        fusion_mode="fallback",
        contributions=[],
    )
    result["calibration"] = []
    result["caveats"] = [
        "AUTHORED SOFTWARE QA. Response values are fixtures; no model inference or accuracy test was performed."
    ]
    result["manifest"]["detectors"] = []
    result["findings"] = [
        {
            **result["findings"][0],
            "id": "F1",
            "family": "T1_ai_text",
            "title": "Authored context comparison",
            "statement": "Inspect the recorded context inputs. Authored scoring fixture only.",
            "strength": "none",
            "evidence_ids": [result["signals"][0]["id"]],
        }
    ]
    evidence = copy.deepcopy(frozen.evidence)
    evidence["original_result_hash"] = canonical_hash(result)
    evidence["sha256"] = canonical_hash({k: v for k, v in evidence.items() if k != "sha256"})
    now = datetime.now(UTC)
    frozen = frozen.model_copy(
        update={
            "result": result,
            "evidence": evidence,
            "case": None,
            "custody": [],
            "model_cards": [],
            "actor_names": {},
            "annotations": [],
            "revisions": [],
            "detector_names": {},
            "custody_verification": None,
            "report": frozen.report.model_copy(
                update={
                    "id": "authored-context-software-qa",
                    "display_id": "AUTHORED-CONTEXT-QA",
                    "revision": 1,
                    "requested_at": now,
                    "requested_by": "software-qa",
                    "requested_by_name": "Chipmunk Verify",
                    "supersedes_report_id": None,
                    "revision_reason": None,
                }
            ),
            "analysis": frozen.analysis.model_copy(
                update={
                    "revision": 1,
                    "created_at": now,
                    "finished_at": now,
                    "supersedes_id": None,
                    "supersedes_revision": None,
                    "mode": None,
                    "coverage": SnapshotCoverage(detectors_total=1, by_status={"ok": 1}),
                }
            ),
            "review": frozen.review.model_copy(
                update={"state": "machine", "conclusion": None, "signoffs": [], "issuable": False}
            ),
            "media": frozen.media.model_copy(
                update={
                    "filename": "AUTHORED-SOFTWARE-QA.pdf",
                    "size_bytes": analysis.size_bytes,
                    "content_type": analysis.content_type,
                    "md5": hashlib.md5(store.read_bytes(analysis.sha256), usedforsecurity=False).hexdigest(),
                    "received_at": now,
                    "date_acquired": now,
                    "supplied_by": "Software QA",
                    "source": "Authored fixture; no neural inference",
                    "acquisition_method": "Authored test text",
                    "intake_notes": "Software export QA only",
                    "custodian_hash_match": None,
                    "custodian_stated_hash": None,
                }
            ),
        }
    )
    receipts = []
    for redacted in (False, True):
        current = frozen.model_copy(
            update={"report": frozen.report.model_copy(update={"profile": "standard", "redacted": redacted})}
        )
        rendered = engine.render_pdf(current)
        text = engine.pdf_text(rendered.pdf)
        normalized = " ".join(text.split())
        assert "How surrounding text affects this check" in normalized
        assert "0.100" in normalized and "0.900" in normalized and "3 distinct saved inputs" in normalized
        assert "No authorship confidence is assigned" in normalized
        assert "AUTHORED SOFTWARE QA" in normalized
        assert ("First sentence contains enough useful words" in normalized) is not redacted
        assert all(name not in text for name in ("PRIVATE MODEL NAME", "txt.desklib", "Local examiner"))
        assert not rendered.refused
        view = multimedia_evidence.view(current, store=store)
        comparison = next(p for p in view["packets"] if p["id"] == "T01")["context_comparison"]
        assert all((w["text"] is None) == redacted for c in comparison["checks"] for w in c["inputs"])
        name = "context-comparison-redacted" if redacted else "context-comparison-full"
        receipt = {
            "name": name,
            "pages": rendered.page_count,
            "sha256": hashlib.sha256(rendered.pdf).hexdigest(),
            "redacted": redacted,
            "authored_scores": True,
            "new_neural_calls": 0,
        }
        receipts.append(receipt)
        folder = os.environ.get("CHIPMUNK_CONTEXT_PDF_QA_DIR")
        if folder:
            path = Path(folder) / "output/pdf"
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{name}.pdf").write_bytes(rendered.pdf)
    folder = os.environ.get("CHIPMUNK_CONTEXT_PDF_QA_DIR")
    if folder:
        (Path(folder) / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
        (Path(folder) / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
