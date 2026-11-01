"""The customer interpretation is exported separately from immutable evidence."""
import copy
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.api import routes
from app.engine.customer_assessment import customer_assessment


@pytest.mark.parametrize("schema_version", [1, 2])
def test_export_adds_leaning_without_replacing_the_recorded_decision(monkeypatch, schema_version):
    stored = {"decision": {"code": "inconclusive", "fusion_mode": "artefact", "probability_ai_calibrated": 0.12,
                           "ci90": [0.10, 0.17]}, "condition_profile": {"within_validated_envelope": True}}
    before = copy.deepcopy(stored)
    compatible = stored if schema_version == 1 else {"verdict": {"p_ai": 0.99}}
    analysis = SimpleNamespace(id="test", status="done", result_schema_version=schema_version, result=compatible,
                               to_summary=lambda: {"id": "test", "result_hash": "original-hash"})
    if schema_version == 2:
        from app.api import v2_evidence

        @contextmanager
        def sessions():
            yield SimpleNamespace(get=lambda *_: analysis)

        monkeypatch.setattr(routes, "get_session", sessions)
        monkeypatch.setattr(v2_evidence, "_result", lambda *_: stored)
    monkeypatch.setattr(routes, "_get", lambda _: analysis)
    monkeypatch.setattr(routes, "_current_revision", lambda _: None)
    monkeypatch.setattr(routes, "_verified", lambda *_: None)
    events = []
    monkeypatch.setattr(routes.audit, "record", lambda *args, **kwargs: events.append(kwargs))
    response = routes.export_json("test", SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id="local"))))
    data = json.loads(response.body)
    assert data["customer_assessment"] == customer_assessment({**stored, "status": "done"})
    assert data["customer_assessment"]["direction"] == "original"
    assert data["customer_assessment"]["confidence_pct"] == 88
    assert data["result"] == compatible
    assert data["result_hash"] == "original-hash"
    assert stored == before
    assert events[0]["after_hash"] == hashlib.sha256(response.body).hexdigest()
