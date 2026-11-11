from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.engine import rerun
from app.engine.stages import document_screening


@pytest.mark.parametrize(
    "threshold,context,calibration_pin,refused",
    [(0.6, "Held-out plain-text business email", document_screening.EMAIL_SHA256, True),
     (0.8, "Held-out plain-text business email", document_screening.EMAIL_SHA256, False),
     (0.6, "Another reference", document_screening.EMAIL_SHA256, False),
     (0.6, "Held-out plain-text business email", "another-calibration", False)],
)
@pytest.mark.parametrize("retry", [False, True])
def test_email_pinned_rerun_refuses_only_old_matching_operating_point(monkeypatch, threshold, context, calibration_pin, refused, retry):
    monkeypatch.setattr(rerun, "pinned_problems", lambda *args: [])
    monkeypatch.setattr(document_screening, "load", lambda: (b"", {}))
    source = SimpleNamespace(status="done", result_schema_version=2, result_hash="original", kind="document", profile="full")
    pin = document_screening.EMAIL_SHA256
    result = {"manifest": {"detectors": []}, "decision": {"fusion_mode": "artefact", "fuser_sha256": pin},
              "calibration": [{"artefact_sha256": calibration_pin,
                               "error_rates": [{"context": context, "threshold": threshold}]}]}
    saved = deepcopy(result)
    retry_pins = {"fusion_mode": "artefact", "fuser_sha256": pin, "profile": "full"} if retry else None
    if refused:
        with pytest.raises(rerun.RerunRefused, match="earlier decision rule.*normal reanalysis"):
            rerun.check(None, None, source, result, stored=retry_pins)
    else:
        assert rerun.check(None, None, source, result, stored=retry_pins)["fuser_sha256"] == pin
    assert result == saved
