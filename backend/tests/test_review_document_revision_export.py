from __future__ import annotations

import json

import pytest

from app.evidence.document_revision import compare
from scripts.review_document_revision_export import review


def test_offline_draft_replay_verifies_changed_ranges_and_rejects_tampering(tmp_path):
    earlier = tmp_path / "earlier.txt"
    later = tmp_path / "later.txt"
    export = tmp_path / "comparison.json"
    earlier.write_text("Café 🙂 staff will arrive today.\n", encoding="utf-8")
    later.write_text("Café 🙂 staff will arrive tomorrow.\n", encoding="utf-8")
    result = compare(earlier.read_bytes(), later.read_bytes(), ".txt")
    export.write_text(json.dumps(result), encoding="utf-8")
    assert review(export, earlier, later) == {"source_hashes_verified": 2,
        "coordinate_space": "decoded_source_unicode_codepoint", "changes_by_role": {"body": 1},
        "changed_token_ranges_verified": 2}
    result["changes"][0]["later"]["changed_tokens"][0][0] -= 1
    export.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="does not replay"):
        review(export, earlier, later)
