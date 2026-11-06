import hashlib
import json

import pytest

from scripts.audit_writing_sources_v2 import audit, check_row


def row(author="person", human_only=False):
    return {
        "metadata": {"author": author, "human_only": human_only, "model": "fixture"},
        "data": [
            {"text": "😀 First human sentence.\n\n", "author": "User", "queries": []},
            {"text": "Then an AI continuation appears.", "author": "Bot", "queries": ["Continue."]},
        ],
    }


def previous(r):
    text = "".join(s["text"] for s in r["data"])
    return {
        "cases": [
            {
                "id": "mixed:0",
                "label": "mixed",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ]
    }


def test_unicode_and_paragraph_normalization_keep_character_provenance():
    text, result = check_row(row())
    assert text.startswith("😀") and result["mixed"] and result["human_characters"] > 0
    assert result["bot_characters"] > 0


@pytest.mark.parametrize("change", ["label", "human_only", "queries", "author"])
def test_unknown_or_conflicting_provenance_is_rejected(change):
    r = row()
    if change == "label":
        r["data"][0]["author"] = "unknown"
    elif change == "human_only":
        r["metadata"]["human_only"] = True
    elif change == "queries":
        r["data"][0]["queries"] = "not a list"
    else:
        r["metadata"]["author"] = ""
    with pytest.raises(ValueError):
        check_row(r)


def test_used_author_other_document_and_exact_text_are_not_called_fresh():
    first, other, duplicate, unused = row(), row(), row("second"), row("third")
    other["data"][0]["text"] = "Different human text. "
    unused["data"][0]["text"] = "Entirely distinct human text. "
    result = audit([first, other, duplicate, unused], previous(first))
    assert result["passed"] and result["prior_mixed_authors"] == 1
    assert result["unused_author_nonreused_pool"]["authors"] == 1
    encoded = json.dumps(result)
    assert all(s not in encoded for s in ("Different human text", "Entirely distinct", '"second"', '"third"'))


def test_failed_rows_remain_visible_and_block_preflight_pass():
    good, bad = row(), row("another")
    bad["data"][0]["author"] = "unknown"
    result = audit([good, bad], previous(good))
    assert not result["passed"] and result["source_rows"] == 2 and result["failed_rows"] == 1
    assert len(result["outcomes"]) == 2


def test_modified_historical_text_cannot_change_the_exclusion_inventory():
    r = row()
    old = previous(r)
    old["cases"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="Historical mixed sample"):
        audit([r], old)


def test_missing_metadata_remains_a_failed_row_in_the_audit_receipt():
    good = row()
    result = audit([good, {"data": []}], previous(good))
    assert not result["passed"] and result["failed_rows"] == 1
    assert len(result["outcomes"]) == result["source_rows"] == 2
