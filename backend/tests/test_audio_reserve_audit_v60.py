from scripts.audio_reserve_audit_v60 import inventory, matches


def test_plan_is_not_a_screening_measurement():
    assert list(matches({"sources": [{"id": "reserved", "role": "confirmation"}]}, {"reserved"})) == [
        {"pointer": "/sources/0", "identity_fields": ["id"], "screening_fields": []}]


def test_source_nested_with_real_measurement_is_flagged():
    result = list(matches({"raw": .4, "source": {"sha256": "reserved"}}, {"reserved"}))
    assert result[0]["screening_fields"] == ["raw"]


def test_sibling_source_measurement_is_not_misattributed():
    result = list(matches([{"id": "elsewhere", "raw": .8}, {"id": "reserved"}], {"reserved"}))
    assert result[0]["screening_fields"] == []


def test_unassessed_and_vector_records_remain_visible():
    result = list(matches([{"id": "reserved", "decision": "inconclusive"},
                           {"sha256": "reserved", "wavlm": [.1, .2]}], {"reserved"}))
    assert [row["screening_fields"] for row in result] == [["decision"], ["wavlm"]]


def test_inventory_follows_moved_research_directory_and_avoids_loop(tmp_path):
    base, outside = tmp_path / "base", tmp_path / "moved"
    base.mkdir()
    outside.mkdir()
    (outside / "result.json").write_text("{}")
    (base / "history").symlink_to(outside, target_is_directory=True)
    (outside / "loop").symlink_to(base, target_is_directory=True)
    assert inventory(base, base / "new-audit") == [base / "history/result.json"]
