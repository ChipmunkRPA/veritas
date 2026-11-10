import copy

import pytest

from scripts import native_mouth_components_v128 as corrected


def source_rows():
    rows = []
    for i, (actor, sentence, ancestry, component) in enumerate(
        [
            ("actor-a", "sentence-a", "ancestry-a", "a"),
            ("actor-b", "sentence-a", "ancestry-b", "a"),
            ("actor-b", "sentence-c", "ancestry-c", "a"),
            ("actor-d", "sentence-d", "ancestry-c", "a"),
            ("actor-e", "sentence-e", "ancestry-e", "e"),
        ]
    ):
        rows.append(
            {
                "id": chr(97 + i),
                "asset_index": i,
                "actor": actor,
                "sentence": sentence,
                "ancestry": ancestry,
                "study_component": component,
                "study_role": "selection",
            }
        )
    return rows


def test_components_join_shared_sentences_actors_and_ancestry_transitively_in_any_input_order():
    rows = source_rows()
    expected = {0: "a", 1: "a", 2: "a", 3: "a", 4: "e"}
    assert corrected.connected(rows) == expected
    assert corrected.connected(list(reversed(rows))) == expected


@pytest.mark.parametrize("problem", ["cross_role", "wrong_component", "missing_ancestry", "duplicate_asset"])
def test_role_leakage_or_wrong_component_inventory_is_rejected(problem):
    rows = source_rows()
    if problem == "cross_role":
        rows[1]["study_role"] = "confirmation"
    elif problem == "wrong_component":
        rows[2]["study_component"] = "c"
    elif problem == "missing_ancestry":
        rows[0]["ancestry"] = ""
    else:
        rows[1]["asset_index"] = rows[0]["asset_index"]
    with pytest.raises(ValueError):
        corrected.connected(rows)


def outcome(group, label, flag):
    metadata = {
        "asset_index": group,
        "source_group": str(group),
        "study_role": "selection",
        "condition": "ai_aligned" if label else "original",
        "input_status": "extracted",
        "label": label,
    }
    return corrected.original.outcome(
        metadata, True, [["1", "2"]] if flag else [], 0.7, ["1", "2"] if label else None
    )


def test_paired_component_counts_do_not_pretend_source_groups_are_independent_samples():
    reference = [outcome(g, y, False) for g in range(4) for y in (0, 1)]
    candidate = [outcome(g, y, bool(y)) for g in range(4) for y in (0, 1)]
    result = corrected.paired_components(
        candidate, reference, {0: "context-a", 1: "context-a", 2: "context-b", 3: "context-b"}
    )
    assert result["source_groups"] == 4 and result["connected_components"] == 2
    assert result["bootstrap_replicates"] == 0
    assert result["additional_localized_ai"] == 4 and result["additional_genuine_flags"] == 0
    assert all(r["group_bootstrap_ci95"] is None for r in result["rate_differences"].values())
    assert result["rate_differences"]["ai_detection_rate_ready"]["candidate_minus_reference"] == 1
    assert [r["source_groups"] for r in result["component_aggregates"]] == [2, 2]
    assert "context-a" not in str(result) and "context-b" not in str(result)
    wrong = copy.deepcopy(reference)
    wrong[0]["condition"] = "blur2"
    with pytest.raises(ValueError):
        corrected.paired_components(candidate, wrong, {0: "a", 1: "a", 2: "b", 3: "b"})


def test_result_projection_replaces_misleading_sample_size_and_keeps_original_context_untouched(
    tmp_path, monkeypatch
):
    inventory = {role: {"sources": 10, "connected_components": 3} for role in ("selection", "confirmation")}
    monkeypatch.setattr(corrected, "component_plan", lambda: ({}, inventory))
    captured = []
    monkeypatch.setattr(corrected, "write", lambda path, value: captured.append(value))
    original = {"schema": "old", "role": "selection", "connected_groups": 10, "scope": "Finite panel."}
    corrected.corrected_write(tmp_path / "result.json", original)
    result = captured[0]
    assert result["source_groups"] == 10 and result["connected_components"] == 3
    assert "connected_groups" not in result and original["connected_groups"] == 10
    assert result["schema"].endswith("/128") and "no bootstrap" in result["uncertainty"]


def test_unrelated_stage_is_not_dispatched():
    with pytest.raises(ValueError):
        corrected.run("fusion_fit")
    with pytest.raises(ValueError):
        corrected.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract"])
