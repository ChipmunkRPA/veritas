"""Extraction parity selection cannot use answers, skip labels or held sources."""

import copy

import pytest

from scripts.qualify_audio_extraction_v98 import KINDS, selection


def rows():
    return [
        {
            "id": f"{kind}-{base}-{condition}",
            "kind": kind,
            "base_id": f"{kind}-{base}",
            "condition": str(condition),
            "role": role,
        }
        for role in ("fusion_fit", "selection", "confirmation")
        for kind in KINDS
        for base in range(3)
        for condition in range(9)
    ]


def test_all_families_and_conditions_are_selected_only_from_fitting_sources():
    selected = selection(rows())
    assert len(selected) == 36
    assert {r["kind"] for r in selected} == set(KINDS)
    assert {r["role"] for r in selected} == {"fusion_fit"}
    assert len({r["base_id"] for r in selected}) == 4


def test_metadata_selection_is_order_and_measured_response_independent():
    source = rows()
    changed = copy.deepcopy(source)
    for i, row in enumerate(changed):
        row.update(raw=float(i % 2), status="not_applicable", decision="original")
    assert [r["id"] for r in selection(source)] == [r["id"] for r in selection(changed[::-1])]


def test_missing_condition_is_refused_without_substituting_another_source():
    source = rows()
    selected = selection(source)
    source = [r for r in source if r != selected[0]]
    with pytest.raises(ValueError, match="nine"):
        selection(source)
