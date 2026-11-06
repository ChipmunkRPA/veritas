import copy

import pytest

from scripts.audit_audio_identity_forwards_v9 import same_native, select


def test_selection_covers_sources_generators_classes_conditions_without_scores():
    rows = [
        {
            "id": f"{dataset}:{generator}:{label}:{condition}:{i}",
            "role": "development",
            "dataset": dataset,
            "generator": generator,
            "label": label,
            "condition": condition,
            "raw": i / 2,
            "status": "ok",
        }
        for dataset in ("a", "b")
        for generator in ("first", "second")
        for label in (0, 1)
        for condition in ("clean", "speech_0")
        for i in range(2)
    ]
    expected = select(rows)
    assert len(expected) == 16 and all(x.endswith(":0") for x in expected)
    for row in rows:
        row.update(raw=0.999, status="not_applicable")
    assert select(list(reversed(rows))) == expected
    assert select(rows + [{**rows[0], "id": "000", "role": "fit"}]) == expected


def test_no_development_inputs_cannot_vacuously_pass():
    with pytest.raises(ValueError, match="development metadata"):
        select([])


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw", 0.8),
        ("speech_spans_s", []),
        ("windows", []),
        ("embedding_sha256", "changed"),
        ("status", "not_applicable"),
    ],
)
def test_native_differences_are_not_masked_by_rounded_or_matching_headlines(key, value):
    expected = {
        "status": "ok",
        "raw": 0.80000001,
        "speech_spans_s": [[0.1, 3.9]],
        "windows": [{"start_sample": 0, "end_sample": 64000}],
        "embedding_sha256": "fixed",
    }
    actual = copy.deepcopy(expected)
    assert same_native(actual, expected)
    actual[key] = value
    assert not same_native(actual, expected)


def test_unassessed_native_case_is_not_a_genuine_speech_verdict():
    row = {"status": "not_applicable", "raw": None, "speech_spans_s": [], "windows": []}
    assert same_native(row, row)
