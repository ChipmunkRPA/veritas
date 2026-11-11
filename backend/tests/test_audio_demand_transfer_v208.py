import pytest

from scripts import audio_demand_transfer_v208 as trial


def test_clean_replay_preserves_actual_direction_and_raw_value():
    reference = {
        "route_status": "ok",
        "decision": "original",
        "windows": [[0, 48000]],
        "stability_probe": "computed",
        "raw": 0.125,
    }
    trial.check_clean({**reference, "raw": 0.125000001}, reference)
    with pytest.raises(ValueError, match="differs"):
        trial.check_clean({**reference, "decision": "ai"}, reference)
    with pytest.raises(ValueError, match="numerical replay"):
        trial.check_clean({**reference, "raw": 0.126}, reference)


def test_transitions_use_each_sources_matched_clean_and_retain_unavailable():
    rows = []
    for source, after in (("one", "ai"), ("two", "unavailable")):
        for condition in trial.CONDITIONS:
            is_clean = condition in {"source_clean", "matched_clean"}
            rows.append(
                {
                    "origin": "genuine",
                    "source": source,
                    "condition": condition,
                    "decision": "original" if is_clean else after,
                    "route_status": "ok" if is_clean or after == "ai" else "not_applicable",
                }
            )
    result = trial.summarize(rows)["genuine"]
    assert result["river_0db"]["speech_gate_rejected"] == 1
    assert result["river_0db"]["paired_transitions"] == {"original->ai": 1, "original->unavailable": 1}
    assert result["matched_clean"]["paired_transitions"] == {"original->original": 2}
    assert result["source_clean"]["transitions_from"] is None
