import json
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import grid_mouth_motion_readout_v158 as readout


def fixture(*, component=104, ordinal=0, missing_reference=None, unavailable=None):
    source = {
        "ordinal": ordinal,
        "source_slot": ordinal // 2,
        "component_index": component,
        "input": {
            "status": readout.planning.reviewed.AVAILABLE,
            "native_video_table": {
                "pts": [9000 + 3600 * i for i in range(75)],
                "time_base": "1/90000",
                "interpolated": [],
                "truncated": False,
            },
        },
    }
    table = source["input"]["native_video_table"]
    q = np.random.default_rng(2).uniform(0.04, 0.4, (75, 4))
    cases, rows = [], []
    for n, condition in enumerate(readout.planning.CASES):
        neural = condition.startswith(("musetalk_", "liveportrait_"))
        ready = condition != unavailable
        case = {
            "condition": condition,
            "case_number": 10 - n,
            "status": "ready" if ready else "unavailable",
            "reason": None if ready else "Authored private cause",
            "neural_provenance_verified": ready and neural,
            "observed_neural_pixel_change": ready and neural,
            "expected": {"rgb_sha256": [f"{i:064x}" for i in range(75)]},
        }
        cases.append(case)
        for i in readout.REFERENCE_FRAMES if condition == "original" else readout.VARIANT_FRAMES:
            clock = Fraction(table["pts"][i], 90000)
            early = condition.endswith("earlier_320ms") or condition == "genuine_frame_retime"
            use = i - 8 if early and i in readout.EDIT else i
            available = ready and (condition != "original" or i != missing_reference)
            row = {
                "request_ordinal": len(rows),
                "case_number": case["case_number"],
                "frame_index": i,
                "repeat": False,
                "case_status": case["status"],
                "case_reason": case["reason"],
                "source_pts": table["pts"][i],
                "source_time_base": table["time_base"],
                "source_time_seconds": str(clock),
                "pts": int(clock * 1000) if ready else None,
                "time_base": "1/1000" if ready else None,
                "rgb_sha256": case["expected"]["rgb_sha256"][i] if ready else None,
                "status": "geometry_available"
                if available
                else "unavailable_case"
                if not ready
                else "unavailable_mesh",
            }
            if available:
                row.update(
                    descriptor_hex=[float(value).hex() for value in q[use]], eye_distance_px_hex=(100.0).hex()
                )
            if condition != "original":
                row["eye_patch_drift"] = {
                    "status": "identical_eye_pixels" if available else "unavailable_geometry",
                    "rms_px_hex": (0.25).hex() if available else None,
                }
            rows.append(row)
    repeat = rows[readout.REFERENCE_FRAMES.index(27)]
    rows.append({**repeat, "repeat": True, "request_ordinal": len(rows)})
    return source, cases, rows


def measure(*, component=104, ordinal=0, missing_reference=None, unavailable=None):
    source, cases, observations = fixture(
        component=component,
        ordinal=ordinal,
        missing_reference=missing_reference,
        unavailable=unavailable,
    )
    geometry = readout.trajectories(source, cases, observations)
    rows, pairs = readout.assess_trajectories(source, geometry)
    return source, cases, rows, pairs


def test_complete_frozen_roster_and_six_exact_pairs_preserve_unavailable_case():
    _, _, rows, pairs = measure(unavailable="brightness_liveportrait")
    outcomes = {row["condition"]: row for row in rows}
    assert tuple(outcomes) == readout.planning.CASES and len(rows) == 11
    assert [(row["earlier"], row["comparator"]) for row in pairs] == list(readout.PAIR_SPECS)
    assert len(pairs) == 6
    assert outcomes["original"]["status"] == "source_time_compatible"
    assert outcomes["audio_delay_320ms"]["status"] == "source_time_compatible"
    assert outcomes["genuine_frame_retime"]["status"] == "earlier_compatible"
    assert outcomes["musetalk_earlier_320ms"]["status"] == "earlier_compatible"
    assert outcomes["brightness_liveportrait"]["unavailable_reason"] == "construction_unavailable"
    assert [pair["status"] for pair in pairs] == ["gain"] * 5 + ["unavailable"]
    assert sum(outcomes["original"]["geometry_counts"].values()) == 44
    assert outcomes["genuine_frame_retime"]["entry_velocity_delta_vs_original"] != 0


def test_target_only_does_not_promote_unavailable_full_curve():
    _, _, rows, pairs = measure(missing_reference=54)
    earlier = next(row for row in rows if row["condition"] == "musetalk_earlier_320ms")
    assert earlier["status"] == "unavailable"
    assert earlier["curve"]["supported_positions"] == 15 and not earlier["curve"]["measured"]
    assert earlier["target_only_contrast"]["measured"]
    assert earlier["target_only_contrast"]["supported_positions"] == 20
    assert all(pair["status"] == "unavailable" for pair in pairs)
    by_condition = readout.aggregate_rows(rows, pairs)["by_condition"]["musetalk_earlier_320ms"]
    assert by_condition["planned_cases"] == 1 and by_condition["full_curves_measured"] == 0
    assert by_condition["target_only_measured_without_full_curve"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_time_seconds", "1/999", "clock or construction availability"),
        ("rgb_sha256", "f" * 64, "bound native construction"),
        ("frame_index", 22, "fixed case schedule"),
        ("repeat", True, "One original repeat"),
        ("descriptor_hex", ["nan"] * 4, "Canonical finite"),
    ],
)
def test_admission_refuses_mutated_observation_metadata(field, value, message):
    source, cases, observations = fixture()
    observations[0][field] = value
    with pytest.raises(ValueError, match=message):
        readout.trajectories(source, cases, observations)


def test_actual_clock_is_rational_and_shift_is_eight_frames():
    source, cases, observations = fixture()
    times = readout.source_times(source)
    assert times[0] == Fraction(1, 10)
    assert times[27] - times[19] == Fraction(8, 25)
    geometry = readout.trajectories(source, cases, observations)
    assert (
        geometry["genuine_frame_retime"]["trajectory"][times[27]]
        == geometry["original"]["trajectory"][times[19]]
    )
    assert len(geometry["original"]["trajectory"]) == 44


def test_private_json_round_trip_preserves_all_signed_lag_keys():
    source, cases, rows, pairs = measure()
    packet = {
        "schema": "chipmunk.grid-mouth-motion-private/158",
        "cases": rows,
        "pairs": pairs,
        "observer_bindings": {},
        "observer_exact_control_checks": {},
    }
    restored = json.loads(json.dumps(packet, allow_nan=False))
    readout.validate_payload(source, cases, {}, {}, restored)
    del restored["cases"][0]["curve"]["lags"]["-8"]
    with pytest.raises(ValueError, match="all exact signed lag"):
        readout.validate_payload(source, cases, {}, {}, restored)


def test_public_aggregation_uses_five_component_quartiles_without_private_fields():
    _, _, rows, pairs = measure()
    small = readout.aggregate_rows(rows, pairs)
    assert small["by_condition"]["original"]["boundaries"]["entry_velocity_norm"]["median"] is None
    all_rows, all_pairs = [], []
    for offset in range(5):
        _, _, rows, pairs = measure(component=104 + offset, ordinal=offset * 2)
        all_rows.extend(rows)
        all_pairs.extend(pairs)
    result = readout.aggregate_rows(all_rows, all_pairs)
    assert (
        result["planned_cases"] == 55 and result["planned_pairs"] == 30 and result["source_components"] == 5
    )
    pair = result["paired_comparisons"][0]
    assert pair["component_medians"]["components"] == 5 and pair["component_medians"]["median"] > 0
    text = json.dumps(result, allow_nan=False)
    for private in (
        "clip_ordinal",
        "source_slot",
        "component_index",
        "case_number",
        "full_support_times",
        "shared_support_times",
        "rgb_sha256",
        "Authored private cause",
    ):
        assert private not in text


def test_public_aggregation_refuses_private_reason_or_status_text():
    _, _, rows, pairs = measure(unavailable="brightness_liveportrait")
    rows[-1]["unavailable_reason"] = "authored private path or text"
    with pytest.raises(ValueError, match="Only fixed aggregate status keys"):
        readout.aggregate_rows(rows, pairs)


def test_only_fixed_data_monitor_worker_is_routed(monkeypatch, tmp_path):
    monkeypatch.setattr(readout, "TRIAL", tmp_path)
    actual = []

    def popen(argv, **kwargs):
        actual.append(argv)
        return SimpleNamespace()

    monkeypatch.setattr(readout.subprocess, "Popen", popen)
    for target in (4, "summary"):
        output = readout.stage_folder(target) / "output"
        readout.routed_popen(
            target,
            ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"],
        )
        assert actual[-1][3:] == [
            "scripts.grid_mouth_motion_readout_v158",
            "_worker",
            str(target),
            str(output),
        ]
        with pytest.raises(ValueError, match="exclusive data-only"):
            readout.routed_popen(
                target,
                ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
            )
