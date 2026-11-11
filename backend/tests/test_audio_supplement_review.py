from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import review_audio_supplement as review

SINGLE = "mostly_single_interferer"
OVERLAP = "overlapping_interferers"


def test_replay_counts_coverage_and_misrouting_by_reference_condition():
    report = review._demo()["summary"]
    assert report["overall"]["planned"] == 12
    single = report["reference_conditions"][SINGLE]
    overlapping = report["reference_conditions"][OVERLAP]
    genuine = single["genuine"]
    synthetic = single["synthetic"]
    assert genuine["added_flags"] == 2
    assert genuine["added_flag_groups"] == 1
    assert synthetic["planned"] == 5
    assert synthetic["supplement_scored"] == synthetic["routed"] == 4
    assert synthetic["eligible"] == 2
    assert synthetic["added_flags"] == 1
    assert synthetic["supplemented"] == {"ai": 1, "original": 1, "inconclusive": 3, "unassessed": 0}
    assert overlapping["genuine"]["observed_routes"][SINGLE] == 2
    assert overlapping["genuine"]["incumbent_assessed"] == 1
    assert overlapping["genuine"]["added_flags_by_route"] == {SINGLE: 1}
    assert overlapping["genuine"]["supplemented"]["unassessed"] == 1
    assert overlapping["synthetic"]["added_flags"] == 0
    assert report["reference_conditions"]["clean"]["synthetic"]["supplemented"]["ai"] == 1
    assert report["overall"]["genuine"]["added_flag_groups"] == 2
    assert report["reference_conditions"]["unknown"]["planned"] == 0
    aggregate = json.dumps(report)
    assert all(secret not in aggregate for secret in ("test-misrouted", "confirmation-1", "authored response units"))


@pytest.mark.parametrize("field", ["router_sha256", "incumbent_sha256", "implementation_sha256"])
def test_comparison_refuses_changed_frozen_identities(field):
    example = review._demo()
    example["confirmation"][field] = "1" * 64
    with pytest.raises(ValueError, match="frozen identities"):
        review.compare(example["manifest"], example["confirmation"], example["frozen"])


def test_confirmation_binds_the_operating_point_and_all_planned_results():
    example = review._demo()
    example["frozen"]["points"][SINGLE] = .7
    with pytest.raises(ValueError, match="frozen benchmark"):
        review.compare(example["manifest"], example["confirmation"], example["frozen"])
    example = review._demo()
    example["confirmation"]["predictions"].pop()
    with pytest.raises(ValueError, match="Every planned case"):
        review.compare(example["manifest"], example["confirmation"], example["frozen"])


def test_freeze_refuses_known_cross_split_lineage():
    example = review._demo()
    example["manifest"]["cases"][0]["lineage"] = ["linked-room"]
    example["manifest"]["cases"][-1]["lineage"] = ["linked-room"]
    example["development"]["manifest_sha256"] = review.digest(example["manifest"])
    with pytest.raises(ValueError, match="lineage leakage"):
        review.freeze(example["manifest"], example["development"], example["points"])


def test_exact_parent_source_cannot_count_as_two_groups():
    example = review._demo()
    cases = example["manifest"]["cases"]
    cases[0]["source_sha256"] = cases[1]["source_sha256"]
    example["development"]["manifest_sha256"] = review.digest(example["manifest"])
    with pytest.raises(ValueError, match="multiple source groups"):
        review.freeze(example["manifest"], example["development"], example["points"])


def test_benchmark_rejects_clean_route_and_out_of_range_calibrated_points():
    example = review._demo()
    with pytest.raises(ValueError, match="noisy routes"):
        review.freeze(example["manifest"], example["development"], {"clean": .6})
    development = example["development"]
    development.update(score_kind="calibrated", calibration_sha256="b" * 64)
    with pytest.raises(ValueError, match="noisy routes"):
        review.freeze(example["manifest"], development, {SINGLE: 1.1})


def test_isolated_cli_reproduces_demo_and_refuses_overwrite(tmp_path):
    script = Path(review.__file__).resolve()
    demo = tmp_path / "demo"
    args = [sys.executable, "-I", str(script)]
    subprocess.run([*args, "demo", str(demo)], check=True, cwd=tmp_path)
    frozen, comparison = tmp_path / "frozen.json", tmp_path / "summary.json"
    freeze_run = subprocess.run(
        [*args, "freeze", str(demo / "manifest.json"), str(demo / "development.json"),
         str(demo / "points.json"), str(frozen)],
        check=True, cwd=tmp_path, capture_output=True, text=True,
    )
    assert freeze_run.stdout.strip() == json.loads((demo / "confirmation.json").read_text())["freeze_sha256"]
    command = [*args, "compare", str(demo / "manifest.json"), str(demo / "confirmation.json"),
               str(frozen), str(comparison)]
    subprocess.run(command, check=True, cwd=tmp_path)
    assert comparison.read_bytes() == (demo / "summary.json").read_bytes()
    repeat = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert repeat.returncode == 1
    assert "File exists" in repeat.stderr
    assert comparison.read_bytes() == (demo / "summary.json").read_bytes()
