"""Native clocks, decoded aliases, missing checks and immutable source admission."""

import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from scripts import video_native_runtime_v92 as runtime
from scripts import video_native_sources_v92 as audit


def test_native_audio_uses_samples_and_rational_clock_not_nominal_fps():
    stream = {"sample_rate": "16000", "time_base": "1/1000", "channels": 1}
    frames = [
        {"pts": 125, "nb_samples": 1600},
        {"pts": "N/A", "best_effort_timestamp": 225, "nb_samples": 1600},
    ]
    result = audit.audio_clocks(stream, frames)
    assert result["clock_qualified"] and result["contiguous"]
    assert result["frames"][0]["start"] == "1/8" and result["frames"][-1]["end"] == "13/40"
    frames[1]["best_effort_timestamp"] = 425
    result = audit.audio_clocks(stream, frames)
    assert result["clock_qualified"] and not result["contiguous"] and result["gap_count"] == 1


@pytest.mark.parametrize(
    "second", [{"pts": 200, "nb_samples": 1600}, {"nb_samples": 1600}, {"pts": 225, "nb_samples": 0}]
)
def test_missing_or_overlapping_audio_support_is_not_qualified(second):
    result = audit.audio_clocks(
        {"sample_rate": "16000", "time_base": "1/1000", "channels": 1},
        [{"pts": 125, "nb_samples": 1600}, second],
    )
    assert not result["clock_qualified"] and not result["contiguous"]


def test_aliases_require_three_distinct_consecutive_informative_frames():
    assert not audit.triplets(["a", "a", "a"])
    assert not audit.triplets(["a", "b", "c"], [True, False, True])
    assert audit.triplets(["a", "b", "c", "d"]) & audit.triplets(["x", "a", "b", "c"])
    assert not (audit.triplets(["a", "b", "c"]) & audit.triplets(["a", "c", "b"]))
    with pytest.raises(ValueError, match="flags"):
        audit.triplets(["a"], [])


def record(group, role, frames=("a", "b", "c")):
    return {
        "group": group,
        "provisional_role": role,
        "status": "decoded",
        "video_clock_qualified": True,
        "three_frame_aliases": list(audit.triplets(list(frames))),
        "full_rgb_sequence_sha256": "\0".join(frames),
        "canonical_audio": None,
    }


def test_decoded_cross_role_aliases_are_joined_before_roles_are_finalized():
    records = [
        record("g1", "fusion_fit"),
        record("g2", "confirmation"),
        record("g3", "selection", ("d", "e", "f")),
    ]
    assignments, graph = audit.refine(records, [])
    assert graph["components"] == 2 and graph["all_roles_reassigned"]
    assert assignments["g1"]["component"] == assignments["g2"]["component"]
    assert assignments["g1"]["role"] == assignments["g2"]["role"]
    assert assignments["g3"]["component"] != assignments["g1"]["component"]


def test_historical_content_and_any_failed_member_exclude_the_entire_component():
    records = [
        record("g1", "fusion_fit"),
        record("g2", "fusion_fit"),
        record("g3", "selection", ("d", "e", "f")),
    ]
    assignments, graph = audit.refine(records, [["x", "a", "b", "c", "y"]])
    assert not assignments["g1"]["eligible"] and not assignments["g2"]["eligible"]
    assert assignments["g3"]["eligible"] and graph["historical_rgb_connected_groups"] == 2
    records += [
        {"group": "g3", "provisional_role": "selection", "status": "failed", "video_clock_qualified": False}
    ]
    assignments, graph = audit.refine(records, [])
    assert not assignments["g3"]["eligible"] and graph["excluded_source_groups"] == 1


def test_static_video_and_silent_audio_do_not_create_identity_links():
    first, second = record("g1", "fusion_fit", ("a", "a", "a")), record("g2", "confirmation", ("a", "a", "a"))
    for row in [first, second]:
        row["canonical_audio"] = {
            "pcm_sha256": "same-silence",
            "informative_block_count": 0,
            "three_block_aliases": [],
        }
    assignments, graph = audit.refine([first, second], [])
    assert graph["components"] == 2 and not graph["all_roles_reassigned"]
    assert assignments["g1"]["eligible"] and assignments["g2"]["eligible"]


@pytest.mark.parametrize("with_audio", [False, True])
def test_real_decoder_roundtrip_preserves_irregular_pts_and_audio_applicability(
    tmp_path, monkeypatch, with_audio
):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    path = tmp_path / "authored.mkv"
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=8:duration=1"]
    if with_audio:
        argv += ["-f", "lavfi", "-i", "sine=frequency=731:sample_rate=16000:duration=1.25"]
    argv += [
        "-vf",
        r"setpts=if(gte(N\,4)\,PTS+0.25/TB\,PTS)",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        "-threads",
        "1",
    ]
    if with_audio:
        argv += ["-c:a", "pcm_s16le"]
    subprocess.run([*argv, str(path)], check=True, capture_output=True, timeout=30)
    monkeypatch.setattr(audit, "MEDIA", tmp_path)
    row = {
        "index": 0,
        "group": "fixture",
        "provisional_role": "selection",
        "source_family": "authored-test",
        "kind": "genuine",
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": audit.sha(path),
    }
    result = audit.decode_asset(row)
    assert result["video_clock_qualified"] and len(result["rgb_sha256"]) == 8
    from fractions import Fraction

    pts = result["frame_table"]["pts"]
    tb = Fraction(result["frame_table"]["time_base"])
    assert (pts[4] - pts[3]) * tb == Fraction(3, 8)
    assert result["has_audio"] is with_audio
    if with_audio:
        assert result["audio_clocks"]["clock_qualified"] and result["canonical_audio"]["samples"] == 20000
    else:
        assert result["audio_clocks"] is None and result["canonical_audio"] is None


def test_existing_audit_is_not_overwritten_or_implicitly_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    existing = tmp_path / "batch-00.jsonl"
    existing.write_text("preserved failed partial attempt\n")
    with pytest.raises(FileExistsError, match="Preserve"):
        runtime.run("qualify", 0)
    assert existing.read_text() == "preserved failed partial attempt\n"
    assert not (tmp_path / "batch-00-profile").exists()


def test_decoder_budget_is_distinct_from_unchanged_neural_limits():
    row = {
        "tree_rss_bytes": 0,
        "ram_available_bytes": 4 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "elapsed_s": 0.0,
    }
    assert runtime.violations(row, initial=True) == []
    assert "initial_ram_headroom" in runtime.original.violations(row, initial=True)
    assert "tree_rss_limit" in runtime.violations({**row, "tree_rss_bytes": 2 * 1024**3})
    assert "ram_reserve" in runtime.violations({**row, "ram_available_bytes": 1024**3})


def test_qualification_imports_no_neural_library():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys,json; import scripts.video_native_sources_v92; print(json.dumps([p for p in ('torch','transformers','onnxruntime') if p in sys.modules]))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == []


def test_changed_acquisition_plan_is_rejected_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ACQUIRED", tmp_path)
    path = tmp_path / "asset-plan.json"
    path.write_text(json.dumps({"planned_assets": 336, "media_root": str(audit.MEDIA), "assets": []}))
    assert hashlib.sha256(path.read_bytes()).hexdigest() != audit.PLAN_SHA
    with pytest.raises(ValueError, match="Exact acquired"):
        audit.assets()
