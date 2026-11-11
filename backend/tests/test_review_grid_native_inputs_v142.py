import copy
import hashlib
import json
import struct
import wave
import zlib
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image

from scripts import review_grid_native_inputs_v142 as review


def mapping(samples=4, origin=160):
    return {
        "timeline_time_base": "1/16000",
        "sample_rate": 16000,
        "channels": 1,
        "sample_format": "s16le",
        "samples": samples,
        "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": origin}],
    }


def clock(samples=4, origin=160):
    return {
        "time_base": "1/16000",
        "sample_rate": 16000,
        "channels": 1,
        "frames": [
            {
                "pts": origin,
                "samples": samples,
                "start": str(Fraction(origin, 16000)),
                "end": str(Fraction(origin + samples, 16000)),
            }
        ],
        "missing_frame_clocks": 0,
        "gap_count": 0,
        "overlap_count": 0,
        "clock_qualified": True,
        "contiguous": True,
    }


def pcm_fixture(tmp_path):
    wav = tmp_path / "authored.wav"
    raw = struct.pack("<hhhh", -32768, -1, 0, 32767)
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 16000, 4, "NONE", "not compressed"))
        stream.writeframes(raw)
    pcm = np.asarray([-1, -1 / 32768, 0, 32767 / 32768], dtype="<f4")
    array = tmp_path / "pcm.npy"
    np.save(array, pcm, allow_pickle=False)
    mapped = mapping()
    mapped.update(
        pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
        source_sha256="source",
        source_sample_rate=16000,
        source_channels=1,
        source_time_base="1/16000",
    )
    return {
        "wav_file": str(wav),
        "wav_file_sha256": review.sha(wav),
        "pcm_file": str(array),
        "pcm_file_sha256": review.sha(array),
        "pcm_mapping": mapped,
        "source_sha256": "source",
        "native_audio_clock": clock(),
        "audio_origin_seconds": "1/100",
        "whole_pcm_source_ranges": review.mapping_ranges(mapped, 4),
    }


def test_independent_wave_decode_matches_saved_float32_and_nonzero_source_clock(tmp_path):
    record = pcm_fixture(tmp_path)
    count, digest = review.verify_pcm(record)
    assert count == 4 and digest == record["pcm_mapping"]["pcm_sha256"]


@pytest.mark.parametrize("damage", ["sample", "dtype", "hash", "origin", "native_end", "range"])
def test_pcm_replay_rejects_wrong_values_clocks_and_mapping(tmp_path, damage):
    row = pcm_fixture(tmp_path)
    if damage in {"sample", "dtype"}:
        values = np.load(row["pcm_file"], allow_pickle=False)
        if damage == "sample":
            values[1] = 0.5
        else:
            values = values.astype("<f8")
        np.save(row["pcm_file"], values, allow_pickle=False)
        row["pcm_file_sha256"] = review.sha(row["pcm_file"])
    elif damage == "hash":
        row["pcm_mapping"]["pcm_sha256"] = "wrong"
    elif damage == "origin":
        row["audio_origin_seconds"] = "0"
    elif damage == "native_end":
        row["native_audio_clock"]["frames"][0]["end"] = "99/100"
    else:
        row["whole_pcm_source_ranges"][0]["source_end_pts"] += 1
    with pytest.raises(ValueError):
        review.verify_pcm(row)


@pytest.mark.parametrize("kind", ["gap", "source_gap", "overlap", "missing_end", "float"])
def test_complete_mapping_cannot_hide_missing_or_discontinuous_samples(kind):
    value = mapping()
    value["timeline"] = [
        {"start_sample": 0, "end_sample": 2, "source_pts": 160},
        {"start_sample": 2, "end_sample": 4, "source_pts": 162},
    ]
    if kind == "gap":
        value["timeline"][1]["start_sample"] = 3
    elif kind == "source_gap":
        value["timeline"][1]["source_pts"] = 163
    elif kind == "overlap":
        value["timeline"][1]["start_sample"] = 1
    elif kind == "missing_end":
        value["timeline"].pop()
    else:
        value["timeline"][0]["source_pts"] = 160.0
    with pytest.raises(ValueError):
        review.mapping_ranges(value, 4)


def test_native_audio_rebuilds_rational_intervals_with_publisher_tick_quantization():
    value = clock(1152, 0)
    value.update(time_base="1/90000", sample_rate=44100, channels=2)
    value["frames"] = [
        {
            "pts": i,
            "samples": 1152,
            "start": str(Fraction(i, 90000)),
            "end": str(Fraction(i, 90000) + Fraction(1152, 44100)),
        }
        for i in (0, 2351, 4702)
    ]
    assert review.native_audio(value)[0] == 0
    value["frames"][1]["pts"] += 1
    with pytest.raises(ValueError, match="rational"):
        review.native_audio(value)


def test_native_audio_preserves_and_checks_unavailable_discontinuous_status():
    value = clock(100, 0)
    value["frames"].append({"pts": 200, "samples": 100, "start": "1/80", "end": "3/160"})
    value.update(gap_count=1, contiguous=False)
    assert review.native_audio(value) == (0, Fraction(3, 160))
    value["contiguous"] = True
    with pytest.raises(ValueError, match="continuity"):
        review.native_audio(value)


def video_fixture():
    table = {
        "pts": [48600, 51600, 54600],
        "time_base": "1/90000",
        "width": 3,
        "height": 2,
        "interpolated": [],
        "truncated": False,
    }
    pictures = [{"frame_index": i, "pts": p} for i, p in enumerate(table["pts"])]
    return table, pictures


@pytest.mark.parametrize("damage", ["pts", "index", "drop", "interpolated", "float"])
def test_video_replay_uses_observed_rational_pts_not_index_fps(damage):
    table, pictures = video_fixture()
    assert review.video_clock(table, pictures) == Fraction(27, 50)
    if damage == "pts":
        pictures[1]["pts"] += 1
    elif damage == "index":
        pictures[1]["frame_index"] = 2
    elif damage == "drop":
        pictures.pop()
    elif damage == "interpolated":
        table["interpolated"] = [1]
    else:
        table["time_base"] = 1 / 90000
    with pytest.raises(ValueError):
        review.video_clock(table, pictures)


def test_partial_unavailable_video_retains_saved_prefix_without_becoming_available():
    table, pictures = video_fixture()
    assert review.video_clock(table, pictures[:1], complete=False) == Fraction(27, 50)
    with pytest.raises(ValueError):
        review.video_clock(table, pictures[:1], complete=True)


def test_unsupported_video_table_is_an_unavailable_observation_not_an_original():
    table, _pictures = video_fixture()
    table["pts"] = [48600, 48000, 54600]
    assert review.supported_video(table) is False
    table["pts"] = []
    assert review.supported_video(table) is False


def test_png_checks_file_bytes_and_independent_rgb_bytes(tmp_path):
    path = tmp_path / "authored.png"
    Image.new("RGB", (3, 2), (1, 2, 3)).save(path)
    raw = bytes([1, 2, 3]) * 6
    record = {"file_sha256": review.sha(path), "rgb_sha256": hashlib.sha256(raw).hexdigest()}
    assert review.verify_png(path, record, (3, 2)) == raw
    Image.new("RGB", (3, 2), (1, 2, 4)).save(path)
    record["file_sha256"] = review.sha(path)  # Correct outer hash cannot conceal wrong pixels.
    with pytest.raises(ValueError, match="RGB"):
        review.verify_png(path, record, (3, 2))


def test_saved_mpg_is_hashed_and_crc_checked_without_a_decoder(tmp_path):
    path = tmp_path / "source.mpg"
    payload = b"authored bytes; deliberately not decodable media" * 40
    path.write_bytes(payload)
    member = {"bytes": len(payload), "crc32": zlib.crc32(payload) & 0xFFFFFFFF}
    review.verify_member(path, member, review.sha(path))
    with pytest.raises(ValueError, match="CRC"):
        review.verify_member(path, {**member, "crc32": member["crc32"] ^ 1}, review.sha(path))


def selection_fixture():
    directories, archives, selected = [], [], []
    for ordinal in range(16):
        source = {"ordinal": ordinal, "publisher_name": f"authored{ordinal}.zip"}
        members = [
            {"name": f"source{ordinal}/word{ordinal}_{j}.mpg", "bytes": 2000, "crc32": j} for j in range(2)
        ]
        archives.append({**source, "real_video_members": members})
        directories.append(
            {"source": source, "media": members, "full_archive_checksum_verified_by_this_stage": False}
        )
        order = sorted(
            members,
            key=lambda m: hashlib.sha256(
                ("chipmunk.grid-clips/134:" + m["name"].split("/")[-1][:-4]).encode()
            ).hexdigest(),
        )
        for j, member in enumerate(order):
            selected.append(
                {
                    "ordinal": len(selected),
                    "source_ordinal": ordinal,
                    "within_source_ordinal": j,
                    "member": member,
                    "source": source,
                    "study_role": "external_transfer",
                    "media_qualified": False,
                }
            )
    return selected, directories, archives


@pytest.mark.parametrize("damage", [None, "drop", "source", "member", "order"])
def test_full32_fixed_selection_rejects_substitution_or_dropping(damage):
    selected, directories, archives = selection_fixture()
    if damage is None:
        review.validate_selection(selected, directories, archives)
        return
    selected = copy.deepcopy(selected)
    if damage == "drop":
        selected.pop()
    elif damage == "source":
        selected[2]["source_ordinal"] = 0
    elif damage == "member":
        selected[2]["member"]["crc32"] ^= 1
    else:
        selected[0], selected[1] = selected[1], selected[0]
    with pytest.raises(ValueError):
        review.validate_selection(selected, directories, archives)


def test_private_exact_content_and_source_slot_components_do_not_invent_independence():
    rows = [
        {"source_slot": 0, "video_content_sha256": "A", "pcm_content_sha256": "x"},
        {"source_slot": 0, "video_content_sha256": "B", "pcm_content_sha256": "y"},
        {"source_slot": 1, "video_content_sha256": "C", "pcm_content_sha256": "y"},
        {"source_slot": 1, "video_content_sha256": None, "pcm_content_sha256": None},
        {"source_slot": 2, "video_content_sha256": None, "pcm_content_sha256": None},
    ]
    result = review.components(rows)
    assert result["duplicate_groups"]["pcm_content_sha256"] == [[1, 2]]
    assert sorted(result["components"], key=len) == [[4], [0, 1, 2, 3]]
    assert "not people" in result["meaning"]


def test_only_fixed_data_review_argv_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(review.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    argv = [
        "python",
        "-B",
        "-m",
        "scripts.video_appearance_native_v108",
        "bind",
        str(review.TRIAL / "bind-00/output"),
        "0",
    ]
    review.routed_popen(argv)
    assert calls[0][3:] == ["scripts.review_grid_native_inputs_v142", "review", argv[5]]
    for wrong in ([*argv[:4], "extract", *argv[5:]], [*argv[:-1], "1"]):
        with pytest.raises(ValueError):
            review.routed_popen(wrong)


def test_data_stage_requires_actual_receipt_resource_and_journal_bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    script = tmp_path / "backend/scripts/authored.py"
    script.parent.mkdir(parents=True)
    script.write_text("# authored caller\n")
    folder = tmp_path / "stage"
    (folder / "output").mkdir(parents=True)
    result = {
        "status": "complete",
        "freeze_sha256": "frozen",
        "new_neural_calls": 0,
        "defaults_changed": False,
    }
    review.write(folder / "output/result.json", result)
    resource = {
        "elapsed_s": 0.1,
        "tree_rss_bytes": 12,
        "ram_available_bytes": 4 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "archive_free_disk_bytes": 4 * 1024**3,
        "violations": [],
    }
    (folder / "resources.jsonl").write_text(json.dumps(resource) + "\n")
    profile = {
        "schema": "chipmunk.video-appearance-resource-profile/108",
        "duration_s": 0.2,
        "child_started": True,
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "error_types": [],
        "resource_limit_observations": [],
        "command": "bind",
        "batch": 0,
        "new_neural_calls": 0,
        "worker_receipt_sha256": review.sha(folder / "output/result.json"),
        "resource_journal_sha256": review.sha(folder / "resources.jsonl"),
        "runtime_sha256": review.sha(script),
        "sampled_tree_rss_lower_bound_bytes": 12,
        "minimum_available_ram_bytes": 4 * 1024**3,
        "terminal_observations": [{"classification": "verified_worker_terminal_transition"}],
    }
    review.write(folder / "profile.json", profile)
    assert review.stage(folder, "frozen", "authored.py", 0, {}) == result
    result["new_neural_calls"] = 1
    (folder / "output/result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="complete bound"):
        review.stage(folder, "frozen", "authored.py", 0, {})


@pytest.mark.parametrize("damage", ["boolean", "float_bytes", "infinite", "backwards", "duration"])
def test_resource_profile_cannot_rehash_invalid_or_nonobserved_samples(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(review, "ROOT", tmp_path)
    script = tmp_path / "backend/scripts/authored.py"
    script.parent.mkdir(parents=True)
    script.write_text("# authored caller\n")
    folder = tmp_path / "stage"
    (folder / "output").mkdir(parents=True)
    review.write(
        folder / "output/result.json",
        {"status": "complete", "freeze_sha256": "frozen", "new_neural_calls": 0, "defaults_changed": False},
    )
    row = {
        "elapsed_s": 0.1,
        "tree_rss_bytes": 12,
        "ram_available_bytes": 4 * 1024**3,
        "disk_free_bytes": 4 * 1024**3,
        "archive_free_disk_bytes": 4 * 1024**3,
        "violations": [],
    }
    rows = [row, {**row, "elapsed_s": 0.15}]
    if damage == "boolean":
        row["tree_rss_bytes"] = False
    elif damage == "float_bytes":
        row["tree_rss_bytes"] = 12.0
    elif damage == "infinite":
        row["ram_available_bytes"] = float("inf")
    elif damage == "backwards":
        rows[1]["elapsed_s"] = 0.05
    (folder / "resources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    review.write(
        folder / "profile.json",
        {
            "schema": "chipmunk.video-appearance-resource-profile/108",
            "duration_s": 0.01 if damage == "duration" else 0.2,
            "child_started": True,
            "child_exit_code": 0,
            "guard": None,
            "sample_errors": 0,
            "error_types": [],
            "resource_limit_observations": [],
            "command": "bind",
            "batch": 0,
            "new_neural_calls": 0,
            "worker_receipt_sha256": review.sha(folder / "output/result.json"),
            "resource_journal_sha256": review.sha(folder / "resources.jsonl"),
            "runtime_sha256": review.sha(script),
            "sampled_tree_rss_lower_bound_bytes": max(r["tree_rss_bytes"] for r in rows),
            "minimum_available_ram_bytes": min(r["ram_available_bytes"] for r in rows),
        },
    )
    with pytest.raises(ValueError):
        review.stage(folder, "frozen", "authored.py", 0, {})
