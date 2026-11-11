from pathlib import Path

import numpy as np
import pytest

from scripts import image_bg_real_jpeg_runtime_v229 as monitor
from scripts import image_bg_real_jpeg_v229 as study


def artificial_pixels():
    source = np.random.default_rng(6).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    edited = source.copy()
    edited[18, 21] ^= 255
    semantic = np.zeros_like(source)
    semantic[14:28, 15:35] = 255
    exact = np.any(source != edited, axis=2)
    return source, edited, semantic, exact


def test_exact_pixel_truth_and_ordinary_controls_use_distinct_regions():
    source, edited, semantic, exact = artificial_pixels()
    images, footprints = study.compose_views(source, edited, semantic, exact)
    assert tuple(images) == tuple(footprints) == study.KINDS
    assert np.array_equal(footprints["ai"], exact)
    assert footprints["ai"].sum() == 1
    assert footprints["original"].sum() == 0
    for kind in study.KINDS[2:]:
        footprint = footprints[kind]
        assert np.array_equal(footprint, np.any(source != np.asarray(images[kind]), axis=2))
        assert footprint.sum() > 1
        assert not np.any(footprint & (semantic[:, :, 0] == 0))
    for quality in study.QUALITIES:
        payload, decoded = study.original.jpeg_view(images["ai"], quality)
        assert study.original.view_binding(payload, decoded)["source_size"] == [64, 64]


def test_source_truth_fails_closed_on_partial_support_or_inexact_labels():
    source, edited, semantic, exact = artificial_pixels()
    with pytest.raises(ValueError, match="Inconsistent exact edit or support"):
        study.compose_views(source, edited, semantic, np.zeros_like(exact))
    edited[0, 0] ^= 255
    with pytest.raises(ValueError, match="Inconsistent exact edit or support"):
        study.compose_views(source, edited, semantic, np.any(source != edited, axis=2))


def test_five_batches_and_unchanged_v189_algorithms():
    batches = [study.tasks(phase) for phase in range(study.PHASES)]
    assert list(map(len, batches)) == [25, 25, 25, 25, 20]
    assert [item for batch in batches for item in batch] == [
        (n, kind) for n in range(24) for kind in study.KINDS
    ]
    assert study.original.PHOTOS == 59
    assert study.original.PHASES == 12
    assert study.score.__code__ is study.original.score.__code__
    assert study.readout.__code__ is study.original.readout.__code__
    assert study.seal.__code__ is study.original.seal.__code__
    assert study.batch_views.__code__ is study.original.batch_views.__code__
    assert monitor.run.__code__ is monitor.original.run.__code__
    assert monitor.LIMITS == monitor.original.LIMITS


def test_published_qualification_and_policy_bindings():
    summary, policy, inherited = study.public_contract()
    assert summary["qualified"] == 24
    assert summary["qualified_by_publisher_split"] == {
        "test_background_ood": 7,
        "test_id": 7,
        "test_subject_ood": 5,
        "val": 5,
    }
    assert policy["sensitive_policy"] == "p0.90-a0.01000"
    assert "backend/scripts/image_jpeg_diagnostic_v189.py" in inherited
    assert "backend/scripts/image_bg_real_transport_v225.py" in inherited


def test_profiles_and_completed_journals_are_v229_specific(tmp_path):
    name = "baseline-q75-00"
    command = "score"
    expected = "a" * 64
    arguments = [command, str(tmp_path), expected, "baseline", "75", "0"]
    folder = tmp_path / f"{name}-profile"
    folder.mkdir()
    resources = folder / "resources.jsonl"
    resources.write_text("{}\n")
    profile = {
        "command": command,
        "worker_arguments": arguments,
        "worker": monitor.MODULE,
        "limits": monitor.LIMITS,
        "monitor_sha256": study.sha(Path(monitor.__file__)),
        "resource_sha256": study.sha(resources),
        "exit_code": 0,
        "failure": None,
        "duration_seconds": 1,
    }
    profile_file = folder / "profile.json"
    monitor.profile_write(profile_file, profile)
    part = tmp_path / name
    part.mkdir()
    journal = part / "rows.jsonl"
    journal.write_text('{"n": 0}\n')
    finished = {"freeze_sha256": expected, "rows_sha256": study.sha(journal)}
    study.write(part / "completed.json", finished)
    assert study.phase_rows(tmp_path, name, command, arguments, expected) == (finished, [{"n": 0}])
    profile_file.unlink()
    study.write(profile_file, {**profile, "schema": "chipmunk.image-jpeg-diagnostic-profile/189"})
    with pytest.raises(ValueError, match="Exact v229 resource profile required"):
        study.phase_rows(tmp_path, name, command, arguments, expected)
    profile_file.unlink()
    monitor.profile_write(profile_file, profile)
    journal.write_text('{"n": 1}\n')
    with pytest.raises(ValueError, match="Completed phase or row journal changed"):
        study.phase_rows(tmp_path, name, command, arguments, expected)


def test_seal_uses_all_matched_batches_and_keeps_source_keys_private(tmp_path, monkeypatch):
    expected = "a" * 64
    sources = [{"source_id": f"private-source-{n}", "component": f"group-{n}"} for n in range(24)]
    inputs = {"sources": [{"changed_pixels": {kind: int(kind != "original") for kind in study.KINDS}}] * 24}
    study.write(
        tmp_path / "freeze.json",
        {
            "unique_model_calls": 480,
            "exact_first_input_repeats": 4,
            "qualification_sha256": "b" * 64,
            "v189_policy_freeze_sha256": "c" * 64,
            "publisher_split_counts": {"test": 24},
        },
    )
    prepare_profile = tmp_path / "prepare-profile"
    prepare_profile.mkdir()
    (prepare_profile / "profile.json").touch()
    visited = []

    def record(name):
        profile = tmp_path / f"{name}-profile"
        profile.mkdir()
        (profile / "profile.json").touch()
        folder = tmp_path / name
        folder.mkdir()
        study.write(folder / "completed.json", {"rows_sha256": "d" * 64})
        visited.append(name)

    def fake_producer(trial, digest, arm, quality, phase):
        assert trial == tmp_path and digest == expected
        record(f"{arm}-q{quality}-{phase:02d}")

    def fake_readout(trial, name, command, args, digest):
        assert trial == tmp_path and digest == expected and command == "readout"
        quality, phase = map(int, args[-2:])
        record(name)
        rows = []
        for n, kind in study.tasks(phase):
            found = kind == "ai" and n < 12
            metrics = {
                "iou": int(found),
                "recall": int(found),
                "intersection": int(found),
                "predicted_positive_pixels": int(found),
                "flagged_area_fraction": int(found) / 64**2,
            }
            rows.append(
                {
                    "n": n,
                    "kind": kind,
                    "quality": quality,
                    "component": sources[n]["component"],
                    "changed_pixels": int(kind != "original"),
                    "methods": {
                        method: {"ai": metrics, "ordinary": None} for method in study.original.METHODS
                    },
                }
            )
        return {"quality": quality, "phase": phase, "neural_calls": 0, "views": len(rows)}, rows

    monkeypatch.setitem(study.seal.__globals__, "lease", lambda *_: (inputs, sources, None))
    monkeypatch.setitem(study.seal.__globals__, "prep_profile", lambda _: prepare_profile)
    monkeypatch.setitem(study.seal.__globals__, "producer_rows", fake_producer)
    monkeypatch.setitem(study.seal.__globals__, "phase_rows", fake_readout)
    study.seal(tmp_path, expected)
    result_path = tmp_path / "summary.json"
    result = study.read(result_path)
    assert result["schema"] == "chipmunk.image-bg-real-jpeg-comparison-result/229"
    assert len(visited) == len(set(visited)) == len(result["phase_output_sha256"]) == 30
    assert len(result["phase_profile_sha256"]) == 31
    assert "private-source-" not in result_path.read_text()
    for quality in study.QUALITIES:
        row = result["quality_results"][str(quality)]
        assert row["photos"] == row["source_groups"] == 24
        assert row["methods"]["sensitive"]["edited_photos_usefully_localized"] == 12
        assert row["methods"]["shared"]["paired_against_current"]["useful"]["difference"] == 0
