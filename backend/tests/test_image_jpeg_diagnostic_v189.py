import numpy as np
import pytest
from PIL import Image

from scripts import image_jpeg_diagnostic_runtime_v189 as runtime
from scripts import image_jpeg_diagnostic_v189 as trial
from scripts.image_native_comparison_readout_v45 import source_mask
from scripts.localization_benchmark import mask_metrics


def test_jpeg75_bytes_are_identical_and_jpeg90_preserves_geometry():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (79, 123, 3), dtype=np.uint8))
    old_bytes, old_image = trial.v28.encoded_view(image, "jpeg75")
    q75, pixels = trial.jpeg_view(image, 75)
    q90, higher = trial.jpeg_view(image, 90)
    assert q75 == old_bytes and np.array_equal(np.asarray(pixels), np.asarray(old_image))
    assert q75 != q90 and higher.size == pixels.size == image.size
    with pytest.raises(ValueError, match="Fixed JPEG"):
        trial.jpeg_view(image, 80)


def test_copy_move_preserves_exact_pixels_and_does_not_invent_changes():
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[:, 3:] = 212
    image = Image.fromarray(pixels)
    alpha = np.zeros((4, 6), dtype=np.uint8)
    alpha[:2, :2] = 255
    alpha[2, 0] = 128
    copied = trial.ordinary_copy(image, Image.fromarray(alpha))
    expected = pixels.copy()
    expected[:2, :2] = 212
    expected[2, 0] = 106
    assert np.array_equal(np.asarray(copied), expected)
    assert np.array_equal(trial.changed_pixels(image, copied), alpha > 0)
    assert not trial.changed_pixels(image, trial.ordinary_copy(image, Image.new("L", image.size))).any()


def test_ordinary_overlap_remains_zero_ai_truth_and_shared_uses_actual_pixels(monkeypatch):
    old = np.zeros((12, 12), dtype=np.float32)
    old[2:8, 2:8] = 0.95
    new = np.zeros_like(old)
    new[5:10, 5:10] = 0.95
    footprint = new > 0

    def fake_mask(raw, gate, threshold, _payload, _pixels, ai_truth):
        assert not ai_truth.any()
        return (raw > threshold) & gate

    monkeypatch.setattr("scripts.image_native_comparison_readout_v45.source_mask", fake_mask)
    monkeypatch.setattr(
        "scripts.score_splice_localizers.score_map",
        lambda raw, gate, threshold, payload, pixels, truth: {
            "metrics": mask_metrics(fake_mask(raw, gate, threshold, payload, pixels, truth), truth)
        },
    )
    metrics = trial.measure_maps(old, new, 0.6, b"", np.zeros((12, 12, 3)), "copy_move", footprint)
    assert metrics["sensitive"]["ai"]["reference_positive_pixels"] == 0
    assert metrics["sensitive"]["ordinary"]["iou"] == 1
    assert metrics["shared"]["ai"]["predicted_positive_pixels"] == 9
    assert metrics["shared"]["ordinary"]["intersection"] == 9
    closed = trial.measure_maps(old, new, 0.49, b"", np.zeros((12, 12, 3)), "copy_move", footprint)
    assert closed["shared"]["ai"]["predicted_positive_pixels"] == 0


def test_unchanged_jpeg_production_mask_is_used_on_real_byte_geometry():
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (120, 170, 3), dtype=np.uint8))
    payload, decoded = trial.jpeg_view(image, 75)
    pixels = np.asarray(decoded)
    truth = np.zeros(pixels.shape[:2], dtype=bool)
    truth[23:77, 37:101] = True
    current = np.zeros((512, 512), dtype=np.float32)
    current[100:350, 100:350] = 0.92
    proposed = np.zeros_like(truth, dtype=np.float32)
    result = trial.measure_maps(current, proposed, 0.6, payload, pixels, "ai", truth)
    expected = source_mask(current, True, 0.5, payload, pixels, truth)
    assert expected.any()
    assert result["current"]["ai"] == mask_metrics(expected, truth)
    assert result["shared"]["ai"]["predicted_positive_pixels"] == 0


def test_planned_support_and_group_paired_metrics_keep_misses_and_controls():
    support = [case for phase in range(trial.PHASES) for case in trial.tasks(phase)]
    assert len(support) == len(set(support)) == 295
    assert all(len(trial.tasks(phase)) == 25 for phase in range(11))
    assert len(trial.tasks(11)) == 20
    components = ["a", "a", "b"]
    positive = np.eye(4, dtype=bool)
    empty = np.zeros_like(positive)
    rows = []
    for n, group in enumerate(components):
        for kind in trial.KINDS:
            truth = positive if kind == "ai" else empty
            result = {
                method: {"ai": mask_metrics(empty, truth), "ordinary": None} for method in trial.METHODS
            }
            if kind == "ai" and n in {0, 2}:
                result["sensitive"]["ai"] = mask_metrics(positive, truth)
            if kind == "copy_move" and n == 0:
                result["sensitive"] = {
                    "ai": mask_metrics(positive, empty),
                    "ordinary": mask_metrics(positive, positive),
                }
            rows.append({"n": n, "kind": kind, "component": group, "quality": 75, "methods": result})
    report = trial.summarize(rows, components, 75)["methods"]["sensitive"]
    assert report["edited_photos_usefully_localized"] == 2
    assert report["group_mean_useful_fraction"] == 0.75
    assert report["paired_against_current"]["edited_iou"]["difference"] == 0.75
    assert report["controls"]["copy_move"]["flagged_groups"] == 1
    assert report["controls"]["copy_move"]["mean_iou_with_ordinary_edit_if_changed"] == 1
    assert report["controls"]["original"]["flagged_photos"] == 0
    with pytest.raises(ValueError, match="Exact full paired support"):
        trial.summarize(rows[:-1], components, 75)


def test_monitor_profile_does_not_accept_other_phase_or_mutated_resource_journal(tmp_path):
    resource = tmp_path / "resources.jsonl"
    resource.write_text("{}\n")
    arguments = ["score", "example", "sha", "baseline", "75", "0"]
    trial.write(
        tmp_path / "profile.json",
        {
            "command": "score",
            "worker_arguments": arguments,
            "worker": runtime.MODULE,
            "limits": runtime.LIMITS,
            "monitor_sha256": trial.sha(runtime.__file__),
            "resource_sha256": trial.sha(resource),
            "exit_code": 0,
            "failure": None,
            "duration_seconds": 1,
        },
    )
    runtime.assert_profile(tmp_path, "score", arguments)
    with pytest.raises(ValueError, match="bounded phase"):
        runtime.assert_profile(tmp_path, "score", arguments[:-1] + ["1"])
    resource.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="bounded phase"):
        runtime.assert_profile(tmp_path, "score", arguments)
