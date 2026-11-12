"""Authored-source and altered-evidence checks for the standalone reviewer."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.review_evidence_measurements import _scheduled_indices, review

CLI = Path(__file__).resolve().parents[1] / "scripts" / "review_evidence_measurements.py"


def _document(kind, source, geometry, packet, *, version=None):
    return {"schema": f"chipmunk.{kind}-evidence/1", "status": "ready", "version": version,
            "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), **geometry},
            "packets": [packet] if packet is not None else []}


def _write(tmp_path, document):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document))
    return path


def test_png_recomputes_unclipped_color_step_from_original_and_refuses_changed_score(tmp_path):
    src = tmp_path / "original.png"
    with Image.new("RGB", (512, 512), (11, 16, 33)) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 15, 130, 142), fill=(31, 36, 53))
        # The first comparison step at x=0 must use the missing-neighbor value,
        # rather than a read wrapping around to this unrelated last column.
        draw.line((511, 0, 511, 511), fill=(81, 86, 103))
        image.save(src)
    geometry = {"coordinate_space": "exif-oriented-original", "exif_orientation": 1, "width": 512, "height": 512}
    response = round(math.log(23 / 3), 6)
    packet = {"id": "E01", "title": "Sharp boundary to review", "evidence_type": "anomaly", "semantics": "residual",
              "region": {"x": 3, "y": 15, "w": 128, "h": 128}, "anchor": {
                  "original_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                  "coordinate_space": geometry["coordinate_space"], "exif_orientation": 1,
                  "region_to_original": [1, 0, 0, 0, 1, 0], "map_grid": [512, 512]},
              "measurements": [{"label": label, "value": response, "unit": "relative response"} for label in
                               ("Left side", "Top side", "Right side", "Bottom side", "Combined boundary response")]}
    document = _document("image", src, geometry, packet)
    result = review("image", src, _write(tmp_path, document))
    assert result["status"] == "matched"
    assert result["packets"] == [{"packet": "E01", "relative_responses_reproduced": 5,
                                  "maximum_absolute_difference": pytest.approx(abs(response - math.log(23 / 3)))}]

    packet["measurements"][0]["value"] += .01
    with pytest.raises(ValueError, match="Reported Left side differs"):
        review("image", src, _write(tmp_path, document))
    packet["region"]["x"] = 2
    with pytest.raises(ValueError, match="native search geometry"):
        review("image", src, _write(tmp_path, document))


def test_no_recorded_measurement_is_explicit_and_wrong_original_is_refused(tmp_path):
    src = tmp_path / "source.bin"
    src.write_bytes(b"source")
    evidence = _write(tmp_path, _document("image", src, {}, None))
    result = review("image", src, evidence)
    assert result["status"] == "no_measurement_recorded" and result["packets"] == []
    src.write_bytes(b"other source")
    with pytest.raises(ValueError, match="source digest"):
        review("image", src, evidence)


def test_exported_image_views_reproduce_resized_pixels_and_reject_tampering(tmp_path):
    source = tmp_path / "source.jpg"
    with Image.new("RGB", (1024, 1536), (44, 65, 139)) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((13, 32, 670, 799), fill=(129, 95, 188))
        draw.rectangle((643, 1043, 950, 1480), fill=(47, 11, 64))
        image.save(source, quality=92)
    crops = []
    with Image.open(source) as original, original.resize((576, 864), Image.Resampling.LANCZOS) as resized:
        for number, (row, y, height) in enumerate(((0, 0, 939), (168, 298, 940), (336, 597, 939)), 1):
            plate = tmp_path / f"crop-{number}.png"
            with resized.crop((24, row, 552, row + 528)) as view:
                view.save(plate)
            digest = hashlib.sha256(plate.read_bytes()).hexdigest()
            plate.rename(tmp_path / f"{digest}.png")
            crops.append({"region": number, "bounds": {"x": 42, "y": y, "w": 940, "h": height}, "sha256": digest})
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = {"source_sha256": digest, "status": "flagged", "crops": crops, "authorship_confidence_contribution": 0}
    document = {"kind": "image", "sha256": digest, "whole_image_review": record}
    evidence = _write(tmp_path, document)
    completed = subprocess.run([sys.executable, "-I", str(CLI), "image-views", str(source), str(evidence),
                                "--assets", str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "matched"
    assert [view["rgb_pixels_reproduced"] for view in result["inspected_views"]] == [278784] * 3
    crops[0]["bounds"]["x"] += 1
    with pytest.raises(ValueError, match="different source coordinates"):
        review("image-views", source, _write(tmp_path, document), asset_dir=tmp_path)
    crops[0]["bounds"]["x"] -= 1
    crops[0]["sha256"] = crops[1]["sha256"]  # A valid, correctly named plate from the wrong source position.
    with pytest.raises(ValueError, match="regenerated source pixels"):
        review("image-views", source, _write(tmp_path, document), asset_dir=tmp_path)
    record.update(status="skipped", crops=[])
    result = review("image-views", source, _write(tmp_path, document), asset_dir=tmp_path)
    assert result["status"] == "skipped" and result["inspected_views"] == []
    source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="supplied image digest"):
        review("image-views", source, evidence, asset_dir=tmp_path)


def test_schedule_resolves_nearest_ties_and_duplicate_pts_in_source_order():
    pts = [-5, -4, -2, -2, -1, 0, 1, 4, 5, 8, 11, 15, 18]
    base = Fraction(1, 24)
    low, high = pts[1] * base - Fraction(5, 8), pts[10] * base + Fraction(5, 8)
    targets = [(pts[0] + tick) * base for tick in range(0, pts[-1] - pts[0] + 1, 2)]
    expected = sorted({min(range(len(pts)), key=lambda i: abs(pts[i] * base - target)) for target in targets})
    expected = [i for i in expected if low <= pts[i] * base <= high]
    assert 1 in expected  # -3 is equally close to -4 and -2.
    assert 2 not in expected and 3 not in expected
    assert _scheduled_indices(pts, base, pts[1], pts[10]) == expected


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="native FFmpeg tools unavailable")
def test_video_recomputes_schedule_and_rejects_changed_clock_and_display_roles(tmp_path):
    source = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=black:size=32x32:rate=24",
                    "-frames:v", "48", "-c:v", "ffv1", "-threads", "1", str(source)], check=True, timeout=20)
    # Matroska ticks round to milliseconds. The 12 Hz schedule selects every
    # other native frame for this authored constant-rate clock.
    pts = [round(Fraction(1000 * i, 24)) for i in range(48)]
    schedule = list(range(0, 40, 2))
    roles = {"before": 10, "during": 18, "after": 24}
    packet = {
        "id": "V01",
        "face_sequence": {"start_pts": pts[12], "end_pts": pts[24], "track_id": "T1",
                          "centers": {"during": [12, 18, 24]}},
        "face_boundary": {"method": "face-appearance-boundary/1", "time_base": "1/1000", "track_id": "T1",
                          "scheduled_frame_indices": schedule,
                          "frame_inputs": [{"frame_index": i, "pts": pts[i]} for i in schedule],
                          "edges": {"entry": {"frame_indices": [8, 10, 12, 14]},
                                    "exit": {"frame_indices": [20, 22, 24, 26]}},
                          "start_pts": pts[12], "end_pts": pts[24], "anchors": roles},
        "face_boundary_frames": [{"interval_role": role, "anchor": {"frame_index": i, "pts": pts[i]}}
                                 for role, i in roles.items()],
    }
    document = _document("video", source, {"frame_count": len(pts), "time_base": "1/1000", "stream_selector": "v:0"},
                         packet, version="video-evidence-11")
    evidence = _write(tmp_path, document)
    completed = subprocess.run([sys.executable, "-I", str(CLI), "video", str(source), str(evidence)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["packets"] == [{
        "packet": "V01", "scheduled_frames_reproduced": 20, "measured_inputs_checked": 20,
        "displayed_frame_roles_checked": 3, "refined_start_pts": 500, "refined_end_pts": 1000,
        "time_base": "1/1000",
    }]

    packet["face_boundary"]["scheduled_frame_indices"] = [i for i in schedule if i != 4]
    with pytest.raises(ValueError, match="schedule differs from the source clock"):
        review("video", source, _write(tmp_path, document))
    packet["face_boundary"]["scheduled_frame_indices"] = schedule
    packet["face_boundary"]["frame_inputs"][0]["pts"] = 1
    with pytest.raises(ValueError, match="inputs do not match"):
        review("video", source, _write(tmp_path, document))
    packet["face_boundary"]["frame_inputs"][0]["pts"] = 0
    packet["face_boundary_frames"][0]["anchor"]["frame_index"] = 8
    with pytest.raises(ValueError, match="frame roles differ"):
        review("video", source, _write(tmp_path, document))
