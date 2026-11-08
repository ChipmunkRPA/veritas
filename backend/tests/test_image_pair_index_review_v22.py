"""Real zero-point reader indexes are valid; altered headers/pixels still fail."""

import gzip
import io
import json
import struct

import numpy as np
import pytest
import test_image_pair_review_v21 as authored

from scripts import image_pair_review_runtime_v22 as runtime
from scripts import review_image_pairs_v22 as reviewer
from scripts.ranged_gzip_v21 import indexed_module


def empty_index():
    payload = b"small authored stream only" * 200
    compressed = gzip.compress(payload, mtime=0)
    module = indexed_module()
    with module.IndexedGzipFile(fileobj=io.BytesIO(compressed), mode="rb", spacing=8 * 1024**2) as reader:
        assert reader.read(len(payload)) == payload
        output = io.BytesIO()
        reader.export_index(fileobj=output)
        assert reader.raw.npoints == 0
    return compressed, payload, output.getvalue()


def test_real_zero_point_index_exports_imports_and_seeks_exact_bytes(tmp_path):
    compressed, payload, data = empty_index()
    assert len(data) == 35
    path = tmp_path / "index.gzi"
    path.write_bytes(data)
    assert reviewer.index_header(path, {"index_sha256": reviewer.sha(path), "index_points": 0}, {"bytes": len(compressed)}) == 0
    with indexed_module().IndexedGzipFile(fileobj=io.BytesIO(compressed), mode="rb") as reader:
        reader.import_index(fileobj=io.BytesIO(data))
        assert reader.raw.npoints == 0 and reader.raw.skip_crc_check
        reader.seek(len(payload) - 21)
        assert reader.read(21) == payload[-21:]
        reader.seek(0)
        assert reader.read(len(payload)) == payload


def test_actual_nonzero_index_header_uses_the_same_strict_archive_binding(tmp_path):
    payload = np.random.default_rng(221).integers(0, 256, 10 * 1024**2, dtype=np.uint8).tobytes()
    compressed = gzip.compress(payload, mtime=0)
    with indexed_module().IndexedGzipFile(fileobj=io.BytesIO(compressed), mode="rb", spacing=8 * 1024**2) as reader:
        reader.build_full_index()
        points = reader.raw.npoints
        assert points > 0
        output = io.BytesIO()
        reader.export_index(fileobj=output)
    path = tmp_path / "nonzero.gzi"
    path.write_bytes(output.getvalue())
    assert reviewer.index_header(path, {"index_sha256": reviewer.sha(path), "index_points": points}, {"bytes": len(compressed)}) == points


@pytest.mark.parametrize("fault", ["truncated", "wrong_magic", "wrong_archive", "wrong_spacing", "wrong_window", "wrong_points", "extra_bytes", "changed_digest"])
def test_new_header_validation_does_not_accept_resealed_invalid_indexes(tmp_path, fault):
    compressed, _payload, data = empty_index()
    header = list(struct.unpack("<5sBBQQIII", data))
    points = 0
    if fault == "truncated":
        data = data[:-1]
    elif fault == "wrong_magic":
        header[0] = b"OTHER"
    elif fault == "wrong_archive":
        header[3] += 1
    elif fault == "wrong_spacing":
        header[5] = 65536
    elif fault == "wrong_window":
        header[6] = 65536
    elif fault == "wrong_points":
        points = 1
    elif fault == "extra_bytes":
        data += b"x"
    if fault not in {"truncated", "extra_bytes"}:
        data = struct.pack("<5sBBQQIII", *header)
    path = tmp_path / "index.gzi"
    path.write_bytes(data)
    checkpoint = {"index_sha256": "0" * 64 if fault == "changed_digest" else reviewer.sha(path), "index_points": points}
    with pytest.raises(ValueError):
        reviewer.index_header(path, checkpoint, {"bytes": len(compressed)})


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(authored, "reviewer", reviewer)
    folder, rows = authored.full_review_fixture(tmp_path, monkeypatch)
    old = tmp_path / "docs/evaluation-panels/image-positive-pairs-v21-review-freeze.json"
    freeze = reviewer.read(old)
    failed_path = tmp_path / "authored-failed-profile.json"
    authored.write(failed_path, {"child_exit_code": 1, "guard": None, "sample_errors": 0})
    freeze["original_failed_review_profile"] = {"path": "authored-failed-profile.json", "sha256": reviewer.sha(failed_path)}
    freeze["preserved_failure_files_sha256"] = {"authored-failed-profile.json": reviewer.sha(failed_path)}
    authored.write(tmp_path / "docs/evaluation-panels/image-positive-pairs-v22-review-freeze.json", freeze)
    authored.seal_rows(folder, rows)
    return folder, rows


def test_successor_reproduces_all_397_authored_truths_without_erasing_original_failure(tmp_path, monkeypatch):
    folder, _rows = fixture(tmp_path, monkeypatch)
    reviewer.review(folder, folder / "review.json")
    result = reviewer.read(folder / "review.json")
    assert result["passed"] and result["qualified_truths_reconstructed"] == 397
    assert result["ordinary_control_hashes_reproduced"] == 794
    assert result["original_strict_review_passed"] is False and result["original_review_failure_preserved"] is True


def test_successor_still_refuses_changed_pixel_evidence_in_full_denominator(tmp_path, monkeypatch):
    folder, rows = fixture(tmp_path, monkeypatch)
    rows[0]["proof"] = json.loads(json.dumps(rows[0]["proof"]))
    rows[0]["proof"]["ordinary_controls"]["tone"]["rgb_sha256"] = "0" * 64
    authored.seal_rows(folder, rows)
    with pytest.raises(ValueError, match="review failed"):
        reviewer.review(folder, folder / "failed.json")
    assert reviewer.read(folder / "failed.json")["failed_case_count"] == 1


def test_actual_successor_help_and_failure_are_both_retained(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    assert runtime.run(["--help"], good)["child_exit_code"] == 0
    with pytest.raises(RuntimeError):
        runtime.run(["--unknown-option"], bad)
    assert reviewer.read(bad / "profile.json")["child_exit_code"] == 2
