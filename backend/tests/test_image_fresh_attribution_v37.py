"""Authored source-collection failures cannot disappear from the denominator."""

import hashlib
import json
from copy import deepcopy

import pytest

from scripts import image_fresh_attribution_v37 as study

LICENSES = {
    1: {"id": 1, "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"},
    4: {"id": 4, "url": "http://creativecommons.org/licenses/by/2.0/"},
    5: {"id": 5, "url": "http://creativecommons.org/licenses/by-sa/2.0/"},
    6: {"id": 6, "url": "http://creativecommons.org/licenses/by-nd/2.0/"},
}


def image(identifier, licence=4, photo=None):
    return {"id": identifier, "license": licence, "flickr_url": f"https://live.staticflickr.com/3350/{photo or 1000 + identifier}_abcdef1234_b.jpg"}


def protocol():
    return {"items": [{"index": n, "source": {"source_id": n + 1}, "url": f"https://example.invalid/authored/{n}"} for n in range(256)]}


def write_unresolved(folder, doc, count=32):
    output = folder / "phase-000"
    output.mkdir()
    rows = [{"index": item["index"], "source_id": item["source"]["source_id"], "url": item["url"], "status": "unresolved", "error": {"type": "Timeout", "message": "Authored timeout"}} for item in doc["items"][:count]]
    (output / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return output, rows


def test_prior_ids_noncommercial_and_no_derivatives_are_excluded_before_ranking():
    rows = [image(1), image(2, 1), image(3, 6), image(4, 5), image(5, 4)]
    selected, stats = study.choose_sources(rows, LICENSES, {1}, count=2)
    assert {row["source_id"] for row in selected} == {4, 5}
    assert stats["eligible_rows"] == stats["eligible_photo_ids"] == 2
    assert all(row["publisher_role"] == "fresh-unassigned" for row in selected)


def test_same_flickr_photo_under_different_coco_ids_counts_once_and_order_is_stable():
    rows = [image(1, photo=5000), image(2, photo=5000), image(3), image(4)]
    first, stats = study.choose_sources(rows, LICENSES, set(), count=3)
    second, _ = study.choose_sources(list(reversed(rows)), LICENSES, set(), count=3)
    assert first == second
    assert len({row["photo_id"] for row in first}) == 3
    assert stats["eligible_rows"] == 4 and stats["eligible_photo_ids"] == 3


def test_unknown_licence_url_and_untrusted_photo_host_do_not_enter_pool():
    licences = deepcopy(LICENSES)
    licences[4]["url"] = "https://example.invalid/claim-of-commercial-permission"
    bad_url = image(2, 5)
    bad_url["flickr_url"] = "https://staticflickr.com.attacker.invalid/3350/1002_abcdef1234_b.jpg"
    selected, stats = study.choose_sources([image(1), bad_url, image(3, 5)], licences, set(), count=1)
    assert [row["source_id"] for row in selected] == [3]
    assert stats["invalid_static_urls"] == 1


def test_insufficient_unique_sources_and_duplicate_coco_ids_fail_without_replacement():
    with pytest.raises(ValueError, match="Insufficient"):
        study.choose_sources([image(1), image(2, 1)], LICENSES, set(), count=2)
    with pytest.raises(ValueError, match="Unique COCO"):
        study.choose_sources([image(1), image(1)], LICENSES, set(), count=1)


@pytest.mark.parametrize("number", [-1, 8, True, None])
def test_only_eight_exact_source_phases_are_allowed(number):
    with pytest.raises(ValueError):
        study.phase_items(protocol(), number)


def test_each_phase_contains_disjoint_sources_and_complete_denominator():
    doc = protocol()
    indices = [row["index"] for number in range(8) for row in study.phase_items(doc, number)]
    assert indices == list(range(256))
    doc["items"].pop()
    with pytest.raises(ValueError, match="Complete ordered"):
        study.phase_items(doc, 7)


def test_all_failed_requests_are_valid_complete_metadata_coverage(tmp_path):
    doc = protocol()
    _, rows = write_unresolved(tmp_path, doc)
    assert study.review_phase(tmp_path, doc, 0) == rows
    assert len(rows) == 32 and all(row["status"] == "unresolved" for row in rows)


def test_dropping_failed_request_or_rebinding_source_is_rejected(tmp_path):
    doc = protocol()
    output, rows = write_unresolved(tmp_path, doc, count=31)
    with pytest.raises(ValueError, match="Complete denominator"):
        study.review_phase(tmp_path, doc, 0)
    rows.append({"index": 31, "source_id": 999, "url": doc["items"][31]["url"], "status": "unresolved"})
    (output / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="binding"):
        study.review_phase(tmp_path, doc, 0)


def test_claimed_success_without_response_bytes_is_rejected(tmp_path):
    doc = protocol()
    output, rows = write_unresolved(tmp_path, doc)
    rows[0]["status"] = "matched_current_provider_metadata"
    (output / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="retained complete"):
        study.review_phase(tmp_path, doc, 0)


def test_modified_response_or_unsafe_filename_is_rejected_even_if_unresolved(tmp_path):
    doc = protocol()
    output, rows = write_unresolved(tmp_path, doc)
    (output / "responses").mkdir()
    body = b"authored retained error body"
    (output / "responses/0000.bin").write_bytes(body + b"changed")
    rows[0].update({"body": "responses/0000.bin", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    path = output / "responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Response bytes changed"):
        study.review_phase(tmp_path, doc, 0)
    rows[0]["body"] = "../other-study.bin"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Exact response filename"):
        study.review_phase(tmp_path, doc, 0)


def test_unverified_or_different_plan_blocks_network_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    doc = {"planned": 256, "chunk": 32, "phases": 8, "input_sha256": {}, "code_sha256": {}}
    (tmp_path / "plan.json").write_text(json.dumps(doc))
    receipt = {"plan_sha256": study.sha(tmp_path / "plan.json"), "remote_verified_before_first_request": False, "remote_freeze_blob_sha": "a" * 40}
    (tmp_path / "execution-start.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="precede requests"):
        study.admission(tmp_path)
    receipt["remote_verified_before_first_request"] = True
    receipt["plan_sha256"] = "f" * 64
    (tmp_path / "execution-start.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="commitment differs"):
        study.admission(tmp_path)
