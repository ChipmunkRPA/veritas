"""Independent source differences, scalar tone and alpha composition checks."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from scripts import review_image_pairs_v21 as reviewer
from scripts.image_pair_truth_v21 import qualify
from scripts.review_image_pairs_v21 import compose, proof, tone_lookup


def fixture():
    original = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    edited = original.copy()
    edited[4:9, 5:12] ^= 17
    alpha = np.zeros((16, 16), np.uint8)
    alpha[2:13, 3:14] = 151
    return original, edited, alpha


def test_independent_library_truth_and_control_proofs_match_all_pixels():
    source, edit, alpha = fixture()
    primary_mask, primary = qualify(source, edit, alpha)
    independent_mask, separate = proof(Image.fromarray(source), Image.fromarray(edit), Image.fromarray(alpha))
    assert np.array_equal(np.asarray(independent_mask) > 0, primary_mask)
    assert all(primary[k] == value for k, value in separate.items())


def test_scalar_tone_lookup_preserves_actual_binary64_half_step():
    table = tone_lookup()
    assert table[1, 40] == 27  # The frozen float recipe is below the ideal27.5.
    assert round((56 * 40 - 865) / 50) == 28
    levels = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (1, 256, 3))
    reference = np.clip(np.rint((levels.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])), 0, 255).astype(np.uint8)
    assert np.array_equal(np.stack([table[c] for c in range(3)], axis=-1)[None], reference)


def test_integer_alpha_composition_matches_actual_pillow_on_every_alpha_value():
    rng = np.random.default_rng(216)
    source = rng.integers(0, 256, (256, 32, 3), dtype=np.uint8)
    changed = rng.integers(0, 256, source.shape, dtype=np.uint8)
    alpha = np.broadcast_to(np.arange(256, dtype=np.uint8)[:, None], source.shape[:2]).copy()
    expected = np.array(Image.composite(Image.fromarray(changed), Image.fromarray(source), Image.fromarray(alpha)))
    assert np.array_equal(compose(changed, source, alpha), expected)


def test_independent_outside_support_change_is_not_admitted():
    source, edit, alpha = fixture()
    edit[0, 0, 1] ^= 1
    with pytest.raises(ValueError, match="outside"):
        proof(Image.fromarray(source), Image.fromarray(edit), Image.fromarray(alpha))


def test_reconstructed_bool_mask_hash_has_exact_primary_representation():
    source, edit, alpha = fixture()
    changed, result = proof(Image.fromarray(source), Image.fromarray(edit), Image.fromarray(alpha))
    expected = np.zeros(source.shape[:2], bool)
    expected[4:9, 5:12] = True
    assert result["changed_pixels"] == 35
    assert hashlib.sha256(expected.tobytes()).hexdigest() == result["truth_mask_sha256"]
    assert set(np.unique(np.asarray(changed))) <= {0, 255}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def chain_fixture(tmp_path, monkeypatch):
    """Small authored chain with a copied seed, six archives and one successor."""
    monkeypatch.setattr(reviewer, "ROOT", tmp_path)
    folder, old = tmp_path / "new", tmp_path / "old"
    plan = {"archives": {}, "cases": []}
    for role, count in (("training", 2), ("validation", 1), ("testing", 1)):
        for number in range(count):
            plan["cases"].append({"publisher_role": role, "edit_member": f"{role}/{number}.png", "mask_member": f"{role}/{number}.mask.png"})
        for kind in ("edits", "masks"):
            plan["archives"][role + "-" + kind] = {"url": "https://example.invalid/" + role + kind, "bytes": 8, "etag": '"fixed"'}
    write(folder / "plan.json", plan)
    plan_digest = reviewer.sha(folder / "plan.json")
    records = []
    for key, archive in plan["archives"].items():
        role, kind = key.split("-")
        members = [c["edit_member" if kind == "edits" else "mask_member"] for c in plan["cases"] if c["publisher_role"] == role]
        base = folder / key
        rows = []
        for member in members:
            payload = ("authored only " + member).encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = base / "objects" / (digest + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            rows.append({"member": member, "file": path.name, "sha256": digest, "bytes": len(payload)})
        previous_sha = None
        for number in range(2 if key == "training-edits" else 1):
            part = base / f"part-{number:03d}"
            part.mkdir()
            (part / "index.gzi").write_bytes(b"authored private index")
            selected = rows[:1] if key == "training-edits" and number == 0 else rows
            checkpoint = {"completed_phase": True, "object": archive, "previous_checkpoint_sha256": previous_sha, "next_tar_offset": (number + 1) * 512, "headers_this_phase": 1, "index_sha256": reviewer.sha(part / "index.gzi"), "index_points": 1, "full_archive_sha256_verified": False, "gzip_crc_validation_disabled_after_index_import": number > 0, "selected": selected, "requested_members": len(rows), "all_wanted_found": len(selected) == len(rows), "archive_end_reached": False, "range_requests": 1, "compressed_bytes_transferred": 8}
            write(part / "checkpoint.json", checkpoint)
            range_path = base / f"ranges-{number:03d}.jsonl"
            write(range_path, {"start": 0, "end": 7, "complete": True, "http_status": 206, "etag": archive["etag"], "content_range": "bytes 0-7/8", "bytes": 8, "sha256": hashlib.sha256(b"12345678").hexdigest()})
            seal = {"plan_sha256": plan_digest, "checkpoint_sha256": reviewer.sha(part / "checkpoint.json"), "range_journal_sha256": reviewer.sha(range_path)}
            write(base / f"completed-{number:03d}.json", seal)
            seeded = key == "training-edits" and number == 0
            profile_path = folder / f"{key}-{number:03d}-profile/profile.json"
            profile = {"module": "scripts.image_positive_pairs_v21" if seeded else "scripts.image_positive_cached_v21", "arguments": ["step", str(old if seeded else folder), "--archive", key, "--number", str(number)], "child_exit_code": 0, "guard": None, "sample_errors": 0, "seconds_limit": 900, "duration_s": 1, "sampled_tree_rss_lower_bound_bytes": 1024, "sampling_error_events": [], "terminal_observation_events": []}
            write(profile_path, profile)
            if seeded:
                for source in (part / "checkpoint.json", profile_path):
                    target = old / source.relative_to(folder)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            else:
                records.append({"archive": key, "phase": number, "checkpoint_sha256": reviewer.sha(part / "checkpoint.json"), "profile_sha256": reviewer.sha(profile_path)})
            previous_sha = reviewer.sha(part / "checkpoint.json")
    write(old / "training-edits-001-profile/profile.json", {"sample_errors": 1})
    seed = {"old_execution_root": "old", "source_checkpoint_sha256": reviewer.sha(old / "training-edits/part-000/checkpoint.json"), "source_profile_sha256": reviewer.sha(old / "training-edits-000-profile/profile.json"), "copied_selected_members": 1}
    write(folder / "seed.json", seed)
    (folder / "controller.py").write_text("# authored controller only\n")
    write(folder / "execution-start.json", {"private_plan_sha256": plan_digest, "controller_sha256": reviewer.sha(folder / "controller.py"), "seed_sha256": reviewer.sha(folder / "seed.json"), "new_neural_forwards": 0, "old_sampling_failure_preserved": True})
    write(tmp_path / "docs/evaluation-panels/image-positive-pairs-v21-cache-monitor-amendment.json", {"new_code_sha256": {}, "seed": seed, "old_phase1_profile_sha256": reviewer.sha(old / "training-edits-001-profile/profile.json"), "old_sampling_failure_cleared": False})
    records.insert(0, {"event": "copied_verified_seed", "executed_in_this_root": False, "source_checkpoint_sha256": seed["source_checkpoint_sha256"]})
    (folder / "execution-journal.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    write(folder / "qualification-profile/profile.json", {"module": "scripts.image_positive_cached_v21", "arguments": ["qualify", str(folder)], "child_exit_code": 0, "guard": None, "sample_errors": 0, "seconds_limit": 900})
    return folder, plan


def test_complete_authored_checkpoint_chain_and_seed_are_reconstructed(tmp_path, monkeypatch):
    folder, plan = chain_fixture(tmp_path, monkeypatch)
    assets, record = reviewer.acquisition(folder, plan)
    assert record["archives"] == 6 and record["phases_including_copied_seed"] == 7
    assert record["selected_members"] == 8 and record["http_receipts"] == 7
    assert record["old_sampling_failure_retained"] is True and record["seed_reexecuted"] is False
    assert len(assets["training-edits"]) == 2


@pytest.mark.parametrize("tamper", ["range_rebound", "selected_dropped", "missing_phase", "profile_failure", "old_failure_erased", "journal_omitted"])
def test_acquisition_refuses_tampering_even_when_a_local_seal_is_recomputed(tmp_path, monkeypatch, tamper):
    folder, plan = chain_fixture(tmp_path, monkeypatch)
    base = folder / "training-edits"
    checkpoint, range_path = base / "part-001/checkpoint.json", base / "ranges-001.jsonl"
    seal_path = base / "completed-001.json"
    seal = reviewer.read(seal_path)
    if tamper == "range_rebound":
        row = reviewer.read(range_path)
        row["sha256"] = "f" * 64
        write(range_path, row)
        seal["range_journal_sha256"] = reviewer.sha(range_path)
        write(seal_path, seal)
    elif tamper == "selected_dropped":
        row = reviewer.read(checkpoint)
        row["selected"] = row["selected"][1:]
        write(checkpoint, row)
        seal["checkpoint_sha256"] = reviewer.sha(checkpoint)
        write(seal_path, seal)
    elif tamper == "missing_phase":
        (base / "part-001").rename(base / "part-002")
    elif tamper == "profile_failure":
        path = folder / "training-edits-001-profile/profile.json"
        row = reviewer.read(path)
        row["sample_errors"] = 1
        write(path, row)
    elif tamper == "old_failure_erased":
        write(tmp_path / "old/training-edits-001-profile/profile.json", {"sample_errors": 0})
    else:
        path = folder / "execution-journal.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError):
        reviewer.acquisition(folder, plan)


def full_review_fixture(tmp_path, monkeypatch):
    """397 tiny authored pairs exercise the full-denominator integration path."""
    monkeypatch.setattr(reviewer, "ROOT", tmp_path)
    folder = tmp_path / "study"
    source, edit, alpha = fixture()
    truth, primary = qualify(source, edit, alpha)
    for kind, value in (("original", source), ("edited", edit), ("mask", alpha)):
        buffer = io.BytesIO()
        Image.fromarray(value).save(buffer, format="PNG")
        data = buffer.getvalue()
        if kind == "original":
            original = tmp_path / "source.png"
            original.write_bytes(data)
        else:
            base = folder / ("training-edits" if kind == "edited" else "training-masks") / "objects"
            base.mkdir(parents=True, exist_ok=True)
            path = base / (hashlib.sha256(data).hexdigest() + ".png")
            path.write_bytes(data)
            if kind == "edited":
                edited_item = {"file": path.name, "sha256": reviewer.sha(path), "bytes": len(data), "root": str(base.relative_to(tmp_path))}
            else:
                mask_item = {"file": path.name, "sha256": reviewer.sha(path), "bytes": len(data), "root": str(base.relative_to(tmp_path))}
    cases, rows = [], []
    objects = {"training-edits": {}, "training-masks": {}}
    qualification = folder / "qualification"
    qualification.mkdir()
    for number in range(397):
        edited = {"member": f"edit/{number}.png", **edited_item}
        matte = {"member": f"mask/{number}.png", **mask_item}
        for key, item in (("training-edits", edited), ("training-masks", matte)):
            objects[key][item["member"]] = {k: v for k, v in item.items() if k != "root"}
        case = {"source_id": number, "publisher_role": "training", "original": "source.png", "original_sha256": reviewer.sha(original), "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(), "edit_member": edited["member"], "mask_member": matte["member"]}
        truth_path = qualification / f"{number}.truth.png"
        Image.fromarray(truth.astype(np.uint8) * 255).save(truth_path)
        rows.append({"source_id": number, "publisher_role": "training", "training_admitted": False, "status": "qualified_source_relative_pair", "edited_image": edited, "compositing_mask": matte, "proof": primary, "truth_mask": truth_path.name, "truth_png_sha256": reviewer.sha(truth_path)})
        cases.append(case)
    write(folder / "plan.json", {"code_sha256": {}, "cases": cases})
    write(tmp_path / "docs/evaluation-panels/image-positive-pairs-v21-freeze.json", {"private_plan_sha256": reviewer.sha(folder / "plan.json")})
    write(tmp_path / "docs/evaluation-panels/image-positive-pairs-v21-review-freeze.json", {"code_sha256": {}, "private_plan_sha256": reviewer.sha(folder / "plan.json"), "runtime_versions": {"numpy": np.__version__, "Pillow": Image.__version__, "opencv": reviewer.cv2.__version__}})
    monkeypatch.setattr(reviewer, "acquisition", lambda *_: (objects, {"authored_separately_tested_chain": True}))
    return folder, rows


def seal_rows(folder, rows):
    journal = folder / "qualification/pairs.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    counts = {status: sum(row["status"] == status for row in rows) for status in {row["status"] for row in rows}}
    write(folder / "qualification/summary.json", {"planned": 397, "completed": 397, "pairs_sha256": reviewer.sha(journal), "statuses": counts, "neural_forwards": 0, "training_admitted": 0})


def test_full_397_pair_review_reproduces_every_truth_and_control(tmp_path, monkeypatch):
    folder, rows = full_review_fixture(tmp_path, monkeypatch)
    seal_rows(folder, rows)
    output = folder / "independent-review.json"
    reviewer.review(folder, output)
    record = reviewer.read(output)
    assert record["passed"] is True and record["qualified_truths_reconstructed"] == 397
    assert record["ordinary_control_hashes_reproduced"] == record["dual_rgb_decoder_checks"] == 794


@pytest.mark.parametrize("tamper", ["truth_pixels", "available_asset_dropped", "ordinary_control_hash"])
def test_full_review_retains_failed_case_even_if_primary_journal_is_resealed(tmp_path, monkeypatch, tamper):
    folder, rows = full_review_fixture(tmp_path, monkeypatch)
    if tamper == "truth_pixels":
        path = folder / "qualification/0.truth.png"
        Image.new("L", (16, 16), 0).save(path)
        rows[0]["truth_png_sha256"] = reviewer.sha(path)
    elif tamper == "available_asset_dropped":
        rows[0] = {"source_id": 0, "publisher_role": "training", "training_admitted": False, "status": "unresolved", "error": {"message": "claimed missing"}}
    else:
        rows[0]["proof"] = json.loads(json.dumps(rows[0]["proof"]))
        rows[0]["proof"]["ordinary_controls"]["tone"]["rgb_sha256"] = "0" * 64
    seal_rows(folder, rows)
    output = folder / "failed-review.json"
    with pytest.raises(ValueError, match="review failed"):
        reviewer.review(folder, output)
    record = reviewer.read(output)
    assert record["passed"] is False and record["failed_case_count"] == 1
    assert record["qualified_truths_reconstructed"] == 396
