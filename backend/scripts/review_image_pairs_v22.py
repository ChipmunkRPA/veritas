"""Explicit index-header correction; original v21 reviewer and failure stay frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parents[2]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def range_records(path, archive, seen):
    """Inspect every stored HTTP receipt; remote payloads are not retained here."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transferred = 0
    for row in rows:
        start, end = row["start"], row["end"]
        need(type(start) is int and type(end) is int and 0 <= start <= end < archive["bytes"], "Range outside frozen archive")
        need(row["complete"] is True and row["http_status"] == 206 and row["etag"] == archive["etag"], "Incomplete or changed HTTP receipt")
        need(row["content_range"] == f"bytes {start}-{end}/{archive['bytes']}" and row["bytes"] == end - start + 1, "Range response dimensions differ")
        digest = row["sha256"]
        need(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest) is not None, "Complete range digest required")
        key = (start, end)
        need(key not in seen or seen[key] == digest, "Repeated archive range has different bytes")
        seen[key] = digest
        transferred += row["bytes"]
    need(bool(rows), "Completed archive phase has no HTTP receipts")
    return len(rows), transferred


def selected_rows(rows, wanted, objects):
    result = {}
    for row in rows:
        member, digest = row["member"], row["sha256"]
        need(member in wanted and member not in result, "Unexpected or repeated selected archive member")
        need(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest) is not None and row["file"] == digest + ".png", "Invalid content-addressed PNG name")
        path = objects / row["file"]
        need(type(row["bytes"]) is int and 0 < row["bytes"] <= 30_000_000 and path.stat().st_size == row["bytes"] and sha(path) == digest, "Selected archive PNG differs")
        result[member] = row
    return result


def acquisition(folder, plan):
    """Separate audit of the complete private chain, without importing its writer."""
    amendment = read(ROOT / "docs/evaluation-panels/image-positive-pairs-v21-cache-monitor-amendment.json")
    need(all(sha(ROOT / name) == expected for name, expected in amendment["new_code_sha256"].items()), "Frozen cache/monitor source differs")
    seed, start = read(folder / "seed.json"), read(folder / "execution-start.json")
    need(seed == amendment["seed"] and start["seed_sha256"] == sha(folder / "seed.json"), "Published copied-seed record differs")
    need(start["private_plan_sha256"] == sha(folder / "plan.json") and start["controller_sha256"] == sha(folder / "controller.py"), "Execution source/plan binding differs")
    need(start["new_neural_forwards"] == 0 and start["old_sampling_failure_preserved"] is True, "Acquisition scope differs")
    old = ROOT / seed["old_execution_root"]
    need(sha(old / "training-edits-001-profile/profile.json") == amendment["old_phase1_profile_sha256"], "Original failed profile must remain retained")
    failed = read(old / "training-edits-001-profile/profile.json")
    need(failed["sample_errors"] == 1 and amendment["old_sampling_failure_cleared"] is False, "Old sampling failure cannot become a clean phase")
    copied = old / "training-edits/part-000/checkpoint.json"
    need(sha(copied) == seed["source_checkpoint_sha256"] and sha(old / "training-edits-000-profile/profile.json") == seed["source_profile_sha256"], "Original successful seed differs")
    journal = [json.loads(line) for line in (folder / "execution-journal.jsonl").read_text().splitlines()]
    need(journal[0] == {"event": "copied_verified_seed", "executed_in_this_root": False, "source_checkpoint_sha256": sha(copied)}, "Copied seed cannot be counted as reexecuted")
    expected_journal, assets = [], {}
    totals = {"archives": 0, "phases_including_copied_seed": 0, "selected_members": 0, "http_receipts": 0, "compressed_bytes_in_receipts": 0, "terminal_observations": 0}
    for key, archive in plan["archives"].items():
        role, kind = key.split("-")
        wanted = {case["edit_member" if kind == "edits" else "mask_member"] for case in plan["cases"] if case["publisher_role"] == role}
        base = folder / key
        parts = sorted(base.glob("part-*"))
        need(bool(parts) and [p.name for p in parts] == [f"part-{i:03d}" for i in range(len(parts))], "Complete contiguous archive checkpoints required")
        previous, previous_sha, offset, seen = {}, None, -1, {}
        for number, part in enumerate(parts):
            checkpoint_path = part / "checkpoint.json"
            checkpoint, seal = read(checkpoint_path), read(base / f"completed-{number:03d}.json")
            ranges = base / f"ranges-{number:03d}.jsonl"
            profile_path = folder / f"{key}-{number:03d}-profile/profile.json"
            profile = read(profile_path)
            need(seal == {"plan_sha256": sha(folder / "plan.json"), "checkpoint_sha256": sha(checkpoint_path), "range_journal_sha256": sha(ranges)}, "Checkpoint/range completion seal differs")
            need(checkpoint["completed_phase"] is True and checkpoint["object"] == archive and checkpoint["previous_checkpoint_sha256"] == previous_sha, "Archive or checkpoint predecessor differs")
            need(checkpoint["next_tar_offset"] > offset and checkpoint["next_tar_offset"] % 512 == 0 and checkpoint["headers_this_phase"] > 0, "Archive checkpoint did not make complete-member progress")
            index_header(part / "index.gzi", checkpoint, archive)
            need(checkpoint["full_archive_sha256_verified"] is False and checkpoint["gzip_crc_validation_disabled_after_index_import"] == (number > 0), "Archive integrity limitations differ")
            need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0 and profile["seconds_limit"] == 900 and 0 < profile["duration_s"] <= 915, "Failed or unbounded acquisition phase cannot pass")
            need(0 <= profile["sampled_tree_rss_lower_bound_bytes"] <= 6 * 1024**3, "Acquisition RSS bound exceeded")
            seeded = key == "training-edits" and number == 0
            module = "scripts.image_positive_pairs_v21" if seeded else "scripts.image_positive_cached_v21"
            expected_root = old if seeded else folder
            need(profile["module"] == module and profile["arguments"] == ["step", str(expected_root), "--archive", key, "--number", str(number)], "Resource profile refers to another worker")
            events = profile.get("terminal_observation_events", [])
            need(not profile.get("sampling_error_events", []), "Live sampling failures must remain failures")
            need(all(e["classification"] == "verified_terminal_transition" and e["child_exit_code"] == 0 for e in events), "Unexplained terminal observation")
            current = selected_rows(checkpoint["selected"], wanted, base / "objects")
            need(all(current.get(member) == value for member, value in previous.items()), "A preceding selected member changed or disappeared")
            need(checkpoint["requested_members"] == len(wanted) and checkpoint["all_wanted_found"] == (set(current) == wanted), "Selected-member denominator differs")
            requests, transferred = range_records(ranges, archive, seen)
            need(checkpoint["range_requests"] == requests and checkpoint["compressed_bytes_transferred"] == transferred, "HTTP receipt totals differ")
            if seeded:
                need(sha(checkpoint_path) == seed["source_checkpoint_sha256"] and sha(profile_path) == seed["source_profile_sha256"] and len(current) == seed["copied_selected_members"], "Copied successful phase differs")
            else:
                expected_journal.append((key, number, sha(checkpoint_path), sha(profile_path)))
            if number < len(parts) - 1:
                need(not checkpoint["all_wanted_found"] and not checkpoint["archive_end_reached"], "Terminal archive cannot have a successor phase")
            previous, previous_sha, offset = current, sha(checkpoint_path), checkpoint["next_tar_offset"]
            totals["phases_including_copied_seed"] += 1
            totals["http_receipts"] += requests
            totals["compressed_bytes_in_receipts"] += transferred
            totals["terminal_observations"] += len(events)
        need(checkpoint["all_wanted_found"] or checkpoint["archive_end_reached"], "Archive acquisition is still in progress")
        assets[key] = previous
        totals["selected_members"] += len(previous)
        totals["archives"] += 1
    recorded = [(r["archive"], r["phase"], r["checkpoint_sha256"], r["profile_sha256"]) for r in journal[1:]]
    need(len(recorded) == len(set(recorded)) and set(recorded) == set(expected_journal), "Execution journal does not bind every actual non-seed phase")
    qualification = read(folder / "qualification-profile/profile.json")
    need(qualification["module"] == "scripts.image_positive_cached_v21" and qualification["arguments"] == ["qualify", str(folder)], "Qualification profile refers to another worker")
    need(qualification["child_exit_code"] == 0 and qualification["guard"] is None and qualification["sample_errors"] == 0 and qualification["seconds_limit"] == 900, "Failed qualification phase cannot pass")
    totals.update({"old_sampling_failure_retained": True, "seed_reexecuted": False, "remote_ranges_reacquired_by_reviewer": False, "full_archive_sha256_verified": False})
    return assets, totals


def index_header(path, checkpoint, archive):
    """Parse the pinned reader's native little-endian v1 header, including zero points."""
    need(sys.byteorder == "little", "Pinned index writer uses native little-endian integers")
    need(sha(path) == checkpoint["index_sha256"], "Private decompression index hash differs")
    with path.open("rb") as stream:
        header = stream.read(35)
    need(len(header) == 35, "Truncated private index header")
    magic, version, flags, compressed, _uncompressed, spacing, window, points = struct.unpack("<5sBBQQIII", header)
    need(magic == b"GZIDX" and version == 1 and flags == 0, "Pinned index format/version differs")
    need(compressed == archive["bytes"] and spacing == 8 * 1024**2 and window == 32768, "Index archive/spacing/window differs")
    need(type(checkpoint["index_points"]) is int and checkpoint["index_points"] == points, "Index point count differs from binary header")
    need(path.stat().st_size == 35 if points == 0 else path.stat().st_size > 35, "Index entry payload length differs")
    return points


def tone_lookup():
    # Scalar binary64 operations deliberately follow the declared float recipe.
    # Ideal rational1.12 is a different quantization path at some half steps.
    return np.asarray([[min(255, max(0, round((float(value) - 127.5) * 1.12 + 127.5 + shift))) for value in range(256)] for shift in (3, -2, 1)], dtype=np.uint8)


def compose(foreground, background, alpha):
    fg, bg, a = foreground.astype(np.uint32), background.astype(np.uint32), alpha.astype(np.uint32)[..., None]
    return ((fg * a + bg * (255 - a) + 127) // 255).astype(np.uint8)


def proof(original, edited, alpha):
    need(original.size == edited.size == alpha.size, "Source/edit/alpha native dimensions differ")
    difference = ImageChops.difference(original, edited)
    channels = difference.split()
    changed = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2]).point(lambda x: 255 if x else 0)
    support = alpha.point(lambda x: 255 if x else 0)
    changed_count = original.width * original.height - changed.histogram()[0]
    support_count = original.width * original.height - support.histogram()[0]
    need(changed_count > 0 and support_count > 0, "Empty intended or observed edit")
    need(ImageChops.multiply(changed, ImageChops.invert(support)).getbbox() is None, "Observed change outside compositing support")
    source = np.array(original)
    table = tone_lookup()
    tone = np.stack([table[c, source[..., c]] for c in range(3)], axis=2)
    # The declared Pillow blur kernel is shared; alpha composition is independent.
    blurred = np.array(original.filter(ImageFilter.GaussianBlur(1.25)))
    controls = {"tone": compose(tone, source, np.array(alpha)), "blur": compose(blurred, source, np.array(alpha))}
    boolean = np.array(changed) != 0
    result = {
        "changed_pixels": changed_count, "support_pixels": support_count, "outside_support_changes": 0,
        "truth_mask_sha256": hashlib.sha256(boolean.tobytes()).hexdigest(),
        "ordinary_controls": {kind: {"rgb_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "changed_pixels": int(np.count_nonzero(np.logical_or.reduce(value != source, axis=2))), "ai_generation_used": False} for kind, value in controls.items()},
    }
    return changed, result


def rgb(path):
    with Image.open(path) as image:
        need(image.format == "PNG" and max(image.size) <= 2048 and image.getexif().get(274, 1) == 1, "Upright bounded source PNG required")
        value = image.convert("RGB").copy()
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    need(decoded is not None and np.array_equal(np.array(value), decoded[..., ::-1]), "Native Pillow/OpenCV RGB differs")
    return value


def review(folder, output):
    plan = json.loads((folder / "plan.json").read_text())
    public = json.loads((ROOT / "docs/evaluation-panels/image-positive-pairs-v21-freeze.json").read_text())
    review_freeze_path = ROOT / "docs/evaluation-panels/image-positive-pairs-v22-review-freeze.json"
    review_freeze = read(review_freeze_path)
    need(all(sha(ROOT / name) == expected for name, expected in review_freeze["code_sha256"].items()), "Separate reviewer freeze differs")
    need(review_freeze["private_plan_sha256"] == sha(folder / "plan.json"), "Reviewer refers to a different source plan")
    retained = review_freeze["original_failed_review_profile"]
    need(sha(ROOT / retained["path"]) == retained["sha256"], "Original failed review resource receipt must remain retained")
    failed = read(ROOT / retained["path"])
    need(failed["child_exit_code"] == 1 and failed["guard"] is None and failed["sample_errors"] == 0, "Original review failure must not become a successful run")
    need(all(sha(ROOT / name) == expected for name, expected in review_freeze["preserved_failure_files_sha256"].items()), "Original failure logs/receipts must remain retained")
    need(review_freeze["runtime_versions"] == {"numpy": np.__version__, "Pillow": Image.__version__, "opencv": cv2.__version__}, "Separate decoder versions differ")
    need(sha(folder / "plan.json") == public["private_plan_sha256"], "Published source plan differs")
    need(all(sha(ROOT / name) == expected for name, expected in plan["code_sha256"].items()), "Frozen pair/reader code differs")
    objects, acquisition_proof = acquisition(folder, plan)
    summary = json.loads((folder / "qualification/summary.json").read_text())
    lines = [json.loads(line) for line in (folder / "qualification/pairs.jsonl").read_text().splitlines()]
    need(len(lines) == len(plan["cases"]) == summary["planned"] == summary["completed"] == 397, "Complete397-case denominator required")
    need(summary["pairs_sha256"] == sha(folder / "qualification/pairs.jsonl"), "Qualified pair journal differs")
    errors, accepted, controls, decoder_checks = [], 0, 0, 0
    for case, row in zip(plan["cases"], lines, strict=True):
        try:
            need(case["source_id"] == row["source_id"] and case["publisher_role"] == row["publisher_role"], "Source/role correspondence differs")
            need(sha(ROOT / case["original"]) == case["original_sha256"], "Original source bytes differ")
            actual_edit = objects[case["publisher_role"] + "-edits"].get(case["edit_member"])
            actual_mask = objects[case["publisher_role"] + "-masks"].get(case["mask_member"])
            if "edited_image" not in row or "compositing_mask" not in row:
                need((actual_edit is None or actual_mask is None) and row["status"] == "unresolved" and "error" in row, "Available asset cannot be dropped as unresolved")
                continue
            edit, matte = row["edited_image"], row["compositing_mask"]
            for item, expected_member, kind in ((edit, case["edit_member"], "edits"), (matte, case["mask_member"], "masks")):
                need(item["member"] == expected_member and sha(ROOT / item["root"] / item["file"]) == item["sha256"], "Selected asset/member binding differs")
                expected = objects[case["publisher_role"] + "-" + kind][expected_member]
                need({k: item[k] for k in expected} == expected and (ROOT / item["root"]).resolve() == folder / (case["publisher_role"] + "-" + kind) / "objects", "Qualified PNG is not the selected checkpoint member")
            original, edited = rgb(ROOT / case["original"]), rgb(ROOT / edit["root"] / edit["file"])
            need(hashlib.sha256(np.asarray(original).tobytes()).hexdigest() == case["source_rgb_sha256"], "Current source pixel-lineage commitment differs")
            decoder_checks += 2
            with Image.open(ROOT / matte["root"] / matte["file"]) as mask:
                need(mask.format == "PNG" and max(mask.size) <= 2048 and mask.getexif().get(274, 1) == 1, "Upright bounded alpha PNG required")
                alpha = mask.convert("L").copy()
            try:
                changed, result = proof(original, edited, alpha)
            except ValueError:
                need(row["status"] == "unresolved" and "error" in row, "Independent source/alpha refusal differs")
                continue
            need(row["status"] == "qualified_source_relative_pair", "Valid pair was not retained")
            need(all(row["proof"].get(k) == value for k, value in result.items()), "Complete independent truth/control proof differs")
            path = folder / "qualification" / row["truth_mask"]
            need(path.name == str(case["source_id"]) + ".truth.png" and sha(path) == row["truth_png_sha256"], "Retained truth PNG differs")
            with Image.open(path) as retained:
                need(retained.convert("L").tobytes() == changed.tobytes(), "Every changed-pixel label must reproduce")
            accepted += 1
            controls += 2
        except Exception as error:
            errors.append({"source_id": case["source_id"], "type": type(error).__name__, "message": str(error)})
    need(dict(Counter(r["status"] for r in lines)) == summary["statuses"], "Status denominator differs")
    need(summary["neural_forwards"] == summary["training_admitted"] == 0 and all(row["training_admitted"] is False for row in lines), "Source qualification cannot admit a model")
    receipt = {"schema": "chipmunk.image-pair-truth-review/22", "original_strict_review_passed": False, "original_review_failure_preserved": True, "cases_reviewed": 397, "qualified_truths_reconstructed": accepted, "ordinary_control_hashes_reproduced": controls, "dual_rgb_decoder_checks": decoder_checks, "acquisition": acquisition_proof, "failed_case_count": len(errors), "passed": not errors, "errors": errors, "training_admitted": 0, "neural_forwards": 0, "outside_human_review": False, "program_sha256": sha(Path(__file__)), "review_freeze_sha256": sha(review_freeze_path), "summary_sha256": sha(folder / "qualification/summary.json"), "scope": "Separate Pillow difference/histogram and integer alpha composition; scalar binary64 tone lookup. Blur kernel/mode conversion shared with the declared Pillow dependency. Retained bytes, not independent remote reacquisition or pixel-authorship proof."}
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "errors"}), flush=True)
    if errors:
        raise ValueError("Full source-pair review failed; original receipt retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.folder.resolve(), args.output)
