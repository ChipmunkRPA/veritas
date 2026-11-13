# SPDX-License-Identifier: Apache-2.0
"""Read back native pictures, guard decisions and authored 720p retry deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence.face_return_binding import validate_second
from scripts import review_video_face_two_returns_hd as review
from scripts import video_face_hd_sparse_retry as trial
from scripts.video_common_sources_v89 import read, require, sha, write


def compact(record, table):
    by_frame = {r["frame_index"]: r for r in record["frame_inputs"]}
    return {"support": [record["support_indices"][role] for role in ("before", "during", "after")],
            "frames": [record["start_frame_index"], record["end_frame_index"]],
            "clock_s": [str(record[key]*table.tb) for key in ("start_pts", "end_pts")],
            "anchors": {role: {"index": i, "pts": by_frame[i]["pts"], "rgb_sha256": by_frame[i]["decoded_rgb_sha256"],
                                "box": [int(by_frame[i]["bbox"][key]) for key in ("x", "y", "w", "h")]}
                        for role, i in record["anchors"].items()},
            "scene_inputs": record["scene_frame_inputs"],
            "scene_frames": [row["frame_index"] for row in record["scene_frame_inputs"]],
            "scene_correlations": record["scene_correlations"],
            "scene_lower_quartiles": record["scene_lower_quartiles"]}


def source_readback(folder, frozen, manifest):
    from app.detectors.video.decode import FrameTable

    ledger = {(row["group"], row["condition"]): row for row in manifest["rows"]}
    picks = trial.earlier.closest_background(frozen["sources"])
    require(sorted(correlation for correlation, _ in picks) == manifest["cut_border_correlations"], "Cut pairing changed")
    exact, crops, rebuilds = 0, 0, 0
    examples, outcomes = {}, {}
    with (folder/"held.jsonl").open() as stream:
        for row, media in zip(map(json.loads, stream), manifest["rows"], strict=True):
            require(all(row[key] == media[key] for key in ("file", "group", "condition", "truth", "sha256")),
                    "Native readback case differs")
            table = FrameTable.from_json(media["clock"])
            records = (row["retry"] or {}).get("ranges", [])
            for index, record in enumerate(records):
                require(float.fromhex(record["response_hex"]) > .1 and record["source_carrier"] == "mp4",
                        "A selected range lacks guarded support")
                if index:
                    validate_second(record, records[0], row["before"]["coarse"], row["before"]["boundary_refinement"])
            actual, checked = review.view_anchors(folder/"media"/row["file"], media, None)
            exact += actual
            require(not checked, "Unexpected unselected crops")
            for record in records:
                candidate = compact(record, table)
                if not any(review.includes_half(candidate["frames"], target) for target in row["truth"]):
                    continue
                output = None
                if row["condition"] not in examples:
                    output = folder/f"private-example-{row['condition']}"
                    output.mkdir(mode=0o700)
                _, anchors = review.view_anchors(folder/"media"/row["file"], media, candidate, output)
                crops += len(anchors)
                if output:
                    write(output/"anchors.json", anchors)
                    examples[row["condition"]] = {"range_s": candidate["clock_s"],
                                                 "anchor_s": [str(value["pts"]*table.tb) for value in anchors.values()]}
            pts = media["clock"]["pts"]
            current = row["before"]
            initial = [current[key] for key in ("coarse", "boundary_refinement", "short_range") if current[key]]
            initial.extend(row["incumbent_second"])
            before = [(pts.index(r["start_pts"]), pts.index(r["end_pts"])) for r in initial]
            after = before + [(pts.index(r["start_pts"]), pts.index(r["end_pts"])) for r in records]
            counts = outcomes.setdefault(row["condition"], Counter())
            counts.update(files=1,
                          before=sum(any(review.includes_half(r, t) for r in before) for t in row["truth"]),
                          after=sum(any(review.includes_half(r, t) for r in after) for t in row["truth"]),
                          flags_before=bool(initial), flags_after=bool(initial or records))
    with TemporaryDirectory(prefix="verify-720p-retry-", dir=folder) as temp:
        for i, pin in enumerate(frozen["sources"], 1):
            base = trial.earlier.native(pin, "original", range(120))
            swap = {**trial.earlier.native(pin, "swap_a", range(*trial.EARLY)), **trial.earlier.native(pin, "swap_b", range(*trial.LATE))}
            cuts = trial.earlier.native(frozen["sources"][-picks[i-1][1]], "original", (*range(*trial.EARLY), *range(*trial.LATE)))
            for role, replacement in (("dual_swap", swap), ("cross_recording_cuts", cuts)):
                row = ledger[i, role]
                require(all(hashlib.sha256(base[j].tobytes()).digest() != hashlib.sha256(rgb.tobytes()).digest()
                            for j, rgb in replacement.items()), "A known source intervention failed exact pixel readback")
                output = Path(temp)/row["file"]
                trial.hd.authored(output, pictures=(replacement.get(j, base[j]) for j in range(120)))
                require(sha(output) == row["sha256"], "Encoded source reconstruction differs")
                if role == "dual_swap":
                    tandem = ledger[i, "tandem_dual_swap"]
                    extra = Path(temp)/tandem["file"]
                    trial.hd.authored(extra, source=output, crf=35)
                    require(sha(extra) == tandem["sha256"], "Tandem copy reconstruction differs")
                    extra.unlink()
                    rebuilds += 1
                output.unlink()
                rebuilds += 1
            print("reviewed source group", i, flush=True)
    return {"exact_native_pictures": exact, "native_review_crops": crops, "exact_encoded_reconstructions": rebuilds,
            "outcomes": outcomes, "examples": examples}


def audit(folder):
    import cv2

    cv2.setNumThreads(1)
    frozen, manifest, aggregate = (read(folder/name) for name in ("freeze.json", "media.json", "aggregate.json"))
    require(frozen == trial.freeze_plan() and aggregate["freeze_sha256"] == manifest["freeze_sha256"] == sha(folder/"freeze.json")
            and aggregate["runtime"]["held_sha256"] == sha(folder/"held.jsonl"), "Frozen inputs or observations differ")
    reviewed = source_readback(folder, frozen, manifest)
    for role, outcome in reviewed.pop("outcomes").items():
        saved = aggregate["strata"][role]
        require(outcome == {"files": saved["files"], "before": saved["before"], "after": saved["after"],
                            "flags_before": saved["before_flag"], "flags_after": saved["after_flag"]}, "Independent totals differ")
    require(reviewed["exact_native_pictures"] == 5760 and reviewed["exact_encoded_reconstructions"] == 24, "Readback is incomplete")
    write(folder/"independent-audit.json", {"status": "verified", "aggregate_sha256": sha(folder/"aggregate.json"), **reviewed})
    print(reviewed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)


if __name__ == "__main__":
    main()
