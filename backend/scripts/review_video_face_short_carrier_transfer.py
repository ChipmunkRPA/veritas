# SPDX-License-Identifier: Apache-2.0
"""Check exact source clocks and outcome counts for short face changes on two carriers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_face_short_prefix_transfer import EXPECTED_ROLES, WINDOW_STARTS, iou, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock_tick(carrier, index):
    return (100 * index + 1) // 3 if carrier == "vp9" else index * 1000


def native_overlap(first, second, carrier):
    return iou([clock_tick(carrier, i) for i in first], [clock_tick(carrier, i) for i in second])


def public_row(measured, incumbent):
    require(
        all(measured[key] == incumbent[key] for key in ("group", "role", "sha256")),
        "Comparison ran on different encoded bytes",
    )
    clock = measured["clock"]
    pts, rgb = clock["pts"], measured["native_hashes"]
    require(len(pts) == len(rgb) == 240, "Missing native clock or pixel commitments")
    first = pts[0]
    require(
        all(value - first == clock_tick(measured["carrier"], i) for i, value in enumerate(pts)),
        "Native ticks differ from the carrier cadence",
    )
    used = {a["frame_index"] for w in measured["windows"] if w["match"] for a in w["match"]["anchors"]}
    return {
        k: measured[k]
        for k in ("group", "role", "carrier", "sha256", "truth", "windows", "face_inputs", "elapsed")
    } | {
        "time_base": clock["time_base"],
        "first_pts": first,
        "last_pts": pts[-1],
        "current": incumbent["ranges"],
        "current_shot_boundaries": incumbent["ordinary_shot_boundaries"],
        "native_pts_sha256": hashlib.sha256(json.dumps(pts, separators=(",", ":")).encode()).hexdigest(),
        "native_rgb_sha256": hashlib.sha256(b"".join(bytes.fromhex(digest) for digest in rgb)).hexdigest(),
        "anchors_rgb": {str(i): rgb[i] for i in sorted(used)},
    }


def assess(rows, carrier):
    totals = {role: Counter() for role in EXPECTED_ROLES}
    groups = set()
    files = set()
    for row in rows:
        require(
            row["carrier"] == carrier
            and row["time_base"] == ("1/1000" if carrier == "vp9" else "1/30000")
            and row["last_pts"] - row["first_pts"] == clock_tick(carrier, 239),
            "Incorrect carrier clock",
        )
        key = row["group"], row["role"]
        require(key not in files, "Duplicated source/delivery role")
        files.add(key)
        require(
            len(row["windows"]) == 3 and tuple(w["window_start"] for w in row["windows"]) == WINDOW_STARTS,
            "Window plan changed",
        )
        truth, current = row["truth"], row["current"]
        observed = []
        for w in row["windows"]:
            match = w["match"]
            if not match:
                continue
            a, b = match["frames"]
            start = w["window_start"]
            require(
                w["assessed"]
                and start + 6 <= a < b <= start + 114
                and 5 <= b - a <= 7
                and match["response"] > 0.1
                and a in w["edges"]
                and b in w["edges"],
                "Native range has insufficient support",
            )
            indices = [f["frame_index"] for f in match["anchors"]]
            require(
                len(indices) == 3 and indices[0] == a - 1 and a <= indices[1] < b and indices[2] == b,
                "The three native roles disagree with the source range",
            )
            for i, f in zip(indices, match["anchors"], strict=True):
                require(
                    f["pts"] == row["first_pts"] + clock_tick(carrier, i)
                    and f["decoded_rgb_sha256"] == row["anchors_rgb"][str(i)]
                    and len(bytes.fromhex(f["aligned_rgb_sha256"])) == 32,
                    "Incorrect original native anchor",
                )
            if not any(
                native_overlap(match["frames"], older, carrier) >= Fraction(1, 2)
                for older in current + observed
            ):
                observed.append(match["frames"])

        def located(p, ranges):
            return any(native_overlap(p, r, carrier) >= Fraction(1, 2) for r in ranges)

        c = totals[row["role"]]
        c["files"] += 1
        c["fully_assessed"] += all(w["assessed"] for w in row["windows"])
        c["known"] += len(truth)
        c["late_known"] += sum(a >= 120 for a, _ in truth)
        c["incumbent_located"] += sum(located(p, current) for p in truth)
        c["combined_located"] += sum(located(p, current + observed) for p in truth)
        c["added_exact"] += sum(p in observed and p not in current for p in truth)
        if truth:
            c["wrong_added"] += sum(not located(r, truth) for r in observed)
            for p in truth:
                if p[0] >= 120 and located(p, observed) and not located(p, current):
                    c["added_late"] += 1
                    c["exact_added_late"] += p in observed
                    groups.add(row["group"])
        else:
            c["incumbent_flagged"] += bool(current)
            c["added_ordinary"] += bool(observed)
            c["new_flagged_file"] += bool(observed and not current)
    count = sum(totals.values(), Counter())
    return {
        "totals": dict(count),
        "roles": {k: dict(v) for k, v in totals.items()},
        "new_late_groups": len(groups),
        "max_face_inputs": max(row["face_inputs"] for row in rows),
        "max_elapsed_s": max(row["elapsed"] for row in rows),
    }


def advancement(count, gate):
    t = count["totals"]
    return {
        "late": t.get("added_late", 0) >= gate["minimum_recovered_late"],
        "groups": count["new_late_groups"] >= gate["minimum_groups"],
        "exact": t.get("exact_added_late", 0) >= gate["minimum_exact"],
        "ordinary": t.get("added_ordinary", 0) <= gate["maximum_new_ordinary"],
        "cuts_and_loops": sum(
            count["roles"][role].get("added_ordinary", 0) for role in ("genuine_cut", "whole_loop")
        )
        <= gate["maximum_new_cut_or_loop"],
        "coverage": t["fully_assessed"] / t["files"] >= gate["minimum_full_coverage"],
        "wrong_edit": t.get("wrong_added", 0) <= gate["maximum_wrong_edited"],
    }


def expected_audit(rows):
    counts = Counter(deliveries=0, intervals=0, pts=0, source_rgb_anchors=0)
    for row in rows:
        if row["anchors_rgb"]:
            counts.update(
                deliveries=1,
                intervals=sum(bool(w["match"]) for w in row["windows"]),
                pts=240,
                source_rgb_anchors=len(row["anchors_rgb"]),
            )
    return dict(counts)


def audit_native(folder, panel):
    carrier = panel["carrier"]
    suffix = "webm" if carrier == "vp9" else "mp4"
    encoding = ("vp9", "[0][0][0][0]") if carrier == "vp9" else ("h264", "avc1")
    for row in panel["rows"]:
        if not row["anchors_rgb"]:
            continue
        media = folder / f"{carrier}-{panel['split']}-{row['group']}" / f"{row['role']}.{suffix}"
        require(sha(media) == row["sha256"], "Encoded source differs")
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,codec_tag_string,width,height,time_base:frame=best_effort_timestamp",
                    "-of",
                    "json",
                    str(media),
                ]
            )
        )
        [stream] = probe["streams"]
        pts = [f["best_effort_timestamp"] for f in probe["frames"]]
        require(
            (stream["codec_name"], stream["codec_tag_string"]) == encoding
            and (stream["width"], stream["height"]) == (1280, 720)
            and stream["time_base"] == row["time_base"]
            and hashlib.sha256(json.dumps(pts, separators=(",", ":")).encode()).hexdigest()
            == row["native_pts_sha256"],
            "Independently probed clock differs",
        )
        indices = sorted(map(int, row["anchors_rgb"]))
        selection = "+".join(f"eq(n\\,{i})" for i in indices)
        rgb = subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(media),
                "-an",
                "-vf",
                f"select={selection}",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        stride = 1280 * 720 * 3
        require(len(rgb) == stride * len(indices), "Independently decoded anchor count differs")
        for j, i in enumerate(indices):
            require(
                hashlib.sha256(rgb[j * stride : (j + 1) * stride]).hexdigest() == row["anchors_rgb"][str(i)],
                "Independently decoded anchor pixels differ",
            )
    return expected_audit(panel["rows"])


def assemble(folder, target):
    lock = read(folder / "freeze.json")
    document = {
        "schema": "chipmunk.video-face-short-carrier-public/1",
        "freeze_sha256": sha(folder / "freeze.json"),
        "gate": lock["gate"],
        "method": lock["method"],
        "formats": lock["formats"],
        "point": lock["point"],
        "binaries": lock["binaries"],
        "weights": lock["weights"],
        "panels": [],
    }
    for carrier in ("avc", "vp9"):
        for split in ("development", "held"):
            path = folder / f"{carrier}-{split}.json"
            scored = read(path)
            baseline = read(folder / f"{carrier}-{split}-incumbent.json")
            require(
                scored["freeze_sha256"] == document["freeze_sha256"]
                and baseline["scored_panel_sha256"] == sha(path),
                "The panel does not match its frozen scoring and incumbent",
            )
            rows = [
                public_row(row, other) for row, other in zip(scored["rows"], baseline["rows"], strict=True)
            ]
            require(
                {(r["group"], r["role"]) for r in rows}
                == {(i, r) for i in range(1, len(lock[split]) + 1) for r in EXPECTED_ROLES},
                "Missing complete carrier and role roster",
            )
            count = assess(rows, carrier)
            native_audit = read(folder / f"{carrier}-native-audit.json") if split == "held" else None
            require(
                native_audit is None or native_audit == expected_audit(rows), "Native audit totals differ"
            )
            document["panels"].append(
                {
                    "carrier": carrier,
                    "split": split,
                    "rows": rows,
                    "summary": count,
                    "advancement": advancement(count, lock["gate"]) if split == "held" else None,
                    "native_audit": native_audit,
                }
            )
    target.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    replay(document)


def replay(document):
    require(document["schema"] == "chipmunk.video-face-short-carrier-public/1", "Unknown study receipt")
    for panel in document["panels"]:
        count = assess(panel["rows"], panel["carrier"])
        require(count == panel["summary"], "Published outcomes do not recompute")
        if panel["split"] == "held":
            require(
                advancement(count, document["gate"]) == panel["advancement"],
                "Published gate does not recompute",
            )
            require(
                expected_audit(panel["rows"]) == panel["native_audit"], "Published native audit totals differ"
            )
        print(panel["carrier"], panel["split"], count["totals"], panel["advancement"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    p.add_argument("--audit-source", type=Path)
    a = p.parse_args()
    if a.assemble:
        assemble(a.assemble, a.receipt)
    else:
        document = read(a.receipt)
        replay(document)
        if a.audit_source:
            for panel in document["panels"]:
                if panel["split"] == "held":
                    print(panel["carrier"], "native source audit", audit_native(a.audit_source, panel))


if __name__ == "__main__":
    main()
