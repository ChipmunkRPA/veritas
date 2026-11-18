# SPDX-License-Identifier: MIT
"""Freeze and audit source-localized cropped reverse replay in a single upload."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-cropped-reverse-panel/1"
DIMENSIONS, RATE, COUNT = (640, 360), "25", 225
ORDINARY = ("original", "original_tandem", "in_place_reverse_crop", "moving_crop", "held_crop")
OTHER_EDITS = ("reverse_only", "crop_only")
EDITS = ("center_reverse", "offset_reverse", "fractional_reverse", "fractional_reverse_tandem")
CENTER_EDITS = ("center_reverse_80", "center_reverse_85", "center_reverse_90")
CROP = "cropped_motion_sequence_review"
GATE = {"min_useful": 20, "min_improvement": 12, "min_sources": 6, "max_wrong": 1, "max_ordinary": 1, "max_other": 0}
CENTER_GATE = {"min_useful": 14, "min_improvement": 12, "min_sources": 6, "max_wrong": 1, "max_ordinary": 1, "max_other": 0}


def sha(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def load(path):
    body = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(body)


def save(path, value):
    body = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)


def authored(digest, role):
    """Assign unaltered and replay positions without looking at the pictures."""
    early = 24 + int(digest[:2], 16) % 17
    late = 135 + int(digest[2:4], 16) % 18
    earlier = role == "offset_reverse" or role in CENTER_EDITS and int(digest[12:14], 16) % 2
    reference, target = (late, early) if earlier else (early, late)
    count = (23 if role.startswith("center_reverse") else 26) + int(digest[4:6], 16) % 7
    left = top = .075
    width = .85
    if role == "offset_reverse":
        width = .80
        left, top = .20, 0.
    if role.startswith("fractional_reverse"):
        width = (.7765, .8265, .8765)[int(digest[14:16], 16) % 3]
        left, top = [(1 - width) * (.28 + int(digest[i:i+2], 16) / 255 * .42) for i in (16, 18)]
    if role in CENTER_EDITS:
        width = int(role.rsplit("_", 1)[1]) / 100
        left = top = (1 - width) / 2
    if role in {"in_place_reverse_crop", "moving_crop"}:
        reference = target
    region = [left, top, left + width, top + width]
    truth = ({"ranges": [[early, early + count - 1], [late, late + count - 1]],
              "wider_range": "first" if reference == early else "second", "crop_box": region}
             if role in (*EDITS, *CENTER_EDITS) else None)
    return reference, target, count, region, truth


def planned_files(roster, *, center_only=False):
    edits = CENTER_EDITS if center_only else EDITS
    return [
        {"source": source["id"], "split": source["split"], "role": role,
         "file": f"{source['id']}-{role}.mp4", "truth": authored(source["sha256"], role)[-1]}
        for source in roster for role in (*ORDINARY, *OTHER_EDITS, *edits)
    ]


def _method_paths():
    backend = Path(__file__).resolve().parents[1]
    return [backend / name for name in (
        "scripts/video_cropped_reverse_panel.py", "scripts/video_cropped_reverse_compare.py",
        "app/detectors/video/picture_crop.py", "app/detectors/video/picture_crop_aligned.py",
        "app/detectors/video/picture_crop_reverse.py",
    )]


def freeze(roster_path, output, *, center_only):
    roster = load(roster_path)
    expected = {"held": 8} if center_only else {"development": 2, "held": 8}
    if (Counter(row["split"] for row in roster) != expected
            or len({row["sha256"] for row in roster}) != len(roster)):
        raise ValueError("The fixed roster has an incorrect split or repeats an original capture group")
    value = {"schema": SCHEMA, "source_roster": roster, "planned_cases": planned_files(roster, center_only=center_only),
             "method_sha256": {path.name: sha(path) for path in _method_paths()},
             "gate": CENTER_GATE if center_only else GATE,
             "profile": {"dimensions": DIMENSIONS, "rate": RATE, "frames": COUNT,
                                          "codec": "h264", "pixel_format": "yuv420p", "carrier": "mp4"},
             "fit": "No tuning on these sources; use the published crop alignment gates in reverse order.",
             "useful": "Both ranges have at least 0.50 inclusive frame IoU, wider occurrence is correct, "
                       "and every crop edge is within two 192x108 working pixels of the authored crop."}
    if center_only:
        value["center_only"] = True
    save(output, value)
    print("Frozen", len(value["planned_cases"]), "deliveries, plan SHA-256", sha(output), flush=True)


def _prepare(source, path):
    import numpy as np

    width, height = DIMENSIONS
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-ss", f"{float(source['offset_s']):.3f}", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
        f"fps={RATE},scale={width}:{height}:force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{height},setsar=1,format=yuv420p",
        "-frames:v", str(COUNT), "-f", "rawvideo", "-threads:v", "1", "pipe:1",
    ]
    decoded = subprocess.run(command, capture_output=True, timeout=60, check=True).stdout
    pixels = width * height * 3 // 2
    if len(decoded) != pixels * COUNT:
        raise ValueError(f"Short published capture {source['id']}")
    return np.frombuffer(decoded, dtype=np.uint8).reshape(COUNT, pixels)


def _clips(pictures, digest, role):
    import numpy as np

    from scripts.video_cropped_reuse_panel import _transform

    if role == "original":
        return (pictures,)
    reference, target, count, region, _ = authored(digest, role)
    donor = pictures[reference:reference + count]
    if role == "reverse_only":
        edited = np.ascontiguousarray(donor[::-1])
    elif role == "crop_only":
        edited = _transform(donor, DIMENSIONS, region)
    elif role == "moving_crop":
        edited = _transform(donor, DIMENSIONS, region, (0.12, 0.16, 0.88, 0.92))
    elif role == "held_crop":
        edited = np.repeat(_transform(donor[:1], DIMENSIONS, region), count, axis=0)
    else:
        edited = _transform(donor[::-1], DIMENSIONS, region)
    return pictures[:target], edited, pictures[target+count:]


def render(pictures, digest, role, destination):
    from scripts.video_reflected_hd_panel import _encode

    if role.endswith("_tandem"):
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            prior = Path(tmp) / "first.mp4"
            render(pictures, digest, role.removesuffix("_tandem"), prior)
            _encode(destination, DIMENSIONS, RATE, "h264", (), prior)
    else:
        _encode(destination, DIMENSIONS, RATE, "h264", _clips(pictures, digest, role), None)


def measure(path, *, center_only=False):
    from contextlib import closing

    import cv2

    from app.detectors.video import picture_crop, picture_hold
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence import picture_sequence as binding
    from scripts import video_cropped_reverse_compare as candidate
    from scripts.video_face_portrait_hevc_transfer import rgb_anchors

    cv2.setNumThreads(1)
    table = probe_frames(path)
    if (len(table) != COUNT or (table.width, table.height) != DIMENSIONS or table.interpolated
            or table.rotation or [value * table.tb for value in table.pts] != [Fraction(i, int(RATE)) for i in range(COUNT)]):
        raise ValueError(f"Authored delivery lacks its exact native clock: {path.name}")
    cover = picture_hold.inspect(
        path, table, origin=0., sequence_review=True, motion_encoding=("h264", "yuv420p"), retimed_review=True,
        fractional_retimed_review=True, reflected_review=True, cropped_review=True,
    )["coverage"]
    current = {}
    for name in ("sequence_review", binding.REVERSE_KEY, *(key for key, _, _ in binding.MOTION_REVIEWS)):
        if pair := cover.get(name, {}).get("pair"):
            current[name] = {side: [pair[side]["start_frame_index"], pair[side]["end_frame_index"]]
                             for side in ("first", "second")}
    working, native = [], []
    with closing(decode_frames(path, table, range(COUNT), timeout_s=20)) as decoded:
        for index, frame in enumerate(decoded):
            if not frame.pts_verified or table.pts[index] != frame.pts or index != frame.index:
                raise ValueError(f"Unverified picture in {path.name}")
            working.append(picture_crop.luminance(frame.rgb))
            native.append([index, frame.pts, hashlib.sha256(frame.rgb).hexdigest()])
    if len(working) != COUNT:
        raise ValueError(f"Incomplete decoded sequence in {path.name}")
    started = time.monotonic()
    match, alignment = candidate.find(working, center_only=center_only)
    elapsed = round(time.monotonic() - started, 3)
    proposal = None
    if match:
        anchors = {f"{side}_{boundary}": native[index]
                   for side, pair in zip(("first", "second"), match.ranges, strict=True)
                   for boundary, index in zip(("start", "end"), pair, strict=True)}
        rgb_anchors(path, {"proposal": {"anchors": anchors}})
        proposal = match._asdict() | {"ranges": [list(pair) for pair in match.ranges],
                                      "alignment": alignment, "anchors": anchors}
    return {"baseline": current, "candidate": proposal, "time_base": table.time_base, "seconds": elapsed,
            "file_sha256": sha(path), "native_frames": len(native)}


def _overlaps(actual, truth):
    return all(2 * (min(b, d) - max(a, c) + 1) >= max(b, d) - min(a, c) + 1
               for (a, b), (c, d) in zip(actual, truth, strict=True))


def _useful(proposal, truth):
    if not proposal or not truth or proposal["wider_range"] != truth["wider_range"]:
        return False
    actual = proposal["crop_box"]
    units = (192 * 32, 108 * 32) * 2
    return (_overlaps(proposal["ranges"], truth["ranges"])
            and all(abs(p - fraction * unit) <= 64 for p, fraction, unit in zip(actual, truth["crop_box"], units, strict=True)))


def report(rows, *, center_only=False):
    edits = CENTER_EDITS if center_only else EDITS
    gate = CENTER_GATE if center_only else GATE
    totals = Counter()
    by_role = {}
    matched_sources = set()
    for row in rows:
        role, proposal, truth = (row[k] for k in ("role", "candidate", "truth"))
        added = proposal if CROP not in row["baseline"] else None
        is_useful = _useful(added, truth)
        target = by_role.setdefault(role, Counter())
        for field, value in {
            "files": 1,
            "legacy_any_alert": bool(row["baseline"]),
            "legacy_localization": bool(truth and any(_overlaps((v["first"], v["second"]), truth["ranges"])
                                                   for v in row["baseline"].values())),
            "candidate_alert": bool(proposal),
            "added_alert": bool(added),
            "useful": is_useful,
            "exact": bool(is_useful and added["ranges"] == truth["ranges"]),
        }.items():
            target[field] += value
        if role in edits:
            totals["positive"] += 1
            totals["useful"] += is_useful
            totals["misplaced"] += bool(added and not is_useful)
            if is_useful:
                matched_sources.add(row["source"])
        else:
            key = "ordinary" if role in ORDINARY else "other"
            totals[key] += 1
            totals[f"added_{key}"] += bool(added)
    totals["matched_sources"] = len(matched_sources)
    totals["legacy_localization"] = sum(metrics["legacy_localization"] for metrics in by_role.values())
    passed = (totals["useful"] >= gate["min_useful"]
              and totals["useful"] - totals["legacy_localization"] >= gate["min_improvement"]
              and totals["matched_sources"] >= gate["min_sources"] and totals["misplaced"] <= gate["max_wrong"]
              and totals["added_ordinary"] <= gate["max_ordinary"] and totals["added_other"] <= gate["max_other"])
    return dict(totals) | {"passed": passed, "by_role": by_role,
                          "median_candidate_seconds": statistics.median(row["seconds"] for row in rows)}


def record(plan_path, sources, split, output):
    plan = load(plan_path)
    center_only = plan.get("center_only", False)
    pins = {path.name: sha(path) for path in _method_paths()}
    if plan["method_sha256"] != pins or plan["planned_cases"] != planned_files(plan["source_roster"], center_only=center_only):
        raise ValueError("Source roster, method or local authoring changed after the freeze")
    chosen = [row for row in plan["planned_cases"] if row["split"] == split]
    scored = []
    for source in plan["source_roster"]:
        if source["split"] != split:
            continue
        media = sources / f"{source['id']}.webm"
        if sha(media) != source["sha256"] or media.stat().st_size != source["size"]:
            raise ValueError(f"Publisher file changed for {source['id']}")
        pictures = _prepare(source, media)
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            for case in (row for row in chosen if row["source"] == source["id"]):
                delivery = Path(folder) / case["file"]
                render(pictures, source["sha256"], case["role"], delivery)
                reading = case | measure(delivery, center_only=center_only)
                scored.append(reading)
                delivery.unlink()
        save(output, plan | {"split": split, "plan_sha256": sha(plan_path), "rows": scored,
                            "summary": report(scored, center_only=center_only),
                            "complete": len(scored) == len(chosen)})
        print(source["id"], len(scored), "of", len(chosen), "deliveries", flush=True)


def verify(path):
    value = load(path)
    rows = value["rows"]
    center_only = value.get("center_only", False)
    assert value["schema"] == SCHEMA and value["complete"] and value["gate"] == (CENTER_GATE if center_only else GATE)
    assert value["planned_cases"] == planned_files(value["source_roster"], center_only=center_only)
    chosen = [row for row in value["planned_cases"] if row["split"] == value["split"]]
    assert len(chosen) == len(rows)
    assert len({row["file_sha256"] for row in rows}) >= len(chosen) - len(value["source_roster"])
    for plan, row in zip(chosen, rows, strict=True):
        assert all(row[k] == v for k, v in plan.items()) and row["native_frames"] == COUNT
        if proposal := row["candidate"]:
            (a, b), (c, d) = proposal["ranges"]
            assert 0 <= a < b < c < d < COUNT and b - a == d - c
            assert 5 * len(proposal["support"]) >= 3 * (b - a)
            assert all(0 <= edge < b - a for edge in proposal["support"])
            for index, key in zip((a, b, c, d), ("first_start", "first_end", "second_start", "second_end"), strict=True):
                frame, tick, digest = proposal["anchors"][key]
                assert index == frame and Fraction(row["time_base"]) * tick == Fraction(index, int(RATE))
                assert len(bytes.fromhex(digest)) == 32
    assert value["summary"] == report(rows, center_only=center_only)
    print(json.dumps(value["summary"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("roster", type=Path)
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--center-only", action="store_true")
    create = commands.add_parser("score")
    create.add_argument("plan", type=Path)
    create.add_argument("sources", type=Path)
    create.add_argument("split", choices=("development", "held"))
    create.add_argument("output", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.roster, args.plan, center_only=args.center_only)
    elif args.action == "score":
        record(args.plan, args.sources, args.split, args.output)
    else:
        verify(args.receipt)


if __name__ == "__main__":
    main()
