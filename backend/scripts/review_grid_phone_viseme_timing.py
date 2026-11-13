# SPDX-License-Identifier: Apache-2.0
"""Replay held phone-shape timing decisions without models or media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

TICK = 640
CONTROLS = ("recorded", "local_gain", "local_lowpass", "local_noise", "constant_early", "constant_late")
EDITS = ("short_early", "short_late", "long_early", "long_late")
REDRAWS = ("original", "source_time_neural", "earlier_drive_neural", "earlier_pixels_ordinary")
SHAPES = {"point": (2, 6, CONTROLS), "held": (3, 8, CONTROLS + EDITS), "redraw": (2, 1, REDRAWS)}
SCHEMA = "chipmunk.grid-phone-viseme-timing-results/1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def useful(location, truth, lag):
    if location is None or truth is None or lag is None or location["relative_lag_ticks"] is None:
        return False
    a, b = location["extent_ticks"]
    c, d = truth
    overlap = max(0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return 5 * overlap >= 2 * union and abs(location["relative_lag_ticks"] - lag) <= 2


def truth_for(case, offset):
    if case in REDRAWS:
        return None if case in REDRAWS[:2] else -8
    if case not in EDITS:
        return None, None
    start = 26 + offset
    sign = -1 if case.endswith("early") else 1
    if case.startswith("short"):
        return [start, start + 16], sign * 5
    return [start - 3, start + 19], sign * 7


def validate_proposal(proposal):
    if proposal is None:
        return
    start, end = proposal["extent_ticks"]
    ref, local = (proposal[key] for key in ("reference_lag_ticks", "local_lag_ticks"))
    gains = proposal["gains"]
    left, middle, right = proposal["pair_counts"]
    require(
        12 <= start < end <= 63
        and end - start in (14, 18, 22)
        and -5 <= ref <= 5
        and -11 <= local <= 11
        and 4 <= abs(local - ref) <= 8
        and proposal["relative_lag_ticks"] == local - ref,
        "Candidate interval or lag differs from the schedule",
    )
    require(
        2 <= left <= 12
        and 3 <= middle <= end - start
        and 2 <= right <= 12
        and len(gains) == 3
        and all(math.isfinite(v) for v in gains)
        and proposal["score"] == min(gains),
        "Candidate support or score differs from the frozen rule",
    )


def summarize(rows, point):
    counts = defaultdict(Counter)
    ranked = Counter()
    extra_groups = set()
    for row in rows:
        positive = row["known_shift"] is not None
        cells = (counts[row["role"]], counts["known_shift" if positive else "controls"])
        proposed = row["best"]
        current = row["current"]
        flag = proposed is not None and proposed["score"] > point
        location = useful(proposed, row["known_extent"], row["known_shift"])
        old = any(useful(result, row["known_extent"], row["known_shift"]) for result in current["ranges"])
        added = flag and location and not old
        if added:
            extra_groups.add(row["group"])
        if positive and proposed:
            ranked.update(
                {
                    "assessed": 1,
                    "useful_before_point": int(location),
                    "correct_sign": int((proposed["relative_lag_ticks"] > 0) == (row["known_shift"] > 0)),
                    "signed_error_at_most_two_ticks": int(
                        abs(proposed["relative_lag_ticks"] - row["known_shift"]) <= 2
                    ),
                }
            )
        for cell in cells:
            cell.update(
                {
                    "files": 1,
                    "measured": int(row["measured"] > 0),
                    "flags": int(flag),
                    "correct": int(flag and location),
                    "current_measured": int(current["measured_windows"] > 0),
                    "current_flags": int(bool(current["ranges"])),
                    "current_correct": int(old),
                    "new_correct": int(added),
                    "lost_current_correct": int(old and not (flag and location)),
                    "new_ordinary_flags": int(not positive and flag and not current["ranges"]),
                }
            )
    return {
        "counts": {key: dict(cell) for key, cell in sorted(counts.items())},
        "new_groups": len(extra_groups),
        "ranking": dict(ranked),
    }


def gate(result):
    counts = result["counts"]
    edit, ordinary = counts["known_shift"], counts["controls"]
    return {
        "new_locations": edit["new_correct"] >= 4,
        "new_groups": result["new_groups"] >= 3,
        "maximum_new_ordinary_flags": ordinary["new_ordinary_flags"] <= 1,
        "minimum_coverage": min(edit["measured"] / edit["files"], ordinary["measured"] / ordinary["files"])
        >= 0.60,
        "maximum_constant_flags": counts["constant_early"]["flags"] + counts["constant_late"]["flags"] == 0,
        "preserve_current_correct": edit["lost_current_correct"] == 0,
    }


def tradeoff(rows):
    """Recount the maximum useful locations under each observed alert budget."""
    scores = {row["best"]["score"] for row in rows if row["best"] and row["best"]["score"] > 0}
    curve = []
    for point in sorted({0.0, *scores}):
        accepted = [row for row in rows if row["best"] and row["best"]["score"] > point]
        ordinary = [row for row in accepted if row["known_shift"] is None]
        found = [row for row in accepted if useful(row["best"], row["known_extent"], row["known_shift"])]
        curve.append(
            (
                len(ordinary),
                len(found),
                len({row["group"] for row in ordinary}),
                len({row["group"] for row in found}),
            )
        )
    result = {}
    for budget in (0, 1, 2, 3, 5, 10):
        flags, located, ordinary_groups, useful_groups = max(
            (row for row in curve if row[0] <= budget), key=lambda row: (row[1], -row[0], row[3])
        )
        result[str(budget)] = {
            "maximum_useful": located,
            "ordinary_flags": flags,
            "ordinary_groups": ordinary_groups,
            "useful_groups": useful_groups,
        }
    return result


def audit(receipt):
    require(receipt["schema"] == SCHEMA and set(receipt["rows"]) == set(SHAPES), "Unexpected receipt schema")
    role_groups = {}
    for split, (group_count, sources_per_group, roles) in SHAPES.items():
        groups = defaultdict(set)
        by_source = defaultdict(Counter)
        source_signatures = defaultdict(set)
        for row in receipt["rows"][split]:
            groups[row["group"]].add(row["recording"])
            by_source[row["recording"]][row["role"]] += 1
            source_signatures[row["recording"]].add((row["group"], row["recipe_offset"]))
            validate_proposal(row["best"])
            require(
                0 <= row["stable_phone_ticks"] <= 75 and (row["best"] is not None) == (row["measured"] > 0),
                "Candidate disposition disagrees with measured support",
            )
            role = row["role"]
            if split == "redraw":
                span = row["known_extent"]
                require(
                    (span is None if role == "original" else span[1] - span[0] == 16)
                    and row["known_shift"] == truth_for(role, None),
                    "Redraw label differs",
                )
            else:
                require(
                    0 <= row["recipe_offset"] < 5
                    and (row["known_extent"], row["known_shift"]) == truth_for(role, row["recipe_offset"]),
                    "Waveform label differs from the frozen recipe",
                )
        require(
            len(groups) == group_count
            and len(by_source) == group_count * sources_per_group
            and all(len(recordings) == sources_per_group for recordings in groups.values())
            and all(count == dict.fromkeys(roles, 1) for count in by_source.values())
            and all(len(signatures) == 1 for signatures in source_signatures.values()),
            "Missing or repeated cases",
        )
        role_groups[split] = set(groups)
    require(
        not role_groups["point"] & (role_groups["held"] | role_groups["redraw"]),
        "A point speaker appears in held evaluation",
    )
    point_rows = receipt["rows"]["point"]
    measured = [row["best"]["score"] for row in point_rows if row["best"] is not None]
    point = {"value": max(0.0, *measured), "controls": len(point_rows), "measured_controls": len(measured)}
    require(receipt["point"] == point, "Calibration point does not match ordinary responses")
    result = {key: summarize(receipt["rows"][key], point["value"]) for key in ("held", "redraw")}
    require(
        receipt["result"] == result and receipt["gate"] == gate(result["held"]), "Saved totals or gate differ"
    )
    require(
        receipt["held_tradeoff_diagnostic"] == tradeoff(receipt["rows"]["held"]), "Held trade-off differs"
    )
    return result


def verify_assets(row, frames, pts, clock, pcm):
    import numpy as np

    exhibits = row["native_evidence"]
    require(bool(exhibits) == row["reported"], "Evidence was attached to the wrong decision")
    if not exhibits:
        return 0
    start, stop = row["best"]["extent_ticks"]
    low, high = start * TICK, stop * TICK
    require(
        exhibits["audio_samples"] == [low, high]
        and digest(pcm[low:high].tobytes()) == exhibits["audio_sha256"],
        "Reported audio slice differs",
    )
    seen = set()
    for exhibit in exhibits["pictures"]:
        i = exhibit["frame"]
        x0, y0, x1, y1 = exhibit["crop_xyxy"]
        require(
            i not in seen
            and i in (start - 1, (start + stop) // 2, stop)
            and exhibit["pts"] == pts[i]
            and Fraction(exhibit["time_base"]) == clock
            and 0 <= x0 < x1 <= 360
            and 0 <= y0 < y1 <= 288
            and exhibit["rgb_sha256"] == digest(frames[i]),
            "Evidence frame differs",
        )
        crop = np.frombuffer(frames[i], dtype=np.uint8).reshape(288, 360, 3)[y0:y1, x0:x1]
        require(exhibit["crop_sha256"] == digest(crop.tobytes()), "Evidence crop differs")
        seen.add(i)
    return len(seen)


def verify_native(freeze, fit, stages):
    from tempfile import TemporaryDirectory

    from scripts.review_grid_articulatory_timing import recreate
    from scripts.review_grid_single_clip_release_timing import RGB_SIZE, native, output

    replay = Counter()
    fit_cases = {item["id"]: item for item in fit["sources"]}
    counts = list(map(sum, zip(*(item["counts"] for item in fit_cases.values()), strict=True)))
    require(len(fit_cases) == 30 and counts == fit["counts"], "Fitted cohort or shape counts differ")
    source_identities = {row["id"]: row for row in freeze["inputs"] + freeze["redraws"]}
    evidence = {item["id"]: item for stage in stages.values() for item in stage["native_records"]}
    case_data = defaultdict(list)
    for stage in stages.values():
        for row in stage["rows"]:
            case_data[row["id"]].append(row)
    require(
        source_identities.keys() == evidence.keys() | fit_cases.keys() and len(evidence) == 44,
        "Native inputs do not match the freeze",
    )
    splits = {
        key: {row["group"] for row in freeze["inputs"] if row["split"] == key}
        for key in ("fit", "point", "held")
    }
    require(
        len(set.union(*splits.values())) == sum(map(len, splits.values()))
        and not {row["group"] for row in freeze["redraws"]} & (splits["fit"] | splits["point"]),
        "Fit, point and held source assignments overlap",
    )
    with TemporaryDirectory(prefix="grid-clock-review-") as folder:
        for row in freeze["excluded_clocks"]:
            archive_path = next(x["archive"] for x in freeze["inputs"] if x["group"] == row["group"])
            with ZipFile(archive_path) as archive:
                payload = archive.read(row["member"])
            require(digest(payload) == row["sha256"], "Clock-excluded archive member differs")
            path = Path(folder) / "clock-excluded.mpg"
            path.write_bytes(payload)
            probe = json.loads(
                output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_frames",
                        "-show_streams",
                        "-show_entries",
                        "stream=time_base:frame=best_effort_timestamp",
                        "-of",
                        "json",
                        str(path),
                    ]
                )
            )
            clock = Fraction(probe["streams"][0]["time_base"])
            pts = [int(frame["best_effort_timestamp"]) * clock for frame in probe["frames"]]
            require(
                len(pts) == row["native_pictures"] and pts != [Fraction(i, 25) for i in range(75)],
                "Clock exclusion failed independent timestamp check",
            )
            replay["excluded_source_clocks"] += 1
    for source in source_identities.values():
        path = Path(source["path"])
        source_id = source["id"]
        require(file_digest(path) == source["sha256"], "Source bytes differ")
        if source["split"] != "redraw":
            with ZipFile(source["archive"]) as archive:
                member = archive.getinfo(source["member"])
                require(
                    source["crc32"] == member.CRC and digest(archive.read(member)) == source["sha256"],
                    "Publisher member does not reproduce the same bytes",
                )
            replay["archive_members"] += 1
        pts, clock, raw, pcm = native(path)
        frames = [raw[i : i + RGB_SIZE] for i in range(0, len(raw), RGB_SIZE)]
        replay.update({"files": 1, "pictures": len(pts), "pcm_samples": len(pcm)})
        if source_id in fit_cases:
            require(fit_cases[source_id]["pcm_sha256"] == digest(pcm.tobytes()), "Fit PCM differs")
            continue
        anchor = evidence[source_id]
        require(
            anchor["pts"] == pts
            and Fraction(anchor["time_base"]) == clock
            and anchor["rgb_sha256"] == [digest(frame) for frame in frames]
            and anchor["original_pcm_sha256"] == digest(pcm.tobytes()),
            "Held source clock or samples differ",
        )
        for row in case_data[source_id]:
            delivered = pcm if source["split"] == "redraw" else recreate(pcm, source["ordinal"], row["role"])
            require(
                digest(delivered.tobytes()) == row["pcm_sha256"],
                "Saved case does not bind to reconstructed PCM",
            )
            replay.update(
                {
                    "case_pcm_commitments": 1,
                    "evidence_crops": verify_assets(row, frames, pts, clock, delivered),
                }
            )
    return dict(replay)


def build(run):
    freeze, fit, calibration = (read(run / f"{name}.json") for name in ("freeze", "fit", "point"))
    freeze_pin, fit_pin, point_pin = (
        file_digest(run / f"{name}.json") for name in ("freeze", "fit", "point")
    )
    stages = {split: read(run / f"{split}-readout.json") for split in SHAPES}
    require(
        fit["freeze_sha256"] == freeze_pin
        and calibration["fit_sha256"] == fit_pin
        and calibration["readout_sha256"] == file_digest(run / "point-readout.json"),
        "Run stages differ",
    )
    for split, stage in stages.items():
        require(
            stage["freeze_sha256"] == freeze_pin
            and stage["fit_sha256"] == fit_pin
            and stage["point_sha256"] == (None if split == "point" else point_pin),
            "Stage commitments differ",
        )
    paths = {row["id"]: row for row in freeze["inputs"] + freeze["redraws"]}
    all_groups = sorted({row["group"] for row in paths.values()})
    public = {}
    for split, stage in stages.items():
        key_by_id = {row["id"]: row["group"] if split == "redraw" else row["id"] for row in stage["rows"]}
        ids = {key: f"{i + 1:02}" for i, key in enumerate(dict.fromkeys(key_by_id.values()))}
        public[split] = []
        for row in stage["rows"]:
            best = row["best"]
            require(
                row["reported"] == bool(split != "point" and best and best["score"] > calibration["value"]),
                "Candidate flag was not at the saved operating point",
            )
            visible = {
                key: row[key]
                for key in (
                    "role",
                    "known_extent",
                    "known_shift",
                    "stable_phone_ticks",
                    "measured",
                    "best",
                    "current",
                )
            }
            public[split].append(
                {
                    **visible,
                    "recording": ids[key_by_id[row["id"]]],
                    "group": f"{all_groups.index(row['group']) + 1:02}",
                    "recipe_offset": paths[row["id"]]["ordinal"] % 5 if split != "redraw" else None,
                }
            )
    result = {split: summarize(public[split], calibration["value"]) for split in ("held", "redraw")}
    receipt = {
        "schema": SCHEMA,
        "method": freeze["spec"],
        "freeze_sha256": freeze_pin,
        "fit_sha256": fit_pin,
        "stage_sha256": {key: file_digest(run / f"{key}-readout.json") for key in SHAPES},
        "point": {key: calibration[key] for key in ("value", "controls", "measured_controls")},
        "shape_counts": {
            "fit": fit["counts"],
            **{
                key: {
                    "counts": stage["shape_class_counts"],
                    "balanced_accuracy": stage["shape_class_accuracy"],
                }
                for key, stage in stages.items()
                if key != "redraw"
            },
        },
        "rows": public,
        "result": result,
        "gate": gate(result["held"]),
        "held_tradeoff_diagnostic": tradeoff(public["held"]),
    }
    audit(receipt)
    receipt["native_replay"] = verify_native(freeze, fit, stages)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        receipt = build(args.run)
        with args.receipt.open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    else:
        receipt = read(args.receipt)
    audit(receipt)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "point",
                    "result",
                    "gate",
                    "shape_counts",
                    "held_tradeoff_diagnostic",
                    "native_replay",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
