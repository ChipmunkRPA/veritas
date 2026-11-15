# SPDX-License-Identifier: MIT
"""Test interleaved observations at a measured lip-timing change."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_aperture_scale_transfer as aperture
from scripts import video_lip_dense_native as native
from scripts import video_lip_dense_posteriors as dense

SCHEMA = "chipmunk.video-lip-timing-block-support/1"
POINT_SOURCE = aperture.PANELS / "video-lip-aperture-scale-2026-11-15.json"
INTAKE_SHA = "1d2cb7f65ab2a87a287f2b09c1809b06bf60bf85b9dadb93bb924968a858c2b8"
PARENT_SHA = {
    "first": "3e96b9d824e5eed3668e08edfd94d1364a82af176cce3794d05e8e2fec89c51e",
    "remaining": "860e4e775d25ea94351b1ce76a5203e0ebe2f71e989eb39fba307f37e221264e",
}
WIDTHS = {"alternating_40ms": 640, "alternating_80ms": 1280}
ARMS = ("sparse", "rank_fallback", *WIDTHS, "change_direction")
METHOD = {
    "check": "Retain the existing sparse finding and point. Use the previous ranked winner only as a fallback. "
             "At that exact interval, group every observation into even and odd time blocks from the audio origin. "
             "Each split must independently prefer the winning signs in all three regions.",
    "variants": "Separately test 40 and 80 ms blocks. Each split must contain an observation on both flanks and "
                "at least two inside. Sum integer score differences at the same candidate lags. Equal or negative "
                "signed support cannot verify the respective region. The splits reuse correlated signals.",
    "change_direction": "After reviewing the first two rosters and before scoring the separate native recording "
                       "bank, add a test of signed rank changes over 80 ms. Require the measured lag to fit "
                       "these changes better than the alternate lag in all three regions, with one observation "
                       "per flank and two inside. Only use original timestamped frames.",
    "parameters": "Use the previous frozen tighter points, lag grid and staged edit overlap rule. No new fitting. "
                  "Keep native audio extents and picture timestamps in every group.",
    "sources": "Recount both existing native actor rosters. Separately decode all 15 archived digest-verified "
               "native recordings from the earlier short-delay intake before comparing any result. "
               "Those actor codes are distinct from both native rosters. Keep even unsupported cases in coverage.",
}
# The separate acquisition does not depend on the later direction comparison.
CAPTURE_PLAN = {key: value for key, value in METHOD.items() if key != "change_direction"}
sha, load = native.study.phone.sha, native.study.load


def require(condition, message):
    if not condition:
        raise ValueError(message)


def support(events, measurement, width=None):
    """Recheck the fixed signed change with optional interleaved time blocks."""
    start, stop = measurement["span"]
    outside = timing.LAGS.index(measurement["baseline_lag"])
    inside = timing.LAGS.index(measurement["baseline_lag"] + measurement["change"])
    blocks = [[[0, 0] for _ in range(3)] for _ in range(2 if width else 1)]
    for sample, _, scores, _ in events:
        region = 0 if sample < start else 2 if sample >= stop else 1
        location = blocks[(sample // width) % 2 if width else 0][region]
        location[0] += 1
        gain = scores[inside] - scores[outside]
        location[1] += gain if region == 1 else -gain
    return blocks


def corroborated(blocks):
    return all(count >= (2 if i == 1 else 1) and gain > 0
               for split in blocks for i, (count, gain) in enumerate(split))


def motion_scores(row, prediction):
    points = timing.observations(row)
    valid = sorted(value for value in prediction if value is not None)
    if points is None or len(valid) < 8 or valid[-1] == valid[0]:
        return []
    raw = {frame["index"]: frame["gap"] for frame in row["frames"] if frame["gap"] is not None}
    gaps = sorted(raw.values())
    ranks = {index: ranked.middle_rank(gaps, value) for index, value in raw.items()}
    events = []
    for i in range(4, len(prediction) - 4, 2):
        before, after = prediction[i - 2], prediction[i + 2]
        if before is None or after is None or before == after:
            continue
        at = ranked.CENTER + i * ranked.STRIDE
        pairs = [(timing.nearest(points, at - lag - timing.STEP), timing.nearest(points, at - lag + timing.STEP))
                 for lag in timing.LAGS]
        if not all(left is not None and right is not None and left[1] != right[1] for left, right in pairs):
            continue
        target = ranked.middle_rank(valid, after) - ranked.middle_rank(valid, before)
        values = [ranks[right[1]] - ranks[left[1]] for left, right in pairs]
        scores = [round(-timing.SCORE_UNITS * ((value - target) / timing.SCALE) ** 2) for value in values]
        events.append((at, "motion", scores, []))
    return events


def flagged(row, events, cutoff):
    result = timing.hypotheses(events, row["samples"])
    if result.get("score", 0) <= cutoff:
        return None
    witness = timing.picture_evidence(row["frames"], result, row["samples"])
    return result if all(witness["picture_brackets"].values()) else None


def outcomes(rows, points, references):
    counts, estimates = defaultdict(Counter), []
    for source in rows:
        scale = Fraction(source["time_base"]) * timing.RATE
        frames = [frame | {"sample": round(frame["pts"] * scale) - source["audio_origin"],
                           "gap": None if frame["gap_units"] is None else frame["gap_units"] / timing.SCORE_UNITS}
                  for frame in source["frames"]]
        row = source | {"frames": frames}
        for role, values in row["cases"].items():
            shift = native.study.parent.EDITS.get(role)
            kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
            phone_events = timing.compatible(row, values["phones"], references)
            rank_events = ranked.scores(row, values["prediction"])
            regular = flagged(row, phone_events, points["sparse"])
            extra = None if regular else flagged(row, rank_events, points["per_clip_ranks"])
            blocks = {arm: support(rank_events, extra, width) for arm, width in WIDTHS.items()} if extra else {}
            if extra:
                blocks["change_direction"] = support(motion_scores(row, values["prediction"]), extra)
            accepted = {"rank_fallback"} | {arm for arm, block in blocks.items() if corroborated(block)}
            for name in ARMS:
                kept = regular or (extra if name in accepted else None)
                useful = bool(kept and shift and native.study.parent.useful(kept["span"], kept["change"], row["edit"], shift))
                counts[f"{name}/{kind}"].update(files=1, flagged=bool(kept), useful=useful,
                                                misplaced=bool(kept and shift and not useful))
                counts[f"{name}/{role}"].update(files=1, flagged=bool(kept), useful=useful)
            if extra:
                estimates.append({"source": row["source"], "role": role, "span": extra["span"], "shift": extra["change"],
                                  "score": extra["score"], "blocked_support": blocks,
                                  "useful": bool(shift and native.study.parent.useful(
                                      extra["span"], extra["change"], row["edit"], shift))})
    return {"counts": counts, "fallback_estimates": estimates}


def definitions():
    authority = load(POINT_SOURCE)
    points = {arm: authority["points"][arm]["tight"] for arm in ("sparse", "per_clip_ranks")}
    require(points == {"sparse": binding.POINT, "per_clip_ranks": ranked.POINT}, "Serving points changed")
    return points


def capture(args, primary):
    import numpy as np

    from app.detectors.video.decode import probe_frames
    from scripts import video_native_sources_v92 as clocks

    intake = args.sources / "intake-receipt.json"
    require(sha(intake) == INTAKE_SHA and sha(args.geometry) == native.study.parent.FACE_SHA256, "Source roster or geometry changed")
    sys.path.insert(0, str(args.runtime.resolve()))
    read_audio, device = dense.reader(args.checkpoint)
    rows, unavailable = [], []
    items = load(intake)["rows"]
    historical = {item["group"] for remainder in (False, True) for item in native.native.roster(args.audit, remaining=remainder)}
    require(all(f"CREMA-D:{item['actor']}" not in historical for item in items), "A comparison actor already appeared")
    for item in items:
        source = args.sources / item["file"]
        require(source.stat().st_size == item["bytes"] and sha(source) == item["sha256"], "A source clip has changed")
        table = probe_frames(source, cap=121, timeout_s=20)
        probed = clocks.probe_json(source, ["-select_streams", "a:0", "-show_streams", "-show_frames"])
        stream, = probed["streams"]
        clock = clocks.audio_clocks(stream, probed["frames"])
        if not (clock["contiguous"] and clock["channels"] == 1 and table.time_base == "1/1000"
                and not table.rotation and not table.interpolated and not table.truncated and 24 <= len(table) <= 120
                and all(b-a in (33, 34) for a, b in zip(table.pts, table.pts[1:]))
                and max(abs(pts-table.pts[0]-Fraction(i*1001, 30)) for i, pts in enumerate(table.pts)) <= Fraction(3, 2)):
            unavailable.append({"source": item["ordinal"], "reason": "native_clock"})
            continue
        exact = clocks.pcm_content(source)
        if not 24_000 <= exact["samples"] <= 64_000:
            unavailable.append({"source": item["ordinal"], "reason": "sample_extent"})
            continue
        qualified = {"input_sha256": item["sha256"], "frame_table": table.to_json(), "canonical_audio": exact}
        frames, table = native.pixels(source, qualified, args.geometry)
        pcm = native.audio(source, qualified)
        require(pcm.dtype == np.dtype("<i2"), "Audio is outside the analysis sample format")
        origin = round(Fraction(clock["frames"][0]["start"]) * timing.RATE)
        edit = [(len(pcm) - native.native.EDIT_WIDTH) // 2, (len(pcm) + native.native.EDIT_WIDTH) // 2]
        cases = {}
        for role, treated in native.study.parent.treatments(pcm, edit).items():
            phonemes, events = read_audio(treated, include_events=True)
            cases[role] = {"pcm_sha256": native.study.phone.digest(treated.tobytes()), "phones": events,
                           "prediction": dense.predict(phonemes, primary["fit"]["weights"])}
        rows.append({"source": item["ordinal"], "sha256": item["sha256"], "samples": len(pcm), "edit": edit,
                     "audio_origin": origin, "time_base": table.time_base, "frames": frames, "cases": cases})
        print(f"Verified alternate native recording {item['ordinal']}/{len(items)}.", flush=True)
    return {"schema": SCHEMA + "/native", "method": CAPTURE_PLAN, "model": dense.MODEL,
            "intake_sha256": INTAKE_SHA, "primary_sha256": sha(aperture.PRIMARY), "device": device,
            "rows": rows, "unavailable": unavailable}


def evaluate(args):
    primary = dense.review(aperture.PRIMARY)
    points = definitions()
    alternate = load(args.alternate)
    require(alternate["schema"] == SCHEMA + "/native" and alternate["primary_sha256"] == sha(aperture.PRIMARY)
            and alternate["method"] == CAPTURE_PLAN and alternate["model"] == dense.MODEL
            and alternate["intake_sha256"] == INTAKE_SHA, "Separate native capture changed")
    report = {}
    for name in ("first", "remaining"):
        path = getattr(args, name)
        require(sha(path) == PARENT_SHA[name], "An earlier native capture changed")
        document = load(path)
        roster = native.native.roster(args.audit, remaining=name == "remaining")
        require(document["result"] == native.evaluate(roster, document["rows"], primary), "Native inputs do not replay")
        report[name] = outcomes(document["rows"], points, binding.REFERENCE)
    accepted, skipped = alternate["rows"], alternate["unavailable"]
    require(sorted(item["source"] for item in accepted + skipped) == list(range(1, 16))
            and all(set(row["cases"]) == set(native.study.parent.ROLES) for row in accepted), "Alternate source coverage differs")
    report["alternate"] = outcomes(accepted, points, binding.REFERENCE)
    report["alternate"]["unassessed_source_clips"] = len(skipped)
    return {"schema": SCHEMA, "method": METHOD, "model": dense.MODEL, "points": points,
            "source": PARENT_SHA | {"earlier_acquisition": INTAKE_SHA, "alternate": sha(args.alternate),
                                   "rank_comparison": sha(POINT_SOURCE)}, "results": report}


def kind_for(role):
    return "local" if role in native.study.parent.EDITS else "constant" if role.startswith("global_") else "ordinary"


def recount(panel):
    base = panel["baseline_by_role"]
    require(set(base) == set(native.study.parent.ROLES) and
            len({row["files"] for row in base.values()}) == 1, "Source role counts do not match")
    out = {f"{arm}/{role}": dict(row) for role, row in base.items() for arm in ARMS}
    for row in panel["fallback"]:
        role, useful, measurements = row["role"], row["useful"], row["support"]
        require(role in base and type(useful) is bool and (not useful or kind_for(role) == "local")
                and set(measurements) == {*WIDTHS, "change_direction"}, "Invalid fallback calibration record")
        for arm, blocks in measurements.items():
            require(len(blocks) == (2 if arm in WIDTHS else 1) and all(len(part) == 3 for part in blocks)
                    and all(type(n) is int and n >= 0 and sign in {-1, 0, 1}
                            and (n > 0 or sign == 0) for part in blocks for n, sign in part),
                    "Invalid block coverage or signed score")
        for arm in ARMS[1:]:
            if arm == "rank_fallback" or corroborated(measurements[arm]):
                record = out[f"{arm}/{role}"]
                record["flagged"] += 1
                record["useful"] += useful
                require(record["useful"] <= record["flagged"] <= record["files"], "Too many flags in this source role")
    for arm in ARMS:
        for role in base:
            kind = kind_for(role)
            current = out[f"{arm}/{role}"]
            record = out.setdefault(f"{arm}/{kind}", dict.fromkeys(("files", "flagged", "useful", "misplaced"), 0))
            for key in ("files", "flagged", "useful"):
                record[key] += current[key]
            record["misplaced"] += current["flagged"] - current["useful"] if kind == "local" else 0
    return out


def public_report(private):
    panels = {}
    for name, panel in private["results"].items():
        cohort = Counter(row["source"] for row in panel["fallback_estimates"] if kind_for(row["role"]) == "ordinary")
        cases = [{"role": row["role"], "useful": row["useful"],
                  "support": {arm: [[[count, (gain > 0) - (gain < 0)] for count, gain in region] for region in blocks]
                              for arm, blocks in row["blocked_support"].items()}}
                 for row in panel["fallback_estimates"]]
        public = {"baseline_by_role": {role: panel["counts"][f"sparse/{role}"] for role in native.study.parent.ROLES},
                  "fallback": sorted(cases, key=lambda row: json.dumps(row, sort_keys=True)),
                  "ordinary_alerts_per_source": sorted(cohort.values(), reverse=True),
                  "unassessed_source_clips": panel.get("unassessed_source_clips", 0)}
        results = recount(public)
        require(results == panel["counts"], "The de-identified recount differs from native results")
        panels[name] = public | {"counts": results}
    return {key: value for key, value in private.items() if key != "results"} | {"results": panels}


def review(path):
    data = load(path)
    require(data["schema"] == SCHEMA and data["method"] == METHOD and data["model"] == dense.MODEL
            and data["points"] == definitions() and set(data["results"]) == {*PARENT_SHA, "alternate"}
            and all(data["source"][key] == value for key, value in
                    (PARENT_SHA | {"earlier_acquisition": INTAKE_SHA, "rank_comparison": sha(POINT_SOURCE)}).items()),
            "The frozen method or public commitments differ")
    summary = {}
    for name, panel in data["results"].items():
        counts = recount(panel)
        sources = panel["ordinary_alerts_per_source"]
        added = counts["rank_fallback/ordinary"]["flagged"] - counts["sparse/ordinary"]["flagged"]
        require(counts == panel["counts"] and all(type(x) is int and 1 <= x <= 3 for x in sources)
                and sum(sources) == added, "The published decisions or grouped alert counts do not replay")
        summary[name] = {"counts": {key: row for key, row in counts.items()
                                    if key.split("/")[1] in {"local", "ordinary", "constant"}},
                         "ordinary_alerts_per_source": sources,
                         "unassessed_source_clips": panel["unassessed_source_clips"]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    collect = phases.add_parser("capture")
    for field in ("sources", "audit", "checkpoint", "geometry", "runtime", "output"):
        collect.add_argument(field, type=Path)
    compare = phases.add_parser("compare")
    for field in ("audit", "first", "remaining", "alternate", "output"):
        compare.add_argument(field, type=Path)
    portable = phases.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "capture":
        dense.seal(args.output, capture(args, dense.review(aperture.PRIMARY)))
    elif args.phase == "compare":
        with args.output.open("x") as output:
            json.dump(public_report(evaluate(args)), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    else:
        print(json.dumps(review(args.receipt), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
