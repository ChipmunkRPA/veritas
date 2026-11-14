# SPDX-License-Identifier: Apache-2.0
"""Measure two or three added-video ranges between copies on cached public footage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_temporal_insert_confirmation as confirmation
from scripts import video_temporal_insert_native as native

SCHEMA = "chipmunk.video-temporal-multiple-inserts/1"
PREFLIGHT = native.ROOT / "docs/evaluation-panels/video-temporal-insert-native-2026-11-13.json"
HELD = native.ROOT / "docs/evaluation-panels/video-temporal-insert-confirmation-2026-11-13.json"
BASE_FRAMES = 150
MIN_FLANK = 24
MIN_INSERT = 16
MAX_ADDED = 144
MAX_FRAMES = 360
MATCH_MEDIAN = .10
MATCH_P90 = .22
RUN_MIN = 12
MOTION_FLOOR = .015
SOURCE_MOTION_LUMA = 1.0
ALTERNATE_RUN_MARGIN = .020
SINGLE_IMPROVEMENT = .30
AMBIGUITY_COST = .035
AMBIGUITY_FRAMES = 3
OPTIONS = (2, 3)
ROLES = ("generated_two", "generated_three", "ordinary_two", "unchanged", "single", "wrong_earlier",
         "mixed_replacement", "mixed_removal", "static_two", "near_static_two")
MULTI = frozenset(("generated_two", "generated_three", "ordinary_two"))
GATES = {"generated_two_complete": 6, "generated_three_complete": 6, "generated_complete_groups": 6,
         "ordinary_two_complete": 5, "maximum_positive_outside_tolerance": 0, "maximum_control_reports": 0,
         "maximum_disrupted_single": 0}
METHOD = {
    "matcher": "Ordered alignment with exactly two or three query insertions and every earlier picture retained. Minimize summed "
               "standardized luma absolute differences. Check temporal content in each retained span, reject substantially better "
               "local alignments, and preserve plausible endpoint bounds across near-ties. Prefer the existing result on ties.",
    "limits": {"min_flank": MIN_FLANK, "min_insert": MIN_INSERT, "max_added": MAX_ADDED, "max_frames": MAX_FRAMES,
               "match_median": MATCH_MEDIAN, "match_p90": MATCH_P90, "match_run_min": RUN_MIN, "motion_floor": MOTION_FLOOR,
               "source_motion_luma": SOURCE_MOTION_LUMA,
               "alternate_run_margin": ALTERNATE_RUN_MARGIN, "single_improvement": SINGLE_IMPROVEMENT,
               "ambiguity_cost": AMBIGUITY_COST, "ambiguity_frames": AMBIGUITY_FRAMES},
    "distribution": "Original 4 cached preflight recordings for development, 8 previously checked but development-disjoint "
                    "cached confirmation recordings for held edit recipes. All media comes from the same CC BY 4.0 deposit.",
    "process": "No learned weights. Exact 24 fps retained-frame clocks verified independently by FFmpeg framehash. "
               "Each delivery preserves the authored source sequence and native frame counts. The comparison carries zero AI weight.",
    "control_gate": "The candidate must add no reports on unchanged, wrong-reference, replacement, removal, static, or "
                    "near-static controls. Count any preexisting single reports on those controls separately.",
    "gates": GATES,
}


def match_cost(earlier, submitted):
    import numpy as np

    from app.evidence.video_reference_insertion import frame_error

    added = len(submitted) - len(earlier)
    return np.stack([frame_error(earlier, submitted[offset:offset+len(earlier)]) for offset in range(added+1)], axis=1)


def cumulative_alignment(cost, max_gaps):
    """Minimal costs for an initial match followed by ordered changes in offset."""
    import numpy as np

    frames, offsets = cost.shape
    cumulative = np.vstack((np.zeros((1, offsets)), np.cumsum(cost, axis=0, dtype=np.float64)))
    states = np.full((max_gaps+1, frames+1, offsets), np.inf)
    states[0, MIN_FLANK:, 0] = cumulative[MIN_FLANK:, 0]
    for count in range(1, max_gaps+1):
        prior_cost = np.minimum.accumulate(states[count-1], axis=1)[:, :-MIN_INSERT]
        new_cost = prior_cost - cumulative[:, MIN_INSERT:]
        best = np.minimum.accumulate(new_cost, axis=0)
        states[count, MIN_FLANK:, MIN_INSERT:] = cumulative[MIN_FLANK:, MIN_INSERT:] + best[:-MIN_FLANK]
    return cumulative, states


def extract_ranges(cumulative, states, count):
    import numpy as np

    last_frame, last_offset = (length-1 for length in cumulative.shape)
    ranges = []
    for gaps in range(count, 0, -1):
        earlier_costs = states[gaps-1, :last_frame-MIN_FLANK+1, :last_offset-MIN_INSERT+1]
        choices = earlier_costs - cumulative[:last_frame-MIN_FLANK+1, last_offset, None]
        frame, offset = (int(value) for value in np.unravel_index(np.argmin(choices), choices.shape))
        ranges.append([frame + offset, frame + last_offset])
        last_frame, last_offset = frame, offset
    return ranges[::-1]


def possible_edges(forward, backward, gaps, bound):
    import numpy as np

    edges = []
    for index in range(gaps):
        earlier = forward[index]
        later = backward[gaps-index-1, ::-1, ::-1]
        min_previous = np.minimum.accumulate(earlier[:, :-MIN_INSERT], axis=1)
        min_next = np.minimum.accumulate(later[:, :MIN_INSERT-1:-1], axis=1)[:, ::-1]
        start_frame, start_offset = np.nonzero(earlier[:, :-MIN_INSERT] + min_next <= bound)
        end_frame, end_offset = np.nonzero(min_previous + later[:, MIN_INSERT:] <= bound)
        starts, stops = start_frame + start_offset, end_frame + end_offset + MIN_INSERT
        edges.append([[int(starts.min()), int(starts.max())], [int(stops.min()), int(stops.max())]])
    return edges


def flank_stats(cost, ranges, pictures):
    import numpy as np

    from app.evidence.video_reference_insertion import frame_error

    support = []
    added = previous = 0
    for start, stop in (*ranges, (len(cost)+cost.shape[1]-1, None)):
        last = start - added
        errors = cost[previous:last, added]
        p50, p90 = (round(float(score), 6) for score in np.percentile(errors, (50, 90)))
        windows = np.lib.stride_tricks.sliding_window_view(errors, RUN_MIN)
        local = round(float(np.max(np.median(windows, axis=1))), 6)
        motion = frame_error(pictures[previous+4:last], pictures[previous:last-4])
        alternatives = np.lib.stride_tricks.sliding_window_view(cost[previous:last, :], RUN_MIN, axis=0).mean(axis=2)
        distinct_improvement = float(np.max(alternatives[:, added] - alternatives.min(axis=1)))
        support.append({"earlier": [previous, last], "offset": added, "error_p50_p90": [p50, p90],
                        "max_12_frame_median": local, "motion_p75": round(float(np.percentile(motion, 75)), 6),
                        "max_alternate_improvement": round(distinct_improvement, 6)})
        if stop is not None:
            added += stop-start
            previous = last
    return support


def detailed_source_motion(pictures):
    import numpy as np

    difference = pictures[4:].astype(np.int16) - pictures[:-4].astype(np.int16)
    difference -= np.median(difference, axis=(1, 2), keepdims=True).astype(np.int16)
    mean = np.mean(np.abs(difference), axis=(1, 2))
    return float(np.percentile(mean, 75)) >= SOURCE_MOTION_LUMA


def additional_ranges(earlier, submitted, existing=None):
    import numpy as np

    from app.evidence.video_reference_insertion import normalized, pair_interval

    if existing is None:
        existing = pair_interval(earlier, submitted)
    lengths = [count for count in OPTIONS if len(earlier) >= MIN_FLANK*(count+1)
               and MIN_INSERT*count <= len(submitted)-len(earlier) <= MAX_ADDED]
    if not lengths or len(submitted) > MAX_FRAMES:
        return {"status": "unsupported_length", "bounds": None}
    if not detailed_source_motion(earlier):
        return {"status": "low_temporal_detail", "bounds": None}
    first = normalized(earlier)
    cost = match_cost(first, normalized(submitted))
    cumulative, forward = cumulative_alignment(cost, max(lengths))
    best = min(lengths, key=lambda count: forward[count, -1, -1])
    minimum = forward[best, -1, -1]
    if not np.isfinite(minimum):
        return {"status": "unsupported_length", "bounds": None}
    ranges = extract_ranges(cumulative, forward, best)
    support = flank_stats(cost, ranges, first)
    detail = {"best": ranges, "flanks": support, "cost": round(float(minimum), 6)}
    if existing["bounds"] is not None and forward[1, -1, -1] - minimum < SINGLE_IMPROVEMENT:
        return {"status": "existing_single", "bounds": None, **detail}
    if any(item["motion_p75"] < MOTION_FLOOR for item in support):
        return {"status": "low_temporal_detail", "bounds": None, **detail}
    if any(item["error_p50_p90"][0] > MATCH_MEDIAN or item["error_p50_p90"][1] > MATCH_P90
           or item["max_12_frame_median"] > MATCH_MEDIAN
           or item["max_alternate_improvement"] > ALTERNATE_RUN_MARGIN for item in support):
        return {"status": "insufficient_retained_match", "bounds": None, **detail}
    if any(forward[count, -1, -1] <= minimum+AMBIGUITY_COST for count in lengths if count != best):
        return {"status": "ambiguous_number_of_inserts", "bounds": None, **detail}
    _, backward = cumulative_alignment(cost[::-1, ::-1], best)
    edges = possible_edges(forward, backward, best, minimum+AMBIGUITY_COST)
    if any(high-low > AMBIGUITY_FRAMES for item in edges for low, high in item):
        return {"status": "ambiguous_boundaries", "bounds": None, "possible_edges": edges, **detail}
    return {"status": "separated_insertions", "bounds": ranges, "possible_edges": edges, **detail}


def edits(slot):
    """Frozen edit recipes expressed as earlier split, dropped pictures, and donor range."""
    two = [(42+slot*2, 0, "donor", 12, 44+2*(slot % 3)), (105+slot*2, 0, "donor", 61, 97+2*(slot % 4))]
    three = [(31+slot, 0, "donor", 8, 24+4*(slot % 3)), (76+slot % 3, 0, "donor", 42, 70+2*(slot % 4)),
             (116+slot, 0, "donor", 84, 120+2*((slot+1) % 3))]
    return {
        "generated_two": two, "generated_three": three,
        "ordinary_two": [(at, 0, "base", 8+i*52, 8+i*52+end-start) for i, (at, _, _, start, end) in enumerate(two)],
        "unchanged": [], "single": [(67+slot, 0, "donor", 34, 76)], "wrong_earlier": two,
        "mixed_replacement": [two[0], (72, 24, "donor", 21, 45), two[1]],
        "mixed_removal": [two[0], (72, 20, "base", 0, 0), two[1]],
        "static_two": two, "near_static_two": two,
    }


def roster():
    plan = confirmation.plan()
    evidence = json.loads(HELD.read_bytes())
    confirmation.review(evidence)
    development = json.loads(PREFLIGHT.read_bytes())
    native.review(development)
    first = {item["example"]: item["members"] for item in native.intake()}
    second = {item["example"]: item["sources"] for item in plan["cases"]}
    groups = {}
    for name, members, doc in (("development", first, development), ("held", second, evidence)):
        groups[name] = [{"example": group["example"], "files": {
            role: {"filename": members[group["example"]][role]["name"], "sha256": group["native"][role]["sha256"]}
            for role in ("authentic", "fake_segments")}, "edits": edits(slot)} for slot, group in enumerate(doc["groups"])]
    return groups


def raw_pictures(path, size, number):
    import numpy as np

    width, height = size
    raw = native.process([*native.ffmpeg(), *native.input_file(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                          "-vf", f"scale={width}:{height}:flags=area,format=yuv420p", "-frames:v", str(number),
                          "-threads", "1", "-f", "rawvideo", "pipe:1"])
    count = width*height*3//2
    native.source.require(len(raw) == count*number, "Incomplete source pictures for frozen edit")
    return np.frombuffer(raw, dtype=np.uint8).reshape(number, count)


def encode(target, size, pictures):
    import subprocess

    width, height = size
    command = [*native.ffmpeg(), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", f"{width}x{height}",
               "-framerate", "24", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1", "-preset", "fast",
               "-crf", "23", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0", "-video_track_timescale", "12288",
               "-map_metadata", "-1", str(target)]
    subprocess.run(command, input=memoryview(pictures), check=True, timeout=90, stdout=subprocess.DEVNULL)


def freeze_controls(base):
    import numpy as np

    still = np.repeat(base[0:1], len(base), axis=0)
    near = still.copy()
    for index in range(len(near)):
        np.clip(still[index].astype(np.int16) + ((index % 7)-3), 0, 255, out=near[index], casting="unsafe")
    return still, near


def construct(base, donor, edit_rows):
    import numpy as np

    pieces, provenance, ranges = [], [], []
    earlier = submitted = 0
    for at, drop, source, start, end in edit_rows:
        keep = base[earlier:at]
        pieces.append(keep)
        provenance.extend((0, i) for i in range(earlier, at))
        submitted += len(keep)
        donor_pictures = (base if source == "base" else donor)[start:end]
        pieces.append(donor_pictures)
        provenance.extend((int(source == "donor"), i) for i in range(start, end))
        if not drop:
            ranges.append([submitted, submitted+len(donor_pictures)])
        submitted += len(donor_pictures)
        earlier = at+drop
    pieces.append(base[earlier:])
    provenance.extend((0, i) for i in range(earlier, len(base)))
    return np.concatenate(pieces), provenance, ranges


def files(case, cache):
    paths = {role: cache / item["filename"] for role, item in case["files"].items()}
    for role, path in paths.items():
        native.source.require(native.file_sha(path) == case["files"][role]["sha256"], "Cached source digest differs")
    return paths


def score_case(case, wrong, cache):
    import numpy as np

    from app.detectors.video.decode import probe_frames
    from app.evidence.video_reference_insertion import pair_interval

    paths = files(case, cache)
    wrong_path = files(wrong, cache)["authentic"]
    table = probe_frames(paths["authentic"], cap=600, timeout_s=60)
    size = native.delivery_size(table)
    base = raw_pictures(paths["authentic"], size, BASE_FRAMES)
    donor = raw_pictures(paths["fake_segments"], size, 125)
    wrong_base = raw_pictures(wrong_path, size, BASE_FRAMES)
    still, near = freeze_controls(base)
    controls = {"wrong_earlier": wrong_base, "static_two": still, "near_static_two": near}
    rows = []
    with TemporaryDirectory(prefix="video-multiple-inserts-") as scratch:
        folder = Path(scratch)
        cached = {}
        for name, content in {"base": base, "donor": donor, **controls}.items():
            target = folder / f"{name}.mp4"
            encode(target, size, content)
            cached[name] = native.native_clip(target)
        for role in ROLES:
            before_name = role if role in controls else "base"
            authored_base = controls.get(role, base) if role != "wrong_earlier" else base
            edited, provenance, known = construct(authored_base, donor, case["edits"][role])
            target = folder / f"{role}-submitted.mp4"
            encode(target, size, edited)
            before, after = cached[before_name], native.native_clip(target)
            reference = cached["base"] if role == "wrong_earlier" else before
            luma = np.stack([(cached["donor"] if source else reference)[1][index] for source, index in provenance])
            splits = [0, *(index for index, (a, b) in enumerate(zip(provenance, provenance[1:]), start=1)
                           if a[0] != b[0] or a[1]+1 != b[1]), len(provenance)]
            render = confirmation.verify_render(after, [(a, luma[a:b]) for a, b in zip(splits, splits[1:])], len(edited))
            single = pair_interval(before[1], after[1])
            candidate = additional_ranges(before[1], after[1], existing=single)
            if candidate["bounds"] is not None:
                candidate["witnesses"] = [[{"frame": frame, "pts": after[0].pts[frame], "sha256": after[2][frame]}
                    for frame in (start-1, (start+stop)//2, stop)] for start, stop in candidate["bounds"]]
            rows.append({"role": role, "truth": known if role in {"generated_two", "generated_three", "ordinary_two", "single"}
                         else None, "render": render, "earlier": before[3], "submitted": after[3], "single": single,
                         "candidate": candidate})
            target.unlink()
    return {"example": case["example"], "wrong_example": wrong["example"], "rows": rows}


def matches(expected, predicted, tolerance):
    return predicted is not None and len(predicted) == len(expected) and all(
        abs(a-c) <= tolerance and abs(b-d) <= tolerance
        for (a, b), (c, d) in zip(expected, predicted, strict=True))


def recount(groups):
    tallies = {role: Counter() for role in ROLES}
    generated_groups = 0
    for group in groups:
        both = 0
        for row in group["rows"]:
            role, expected = row["role"], row["truth"]
            current = row["single"]["bounds"]
            candidate = row["candidate"]["bounds"]
            combined = candidate if candidate is not None else ([current] if current is not None else None)
            counts = tallies[role]
            for method, proposed in (("baseline", [current] if current is not None else None), ("combined", combined)):
                counts[f"{method}_reports"] += proposed is not None
                if expected is not None:
                    counts[f"{method}_complete"] += matches(expected, proposed, AMBIGUITY_FRAMES)
                    counts[f"{method}_exact"] += matches(expected, proposed, 0)
            counts[f"candidate_{row['candidate']['status']}"] += 1
            if role in MULTI:
                counts["outside_tolerance"] += combined is not None and not matches(expected, combined, AMBIGUITY_FRAMES)
                both += role.startswith("generated_") and matches(expected, combined, AMBIGUITY_FRAMES)
            if role == "single":
                counts["disrupted"] += combined != ([current] if current is not None else None)
            if expected is None:
                counts["new_control_reports"] += candidate is not None
        generated_groups += both == 2
    gates = {
        "generated_two_complete": tallies["generated_two"]["combined_complete"] >= GATES["generated_two_complete"],
        "generated_three_complete": tallies["generated_three"]["combined_complete"] >= GATES["generated_three_complete"],
        "generated_complete_groups": generated_groups >= GATES["generated_complete_groups"],
        "ordinary_two_complete": tallies["ordinary_two"]["combined_complete"] >= GATES["ordinary_two_complete"],
        "maximum_positive_outside_tolerance": sum(tallies[role]["outside_tolerance"] for role in MULTI)
        <= GATES["maximum_positive_outside_tolerance"],
        "maximum_control_reports": sum(tallies[role]["new_control_reports"] for role in set(ROLES)-MULTI-{"single"})
        <= GATES["maximum_control_reports"],
        "maximum_disrupted_single": tallies["single"]["disrupted"] <= GATES["maximum_disrupted_single"],
    }
    return {"groups": len(groups), "generated_complete_groups": generated_groups, "tallies": tallies,
            "gates": gates, "qualified": all(gates.values())}


def truth(edits):
    total = 0
    spans = []
    cuts = [0]
    for at, drop, _, start, end in edits:
        cuts.append(at+total)
        if not drop:
            spans.append([at+total, at+total+end-start])
        length = end-start
        if length:
            cuts.append(at+total+length)
        total += length-drop
    cuts.append(BASE_FRAMES+total)
    return spans, cuts


def review(document):
    require = native.source.require
    require(document["schema"] == SCHEMA and document["method"] == METHOD, "Different frozen experiment")
    partition = document["partition"]
    cases = roster()[partition]
    require(document["roster_sha256"] == native.exact_json_sha(cases), "Frozen source and recipe digest differs")
    for index, (case, group) in enumerate(zip(cases, document["groups"], strict=True)):
        require(group["example"] == case["example"] and group["wrong_example"] == cases[(index+1)%len(cases)]["example"],
                "Different frozen source order")
        require([row["role"] for row in group["rows"]] == list(ROLES), "Different conditions")
        by_role = {row["role"]: row for row in group["rows"]}
        require(by_role["wrong_earlier"]["earlier"]["sha256"] != by_role["generated_two"]["earlier"]["sha256"]
                and by_role["wrong_earlier"]["submitted"] == by_role["generated_two"]["submitted"],
                "Wrong reference did not reuse the same submitted copy")
        for row in group["rows"]:
            role, candidate = row["role"], row["candidate"]
            before, after = row["earlier"], row["submitted"]
            native.verify_clock(before)
            native.verify_clock(after)
            intervals, cuts = truth(case["edits"][role])
            after_count = cuts[-1]
            require(row["truth"] == (intervals if role in MULTI or role == "single" else None) and
                    before["frames"] == BASE_FRAMES and after["frames"] == after_count and
                    (before["width"], before["height"]) == (after["width"], after["height"]) and
                    before["codec"] == after["codec"] == "h264" and max(after["width"], after["height"]) == 480,
                    "Inputs differ from the frozen carrier or edit truth")
            require([item["bounds"] for item in row["render"]] == [list(span) for span in zip(cuts, cuts[1:])]
                    and all(0 <= median <= .10 and median <= p90 <= .22
                            for median, p90 in (item["error_p50_p90"] for item in row["render"])),
                    "Rendered pieces do not match the expected source pictures")
            spans = candidate["bounds"]
            if spans is None:
                require("witnesses" not in candidate and candidate["status"] != "separated_insertions", "Incomplete output")
                continue
            require(candidate["status"] == "separated_insertions" and candidate["best"] == spans and len(spans) in OPTIONS,
                    "Candidate result has an invalid range count")
            final = added = 0
            prior_frame = 0
            expected_flanks = []
            for bounds, edges, seen in zip(spans, candidate["possible_edges"], candidate["witnesses"], strict=True):
                start, stop = bounds
                require(start-final >= MIN_FLANK and stop-start >= MIN_INSERT and stop <= after_count-MIN_FLANK,
                        "Invalid added footage range")
                require(all(lo <= x <= hi and hi-lo <= AMBIGUITY_FRAMES for x, (lo, hi) in zip(bounds, edges, strict=True)),
                        "Plausible endpoints are inconsistent")
                for frame, item in zip((start-1, (start+stop)//2, stop), seen, strict=True):
                    require(item["frame"] == frame and item["pts"] == frame*after["tick_step"] and
                            native.SHA256.fullmatch(item["sha256"]), "Changed screenshot clock or missing hash")
                expected_flanks.append(([prior_frame, start-added], added))
                prior_frame = start-added
                final = stop
                added += stop-start
            require(added == after_count-BASE_FRAMES, "Ranges fail to account for all added frames")
            expected_flanks.append(([prior_frame, BASE_FRAMES], added))
            for flank, (earlier, offset) in zip(candidate["flanks"], expected_flanks, strict=True):
                require(flank["earlier"] == earlier and flank["offset"] == offset
                        and 0 <= flank["error_p50_p90"][0] <= MATCH_MEDIAN
                        and 0 <= flank["error_p50_p90"][1] <= MATCH_P90
                        and 0 <= flank["max_12_frame_median"] <= MATCH_MEDIAN
                        and 0 <= flank["max_alternate_improvement"] <= ALTERNATE_RUN_MARGIN
                        and flank["motion_p75"] >= MOTION_FLOOR, "Alignment flank omitted expected match or support")
    result = recount(document["groups"])
    require(document["result"] == result, "Stored counts changed")
    return result


def score(cache, partition):
    cases = roster()[partition]
    groups = []
    for index, case in enumerate(cases):
        group = score_case(case, cases[(index+1)%len(cases)], cache)
        groups.append(group)
        print(case["example"], [(row["role"], row["candidate"]["status"]) for row in group["rows"]], flush=True)
    return {"schema": SCHEMA, "method": METHOD, "partition": partition, "roster_sha256": native.exact_json_sha(cases),
            "groups": groups, "result": recount(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "score", "review"))
    parser.add_argument("path", type=Path, nargs="?")
    parser.add_argument("--partition", choices=("development", "held"))
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    if args.operation == "plan":
        print(json.dumps({"schema": SCHEMA, "method": METHOD, "partitions": roster()}, indent=2, sort_keys=True))
    elif args.operation == "review" and args.path:
        print(json.dumps(review(json.loads(args.path.read_bytes())), sort_keys=True))
    elif args.operation == "score" and args.path and args.partition and args.write and not args.write.exists():
        document = score(args.path, args.partition)
        review(document)
        with args.write.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(document["result"], sort_keys=True))
    else:
        parser.error("review requires a receipt path, and score requires the cache path, --partition, and a new --write path")


if __name__ == "__main__":
    main()
