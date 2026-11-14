# SPDX-License-Identifier: Apache-2.0
"""Confirm frozen earlier-copy insertion intervals on eight new video sources."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_temporal_insert_native as prior

PLAN = prior.ROOT / "docs/evaluation-panels/video-temporal-insert-confirmation-plan-2026-11-13.json"
PLAN_PIN = "f1e40d833f0ac3861cc7a63913a0b264c6d0c711429785bba8631b2b6b47abfd"
SOURCE_RECEIPT = prior.ROOT / "docs/evaluation-panels/video-temporal-insert-native-2026-11-13.json"
SCHEMA = "chipmunk.video-temporal-insert-confirmation/1"
ROLES = ("generated_insert", "matched_ordinary_insertion", "wrong_reference", "equal_length_replacement")
GATES = {"minimum_generated_exact": 8, "minimum_generated_within_three": 12, "minimum_generated_groups": 6,
         "maximum_generated_outside_three": 0, "minimum_ordinary_within_three": 8,
         "maximum_unchanged_replacement_wrong_reports": 0}
METHOD = {
    "candidate": prior.PROTOCOL["paired_check"],
    "source_separation": "Check exact file and decoded luma commitments against the eight candidates and all eight preflight "
                         "original/generated clips. Search for exact 96-square frames and close 24-square downsampled "
                         "sequences. A run of at least 12 nearly identical detailed pictures at the same relative offset "
                         "counts as a possible near-duplicate. Retain any rejected candidate in the denominator.",
    "confirmed_deliveries": "Render a matching 480-pixel-long-side H.264 picture stream at exact 24 fps. Use the frozen "
                            "generated slice and insertion split. Ordinary insertion prepends a repeated authentic part "
                            "from the reference at that split. The equal-length replacement uses the same generated "
                            "slice and clamps the split when needed to preserve an authentic 24-frame suffix.",
    "unchanged_and_wrong": "One unchanged control per source, with independent re-encoding settings identical to the "
                           "earlier delivery. For each generated delivery, use the next source in plan order as a wrong "
                           "reference. Match its geometry, cadence and earlier length, looping if needed.",
    "measurement": "Keep candidate settings unchanged and report exact, within three frames and outside-three results. "
                   "Run the existing earlier-copy app comparison with all locally enabled picture options on the same "
                   "delivered pairs. Count ordinary insertions as physical edits, regardless of generation.",
    "scope": "Physical insertion between two copies. Zero AI-confidence weight. Authored deliveries use new source "
             "groups from one publisher, no sound, and no learned weights.",
    "release_gates": GATES,
}


def plan():
    prior.source.require(prior.file_sha(PLAN) == PLAN_PIN, "Frozen confirmation roster changed")
    return json.loads(PLAN.read_bytes())


def acquire(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(prior.fetch_member, directory, role, member) for case in plan()["cases"]
                 for role, member in case["sources"].items()]
        for task in tasks:
            task.result()


def same_content(first, second):
    """Check hashes and count consecutive close detailed thumbnails at a fixed offset."""
    import numpy as np
    from scipy.spatial.distance import cdist

    if first[3]["sha256"] == second[3]["sha256"] or first[3]["luma_sha256"] == second[3]["luma_sha256"]:
        return True
    if len(set(first[2]) & set(second[2])) >= 12:
        return True

    def reduced(pixels):
        signal = pixels.reshape(-1, 24, 4, 24, 4).mean(axis=(2, 4)).astype(np.float64)
        return signal.reshape(-1, 576), signal.std(axis=(1, 2)) >= 15

    a, a_active = reduced(first[1])
    b, b_active = reduced(second[1])
    near = (cdist(a, b, metric="cityblock") <= 3 * a.shape[1]) & a_active[:, None] & b_active
    for offset in range(12 - len(a), len(b) - 11):
        diagonal = near.diagonal(offset=offset)
        if np.convolve(diagonal, np.ones(12, dtype=int), "valid").max(initial=0) >= 12:
            return True
    return False


def read_sources(directory, cases, *, preflight):
    found = {}
    rejected = {}
    prior_receipt = json.loads(SOURCE_RECEIPT.read_bytes())
    prior.review(prior_receipt)
    verified = {}
    for case, group in zip(prior.intake(), prior_receipt["groups"], strict=True):
        for role in ("authentic", "fake_segments"):
            member = case["members"][role]
            path = preflight / member["name"]
            prior.file_receipt(path, member)
            clip = prior.native_clip(path)
            prior.source.require(clip[3] == group["native"][role], "Prior native content does not match its commitments")
            verified[case["example"], role] = clip
    for case in cases:
        example = case["example"]
        pair = {}
        for role, member in case["sources"].items():
            path = directory / member["name"]
            prior.file_receipt(path, member)
            try:
                clip = prior.native_clip(path)
            except (ValueError, RuntimeError):
                rejected[example] = {"reason": "unsupported_native_clock", "role": role}
                break
            expected = case["earlier_frames"] if role == "authentic" else case["generated_frames"]
            if len(clip[0]) != expected:
                rejected[example] = {"reason": "unexpected_native_frame_count", "role": role, "frames": len(clip[0])}
                break
            pair[role] = clip
        if example not in rejected:
            found[example] = pair
    all_sources = {**verified, **{(example, role): clip for example, pair in found.items() for role, clip in pair.items()}}
    for (example, role), clip in all_sources.items():
        if example not in found or example in rejected:
            continue
        for (other, other_role), sample in all_sources.items():
            if example != other and same_content(clip, sample):
                rejected[example] = {"reason": "cross_group_content_overlap", "role": role,
                                     "other_example": other, "other_role": other_role}
                break
    return found, rejected


def edited_delivery(original, donor, output, size, split, excerpt, *, replace=False):
    width, height = size
    start, end = excerpt
    after = split + (end-start if replace else 0)
    scaling = f"scale={width}:{height}:flags=area,setsar=1"
    graph = (
        f"[0:v:0]{scaling},split=2[a][b];"
        f"[a]trim=end_frame={split},setpts=PTS-STARTPTS[left];"
        f"[1:v:0]{scaling},trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[insert];"
        f"[b]trim=start_frame={after},setpts=PTS-STARTPTS[right];"
        "[left][insert][right]concat=n=3:v=1:a=0[v]"
    )
    prior.process([*prior.ffmpeg(), *prior.input_file(original), *prior.input_file(donor),
                   "-filter_complex_threads", "1", "-filter_complex", graph, "-map", "[v]", "-an", "-sn", "-dn",
                   "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-g", "48", "-keyint_min", "48", "-sc_threshold", "0", "-r", "24", "-video_track_timescale",
                   "12288", "-map_metadata", "-1", str(output)])


def verify_application(earlier_path, submitted_path, earlier, submitted, candidate):
    from app.evidence import video_reference

    require = prior.source.require
    record, assets = video_reference.compare(earlier_path, submitted_path, include_picture_insertion=True)
    insertion = record.get("picture_insertion")
    if candidate["bounds"] is None:
        require(insertion is None, "Opt-in app reported an interval withheld by the frozen candidate")
        return
    start, stop = candidate["bounds"]
    require(record["schema"] == video_reference.SCHEMA_WITH_PICTURE_INSERTION and record["status"] == "measured"
            and record["affects_overall_confidence"] is False and insertion["audio_checked"] is False
            and [insertion["start_frame"], insertion["end_frame_exclusive"]] == [start, stop]
            and insertion["source_ticks"] == {"start_pts": submitted[0].pts[start],
                                              "end_exclusive_pts": submitted[0].pts[stop],
                                              "time_base": submitted[0].time_base},
            "Opt-in app differed from frozen candidate range, clock or confidence scope")
    witnesses = insertion["witnesses"]
    expected = {"before": [("earlier", start-1), ("submitted", start-1)],
                "during": [("submitted", (start+stop)//2)],
                "after": [("earlier", start), ("submitted", stop)]}
    require(set(witnesses) == set(expected) and len(assets) == 5, "Opt-in app omitted insertion witnesses")
    for role, items in expected.items():
        require(set(witnesses[role]) == {side for side, _ in items}, "Opt-in app used a different witness role")
        for side, index in items:
            table = (earlier if side == "earlier" else submitted)[0]
            item = witnesses[role][side]
            asset = item["asset"]
            require((item["frame_index"], item["pts"], item["time_base"], item["time_s"]) ==
                    (index, table.pts[index], table.time_base, table.seconds(index))
                    and asset["media_type"] == "image/png"
                    and prior.source.sha(assets[asset["file"]]) == asset["sha256"],
                    "Opt-in app used a different witness time or invalid screenshot")


def compare(earlier_path, submitted_path, earlier, submitted, *, verify_app):
    from app.evidence import video_reference

    candidate = prior.pair_interval(earlier[1], submitted[1])
    if candidate["bounds"]:
        start, stop = candidate["bounds"]
        candidate["evidence"] = [{"frame": i, "pts": submitted[0].pts[i], "gray_96_sha256": submitted[2][i]}
                                 for i in (start-1, (start+stop)//2, stop)]
    existing, _ = video_reference.compare(earlier_path, submitted_path, include_trimmed=True,
                                         include_trim_reencoded=True, include_local_area=True,
                                         include_measured_interval=True, include_soundtrack_reencode=True)
    if verify_app:
        verify_application(earlier_path, submitted_path, earlier, submitted, candidate)
    return {"earlier": earlier[3], "submitted": submitted[3], "status": existing["status"],
            "existing_picture_ranges": [[row["start_frame"], row["end_frame_exclusive"]] for row in
                                        existing.get("picture", {}).get("ranges", ())],
            "paired": candidate}


def verify_render(clip, sources, expected_frames):
    """Check each authored segment against the corresponding decoded source pictures."""
    import numpy as np

    prior.source.require(len(clip[0]) == expected_frames, "Delivery changed the expected 24 fps frame count")
    checks = []
    for start, expected in sources:
        errors = prior.frame_error(prior.normalized(expected), prior.normalized(clip[1][start:start+len(expected)]))
        median, p90 = (round(float(value), 6) for value in np.percentile(errors, (50, 90)))
        prior.source.require(median <= .10 and p90 <= .22, "Encoded pictures do not match the authored source sequence")
        checks.append({"bounds": [start, start+len(expected)], "error_p50_p90": [median, p90]})
    return checks


def measure_group(directory, case, candidate, wrong_case, *, verify_app):
    paths = {role: directory / item["name"] for role, item in case["sources"].items()}
    width, height = prior.delivery_size(candidate["authentic"][0])
    original_frames = case["earlier_frames"]
    with TemporaryDirectory(prefix="video-insert-confirm-") as temp:
        folder = Path(temp)
        earlier_path = folder / "earlier.mp4"
        untouched_path = folder / "untouched.mp4"
        wrong_path = folder / "wrong-reference.mp4"
        for path, raw in ((earlier_path, paths["authentic"]), (untouched_path, paths["authentic"]),
                          (wrong_path, directory / wrong_case["sources"]["authentic"]["name"])):
            prior.same_delivery(raw, path, width=width, height=height, length=original_frames if path == wrong_path else None)
        earlier, untouched, wrong = (prior.native_clip(path) for path in (earlier_path, untouched_path, wrong_path))
        prior.source.require(len(earlier[0]) == len(untouched[0]) == len(wrong[0]) == original_frames,
                             "Earlier or matched control frame clock differs")
        rows = [{"render": None, "role": "untouched_delivery", "truth": None,
                 **compare(earlier_path, untouched_path, earlier, untouched, verify_app=verify_app)}]
        renders = []
        for number, authored in enumerate(case["authored_renders"]):
            split = authored["earlier_split"]
            start, end = authored["generated_slice"]
            duration = end-start
            replacement = min(split, original_frames - duration - 24)
            rendered = {}
            validation = {}
            for role, donor, at, excerpt, replace in (
                ("generated_insert", paths["fake_segments"], split, (start, end), False),
                ("matched_ordinary_insertion", paths["authentic"], split, (0, duration), False),
                ("equal_length_replacement", paths["fake_segments"], replacement, (start, end), True),
            ):
                target = folder / f"{number}-{role}.mp4"
                edited_delivery(paths["authentic"], donor, target, (width, height), at, excerpt, replace=replace)
                clip = prior.native_clip(target)
                original = earlier[1]
                same_source = role == "matched_ordinary_insertion"
                inserted = original[:duration] if same_source else candidate["fake_segments"][1][start:end]
                suffix = original[at+duration if replace else at:]
                validation[role] = verify_render(clip,
                    [(0, original[:at]), (at, inserted), (at+duration, suffix)],
                    original_frames if replace else original_frames + duration)
                rendered[role] = target, clip
            for role in ROLES:
                target, clip = rendered["generated_insert" if role == "wrong_reference" else role]
                reference, reference_path = (wrong, wrong_path) if role == "wrong_reference" else (earlier, earlier_path)
                truth = [split, split+duration] if role in {"generated_insert", "matched_ordinary_insertion"} else None
                rows.append({"render": number, "role": role, "truth": truth,
                             **compare(reference_path, target, reference, clip, verify_app=verify_app)})
            renders.append({"render": number, "verification": validation, "replacement_start": replacement})
    return {"example": case["example"], "native": {role: clip[3] for role, clip in candidate.items()},
            "wrong_example": wrong_case["example"], "renders": renders, "rows": rows}


def summarize(groups):
    counts = {"untouched_delivery": Counter(planned=8), **{role: Counter(planned=16) for role in ROLES}}
    covered = set()
    for group in groups:
        for row in group.get("rows", ()):
            tally = counts[row["role"]]
            tally["seen"] += 1
            tally[f"baseline_{row['status']}"] += 1
            bounds = row["paired"]["bounds"]
            tally[f"candidate_{row['paired']['status']}"] += 1
            if bounds is None:
                continue
            tally["reported"] += 1
            truth = row["truth"]
            if truth is not None:
                error = max(abs(a-b) for a, b in zip(bounds, truth, strict=True))
                tally["exact"] += error == 0
                tally["within_three"] += error <= 3
                tally["outside_three"] += error > 3
                if error <= 3 and row["role"] == "generated_insert":
                    covered.add(group["example"])
    g = counts["generated_insert"]
    checks = {
        "generated_exact": g["exact"] >= GATES["minimum_generated_exact"],
        "generated_within_three": g["within_three"] >= GATES["minimum_generated_within_three"],
        "generated_groups": len(covered) >= GATES["minimum_generated_groups"],
        "generated_outside_three": g["outside_three"] <= GATES["maximum_generated_outside_three"],
        "ordinary_insertions": counts["matched_ordinary_insertion"]["within_three"] >= GATES["minimum_ordinary_within_three"],
        "unchanged_replacement_wrong": sum(counts[role]["reported"]
            for role in ("untouched_delivery", "equal_length_replacement", "wrong_reference"))
        <= GATES["maximum_unchanged_replacement_wrong_reports"],
    }
    return {"counts": counts, "generated_source_groups_within_three": sorted(covered),
            "checks": checks, "supports_local_evidence": all(checks.values())}


def score(directory, preflight, *, verify_app):
    import cv2

    cv2.setNumThreads(1)
    cases = plan()["cases"]
    found, rejected = read_sources(directory, cases, preflight=preflight)
    groups = []
    for index, case in enumerate(cases):
        example = case["example"]
        group = ({"example": example, "declined": rejected[example]} if example in rejected
                 else measure_group(directory, case, found[example], cases[(index+1) % len(cases)], verify_app=verify_app))
        groups.append(group)
        print(f"Verified and measured planned source {example} ({index+1}/{len(cases)})", flush=True)
    return {"schema": SCHEMA, "plan_sha256": PLAN_PIN, "method": METHOD, "groups": groups, "results": summarize(groups)}


def review(document):
    require = prior.source.require
    require((document["schema"], document["plan_sha256"], document["method"]) == (SCHEMA, PLAN_PIN, METHOD),
            "Incorrect confirmation protocol")
    cases = plan()["cases"]
    for index, (case, group) in enumerate(zip(cases, document["groups"], strict=True)):
        require(group["example"] == case["example"], "Different frozen example roster")
        if "declined" in group:
            require(group["declined"]["reason"] in ("unsupported_native_clock", "unexpected_native_frame_count",
                                                     "cross_group_content_overlap") and "rows" not in group,
                    "Unexpected or incomplete source decline")
            continue
        require(group["wrong_example"] == cases[(index+1) % len(cases)]["example"], "Wrong reference differs")
        for role, field in (("authentic", "earlier_frames"), ("fake_segments", "generated_frames")):
            prior.verify_clock(group["native"][role])
            require(group["native"][role]["frames"] == case[field], "Native source violates the frozen length")
        earlier_frames = case["earlier_frames"]
        rows = group["rows"]
        roster = [(None, "untouched_delivery"), *((i, role) for i in (0, 1) for role in ROLES)]
        require([(row["render"], row["role"]) for row in rows] == roster, "Missing or extra delivery conditions")
        bindings = {(row["render"], row["role"]): row for row in rows}
        for number in (0, 1):
            generated = bindings[number, "generated_insert"]
            wrong = bindings[number, "wrong_reference"]
            require(wrong["submitted"] == generated["submitted"] and wrong["earlier"] != generated["earlier"],
                    "Wrong-reference control must reuse the exact generated delivery with a different earlier copy")
        for i, (entry, record) in enumerate(zip(case["authored_renders"], group["renders"], strict=True)):
            split = entry["earlier_split"]
            length = entry["generated_slice"][1]-entry["generated_slice"][0]
            require(record["render"] == i and record["replacement_start"] == min(split, earlier_frames-length-24)
                    and set(record["verification"]) == set(ROLES) - {"wrong_reference"}, "Unexpected authored control")
            for role, segments in record["verification"].items():
                start = record["replacement_start"] if role == "equal_length_replacement" else split
                stop = earlier_frames if role == "equal_length_replacement" else earlier_frames + length
                require([item["bounds"] for item in segments] == [[0, start], [start, start+length], [start+length, stop]]
                        and all(0 <= mid <= .10 and mid <= upper <= .22
                                for mid, upper in (item["error_p50_p90"] for item in segments)),
                        "Authored source profile differs")
        for row in rows:
            before, after = row["earlier"], row["submitted"]
            for clip in (before, after):
                prior.verify_clock(clip)
            role, number = row["role"], row["render"]
            length = 0
            want = None
            if number is not None and role != "equal_length_replacement":
                entry = case["authored_renders"][number]
                length = entry["generated_slice"][1] - entry["generated_slice"][0]
                if role in {"generated_insert", "matched_ordinary_insertion"}:
                    want = [entry["earlier_split"], entry["earlier_split"]+length]
            require(row["truth"] == want and before["frames"] == earlier_frames and after["frames"] == earlier_frames+length
                    and (before["width"], before["height"]) == (after["width"], after["height"])
                    and before["codec"] == after["codec"] == "h264", "Paired source geometry or truth differs")
            if role != "wrong_reference":
                require(before == rows[0]["earlier"], "Earlier copy changed between authored conditions")
            require(row["status"] in ("measured", "picture_timelines_differ", "comparison_limit")
                    and all(0 <= a < b <= after["frames"] for a, b in row["existing_picture_ranges"]),
                    "Invalid app coverage or picture range")
            paired = row["paired"]
            if paired["bounds"] is not None:
                start, stop = paired["bounds"]
                lo, hi = paired["uncertain_starts"]
                require(paired["status"] == "paired_insertion" and paired["best"] == paired["bounds"]
                        and 24 <= start < stop <= after["frames"]-24 and stop-start == length
                        and lo <= start <= hi and hi-lo <= 3
                        and all(0 <= mid <= .10 and mid <= upper <= .22 for mid, upper in
                                (paired["before_error_p50_p90"], paired["after_error_p50_p90"])),
                        "Candidate does not match its frozen boundary, flanks or ambiguity limits")
                for frame, item in zip((start-1, (start+stop)//2, stop), paired["evidence"], strict=True):
                    require(item["frame"] == frame and item["pts"] == frame * after["tick_step"]
                            and prior.SHA256.fullmatch(item["gray_96_sha256"]), "Altered evidence frame or clock")
            else:
                require(paired["status"] in ("not_a_longer_pair", "insufficient_matching_frames", "ambiguous_insertion")
                        and "evidence" not in paired, "Unassessed pair reports evidence")
    result = summarize(document["groups"])
    require(document["results"] == result, "Outcome counts differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--verify-app", action="store_true", help="Verify opt-in app bounds, clocks and witness assets")
    args = parser.parse_args()
    if args.phase == "acquire":
        acquire(args.path)
    elif args.phase == "review":
        print(json.dumps(review(json.loads(args.path.read_bytes())), sort_keys=True))
    else:
        if args.preflight is None or args.write is None or args.write.exists():
            parser.error("score requires --preflight and a new --write path")
        result = score(args.path, args.preflight, verify_app=args.verify_app)
        review(result)
        with args.write.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result["results"], sort_keys=True))


if __name__ == "__main__":
    main()
