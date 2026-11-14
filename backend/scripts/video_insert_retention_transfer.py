# SPDX-License-Identifier: Apache-2.0
"""Replay a stricter check of retained pictures around one video insertion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_temporal_insert_confirmation as confirmation
from scripts import video_temporal_insert_native as native
from scripts import video_temporal_multiple_inserts as multiple

SCHEMA = "chipmunk.video-insert-retention-transfer/1"
SPEECH = native.ROOT / "docs/evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json"
WINDOW = 8
MAX_MEDIAN = .035
ALTERNATE_MARGIN = .020
DURATION = 150
SIZE = (480, 270)
GUARDS = ("median", "alternate", "both")
RECIPES = {
    "ordinary_single_early": [(31, 0, "donor", 9, 39)],
    "ordinary_single_late": [(105, 0, "donor", 9, 51)],
    "ordinary_two": [(36, 0, "donor", 9, 39), (104, 0, "donor", 51, 87)],
    "ordinary_three": [(30, 0, "donor", 6, 26), (74, 0, "donor", 42, 70), (119, 0, "donor", 84, 108)],
    "mixed_replacement": [(36, 0, "donor", 9, 39), (67, 24, "donor", 67, 91), (104, 0, "donor", 51, 87)],
    "mixed_removal": [(36, 0, "donor", 9, 39), (72, 20, "base", 0, 0), (104, 0, "donor", 51, 87)],
    "unchanged": [],
    "wrong_earlier": [(31, 0, "donor", 9, 39)],
    "static_two": [(36, 0, "donor", 9, 39), (104, 0, "donor", 51, 87)],
    "brightness_grade_single": [(31, 0, "donor", 9, 39)],
}
SINGLES = {"single", "ordinary_single_early", "ordinary_single_late", "brightness_grade_single"}
NEGATIVE = {*multiple.MULTI, "ordinary_three", "mixed_replacement", "mixed_removal", "static_two", "near_static_two",
            "unchanged", "wrong_earlier"}
METHOD = {
    "retention_window_frames": WINDOW,
    "maximum_retained_median": MAX_MEDIAN,
    "maximum_gain_at_another_offset": ALTERNATE_MARGIN,
    "source": "Four prior equal-cadence development groups. Held tests use eight distinct older source groups with frozen "
              "multi-edit recipes, plus four disjoint authentic sources across two publishers with new edit recipes.",
    "decision": "Intersect the two checks. Median error tests for a run of mismatched retained pictures. The offset check "
                "measures whether any alternate forward offset produces a closer eight-picture correspondence.",
    "primary_gate": "Keep at least 90% of previously correct single-insertion reports in each held bank; withhold at least "
                    "half the inaccurate multi/mixed reports overall. No new flagged controls or shifted boundaries. "
                    "Report grade transfer separately.",
}


def plan():
    candidates = multiple.roster()
    return {"schema": SCHEMA, "method": METHOD, **candidates,
            "public_sources": json.loads(SPEECH.read_bytes())["sources"], "recipes": RECIPES}


def local_stats(earlier, submitted, selected):
    from app.evidence import video_reference_insertion

    native.source.require((WINDOW, MAX_MEDIAN, ALTERNATE_MARGIN) ==
                          (video_reference_insertion.RETENTION_WINDOW, video_reference_insertion.MAX_RETENTION_MEDIAN,
                           video_reference_insertion.MAX_ALTERNATE_GAIN), "Runtime and frozen points differ")
    return video_reference_insertion.retention_support(earlier, submitted, selected)


def outcomes(local):
    if local is None:
        return {name: False for name in GUARDS}
    median = max(row["median"] for row in local) <= MAX_MEDIAN
    alternate = max(row["alternate_gain"] for row in local) <= ALTERNATE_MARGIN
    return {"median": median, "alternate": alternate, "both": median and alternate}


def cut_decode(path, second):
    import numpy as np

    w, h = SIZE
    data = native.process([*native.ffmpeg(), "-ss", str(second), *native.input_file(path), "-map", "0:v:0",
                           "-vf", f"fps=24,scale={w}:{h}:flags=area,format=yuv420p", "-an", "-sn", "-dn",
                           "-frames:v", str(DURATION), "-threads", "1", "-f", "rawvideo", "pipe:1"])
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(-1, w * h * 3 // 2)
    native.source.require(len(pixels) == DURATION, "Public source did not supply every planned picture")
    return pixels


def measure(earlier, submitted, label, exact, render):
    from app.evidence.video_reference_insertion import pair_interval

    single = pair_interval(earlier[1], submitted[1])
    selected = single["bounds"]
    witness = ([{"frame": frame, "pts": submitted[0].pts[frame], "luma_sha256": submitted[2][frame]}
                for frame in (selected[0] - 1, sum(selected) // 2, selected[1])] if selected else [])
    local = local_stats(earlier[1], submitted[1], selected) if selected else None
    return {"role": label, "truth": exact, "earlier": earlier[3], "submitted": submitted[3],
            "render": render, "single": single, "support": local, "arms": outcomes(local), "witnesses": witness}


def do_group(folder, *, group, size, base, donor, wrong, recipes):
    import numpy as np

    def delivery(name, pictures):
        target = folder / f"{name}.mp4"
        multiple.encode(target, size, pictures)
        return native.native_clip(target)

    still, near = multiple.freeze_controls(base)
    flavors = {"wrong_earlier": wrong, "static_two": still, "near_static_two": near}
    original = delivery("base", base)
    alternate = delivery("donor", donor)
    rows = []
    for label, recipe in recipes.items():
        real_base = flavors.get(label, base) if label != "wrong_earlier" else base
        real_original = delivery(label + "-earlier", real_base) if label in flavors and label != "wrong_earlier" else original
        earlier = delivery(label + "-earlier", wrong) if label == "wrong_earlier" else real_original
        edited, provenance, truth = multiple.construct(real_base, donor, recipe)
        if label == "brightness_grade_single":
            edited[:, :size[0] * size[1]] = np.clip(edited[:, :size[0] * size[1]].astype(np.int16) + 10, 0, 255)
        submitted = delivery(label, edited)
        expected = np.stack([(alternate if source else real_original)[1][index] for source, index in provenance])
        _, cuts = multiple.truth(recipe)
        spans = list(zip(cuts, cuts[1:]))
        render = confirmation.verify_render(submitted, [(a, expected[a:b]) for a, b in spans], len(edited))
        snippets = truth if label in multiple.MULTI | SINGLES | {"ordinary_three"} else None
        rows.append(measure(earlier, submitted, label, snippets, render))
    return {"group": group, "rows": rows}


def measure_groups(input_plan, category, archive):
    from app.detectors.video.decode import probe_frames

    if category == "public":
        sources = input_plan["public_sources"]
        paths = [archive / f"{row['id']}.webm" for row in sources]
        for row, path in zip(sources, paths, strict=True):
            native.source.require(path.stat().st_size == row["bytes"] and native.file_sha(path) == row["sha256"],
                                  "Public recording differed from its pinned attribution")
        for index, (row, path) in enumerate(zip(sources, paths, strict=True)):
            with TemporaryDirectory(prefix="chipmunk-retention-public-") as directory:
                wrong = (index + 1) % len(paths)
                yield do_group(Path(directory), group=row["id"], size=SIZE, base=cut_decode(path, row["offsets"][0]),
                               donor=cut_decode(path, row["offsets"][1]),
                               wrong=cut_decode(paths[wrong], sources[wrong]["offsets"][0]),
                               recipes=RECIPES)
        return
    cases = input_plan[category]
    for index, case in enumerate(cases):
        paths = multiple.files(case, archive)
        wrong = multiple.files(cases[(index + 1) % len(cases)], archive)
        geometry = native.delivery_size(probe_frames(paths["authentic"], cap=600, timeout_s=60))
        with TemporaryDirectory(prefix="chipmunk-retention-archive-") as directory:
            yield do_group(Path(directory), group=case["example"], size=geometry,
                           base=multiple.raw_pictures(paths["authentic"], geometry, DURATION),
                           donor=multiple.raw_pictures(paths["fake_segments"], geometry, 125),
                           wrong=multiple.raw_pictures(wrong["authentic"], geometry, DURATION),
                           recipes=case["edits"])


def count(groups):
    metrics = {}
    for group in groups:
        for row in group["rows"]:
            role, selected = row["role"], row["single"]["bounds"]
            exact = row["truth"]
            correct = role in SINGLES and multiple.matches(exact, [selected] if selected else None, 3)
            poor = selected is not None and (role in NEGATIVE or (role in SINGLES and not correct))
            counter = metrics.setdefault(role, Counter())
            counter.update(planned=1, existing_reports=int(selected is not None), existing_correct=int(correct), existing_bad=int(poor))
            for arm in GUARDS:
                counter.update({f"{arm}_reports": int(row["arms"][arm]), f"{arm}_correct": int(correct and row["arms"][arm]),
                                f"{arm}_bad": int(poor and row["arms"][arm])})
    return metrics


def gate(counts):
    core = {name: stats for name, stats in counts.items() if name != "brightness_grade_single"}
    supported = sum(stats["both_correct"] for stats in core.values())
    original = sum(stats["existing_correct"] for stats in core.values())
    wrong = sum(stats["existing_bad"] for stats in core.values())
    retained_wrong = sum(stats["both_bad"] for stats in core.values())
    return {"retains_at_least_90_percent_of_correct_singles": bool(original) and 10 * supported >= 9 * original,
            "withholds_at_least_half_of_wrong_ranges": bool(wrong) and 2 * retained_wrong <= wrong,
            "changes_no_old_bounds_or_unflagged_controls": all(stats["both_reports"] <= stats["existing_reports"] for stats in core.values())}


def review(result):
    native.source.require(result["schema"] == SCHEMA and result["method"] == METHOD, "Wrong fixed protocol")
    category = result["partition"]
    source_plan = plan()
    native.source.require(result["plan_sha256"] == native.exact_json_sha(source_plan), "Edit or publisher plan changed")
    roster = source_plan[category] if category != "public" else source_plan["public_sources"]
    for expected, group in zip(roster, result["groups"], strict=True):
        recipes = RECIPES if category == "public" else expected["edits"]
        native.source.require(group["group"] == expected.get("id", expected.get("example")) and
                              [row["role"] for row in group["rows"]] == list(recipes), "Missing source or condition")
        for row in group["rows"]:
            role, earlier, submitted = (row[k] for k in ("role", "earlier", "submitted"))
            native.verify_clock(earlier)
            native.verify_clock(submitted)
            exact, boundaries = multiple.truth(recipes[role])
            native.source.require(earlier["frames"] == DURATION and submitted["frames"] == boundaries[-1]
                                  and (earlier["width"], earlier["height"]) == (submitted["width"], submitted["height"])
                                  and earlier["codec"] == submitted["codec"] == "h264"
                                  and earlier["pixel_format"] == submitted["pixel_format"] == "yuv420p"
                                  and max(earlier["width"], earlier["height"]) == 480
                                  and row["truth"] == (exact if role in multiple.MULTI | SINGLES | {"ordinary_three"} else None)
                                  and [span["bounds"] for span in row["render"]] == [list(span) for span in zip(boundaries, boundaries[1:])]
                                  and all(0 <= median <= .10 and median <= p90 <= .22
                                          for median, p90 in (span["error_p50_p90"] for span in row["render"]))
                                  and row["arms"] == outcomes(row["support"]), "Unexpected delivery or decision")
            if selected := row["single"]["bounds"]:
                a, b = selected
                native.source.require(24 <= a < b <= submitted["frames"]-24 and b-a == submitted["frames"]-DURATION
                                      and len(row["support"]) == 2 and all(
                                          0 <= side["alternate_gain"] <= 16 and 0 <= side["median"] <= 16
                                          for side in row["support"]), "Retained picture measurement is impossible")
                for witness, frame in zip(row["witnesses"], (a-1, (a+b)//2, b), strict=True):
                    native.source.require(witness["frame"] == frame and witness["pts"] == submitted["tick_step"] * frame
                                          and native.SHA256.fullmatch(witness["luma_sha256"]), "Incorrect witness clock")
            else:
                native.source.require(row["support"] is None and not row["witnesses"], "Unsupported comparison carries evidence")
    metrics = count(result["groups"])
    native.source.require(metrics == result["counts"] and result["gates"] == gate(metrics), "Incorrect totals or gates")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "development", "held", "public", "review"))
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(json.loads(args.path.read_bytes())), sort_keys=True))
        return
    locked = plan()
    if args.phase == "plan":
        print(json.dumps(locked, indent=2))
        return
    if not args.write:
        parser.error("Provide a new --write path")
    groups = []
    for row in measure_groups(locked, args.phase, args.path):
        groups.append(row)
        print(f"Measured {args.phase} {row['group']}: {len(row['rows'])} comparisons", flush=True)
    metrics = count(groups)
    result = {"schema": SCHEMA, "method": METHOD, "partition": args.phase, "plan_sha256": native.exact_json_sha(locked),
              "groups": groups, "counts": metrics, "gates": gate(metrics)}
    review(result)
    with args.write.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
