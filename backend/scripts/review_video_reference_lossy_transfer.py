# SPDX-License-Identifier: Apache-2.0
"""Replay delivery clocks, paired decisions and selected native video pixels offline."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_reference_lossy_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def _interval_intersects(first, second):
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return 2 * intersection >= union


def _qualify(row, threshold):
    chosen = row["best"]
    truth = row["known_edit"]
    seen = row["status"] == "measured"
    require(seen == (chosen is not None), "Comparison scope disagrees with the score")
    flag = seen and chosen["score"] > threshold
    time = bool(flag and truth and _interval_intersects(chosen["frames"], truth))
    native_overlap = row["spatial"]
    space = bool(flag and native_overlap and native_overlap[0] >= max(native_overlap[1] / 12, native_overlap[2] / 8))
    exact = bool(truth and any(_interval_intersects(interval, truth) for interval in row["exact"]))
    return seen, flag, time, space, exact


def _frames(path, table, indexes):
    wanted = sorted(set(indexes))
    found = {f.index: f for f in decode_frames(path, table, wanted, timeout_s=30, max_pixels=study.MAX_PIXELS)}
    require(list(found) == wanted and all(f.pts_verified and f.pts == table.pts[i] for i, f in found.items()),
            "Source frame missing or not timestamp verified")
    return found


def _clock(table):
    return [(pts - table.pts[0]) * table.tb for pts in table.pts]


def _verify_tile(earlier, dest, first, last, best):
    """Recompute the chosen tile statistics using box-filtered means."""
    clock = _clock(first)
    step = float(np.median(np.diff([float(tick) for tick in clock])))
    flank = round(.24 / step)
    start, end = best["frames"]
    indexes = range(start - flank, end + flank)
    earlier_frames = _frames(earlier, first, indexes)
    delivered_frames = _frames(dest, last, indexes)
    x, y, w, h = best["tile_xywh"]
    all_x = np.arange(0, first.width - w + 1, max(1, w // 2))
    all_y = np.arange(0, first.height - h + 1, max(1, h // 2))
    core = []
    for index in indexes:
        images = [earlier_frames[index].rgb, delivered_frames[index].rgb]
        planes = [cv2.GaussianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 1.2)
                  for pixels in images]
        stats = [np.quantile(cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA), [.2, .5, .8]) for p in planes]
        a, b = stats
        require(min(a[2] - a[0], b[2] - b[0]) >= 10, "Replay window has unsupported brightness")
        normal = np.clip((planes[1] - b[1]) * np.clip((a[2] - a[0]) / (b[2] - b[0]), .75, 1.35) + a[1], 0, 255)
        difference = np.abs(planes[0] - normal)
        averages = cv2.boxFilter(difference, cv2.CV_64F, (w, h), anchor=(0, 0))
        baseline = np.quantile(averages[np.ix_(all_y, all_x)], .65)
        core.append(max(0.0, averages[y, x] - baseline))
    medians = [float(np.median(part)) for part in (core[:flank], core[flank:-flank], core[-flank:])]
    require(np.allclose(medians, best["tile_response"], atol=1e-5, rtol=1e-6)
            and abs(medians[1] - max(medians[0], medians[2]) - best["score"]) < 1e-5,
            "Selected tile response does not reproduce from the decoded sources")
    return 2 * len(indexes)


def verify(run):
    freeze = run / "freeze.json"
    plan = read(freeze)
    aggregate = read(run / "aggregate.json")
    calibration = read(run / "calibration.json")
    held = study.rows(run, "held")
    development = study.rows(run, "development")
    neutral = [row["best"]["score"] for row in development
               if row["kind"] in ("neutral", "different_source") and row["best"]]
    point = max(.8, max(neutral) + .15)
    require(aggregate["point"] == calibration["point"] == point and len(neutral) == calibration["development_controls_measured"]
            and sha(run / "development.jsonl") == calibration["development"] and sha(freeze) == calibration["freeze"]
            == aggregate["freeze_sha256"] and sha(run / "held.jsonl") == aggregate["held_sha256"],
            "Calibration or held commitments do not match")
    totals, groups, examples = {}, {}, {}
    for row in held:
        seen, flag, time, space, exact = _qualify(row, point)
        joint = time and space
        name, kind = row["bank"], row["kind"]
        totals.setdefault((name, kind), Counter()).update(
            views=1, assessed=seen, flag=flag, time_correct=time, spatial_correct=space,
            exact_correct=exact, joint_correct=joint, visible_source_time=row["visible_source_time"],
            visible_source_time_joint=joint and row["visible_source_time"],
        )
        if kind == "neural" and joint and not exact:
            groups.setdefault(name, set()).add(row["source_group"])
            old = examples.get(name)
            if old is None or row["visible_source_time"] or not old["visible_source_time"] and row["best"]["score"] > old["best"]["score"]:
                examples[name] = row
    for name, panel in aggregate["panels"].items():
        require(all(Counter(value) == totals.get((name, kind), Counter()) for kind, value in panel["counts"].items())
                and panel["added_correct_source_groups"] == len(groups.get(name, set())), "Published decision counts differ")
        counts = panel["counts"]
        advances = (len(groups.get(name, set())) >= 3 and counts["neutral"]["flag"] <= 1
                    and not counts["different_source"].get("flag", 0)
                    and all(5 * counts[k]["assessed"] >= 4 * counts[k]["views"] for k in ("neural", "neutral")))
        require(advances == panel["advances"], "Advancement outcome differs")
    require(aggregate["advances_both"] == all(panel["advances"] for panel in aggregate["panels"].values()),
            "Transfer outcome differs")

    items = {(row["bank"], row["slot"], row["condition"]): row for row in held}
    pictures, crops, checked = 0, 0, 0
    response_pictures = 0
    cv2.setNumThreads(1)
    out = run / "private-examples"
    out.mkdir(mode=0o700)
    for bank, index, _, source_rows, alternative in study.prepare_groups(plan, "held"):
        earlier, earlier_row, _ = source_rows[0]
        first = probe_frames(earlier, cap=study.MAX_FRAMES, max_pixels=study.MAX_PIXELS)
        clock = _clock(first)
        for kind, label, authored, source_record, _, _, truth, _ in study.cases_for(source_rows, alternative):
            row = items.pop((bank, index, label))
            dest = run / "held" / f"{bank}-{index:03d}-{label}.mkv"
            require((row["earlier_sha256"], row["authored_sha256"], row["submitted_sha256"])
                    == (sha(earlier), sha(authored), sha(dest)) and row["earlier_sha256"] == earlier_row["file_sha256"]
                    and row["authored_sha256"] == source_record["file_sha256"], "Held file identity changed")
            last = probe_frames(dest, cap=study.MAX_FRAMES, max_pixels=study.MAX_PIXELS)
            require(_clock(last) == clock and not (last.interpolated or last.truncated or last.rotation)
                    and (first.width, first.height) == (last.width, last.height), "Held native presentation clock differs")
            checked += 1
            best = row["best"]
            if best is None:
                continue
            require([Fraction(x) for x in best["bounds_s"]] == [clock[i] for i in best["frames"]]
                    and [Fraction(x) for x in best["sample_s"]] == [clock[i] for i in best["samples"]],
                    "Selected range does not map to native pictures")
            start, end = best["frames"]
            before, middle, after = best["samples"]
            x, y, w, h = best["tile_xywh"]
            require(before == start - 1 < start <= middle < end == after and x >= 0 and y >= 0
                    and x + w <= first.width and y + h <= first.height, "Selected tile or context is outside the native image")
            should_save = examples.get(bank) is row
            if not (should_save or truth and _qualify(row, point)[1]):
                continue
            source = probe_frames(authored, cap=study.MAX_FRAMES, max_pixels=study.MAX_PIXELS)
            reference = _frames(earlier, first, best["samples"] if should_save else [middle])
            truth_frame = _frames(authored, source, [middle])[middle]
            mask = np.max(np.abs(reference[middle].rgb.astype(np.int16) - truth_frame.rgb.astype(np.int16)), axis=2) != 0
            require(row["spatial"] == [int(np.count_nonzero(mask[y:y+h, x:x+w])), int(mask.sum()), w * h]
                    and truth == row["known_edit"] and kind == row["kind"], "Authored overlap does not replay")
            pictures += len(reference) + 1
            if should_save:
                delivered = _frames(dest, last, best["samples"])
                pictures += len(delivered)
                response_pictures += _verify_tile(earlier, dest, first, last, best)
                example = out / bank
                example.mkdir(mode=0o700)
                evidence = []
                for role, frame in zip(("before", "during", "after"), best["samples"], strict=True):
                    for side, sample in (("earlier", reference[frame]), ("submitted", delivered[frame])):
                        pixels = sample.rgb[y:y+h, x:x+w]
                        name = f"{role}-{side}.png"
                        Image.fromarray(pixels).save(example / name)
                        evidence.append({"role": role, "side": side, "native_pts": sample.pts,
                                         "relative_s": str(clock[frame]), "file": name, "sha256": sha(example / name),
                                         "native_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
                        crops += 1
                write(example / "evidence.json", {"tile_xywh": best["tile_xywh"], "reported_s": best["bounds_s"],
                                                    "truth_s": [str(clock[i]) for i in truth], "samples": evidence})
    require(not items and checked == len(held), "Held delivery roster differs")
    write(run / "independent-audit.json", {
        "scope": "Offline decisions, exact held file identities, clocks, native changed-pixel counts and example crops.",
        "held_deliveries_verified": checked, "original_and_authored_or_delivery_pictures_reopened": pictures,
        "native_picture_pairs_for_independent_tile_response": response_pictures // 2,
        "paired_native_crops_written": crops, "aggregate_sha256": sha(run / "aggregate.json"),
        "data_redistributed": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    verify(args.run)


if __name__ == "__main__":
    main()
