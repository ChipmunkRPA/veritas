# SPDX-License-Identifier: Apache-2.0
"""Test the frozen paired local picture check with independently encoded H.264 earlier copies."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import resource
import subprocess
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import probe_frames
from app.evidence import video_reference
from app.evidence import video_reference_local as local
from scripts import video_reference_lossy_transfer as previous
from scripts.video_common_sources_v89 import read, require, sha, write

PRIOR = previous.bank.DATA / "video-reference-lossy-transfer-2026-11-11"
SCHEMA = "chipmunk.video-reference-h264-earlier-transfer/1"
SPEC = {
    "schema": SCHEMA,
    "reference_delivery": "H.264 yuv420p, CRF 23, medium preset, fixed 48-picture GOP, independent of submitted encodes.",
    "submitted_deliveries": "All 184 exact previous submissions; two tandem encodes of each new earlier copy at CRF 28 "
                            "and 33; nine unrelated CREMA-D sources with exactly matching native picture-clock prefixes.",
    "alignment": "Identical dimensions and relative native presentation timestamps, complete upright picture tables. "
                 "No resampling, time matching, cropping, or other alignment during comparison.",
    "truth": "Exact pre-encoding authored picture difference and half-open time range; 0.50 temporal IoU, "
             "tile covers at least one eighth changed pixels and one twelfth of the authored changed area.",
    "gate": {"prior_joint_recall_min": .90, "minimum_new_groups": 3, "minimum_assessed": .80,
             "neutral_flags_per_bank": 1, "tandem_flags": 0, "unrelated_flags": 0},
    "point": local.POINT,
}
LOCKED = ("freeze.json", "calibration.json", "held.jsonl", "aggregate.json", "independent-audit.json")


def bindings(prior):
    implementations = (*[getattr(local, name) for name in ("rectangles", "measure", "assess", "cadence")],
                       video_reference._comparison_issue, video_reference._elapsed)
    method = "\n".join(inspect.getsource(function) for function in implementations)
    return {
        "runner": sha(Path(__file__)), "method": hashlib.sha256(method.encode()).hexdigest(),
        "minimum_fraction": local.MIN_FRACTION, "width_ms": list(local.WIDTH_MS), "point": local.POINT,
        "previous_sources": {name: sha(prior / name) for name in LOCKED},
        "previous_runner": sha(Path(previous.__file__)),
        "versions": [cv2.__version__, np.__version__,
                     subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=4).splitlines()[0]],
    }


def group_records(prior):
    source_plan = read(prior / "freeze.json")
    groups = list(previous.prepare_groups(source_plan, "held"))
    require(len(groups) == 18, "Need the exact two held source banks")
    old = {(row["bank"], row["slot"], row["condition"]): row for row in previous.rows(prior, "held")}
    require(len(old) == 184 and read(prior / "aggregate.json")["point"] == local.POINT, "Previous cohort or point differs")
    return groups, old


def relative_native(row):
    times = row["expected"]["picture_pts_ms"]
    return [value - times[0] for value in times]


def unrelated(groups, name, index, original):
    if name != "cremad":
        return None
    target = relative_native(original)
    choices = [(len(clock), j, path, other)
               for bank, j, _, rows, _ in groups if bank == name and j != index
               for path, other, _ in rows[:1]
               if other["source_group"] != original["source_group"]
               and (clock := relative_native(other))[:len(target)] == target]
    return min(choices, default=None)


def roster(prior):
    groups, old = group_records(prior)
    for name, index, group, rows, alternate in groups:
        earlier, original, _ = rows[0]
        cases = []
        for kind, condition, source, authored, _, _, truth, compatible in previous.cases_for(rows, alternate):
            key = name, index, condition
            record = old[key]
            require((record["kind"], record["earlier_sha256"], record["authored_sha256"], record["known_edit"],
                     record["visible_source_time"]) == (kind, original["file_sha256"], authored["file_sha256"],
                                                       truth, compatible), "Prior case binding differs")
            dest = prior / "held" / f"{name}-{index:03d}-{condition}.mkv"
            cases.append((kind, condition, source, authored, dest, truth, compatible, record))
        yield name, index, group, earlier, original, cases, unrelated(groups, name, index, original)


def freeze(run, prior):
    plan = []
    for name, index, _, earlier, original, cases, donor in roster(prior):
        require(sha(earlier) == original["file_sha256"], "Reference bytes differ")
        plan.append({"bank": name, "slot": index, "native_earlier_sha256": sha(earlier),
                     "submitted": {case[1]: case[-1]["submitted_sha256"] for case in cases},
                     "unrelated_slot": donor[1] if donor else None,
                     "unrelated_original_sha256": donor[3]["file_sha256"] if donor else None})
    require(sum(row["unrelated_slot"] is not None for row in plan) == 9, "Unrelated source controls differ")
    run.mkdir(mode=0o700, parents=True)
    write(run / "freeze.json", {"spec": SPEC, "binding": bindings(prior), "roster": plan})


def validate(run, prior):
    locked = read(run / "freeze.json")
    require(locked["spec"] == SPEC and locked["binding"] == bindings(prior), "Frozen production measurement or source data differs")
    return {(item["bank"], item["slot"]): item for item in locked["roster"]}


def encode_h264(source, dest, crf, *, earlier=False, frames=None):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-filter_threads", "1", "-threads", "1", "-copyts", "-i", str(source),
            "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-enc_time_base", "1:1000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", "-crf", str(crf)]
    args += ["-preset", "medium", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0"] if earlier else ["-preset", "veryfast"]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    subprocess.run([*args, str(dest)], check=True, timeout=25)


def check_clock(first, later):
    problem = video_reference._comparison_issue((first, later))
    require(problem is None and first.rotation == later.rotation == 0 and len(first) <= local.MAX_FRAMES
            and (first.width, first.height) in local.SHAPES, "Exact geometry or picture clock differed")
    dt = local.cadence(video_reference._elapsed(first))
    require(dt is not None, "Native picture cadence changed")
    return dt


def observe(earlier_frames, submitted, early, later):
    dt = check_clock(early, later)
    rects = local.rectangles(early.height, early.width)
    signals, errors, differing = [], [], []
    with closing(previous.verified_frames(submitted, later)) as decoded:
        for first, second in zip(earlier_frames, decoded, strict=True):
            value, error = local.measure(first, second, rects)
            signals.append(value)
            errors.append(error)
            differing.append(not np.array_equal(first, second))
    status, best = local.assess(signals, errors, dt, rects)
    exact = [[int(a), int(b)] for a, b in video_reference.marked_ranges(np.array(differing))]
    if best:
        x, y, width, height = (best["tile"][field] for field in ("x", "y", "w", "h"))
        best = {"score": best["score"], "frames": best["frames"], "tile_xywh": [x, y, width, height],
                "samples": list(best["samples"].values())}
    return {"status": status, "best": best, "exact": exact, "errors": errors}


def table(path):
    require(local._carrier(path) == [("h264", "yuv420p")], "Expected a color H.264 delivery")
    return probe_frames(path, cap=local.MAX_FRAMES, timeout_s=10, max_pixels=previous.MAX_PIXELS)


def run_study(run, prior):
    locked = validate(run, prior)
    folder = run / "deliveries"
    folder.mkdir(mode=0o700)
    start = time.monotonic()
    with (run / "held.jsonl").open("x") as out:
        for name, index, group, earlier, original, cases, donor in roster(prior):
            frozen = locked[name, index]
            require(frozen["native_earlier_sha256"] == sha(earlier)
                    and frozen["unrelated_slot"] == (donor[1] if donor else None), "Frozen case source differs")
            source_table = previous.earlier_table(earlier, original)
            original_frames = list(previous.verified_frames(earlier, source_table))
            require([previous.bank.raw_digest(rgb) for rgb in original_frames] == original["expected"]["rgb_sha256"],
                    "Native picture identities differ from the authored earlier reference")
            reference = folder / f"{name}-{index:03d}-earlier.mkv"
            encode_h264(earlier, reference, 23, earlier=True)
            early = table(reference)
            check_clock(source_table, early)
            earlier_frames = list(previous.verified_frames(reference, early))
            early_sha = sha(reference)
            generated = []
            for crf in (28, 33):
                role = f"tandem_crf{crf}"
                delivery = folder / f"{name}-{index:03d}-{role}.mkv"
                encode_h264(reference, delivery, crf)
                generated.append(("tandem", role, None, None, delivery, None, False, None))
            if donor:
                _, _, source, authored = donor
                previous.earlier_table(source, authored)
                require(sha(source) == frozen["unrelated_original_sha256"], "Unrelated source bytes differ")
                delivery = folder / f"{name}-{index:03d}-unrelated.mkv"
                encode_h264(source, delivery, 28, frames=len(early))
                generated.append(("different_source", "different_source_native_prefix", source, authored,
                                  delivery, None, False, None))
            for kind, role, source, authored, dest, truth, compatible, prior_case in [*cases, *generated]:
                digest = sha(dest)
                if prior_case:
                    require(digest == prior_case["submitted_sha256"] == frozen["submitted"][role]
                            and sha(source) == authored["file_sha256"], "Exact submitted or authored input differs")
                require(early_sha != digest, "Reference and delivery must have distinct encodings")
                later = table(dest)
                values = observe(earlier_frames, dest, early, later)
                spatial = None
                if truth and values["best"]:
                    authored_table = previous.earlier_table(source, authored)
                    spatial = previous.authored_spatial(original_frames, source, authored_table,
                                                         authored["expected"], values["best"])
                out.write(json.dumps({"bank": name, "slot": index, "source_group": group, "kind": kind,
                    "condition": role, "earlier_sha256": early_sha, "submitted_sha256": digest,
                    "authored_sha256": sha(source) if source else None, "known_edit": truth,
                    "visible_source_time": compatible, "spatial": spatial, **values}, sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            print(name, index, len(cases) + len(generated), flush=True)
    write(run / "runtime.json", {"elapsed_s": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "journal_sha256": sha(run / "held.jsonl"), "freeze_sha256": sha(run / "freeze.json")})


def report(run, prior):
    validate(run, prior)
    runtime = read(run / "runtime.json")
    require(runtime["journal_sha256"] == sha(run / "held.jsonl") and runtime["freeze_sha256"] == sha(run / "freeze.json"),
            "Held run commitment differs")
    rows = previous.rows(run, "held")
    _, old = group_records(prior)
    require(len(rows) == 229 and len({(row["bank"], row["slot"], row["condition"]) for row in rows}) == 229,
            "Incomplete held roster")
    point, gate = local.POINT, SPEC["gate"]
    panels = {}
    for name in ("grid", "cremad"):
        counts, status_counts, groups = {}, Counter(), set()
        for kind in ("neutral", "tandem", "different_source", "ordinary_local", "neural"):
            summary = Counter()
            for row in rows:
                if (row["bank"], row["kind"]) != (name, kind):
                    continue
                case = old.get((name, row["slot"], row["condition"]))
                obs = previous.decide(row, point)
                obs["assessed"] = row["status"] in ("local_difference", "no_local_range")
                joint = obs["time_correct"] and obs["spatial_correct"]
                legacy = previous.decide(case, point) if case else None
                before = bool(legacy and legacy["time_correct"] and legacy["spatial_correct"])
                summary.update({"views": 1, **obs, "joint_correct": joint, "prior_joint_correct": before,
                                "prior_correct_retained": before and joint, "prior_correct_lost": before and not joint,
                                "new_correct": joint and not before, "visible_source_time": row["visible_source_time"],
                                "visible_source_time_joint": row["visible_source_time"] and joint})
                status_counts[f"{kind}/{row['status']}"] += 1
                if kind == "neural" and joint:
                    groups.add(row["source_group"])
            counts[kind] = dict(summary)
        neural, neutral, tandem, unrelated_rows = (counts[kind] for kind in ("neural", "neutral", "tandem", "different_source"))
        criteria = {"prior_recall": neural["prior_correct_retained"] >= gate["prior_joint_recall_min"] * neural["prior_joint_correct"],
                    "group_coverage": len(groups) >= gate["minimum_new_groups"],
                    "assessed": all(arm["assessed"] >= gate["minimum_assessed"] * arm["views"]
                                    for arm in (neural, neutral, tandem)),
                    "neutral_flags": neutral["flag"] <= gate["neutral_flags_per_bank"],
                    "tandem_flags": tandem["flag"] == gate["tandem_flags"],
                    "unrelated_flags": unrelated_rows["flag"] == gate["unrelated_flags"]}
        panels[name] = {"counts": counts, "statuses": dict(status_counts), "correct_source_groups": len(groups),
                        "criteria": criteria, "advances": all(criteria.values())}
    write(run / "aggregate.json", {"spec": SPEC, "runtime": runtime, "panels": panels,
                                    "advances_both": all(row["advances"] for row in panels.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", type=Path, default=PRIOR)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    {"freeze": freeze, "held": run_study, "report": report}[args.phase](args.run, args.prior)


if __name__ == "__main__":
    main()
