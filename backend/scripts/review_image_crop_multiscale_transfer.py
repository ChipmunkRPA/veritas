# SPDX-License-Identifier: Apache-2.0
"""Replay the multi-scale crop study without weights, or verify its retained native inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from types import SimpleNamespace

SCHEMA = "chipmunk.image-crop-multiscale-transfer/1"
SCALES = ("6", "4", "3")
POSITIVES = {"edited", "feathered_edit", "edit"}
ROLES = {"fit": (8, ("original", "tone", "blur")),
         "primary": (12, ("original", "edited", "feathered_edit", "tone", "local_blur", "resample")),
         "legacy": (7, ("original", "edited", "feathered_edit", "global_processing")),
         "screen": (12, ("template", "photo", "edit"))}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rectangles(shape, scale):
    h, w = shape
    n = int(scale)
    denominator = 2*n
    return [((w*x+denominator-1)//denominator, (h*y+denominator-1)//denominator,
             (w*(x+2)+denominator-1)//denominator, (h*(y+2)+denominator-1)//denominator)
            for y in range(denominator-1) for x in range(denominator-1)]


def exact_selection(row, points):
    shape = row["shape"]
    need(len(shape) == 2 and min(shape) >= 24, "Invalid source geometry")
    picks = {}
    for scale in SCALES:
        values = row["logits"][scale]
        boxes = rectangles(shape, scale)
        need(len(values) == len(boxes) and all(math.isfinite(value) for value in values), "Invalid scale score vector")
        i = values.index(max(values))
        picks[scale] = {"box": list(boxes[i]), "response": values[i]-median(values)}
    winner = max(SCALES, key=lambda scale: picks[scale]["response"])
    ax, ay, bx, by = picks["3"]["box"]
    others = [value for (x, y, right, bottom), value in zip(rectangles(shape, "3"), row["logits"]["3"], strict=True)
              if right <= ax or bx <= x or bottom <= ay or by <= y]
    remote = median(others)
    baseline = {"box": [ax, ay, bx, by], "relative": picks["3"]["response"], "remote": remote}
    baseline["flag"] = baseline["relative"] >= points["relative"] and remote < points["remote_upper_strict"]
    return picks, winner, baseline


def scalar_overlap(box, overlap, target):
    x, y, right, bottom = box
    area = (right-x)*(bottom-y)
    need(type(overlap) is int and type(target) is int and 0 <= overlap <= min(area, target), "Invalid native pixel counts")
    useful = bool(target and overlap/target >= .5 and overlap/(area+target-overlap) >= .30)
    return {"reported": area, "overlap": overlap, "target": target, "useful": useful}


def roster(rows, panels):
    expected = {(name, slot, role) for name in panels for slot in range(ROLES[name][0]) for role in ROLES[name][1]}
    keys = [tuple(row["key"]) for row in rows]
    need(len(keys) == len(expected) and set(keys) == expected, "Incomplete or duplicate study cases")


def verify_fit(rows, points):
    roster(rows, ("fit",))
    maxima = {}
    for row in rows:
        peaks, winner, _ = exact_selection(row, points)
        slot = str(row["key"][1])
        maxima[slot] = max(maxima.get(slot, -math.inf), peaks[winner]["response"])
    return {"point": sorted(maxima.values())[-2], "maxima": maxima}


def verify_results(rows, baseline_points, point):
    roster(rows, ("primary", "legacy", "screen"))
    metrics, geometry = {}, {}
    flags = {"baseline": set(), "candidate": set()}
    gains = {"added_primary_feathered": [], "added_screen": [], "lost_useful": []}
    for row in rows:
        panel, slot, role = row["key"]
        peaks, winner, baseline = exact_selection(row, baseline_points)
        need((row["target"] > 0) == (role in POSITIVES) and row["target"] <= math.prod(row["shape"]), "Invalid target size")
        need(row["winner"] == winner and row["candidate_flag"] == (peaks[winner]["response"] > point), "Candidate rule differs")
        counter = metrics.setdefault(f"{panel}/{role}", Counter(views=0))
        counter["views"] += 1
        for scale in SCALES:
            measured = scalar_overlap(peaks[scale]["box"], row["peaks"][scale]["overlap"]["overlap"], row["target"])
            need(row["peaks"][scale] == peaks[scale] | {"overlap": measured}, "Per-scale crop selection differs")
            counter[f"top_{scale}_useful"] += measured["useful"]
            maximum = row["geometry"][scale]
            need(0 <= maximum["best_iou"] <= 1 and 0 <= maximum["useful_proposals"] <= len(row["logits"][scale])
                 and (not measured["useful"] or maximum["useful_proposals"] > 0), "Impossible geometry maximum")
        if role in POSITIVES:
            ceiling = geometry.setdefault(f"{panel}/{role}", Counter(views=0))
            ceiling["views"] += 1
            for scale in SCALES:
                ceiling[scale] += row["geometry"][scale]["useful_proposals"] > 0
        old = scalar_overlap(baseline["box"], row["baseline"]["overlap"]["overlap"], row["target"])
        need(row["baseline"] == baseline | {"overlap": old}, "Remote control comparator differs")
        current = row["peaks"][winner]["overlap"]
        counter["top_winner_useful"] += current["useful"]
        for arm, active, hit in (("baseline", baseline["flag"], old["useful"]),
                                 ("candidate", row["candidate_flag"], current["useful"])):
            counter[arm+"_flags"] += active
            counter[arm+"_useful"] += active and hit
            if active and role not in POSITIVES:
                flags[arm].add((panel, slot))
        old_useful, useful = baseline["flag"] and old["useful"], row["candidate_flag"] and current["useful"]
        if old_useful and not useful:
            gains["lost_useful"].append(row["key"])
        if useful and not old_useful:
            if (panel, role) == ("primary", "feathered_edit"):
                gains["added_primary_feathered"].append(slot)
            if (panel, role) == ("screen", "edit"):
                gains["added_screen"].append(slot)
    photo_extra = len({group for group in flags["candidate"] - flags["baseline"] if group[0] != "screen"})
    ui_extra = len({group for group in flags["candidate"] - flags["baseline"] if group[0] == "screen"})
    counts = {arm: {name: sum(group[0] == name for group in selected) for name in ("primary", "legacy", "screen")}
              for arm, selected in flags.items()}
    passed = (len(gains["added_primary_feathered"]) >= 3 and len(gains["added_screen"]) >= 1
              and not gains["lost_useful"] and photo_extra <= 2 and ui_extra <= 1)
    return {"breakdown": {key: dict(values) for key, values in metrics.items()}, "control_groups": counts,
            "changes": gains, "extra_control_photo_groups": photo_extra, "extra_control_ui_groups": ui_extra,
            "geometry": {key: dict(values) for key, values in geometry.items()}, "passed": passed}


def sharp_diagnostic(rows):
    counts = {}
    for row in rows:
        kind, _, role = row["key"]
        measured = row["shipped_sharp"]
        need(type(measured["assessed"]) is bool and type(measured["flag"]) is bool and type(measured["useful"]) is bool
             and (not measured["flag"] or measured["assessed"])
             and (not measured["useful"] or (measured["flag"] and role in POSITIVES)),
             "Invalid existing sharp-boundary diagnostic")
        counts.setdefault(f"{kind}/{role}", Counter()).update(views=1, assessed=measured["assessed"],
                                                                 flags=measured["flag"], useful=measured["useful"])
    return {key: dict(values) for key, values in counts.items()}


def verify_public(record):
    need(record["schema"] == SCHEMA, "Unsupported study schema")
    design = record["design"]
    need(design["divisors"] == design["scale_order_for_ties"] == [6, 4, 3]
         and design["grid"] == 24 and design["confidence_weight"] == 0
         and design["useful"] == {"minimum_iou": .30, "minimum_recall": .50}
         and design["gate"] == {"new_primary_feathered_groups": 3, "new_screen_groups": 1,
                                "lost_useful_views": 0, "extra_ordinary_photo_groups": 2, "extra_ordinary_ui_groups": 1},
         "Study design changed")
    points = record["baseline_points"]
    need(points["relative"] == 8.313690900802614 and points["remote_upper_strict"] == -4.78482723236084,
         "Historical operating points changed")
    need(verify_fit(record["fit_rows"], points) == record["fit"], "Fitted point changed")
    need(verify_results(record["held_rows"], points, record["fit"]["point"]) == record["result"], "Held report changed")
    need(sharp_diagnostic(record["held_rows"]) == record["shipped_sharp_diagnostic"], "Existing sharp-boundary counts changed")
    actual = record["native_audit"]
    need(actual["views"] == 160 and actual["crop_inputs"] == 31200 and actual["positive_geometry_inputs"] == 50*195,
         "Native audit denominator changed")
    need(actual["cross_role_source_pairs"] == 236 and actual["cross_role_source_matches"] == 0,
         "Cross-role source review changed")
    need(actual["historical_decisions_matched"] == 28 and actual["historical_crop_values"] == 700
         and 0 <= actual["historical_max_abs_logit_delta"] <= 2e-5,
         "Historical one-third comparison changed")
    return record["result"]


def source_audit(args, run):
    import numpy as np
    from PIL import Image, ImageOps

    from scripts.image_source_groups_v22 import near, sketch

    def token(rgb):
        with Image.fromarray(rgb) as image:
            return {"rgb_sha256": digest(rgb.tobytes()), "sketch": f"{sketch(image):064x}", "size": image.size}

    fit = []
    for record in run.fit_roster(args):
        with Image.open(args.commons / record["file"]) as original, ImageOps.exif_transpose(original) as oriented:
            fit.append(token(np.asarray(oriented.convert("RGB"))))
    primary = [token(rgb) for _, _, rgb, _, _ in run.primary_sources(args)]
    legacy = [token(views["original"]) for _, views, _ in run.prior.REGIONAL.paired.inputs(args.legacy)]
    tested = sum(len(left)*len(right) for left, right in ((fit, primary), (fit, legacy), (primary, legacy)))
    matches = sum(a["rgb_sha256"] == b["rgb_sha256"] or near(a, b)
                  for left, right in ((fit, primary), (fit, legacy), (primary, legacy)) for a in left for b in right)
    return {"cross_role_source_pairs": tested, "cross_role_source_matches": matches}


def native(args, out):
    import numpy as np
    from PIL import Image

    from app.detectors.image import sharp_boundary
    from scripts import image_crop_multiscale_transfer as run

    lock_path = args.run / "freeze.json"
    lock = read(lock_path)
    result_path = args.run / "result.json"
    result = read(result_path)
    need(digest(lock_path.read_bytes()) == result["freeze_sha256"] and lock == run.freeze(args), "Frozen source or code changed")
    need(result["geometry_sha256"] == digest((args.run / "geometry.json").read_bytes()), "Pre-inference geometry changed")
    audit = Counter(views=0, crop_inputs=0, positive_geometry_inputs=0, reused_values=0)
    audit.update(source_audit(args, run))
    historical_path = args.baseline / "fresh-rows.json"
    historical_sha = digest(historical_path.read_bytes())
    need(read(args.baseline / "result.json")["journals"]["fresh"] == historical_sha, "Earlier crop journal changed")
    historical = {tuple(row["key"][1:3]): row for row in read(historical_path) if row["key"][-1] == "png"}
    audit["historical_max_abs_logit_delta"] = 0.
    fit_rows, held_rows = [], []
    for panel, fitting, output in (("fit", True, fit_rows), ("held", False, held_rows)):
        path = args.run / f"{panel}-rows.jsonl"
        expected = result["fit"]["journal_sha256"] if fitting else result["held_sha256"]
        need(expected == digest(path.read_bytes()), "Response journal changed")
        seen = {}
        rows = (json.loads(line) for line in path.read_text().splitlines())
        for case, row, frozen in zip(run.case_views(args, fitting), rows, lock["cases"][panel], strict=True):
            key, family, body, rgb, mask = case
            truth = int(np.count_nonzero(mask))
            receipt = {"key": list(key), "family": family, "input": digest(body), "rgb": digest(rgb.tobytes()),
                       "truth": digest(mask.tobytes()), "shape": list(mask.shape), "target": truth}
            need(frozen == receipt and all(row[name] == value for name, value in receipt.items()), "Native input differs")
            audit["views"] += 1
            with Image.fromarray(rgb) as image:
                baseline = list(run.prior.baseline.crops.crops(image))
                for scale in SCALES:
                    regions = rectangles(row["shape"], scale)
                    if scale == "3":
                        need(regions == [box for box, _ in baseline], "One-third baseline rectangles changed")
                    native_values = []
                    for i, ((x, y, right, bottom), score, committed) in enumerate(zip(
                            regions, row["logits"][scale], row["hashes"][scale], strict=True)):
                        patch = Image.fromarray(rgb[y:bottom, x:right]).resize((384, 384), Image.Resampling.BILINEAR)
                        prepared = digest(patch.tobytes())
                        need(prepared == committed, "Native prepared crop input differs")
                        if scale == "3":
                            need(prepared == digest(baseline[i][1].tobytes()), "One-third baseline model input changed")
                        audit["crop_inputs"] += 1
                        if committed in seen:
                            need(seen[committed] == score, "Reused crop response differs")
                            audit["reused_values"] += 1
                        seen[committed] = score
                        if truth:
                            overlap = int(np.count_nonzero(mask[y:bottom, x:right]))
                            native_values.append(scalar_overlap((x, y, right, bottom), overlap, truth))
                            audit["positive_geometry_inputs"] += 1
                    useful = sum(item["useful"] for item in native_values)
                    iou = max((item["overlap"]/(item["reported"]+truth-item["overlap"]) for item in native_values), default=0.)
                    need(row["geometry"][scale] == {"useful_proposals": useful, "best_iou": iou}, "Native geometry differs")
                    peak = row["peaks"][scale]
                    x, y, right, bottom = peak["box"]
                    overlap = int(np.count_nonzero(mask[y:bottom, x:right]))
                    need(peak["overlap"] == scalar_overlap(peak["box"], overlap, truth), "Selected native crop differs")
            if key[0] == "legacy":
                previous = historical[tuple(key[1:])]
                need(previous["input_sha256"] == row["input"] and previous["truth_sha256"] == row["truth"]
                     and previous["crop_sha256"] == row["hashes"]["3"], "Historical model input differs")
                delta = max(abs(old-new) for old, new in zip(previous["logits"], row["logits"]["3"], strict=True))
                audit["historical_max_abs_logit_delta"] = max(audit["historical_max_abs_logit_delta"], delta)
                need(previous["response"]["remote_control"] == row["baseline"]["flag"]
                     and previous["response"]["box"] == row["baseline"]["box"], "Historical candidate or decision changed")
                audit["historical_decisions_matched"] += 1
                audit["historical_crop_values"] += len(previous["logits"])
            fields = ("key", "family", "shape", "target", "logits") if fitting else (
                "key", "family", "shape", "target", "logits", "winner", "peaks", "baseline", "geometry", "candidate_flag")
            public_row = {name: row[name] for name in fields}
            if not fitting:
                h, w = mask.shape
                boundary = sharp_boundary.measure(rgb) if sharp_boundary.supported_size(w, h) else None
                flagged = boundary is not None and boundary.score > sharp_boundary.REVIEW_POINT
                useful = False
                if flagged:
                    x, y, right, bottom = boundary.box
                    overlap = int(np.count_nonzero(mask[y:bottom, x:right]))
                    useful = scalar_overlap(boundary.box, overlap, truth)["useful"]
                public_row["shipped_sharp"] = {"assessed": boundary is not None, "flag": flagged, "useful": useful}
            output.append(public_row)
        counters = result["fit"]["counters"] if fitting else result["held_counters"]
        need(len(seen) == counters["new_model_inputs"], "Scorer crop reuse count differs")
    public = {"schema": SCHEMA, "checkpoint": lock["checkpoint"], "design": lock["design"], "code": lock["code"],
              "source_commitments": lock["source_commitments"] | {"historical_crop_journal": historical_sha},
              "freeze_sha256": result["freeze_sha256"],
              "geometry_sha256": result["geometry_sha256"], "result_sha256": digest(result_path.read_bytes()),
              "baseline_points": lock["baseline_points"], "fit": {key: result["fit"][key] for key in ("point", "maxima")},
              "fit_rows": fit_rows, "held_rows": held_rows, "native_audit": dict(audit),
              "inference": {"fit": result["fit"]["counters"], "held": result["held_counters"]}, "result": result["result"],
              "shipped_sharp_sha256": digest(Path(sharp_boundary.__file__).read_bytes()),
              "shipped_sharp_diagnostic": sharp_diagnostic(held_rows)}
    verify_public(public)
    run.write(out, public)
    print(dict(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--native", nargs=8, type=Path,
                        metavar=("COMMONS", "PRIMARY", "BACKUP", "LEGACY", "PREVIOUS", "BASELINE", "WEIGHT", "RUN"))
    args = parser.parse_args()
    if args.native:
        keys = ("commons", "primary", "backup", "legacy", "previous", "baseline", "weight", "run")
        native(SimpleNamespace(**dict(zip(keys, args.native, strict=True))), args.record)
    else:
        print(json.dumps(verify_public(read(args.record)), sort_keys=True))


if __name__ == "__main__":
    main()
