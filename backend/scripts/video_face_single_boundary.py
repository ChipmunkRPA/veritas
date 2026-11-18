# SPDX-License-Identifier: Apache-2.0
"""Measure standalone face changes that reach the start or end of the clip."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_common_sources_v89 as io
from scripts import video_lip_correspondence as receipt

SCHEMA = "chipmunk.video-face-single-boundary/1"
SPLITS = {"development": 6, "held": 12}
SAMPLE = tuple(range(0, 100, 2))
BOUNDARIES = (32, 39, 48, 57, 64, 51)
FAMILIES = ("inswapper", "uniface")
ENCODES = ("direct", "tandem")
CONTROLS = ("original", "exposure_step", "focus_step", "same_source_jump", "different_shot")
ARMS = ("persistent", "edge_supported")
RULE = {"minimum_additions": 12, "minimum_groups": 6, "minimum_per_family": 4,
        "maximum_ordinary_groups": 2, "maximum_hard_cut_flags": 0, "maximum_misplaced": 2}
CONTEXT = {"flank": 6, "minimum_presence": .76, "minimum_iou": .35, "minimum_scene": .60,
           "maximum_boundary_error": 1, "point_floor": .06, "development_ordinary_group_budget": 1}
METHOD = {
    "prepare": "Use publisher-aligned original and swapped tracks. Preserve the authored boundary through each 25 fps encode.",
    "persistent": "Require a sustained appearance step relative to within-side variation, with a consistent scene and one dominant face.",
    "edge_supported": "Require the sustained step and an abrupt local adjacent-face jump relative to its near neighbors.",
    "scope": "A localized appearance transition can have ordinary causes. Which side changed stays unresolved.",
}


def code_hashes():
    from app.detectors.video import face_local_detail, face_return
    from scripts import video_face_local_panel, video_face_temporal_delivery

    return {Path(module.__file__).name: io.sha(Path(module.__file__)) for module in (
        face_local_detail, face_return, video_face_local_panel, video_face_temporal_delivery,
    )} | {Path(__file__).name: io.sha(Path(__file__))}


def read(path):
    return receipt.read(path) if path.suffix == ".gz" else json.loads(path.read_bytes())


def raw_identities(row):
    return {row["group"], io.source_key(row["source_video_url"]), *row["donors"],
            *(file["sha256"] for file in row["files"].values())}


def reserve(development_roster, held_roster, output):
    from scripts.video_face_local_measure import assets

    bindings = {}
    selected, blocked = {}, set()
    for split, path in (("development", development_roster), ("held", held_roster)):
        value = read(path)
        if value["revision"] != io.RTFS_REV:
            raise ValueError("Only the reviewed publisher release can enter this study")
        bindings[split] = io.sha(path)
        selected[split] = []
        order = sorted(value["groups"], key=lambda row: hashlib.sha256(f"{SCHEMA}/{split}/{row['group']}".encode()).hexdigest())
        for row in order:
            ids = raw_identities(row)
            # Keep both target and donor components separate across the two splits.
            if ids & blocked or set(row["files"]) != {"ordinary", *FAMILIES}:
                continue
            selected[split].append(row)
            blocked.update(ids)
            if len(selected[split]) == SPLITS[split]:
                break
        if len(selected[split]) != SPLITS[split]:
            raise ValueError(f"Insufficient disjoint publisher sources in {split}")
    receipt.save(output, {"schema": SCHEMA, "source_rosters": bindings, "code": code_hashes(), "weights": assets(),
                          "groups": selected, "rules": RULE, "context": CONTEXT, "method": METHOD,
                          "boundaries": list(BOUNDARIES), "encodes": list(ENCODES)})
    print("Frozen source split", {phase: len(rows) for phase, rows in selected.items()}, flush=True)


def supported(samples, onset):
    from app.detectors.image.faces import iou
    from app.detectors.video.face_return import scene_lower_quartile

    a, b = samples[onset - 2], samples[onset]
    if a.face is None or b.face is None or iou(a.face, b.face) < CONTEXT["minimum_iou"]:
        return None
    result = scene_lower_quartile(a, b)
    return result if result is not None and result >= CONTEXT["minimum_scene"] else None


def proposals(samples, cuts):
    import numpy as np

    flank = CONTEXT["flank"]
    present = [index for index in SAMPLE if samples[index].vector is not None]
    if len(present) < 2 * flank or cuts:
        return []
    vectors = np.stack([samples[index].vector for index in present])
    distances = 1 - np.clip(vectors @ vectors.T, -1, 1)
    positions = {index: pos for pos, index in enumerate(present)}

    def within(indices):
        values = distances[np.ix_(indices, indices)]
        return np.quantile(values[np.triu_indices(len(indices), k=1)], .75)

    def separation(before, after):
        return float(np.quantile(distances[np.ix_(before, after)], .25) - max(within(before), within(after)))

    result = []
    for edge in range(22, 79, 2):
        near_before = list(range(edge - 2 * flank, edge, 2))
        near_after = list(range(edge, edge + 2 * flank, 2))
        if not set(near_before + near_after) <= positions.keys() or (scene := supported(samples, edge)) is None:
            continue
        before = [positions[index] for index in present if index < edge]
        after = [positions[index] for index in present if index >= edge]
        if min(len(before) / (edge // 2), len(after) / ((100 - edge) // 2)) < CONTEXT["minimum_presence"]:
            continue
        left, right = ([positions[index] for index in group] for group in (near_before, near_after))
        persistent = min(separation(before, after), separation(left, right))
        neighbors = np.r_[distances[left[:-1], left[1:]], distances[right[:-1], right[1:]]]
        bump = float(distances[left[-1], right[0]] - np.quantile(neighbors, .75))
        result.append({"coarse": edge, "scene": round(scene, 9), "persistent": round(persistent, 9),
                       "edge_supported": round(min(persistent, bump), 9), "jump_excess": round(bump, 9)})
    return result


def shot_observer(table):
    from app.detectors.video.shots import ShotDetector

    return ShotDetector(table, min_shot_s=0.)


def locate(path, table, samples, reading, locator, encoder):
    import numpy as np

    from app.detectors.image.faces import iou
    from app.detectors.video import face_local_detail, face_return
    from app.detectors.video.decode import decode_frames

    coarse = reading["coarse"]
    centers = []
    for region in (range(coarse - 12, coarse - 3, 2), range(coarse + 2, coarse + 12, 2)):
        center = np.mean([samples[index].vector for index in region], axis=0)
        centers.append(center / max(np.linalg.norm(center), 1e-6))
    wanted = sorted({coarse - 3, coarse - 2, coarse - 1, coarse, coarse + 1, coarse + 2, coarse + 11})
    fine, anchors = {}, {}
    with closing(decode_frames(path, table, wanted, timeout_s=15)) as frames:
        for frame in frames:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("The precise face boundary lacks native frame timing")
            sample, _, pin, _ = face_local_detail.observe(frame.rgb, locator, encoder, with_texture=True)
            fine[frame.index] = sample
            if pin:
                anchors[frame.index] = pin | {"index": frame.index, "pts": frame.pts}
    possibilities = []
    for edge in range(coarse - 2, coarse + 3):
        left, right = fine.get(edge - 1), fine.get(edge)
        if any(sample is None or sample.vector is None for sample in (left, right)):
            continue
        if iou(left.face, right.face) < CONTEXT["minimum_iou"]:
            continue
        scene = face_return.scene_lower_quartile(left, right)
        if scene is None or scene < CONTEXT["minimum_scene"]:
            continue
        old, new = centers
        prior = float(left.vector @ old - left.vector @ new)
        following = float(right.vector @ new - right.vector @ old)
        score = min(prior, following)
        if score > 0:
            possibilities.append((score, -abs(edge - coarse), -edge, scene))
    if not possibilities:
        return None
    margin, _, neg_edge, scene = max(possibilities)
    edge = -neg_edge
    roles = {"before": edge - 1, "transition": edge, "after": coarse + 11}
    if not set(roles.values()) <= anchors.keys():
        return None
    return reading | {"boundary": edge, "pts": table.pts[edge], "time_base": table.time_base,
                      "fine_margin": round(margin, 9), "fine_scene": round(scene, 9),
                      "source_frames": {role: anchors[index] for role, index in roles.items()}}


def observe(path, locator, encoder):
    from app.detectors.video import face_local_detail, face_return
    from app.detectors.video.decode import decode_frames
    from app.evidence import face_local_return_binding as local
    from scripts.video_face_temporal_readout import table_for

    table = table_for(path)
    if not face_local_detail.supported(table, ("h264", "yuv420p")):
        raise ValueError("The delivered clip left the frozen H.264 profile")
    times, sparse = face_return.schedule(table)
    schedule = set(SAMPLE) | set(sparse)
    samples, details, multi = {}, {}, set()
    observer = shot_observer(table)
    with closing(decode_frames(path, table, None, timeout_s=22)) as frames:
        for index, frame in enumerate(frames):
            if frame.index != index or not frame.pts_verified or frame.pts != table.pts[index]:
                raise ValueError("An encoded picture lacks its native timestamp")
            observer.update(index, frame.rgb)
            if index in schedule:
                sample, detail, _, many = face_local_detail.observe(frame.rgb, locator, encoder, with_texture=index in SAMPLE)
                samples[index] = sample
                if detail is not None:
                    details[index] = detail
                if many:
                    multi.add(index)
    if observer.frames_seen != len(table) or set(samples) != schedule:
        raise ValueError("Native decode omitted a planned comparison frame")
    cuts = [row["frame_index"] for row in observer.boundaries]
    base = face_return.highest(times, sparse, samples) if not multi.intersection(sparse) else None
    baseline = {"brief_return": [base.start, base.end] if base and face_return.distinct(base, table, None) else None,
                "local_return": None}
    possible = face_local_detail.proposals(samples, details)
    if possible:
        score, best = max(((local.response([round(v, 9) for v in item.features[:2]]), item) for item in possible),
                          key=lambda pair: (pair[0], pair[1].end - pair[1].start, -pair[1].start))
        support = best.support
        if score > local.POINT and not any(support[0][0] < cut <= support[-1][-1] for cut in cuts):
            baseline["local_return"] = [best.start, best.end]
    pool = proposals(samples, cuts)
    selected = {}
    native = {}
    for arm in ARMS:
        top = max(pool, key=lambda item: (item[arm], item["jump_excess"], -item["coarse"]), default=None)
        if top is not None and top["coarse"] not in native:
            native[top["coarse"]] = locate(path, table, samples, top, locator, encoder)
        selected[arm] = native.get(top["coarse"]) if top else None
    return {"baseline": baseline, "candidates": selected, "proposals": len(pool), "cuts": cuts,
            "pictures": len(table), "face_samples": sum(samples[i].vector is not None for i in SAMPLE)}


def variants(original, alternate, swaps, pos, locator):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from scripts.video_face_local_panel import softened_face

    yield "original", None, None, original
    step = BOUNDARIES[pos % len(BOUNDARIES)]
    yield "exposure_step", step, None, [*original[:step], *(np.rint(np.clip(frame * 1.17 + 5, 0, 255)).astype(np.uint8)
                                                           for frame in original[step:])]
    edited = []
    for image in original[step:]:
        faces = detect_faces(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), detector=locator, min_size=48)
        edited.append(softened_face(image, faces[0]) if len(faces) == 1 else image)
    yield "focus_step", step, None, [*original[:step], *edited]
    yield "same_source_jump", step, None, [*original[:step], *original[step + 5:], *([original[-1]] * 5)]
    yield "different_shot", step, None, [*original[:step], *alternate[step:]]
    for j, family in enumerate(FAMILIES):
        if family not in swaps:
            continue
        edge = BOUNDARIES[(pos + j * 2) % len(BOUNDARIES)]
        swap = swaps[family]
        yield family + "_entry", edge, "suffix", [*original[:edge], *swap[edge:]]
        yield family + "_exit", edge, "prefix", [*swap[:edge], *original[edge:]]


def measure(frozen, sources, split, output, lock=None):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_face_appearance
    from scripts import video_face_local_panel as build
    from scripts import video_face_temporal_delivery as encode
    from scripts.video_face_local_measure import ENCODER, LOCATOR, assets

    cv2.setNumThreads(1)
    plan = read(frozen)
    if plan["code"] != code_hashes() or plan["weights"] != assets():
        raise ValueError("The frozen code or commercially permitted assets changed")
    if split == "held" and (lock is None or read(lock)["frozen"] != io.sha(frozen)):
        raise ValueError("The development operating point must be locked before scoring held media")
    locator = load_yunet(LOCATOR, "cpu", "float32").obj
    encoder = load_face_appearance(ENCODER, "cpu", "float32").obj
    groups = plan["groups"][split]
    header = {"schema": SCHEMA, "frozen": io.sha(frozen), "split": split, "locked": io.sha(lock) if lock else None}
    if output.exists():
        with output.open() as prior:
            if json.loads(next(prior)) != header:
                raise ValueError("The journal belongs to a different frozen comparison")
            done = {(row["group"], row["role"], row["encoding"]) for line in prior if (row := json.loads(line))}
    else:
        done = set()
    with TemporaryDirectory(prefix="chipmunk-one-face-", dir=output.parent) as stage, output.open("a") as journal:
        if not done and not output.stat().st_size:
            journal.write(json.dumps(header, sort_keys=True) + "\n")
        scratch = Path(stage)
        for pos, group in enumerate(groups):
            raw, mapping, identity, source_start = build.source_pictures(group, sources, "ordinary")
            opposite = groups[(pos + 1) % len(groups)]
            alternate, _, counter, _ = build.source_pictures(opposite, sources, "ordinary")
            swaps, commitments = {}, {}
            for family in FAMILIES:
                pictures, clock, digest, first = build.source_pictures(group, sources, family)
                if build.compatible_clocks(mapping, clock, source_start, first):
                    swaps[family] = pictures
                    commitments[family] = (digest, clock)
            for role, edge, side, pictures in variants(raw, alternate, swaps, pos, locator):
                encodings = ENCODES
                want = [mode for mode in encodings if (group["group"], role, mode) not in done]
                if not want:
                    continue
                direct, tandem = scratch / "native.mp4", scratch / "recompressed.mp4"
                encode.encode(pictures, direct)
                if "tandem" in want:
                    encode.tandem(direct, tandem)
                family = role.split("_")[0] if side else None
                derived, matched = commitments.get(family, (counter if role == "different_shot" else identity, mapping))
                reference_mapping = None
                if side:
                    if not all(pictures[i] is raw[i] for i in (range(edge) if side == "suffix" else range(edge, 100))):
                        raise ValueError("One-sided edit changed a reference picture")
                    reference_mapping = [matched[edge - 1], mapping[edge]] if side == "prefix" else [mapping[edge - 1], matched[edge]]
                for mode, path in (("direct", direct), ("tandem", tandem)):
                    if mode not in want:
                        continue
                    row = {"group": group["group"], "role": role, "family": family, "encoding": mode,
                           "authored_edge": edge, "edited_side": side, "publisher_sha256": identity,
                           "treated_sha256": derived, "edit_source_clocks": reference_mapping, "input_sha256": io.sha(path)}
                    row.update(observe(path, locator, encoder))
                    journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                direct.unlink()
                tandem.unlink(missing_ok=True)
            print(f"{split} native observations {pos + 1}/{len(groups)} source groups", flush=True)


def journal(path):
    with path.open() as source:
        header = json.loads(next(source))
        return header, [json.loads(line) for line in source]


def selected(row, arm, point):
    reading = row["candidates"][arm]
    return reading if reading is not None and reading[arm] > point else None


def count(rows, arm, point):
    total = Counter()
    edit_groups, ordinary_groups = set(), set()
    by_family = Counter()
    by_encoding = defaultdict(Counter)
    for row in rows:
        observation = selected(row, arm, point)
        truth = row["authored_edge"]
        old = any(value and any(abs(k - truth) <= CONTEXT["maximum_boundary_error"] for k in value)
                  for value in row["baseline"].values()) if row["edited_side"] else False
        if row["edited_side"]:
            useful = observation is not None and abs(observation["boundary"] - truth) <= CONTEXT["maximum_boundary_error"]
            counts = {"edits": 1, "baseline_boundary_matches": old, "candidate_boundary_matches": useful,
                      "additional_boundary_matches": useful and not old,
                      "exact_boundary_matches": bool(observation and observation["boundary"] == truth),
                      "misplaced": bool(observation and not useful), "unassessed": row["candidates"][arm] is None}
            if useful and not old:
                by_family[row["family"]] += 1
                edit_groups.add(row["group"])
        else:
            if observation:
                ordinary_groups.add(row["group"])
            counts = {"ordinary": 1, "ordinary_alerts": observation is not None,
                      "hard_cut_alerts": row["role"] == "different_shot" and observation is not None}
        total.update(counts)
        by_encoding[row["encoding"]].update(counts)
    return dict(total) | {"added_groups": len(edit_groups), "ordinary_alert_groups": len(ordinary_groups),
                          "by_family": dict(sorted(by_family.items())),
                          "by_encoding": {key: dict(counts) for key, counts in sorted(by_encoding.items())}}


def calibrate(frozen, development, output):
    header, rows = journal(development)
    if header != {"schema": SCHEMA, "split": "development", "frozen": io.sha(frozen), "locked": None}:
        raise ValueError("The development measurements do not come from this source freeze")
    options = {}
    for arm in ARMS:
        controls = defaultdict(list)
        for row in rows:
            if not row["edited_side"] and (reading := row["candidates"][arm]) is not None:
                controls[row["group"]].append(reading[arm])
        maxima = sorted((max(values) for values in controls.values()), reverse=True)
        budget = CONTEXT["development_ordinary_group_budget"]
        point = max(CONTEXT["point_floor"], maxima[budget] if len(maxima) > budget else 0.)
        options[arm] = {"point": point, "counts": count(rows, arm, point)}
    name = max(ARMS, key=lambda arm: (options[arm]["counts"].get("additional_boundary_matches", 0),
                                     -options[arm]["counts"].get("misplaced", 0),
                                     -options[arm]["counts"].get("ordinary_alert_groups", 0)))
    receipt.save(output, {"schema": SCHEMA, "frozen": io.sha(frozen), "development": io.sha(development),
                          "selected": name, "options": options})
    print(json.dumps({"selected": name, **options[name]}, sort_keys=True), flush=True)


def report(frozen, lock, held, output):
    header, rows = journal(held)
    thresholds = read(lock)
    if header != {"schema": SCHEMA, "split": "held", "frozen": io.sha(frozen), "locked": io.sha(lock)}:
        raise ValueError("The held observations do not come from this locked comparison")
    name = thresholds["selected"]
    point = thresholds["options"][name]["point"]
    totals = count(rows, name, point)
    advances = (totals["additional_boundary_matches"] >= RULE["minimum_additions"]
                and totals["added_groups"] >= RULE["minimum_groups"]
                and all(totals["by_family"].get(family, 0) >= RULE["minimum_per_family"] for family in FAMILIES)
                and totals["misplaced"] <= RULE["maximum_misplaced"]
                and totals["ordinary_alert_groups"] <= RULE["maximum_ordinary_groups"]
                and totals["hard_cut_alerts"] <= RULE["maximum_hard_cut_flags"])
    receipt.save(output, {"schema": SCHEMA, "frozen": io.sha(frozen), "held": io.sha(held), "rows": rows,
                          "thresholds": thresholds, "counts": totals, "advances": advances})
    print(json.dumps({"advances": advances, "counts": totals}, indent=2), flush=True)


def review(path):
    recorded = read(path)
    threshold = recorded["thresholds"]
    name = threshold["selected"]
    point = threshold["options"][name]["point"]
    actual = count(recorded["rows"], name, point)
    if actual != recorded["counts"]:
        raise ValueError("The published measurements do not reproduce")
    print(json.dumps({"mode": name, "point": point, "outcomes": actual, "advance": recorded["advances"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"reserve": (reserve, ("development_roster", "held_roster", "output")),
                "measure": (measure, ("frozen", "sources", "output")),
                "calibrate": (calibrate, ("frozen", "development", "output")),
                "report": (report, ("frozen", "lock", "held", "output")),
                "review": (review, ("path",))}
    for name, (_, positional) in commands.items():
        cli = sub.add_parser(name)
        for arg in positional:
            cli.add_argument(arg, type=Path)
        if name == "measure":
            cli.add_argument("--split", choices=SPLITS, required=True)
            cli.add_argument("--lock", type=Path)
    args = vars(parser.parse_args())
    commands[args.pop("command")][0](**args)
