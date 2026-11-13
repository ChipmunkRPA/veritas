# SPDX-License-Identifier: Apache-2.0
"""Measure paired native face edges within an incumbent appearance range."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
SOURCES = {
    "selection_edits": ("video-face-hd-sparse-retry-2026-11-12-v2", "held.jsonl"),
    "selection_controls": ("video-face-scale-context-holdout-2026-11-12-v2", "full.jsonl"),
    "held_edits_and_cuts": ("video-face-hd-retry-scene-challenge-2026-11-12", "held.jsonl"),
    "held_two_edits": ("video-face-two-returns-hd-2026-11-12", "held.jsonl"),
}
WEIGHTS = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_recognition_sface_2021dec.onnx": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
}
METHOD = {
    "profile": [1280, 720, 30, 120, "h264", "yuv420p", "mp4"],
    "selection": "Only the production coarse range routes a file. Score native boundaries within 12 frames of its endpoints.",
    "transition": "Compare three source-bound consecutive face embeddings on each side. Use cosine separation less "
                  "within-side dispersion. Require positive raw change of at least .08 at each edge.",
    "geometry": "Five through sixteen changed frames. The before and after medians must be more similar than either "
                "is to the proposed middle. Use observed unambiguous faces with adjacent box IoU at least .25.",
    "guard": "Measure one central ellipse and four adjacent head sectors at each edge, subtract ordinary nearby motion. "
             "Reject a broad edge where at least two available sectors change by at least 6 grayscale levels and half "
             "the central residual. Require two visible sectors and central residual of at least 3 at both edges.",
    "point": "Choose the stricter of 0.02 or the largest guarded candidate from ordinary selection files.",
    "measure": "At least .5 temporal IoU. Ordinary locations do not count as recovered synthetic edits.",
    "gate": {"min_added": 3, "min_groups": 2, "max_new_ordinary_files": 1, "max_new_cut_intervals": 1,
             "max_wrong_on_edit": 1, "min_assessed_routed_edit_share": .75},
}
SCHEMA = "chipmunk.video-face-paired-edge-transfer/1"
NORMALIZED_SIZE = (384, 216)
COARSE_MARGIN = 12


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_bytes())


def save(path, payload):
    with path.open("xb") as stream:
        stream.write(encode(payload))


def bound_paths():
    app = ROOT / "backend/app"
    files = ["detectors/video/decode.py", "detectors/video/face_appearance.py", "detectors/image/faces.py",
             "detectors/video/adapters/face_appearance.py"]
    paths = [app / name for name in files] + [Path(__file__)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def intervals(baseline, pts):
    return [[pts.index(value["start_pts"]), pts.index(value["end_pts"])] for key in
            ("coarse", "boundary_refinement", "short_range") if (value := baseline.get(key))]


def intake():
    rows, pins = [], {}
    for panel, (name, journal) in SOURCES.items():
        folder = DATA / name
        media, observed = folder / "media.json", folder / journal
        pins[panel] = {"manifest_sha256": sha(media), "journal_sha256": sha(observed), "freeze_sha256": sha(folder / "freeze.json")}
        ledger = read(media)
        require(ledger["freeze_sha256"] == pins[panel]["freeze_sha256"], "Prepared panel does not match the archived freeze")
        if journal == "full.jsonl":
            require(read(folder / "full.json")["journal_sha256"] == pins[panel]["journal_sha256"], "Current baseline changed")
        else:
            require(read(folder / "runtime.json")["held_sha256"] == pins[panel]["journal_sha256"], "Current baseline changed")
        measured = {row["file"]: row for row in map(json.loads, observed.read_text().splitlines())}
        require(len(measured) == len(ledger["rows"]), "Incomplete current panel")
        for media in ledger["rows"]:
            row = measured[media["file"]]
            require(row["sha256"] == media["sha256"], "Current result and input differ")
            baseline = row.get("baseline", row.get("before"))
            prior = intervals(baseline, media["clock"]["pts"])
            prior.extend(item["frames"] for item in row.get("incumbent_second", []))
            if second := row.get("second", {}).get("range"):
                prior.append(second["frames"])
            truth = [] if panel == "selection_controls" else row["truth"]
            rows.append({"id": len(rows), "panel": panel, "group": row["group"], "condition": row["condition"],
                         "sha256": row["sha256"], "truth": truth, "incumbent": prior,
                         "coarse": prior[0] if baseline["coarse"] else None})
    require(len(rows) == len({row["sha256"] for row in rows}) == 203, "Incomplete or repeated frozen delivery")
    return rows, pins


def freeze():
    import cv2

    rows, pins = intake()
    require({name: sha(ROOT / "models/weights" / name) for name in WEIGHTS} == WEIGHTS, "Model weight changed")
    return {"schema": SCHEMA, "method": METHOD, "inputs": pins, "weights": WEIGHTS, "code": bound_paths(),
            "opencv": cv2.__version__, "rows": rows}


def read_media(lock, role):
    for panel, (name, _) in SOURCES.items():
        if not panel.startswith(role):
            continue
        folder = DATA / name
        manifest = folder / "media.json"
        require(sha(manifest) == lock["inputs"][panel]["manifest_sha256"], "Source manifest changed")
        by_digest = {row["sha256"]: row for row in read(manifest)["rows"]}
        for row in lock["rows"]:
            if row["panel"] == panel:
                binding = by_digest[row["sha256"]]
                yield row, folder / "media" / binding["file"], binding


def frames(path, receipt, desired, detector, model):
    import cv2

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.adapters.face_appearance import _measure
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.detectors.video.face_appearance import unit
    from app.detectors.video.face_return import source_carrier

    table = FrameTable.from_json(receipt["clock"])
    require(sha(path) == receipt["sha256"] and source_carrier(path) == "mp4" and
            not (table.rotation or table.interpolated or table.truncated) and
            (table.width, table.height, table.coded_width, table.coded_height, len(table)) == (1280, 720, 1280, 720, 120)
            and all((pts - table.pts[0]) * table.tb == Fraction(i, 30) for i, pts in enumerate(table.pts)),
            "Source media or native geometry changed")
    values, reduced = {}, {}
    for frame in decode_frames(path, table, desired, timeout_s=45):
        i, rgb = frame.index, frame.rgb
        native = hashlib.sha256(rgb.tobytes()).hexdigest()
        require(frame.pts_verified and frame.pts == table.pts[i] and native == receipt["rgb_sha256"][i], "Native source picture changed")
        reduced[i] = cv2.cvtColor(cv2.resize(rgb, NORMALIZED_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = detect_faces(bgr, detector=detector, min_size=48)
        if len(faces) == 1 and (measured := _measure(rgb, bgr, i, table, faces[0], model)) is not None:
            vector, binding = measured
            values[i] = (unit(vector), faces[0], binding)
        del rgb, bgr
    require(set(reduced) == set(desired), "Incomplete native source scan")
    return table, values, reduced


def overlaps_half(a, b):
    return 2 * (min(a[1], b[1]) - max(a[0], b[0])) >= max(a[1], b[1]) - min(a[0], b[0])


def edge_pixels(index, reduced, face):
    import cv2
    import numpy as np

    from scripts.video_face_near_head_transfer import geometry

    box = tuple(value * 0.3 for value in face.box())
    region = geometry(reduced[index].shape, box)
    if region is None:
        return None
    mx, my, masks, sectors = region
    indices = [index + offset for offset in (-5, -4, -3, 0, 3, 4, 5) if index+offset-1 in reduced and index+offset in reduced]
    if len(indices) < 5:
        return None
    reprojected = {i: cv2.remap(reduced[i], mx, my, cv2.INTER_LINEAR).astype("float32")
                   for j in indices for i in (j-1, j)}
    changes = {i: [float(np.abs(reprojected[i]-reprojected[i-1])[mask].mean()) for mask in masks] for i in indices}
    baseline = np.median([changes[i] for i in indices if i != index], axis=0)
    residual = np.maximum(0., np.asarray(changes[index])-baseline)
    central = float(residual[0])
    return {"central": central, "sectors": [float(v) for v in residual[1:]], "sector_ids": sectors[1:],
            "supported": central >= 3., "broad": sum(float(x) >= max(6., .5*central) for x in residual[1:]) >= 2}


def candidates(coarse, values, reduced, pts):
    import numpy as np

    from app.detectors.image.faces import iou
    from app.detectors.video.face_appearance import distance

    first, last = coarse
    entries = range(max(5, first-COARSE_MARGIN), min(115, first+COARSE_MARGIN)+1)
    exits = range(max(5, last-COARSE_MARGIN), min(115, last+COARSE_MARGIN)+1)
    edges = {}
    for i in sorted(set(entries) | set(exits)):
        left, right = range(i-3, i), range(i, i+3)
        seq = list(left)+list(right)
        if not all(j in values for j in seq) or any(iou(values[a][1], values[b][1]) < .25 for a,b in zip(seq,seq[1:])):
            continue
        a, b = [np.median([values[j][0] for j in side], axis=0) for side in (left, right)]
        separation = distance(a, b)
        if separation < .08:
            continue
        floor = max(distance(values[j][0], a if j<i else b) for j in seq)
        geometry = edge_pixels(i, reduced, values[i-1][1])
        if geometry is not None and geometry["supported"]:
            edges[i] = (a, b, float(separation-floor), geometry)
    observed, guarded = [], []
    for start in entries:
        if start not in edges:
            continue
        before, first_middle, left_score, left_geometry = edges[start]
        for stop in exits:
            if stop not in edges or not 5 <= stop-start <= 16:
                continue
            shared = min(stop,last)-max(start,first)
            if 2*shared < min(stop-start,last-first):
                continue
            last_middle, after, right_score, right_geometry = edges[stop]
            between = distance(first_middle,last_middle)
            return_gap = min(distance(before,first_middle),distance(after,last_middle)) - max(distance(before,after),between)
            score = min(left_score, right_score, float(return_gap))
            during = min((i for i in range(start,stop) if i in values), key=lambda i:(abs(2*i-start-stop),i))
            anchors = [start-1,during,stop]
            result = {"frames":[start,stop],"score":score,"edges":[left_geometry,right_geometry],
                      "pts":[pts[start],pts[stop]],"anchors":[{k:values[i][2][k] for k in
                      ("frame_index","pts","decoded_rgb_sha256","bbox","aligned_rgb_sha256")} for i in anchors]}
            observed.append(result)
            if not (left_geometry["broad"] or right_geometry["broad"]):
                guarded.append(result)
    def rank(item):
        return item["score"], -item["frames"][0], -item["frames"][1]

    return max(observed,key=rank,default=None), max(guarded,key=rank,default=None)


def measure(path, receipt, row, detector, model):
    require(sha(path) == row["sha256"], "Prepared media changed")
    if not row["coarse"]:
        return {"id":row["id"],"status":"not_routed","native_reads":0,"raw":None,"guarded":None}
    a,b = row["coarse"]
    desired = list(range(max(0, a-COARSE_MARGIN-6), min(120,b+COARSE_MARGIN+6)))
    table, values, reduced = frames(path,receipt,desired,detector,model)
    raw, guarded = candidates(row["coarse"], values, reduced, table.pts)
    return {"id":row["id"],"status":"assessed" if raw is not None else "no_supported_pair","native_reads":len(desired),
            "raw":raw,"guarded":guarded}


def stage(run, lock, role):
    import cv2

    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(2)
    detector = load_yunet(ROOT/"models/weights/face_detection_yunet_2023mar.onnx")
    model = cv2.FaceRecognizerSF.create(str(ROOT/"models/weights/face_recognition_sface_2021dec.onnx"), "")
    result = []
    started = time.monotonic()
    for row,path,receipt in read_media(lock,role):
        item = measure(path,receipt,row,detector,model)
        result.append(item)
        if row["coarse"]:
            print(role,row["condition"],row["group"],item["status"],
                  item["guarded"]["frames"] if item["guarded"] else None,flush=True)
    save(run/f"{role}.json",{"observations":result,"elapsed_s":time.monotonic()-started,
                             "freeze_sha256":sha(run/"freeze.json")})


def fit(lock, selection):
    rows = {r["id"]:r for r in lock["rows"] if r["panel"].startswith("selection")}
    require(set(rows)=={o["id"] for o in selection},"Missing fixed selection observations")
    controls = [o["guarded"]["score"] for o in selection if not rows[o["id"]]["truth"] and o["guarded"]]
    return {"point":max([.02,*controls]),"guarded_ordinary_candidates":len(controls)}


def recount(lock, observations, point):
    rows = {r["id"]:r for r in lock["rows"] if r["panel"].startswith("held")}
    require(set(rows)=={o["id"] for o in observations},"Missing fixed held observations")
    counts, gained = {},set()
    totals = Counter()
    for item in observations:
        row = rows[item["id"]]
        prior, truth = row["incumbent"],row["truth"]
        count = counts.setdefault(row["panel"]+"/"+row["condition"],Counter())
        count.update(files=1,truth=len(truth),routed=bool(row["coarse"]),assessed=item["status"]=="assessed")
        before = {i for i,x in enumerate(truth) if any(overlaps_half(x,y) for y in prior)}
        cut = row["condition"] in {"person_switch","tandem_person_switch","cross_recording_cuts"}
        value = item["guarded"]
        raw = item["raw"]
        span = value["frames"] if value and value["score"] > point else None
        count.update(ordinary_files_already_flagged=bool(not truth and prior),
                     raw_above_point=bool(raw and raw["score"]>point),guarded_above_point=bool(span),
                     before_useful=len(before))
        if span and any(overlaps_half(span,x) for x in prior[1:]) and any(overlaps_half(span, x) for x in truth if x not in [truth[i] for i in before]):
            count["overlaps_existing_refinement"] += 1
        after = before | {i for i,x in enumerate(truth) if span and overlaps_half(span,x)}
        count.update(combined_useful=len(after),additional_useful=len(after-before))
        totals.update(added=len(after-before),edit_files=bool(truth),edit_files_routed=bool(truth and row["coarse"]),
                      edit_files_assessed=bool(truth and item["status"]=="assessed"))
        if after-before:
            gained.add((row["panel"],row["group"]))
        if span and not truth:
            count.update(new_ordinary_file=not prior, extra_cut_interval=cut)
            totals.update(new_ordinary_files=not prior,new_cut_intervals=cut)
        if span and truth and not any(overlaps_half(span,x) for x in truth):
            count["wrong_edit_range"] += 1
            totals["wrong_on_edit"] += 1
    limits=METHOD["gate"]
    gate={"added":totals["added"]>=limits["min_added"],"groups":len(gained)>=limits["min_groups"],
          "ordinary_files":totals["new_ordinary_files"]<=limits["max_new_ordinary_files"],
          "ordinary_cut_intervals":totals["new_cut_intervals"]<=limits["max_new_cut_intervals"],
          "wrong_on_edit":totals["wrong_on_edit"]<=limits["max_wrong_on_edit"],
          "assessment":totals["edit_files_assessed"]>=limits["min_assessed_routed_edit_share"]*totals["edit_files_routed"]}
    return {"cells":counts,"totals":totals,"gained_groups":len(gained),"gate":gate,"passed":all(gate.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","selection","held","report"))
    parser.add_argument("run",type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.run / "mpl"))
    run = args.run
    if args.phase=="freeze":
        run.mkdir(mode=0o700)
        save(run/"freeze.json",freeze())
        return
    lock=read(run/"freeze.json")
    require(lock==freeze(),"Frozen study changed")
    if args.phase=="selection":
        stage(run,lock,"selection")
        save(run/"point.json",fit(lock,read(run/"selection.json")["observations"])|{"selection_sha256":sha(run/"selection.json")})
    elif args.phase=="held":
        point = read(run/"point.json")
        require(point==fit(lock,read(run/"selection.json")["observations"])|{"selection_sha256":sha(run/"selection.json")},
                "Fixed selection point differs")
        stage(run,lock,"held")
    else:
        selected, held, point = (read(run/name) for name in ("selection.json","held.json","point.json"))
        require(selected["freeze_sha256"]==held["freeze_sha256"]==sha(run/"freeze.json")
                and point["selection_sha256"]==sha(run/"selection.json"),"Measurement freeze changed")
        save(run/"receipt.json",{"freeze":lock,"selection":selected,"held":held,"point":point,
                                 "result":recount(lock,held["observations"],point["point"])})


if __name__=="__main__":
    main()
