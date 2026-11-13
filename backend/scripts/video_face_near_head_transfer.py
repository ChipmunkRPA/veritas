# SPDX-License-Identifier: Apache-2.0
"""Transfer a fixed near-head change check to source-timed video cuts and swaps."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from scripts import video_face_subject_context as development
from scripts import video_mouth_jaw_transfer as crema
from scripts.video_common_sources_v89 import read, require, sha, write

hd = development.source.hd
RTFS = development.source.retry.PREVIOUS.parent
ROLES = ("unchanged", "cutaway", "tandem_cutaway", "jump", "punch_in", "gain")
SPEC = {
    "schema": "chipmunk.video-face-near-head-transfer/1",
    "delivery": [1280, 720, 30, 120, "h264", "yuv420p"],
    "ordinary_sources": "Ten distinct qualified CREMA-D originals in archive order, paired into five fixed-layout cutaways.",
    "swap_sources": "Eight previously evaluated RTFS source groups from the 720p two-return study, separated from this assay's "
                    "GRID/RTFS development. Both swap windows on both primary and tandem deliveries.",
    "truth": "Source-timed engineering assay only. All boundaries are supplied. This does not measure report coverage.",
    "crema_placements": [[56, 67], [61, 72]],
    "crema_tile_xywh": [360, 150, 560, 420],
    "face_ellipse": [.43, .47],
    "head_ellipse": [.77, .76],
    "sector_support": 2,
    "face_excess_minimum": 4.0,
    "sector_excess_minimum": 8.0,
    "sector_to_face_minimum": 0.55,
    "minimum_sector_visible_fraction": .8,
    "motion_reference": "Median of adjacent differences at offsets [-10,-9,-8,-7,7,8,9,10] around each supplied edge.",
    "gate": {"cutaway_groups": 4, "swap_annotation_max": 2, "ordinary_processing_max": 1, "assessment_minimum": .75},
}


def plan(dev_run):
    from app.detectors.video import decode
    from scripts import video_face_six_picture_transfer as archived

    result = read(dev_run/"aggregate.json")
    base = RTFS.parent/"video-face-hd-retry-scene-challenge-2026-11-12"
    developed, transferred = read(base/"freeze.json"), read(RTFS/"freeze.json")
    require(not result["advances"] and result["runtime"]["freeze_sha256"] == sha(dev_run/"freeze.json")
            and read(dev_run/"freeze.json")["source_manifest_sha256"] == sha(base/"media.json"),
            "Exact completed subject-context development is required")
    old_manifest, old_run = read(RTFS/"media.json"), read(RTFS/"runtime.json")
    require(old_manifest["freeze_sha256"] == sha(RTFS/"freeze.json")
            and old_run["held_sha256"] == sha(RTFS/"held.jsonl"), "RTFS transfer bank differs")
    metadata = archived.trial.prior.comparison.metadata()
    documents = {doc["index"]: doc for path in archived.common.DATA.glob("group-*-complete.json") if (doc := read(path))}
    prior_ids = {row["pin"]["index"] for row in developed["swaps"]}
    transfer_ids = {row["index"] for row in transferred["sources"]}
    require(len(prior_ids) == 7 and len(transfer_ids) == 8 and not prior_ids & transfer_ids, "RTFS group IDs overlap")
    prior_meta, transfer_meta = ([metadata[documents[index]["group"]] for index in identifiers]
                                 for identifiers in (prior_ids, transfer_ids))
    require(not {row["url"] for row in prior_meta} & {row["url"] for row in transfer_meta}
            and not set().union(*(row["donors"] for row in prior_meta)) & set().union(*(row["donors"] for row in transfer_meta)),
            "Known RTFS host URLs or donors overlap")
    observed = [row for row in old_manifest["rows"] if row["condition"] in ("two_swaps", "tandem_two_swaps")]
    require(len(observed) == 16 and all(len(row["truth"]) == 2 for row in observed), "Need all 32 source-timed swap intervals")
    crease = [group[0] for group in crema.native_groups(crema.ARCHIVE).values()]
    require(len({row["source_group"] for row in crease}) == len(crease) == 10, "Need ten different retained genuine actors")
    return {"protocol": SPEC, "development_sha256": sha(dev_run/"aggregate.json"),
            "rtfs_development_groups": len(prior_ids), "rtfs_transfer_groups": len(transfer_ids),
            "rtfs_manifest_sha256": sha(RTFS/"media.json"), "crema_originals": crease,
            "rtfs_file_sha256": [row["sha256"] for row in observed],
            "bindings": {Path(module.__file__).name: sha(module.__file__) for module in (development, hd, crema, decode)}
            | {"runner": sha(__file__), "face_asset": sha(development.ROOT/"models/weights/face_detection_yunet_2023mar.onnx")},
            "opencv": cv2.__version__}


def genuine_frames(row):
    from app.detectors.video.decode import decode_frames, probe_frames

    path = crema.ARCHIVE/row["file"]
    require(sha(path) == row["file_sha256"], "Genuine input file differs")
    table = probe_frames(path, timeout_s=15)
    proof = row["expected"]
    result = []
    for frame in decode_frames(path, table, range(len(table)), timeout_s=20):
        require(frame.index == len(result) and frame.pts_verified
                and frame.pts*table.tb == Fraction(proof["picture_pts_ms"][frame.index],1000)
                and development.source.review.digest(frame.rgb) == proof["rgb_sha256"][frame.index], "Genuine input picture differs")
        result.append(frame.rgb)
    require(len(result) == len(table) >= 55 and len(result) == len(proof["rgb_sha256"]), "Genuine input is incomplete")
    return result


def compose(first, second, span, role):
    def sample(frames, n):
        extent = len(frames)-1
        k = (extent//2+n-64) % (2*extent)
        return frames[min(k, 2*extent-k)]

    poster = development.source.poster()
    x, y, w, h = SPEC["crema_tile_xywh"]
    for n in range(120):
        changed = span[0] <= n < span[1]
        frames = second if changed and role == "cutaway" else first
        picture = sample(frames, n-13 if changed and role == "jump" else n)
        if changed and role == "punch_in":
            picture = picture[25:-25,32:-32]
        if changed and role == "gain":
            picture = np.clip(picture.astype("float32")*1.10,0,255).astype("uint8")
        result = poster.copy()
        result[y:y+h,x:x+w] = cv2.resize(picture, (w,h), interpolation=cv2.INTER_LINEAR)
        yield result


def prepare(output, freeze):
    cv2.setNumThreads(1)
    media = output/"media"
    media.mkdir(mode=0o700)
    results = []
    sources = freeze["crema_originals"]
    for group, (a,b) in enumerate(zip(sources[::2], sources[1::2], strict=True),1):
        first, second = genuine_frames(a), genuine_frames(b)
        span = SPEC["crema_placements"][(group-1)%2]
        for role in ROLES:
            path = media/f"c{group}-{role}.mp4"
            if role == "tandem_cutaway":
                hd.authored(path,source=media/f"c{group}-cutaway.mp4",crf=35)
            else:
                hd.authored(path,pictures=compose(first,second,span,role))
            results.append(hd.describe(path,group,role,[span]))
        print("prepared",group,flush=True)
    require(len(results)==30 and len({r["sha256"] for r in results})==30, "Expected all 30 different ordinary deliveries")
    write(output/"media.json",{"freeze_sha256":sha(output/"freeze.json"),"rows":results})


def geometry(shape, box):
    height,width = shape[:2]
    x,y,w,h = box
    cx,cy = x+w/2,y+h/2
    nx,ny = np.meshgrid(np.linspace(-.8,.8,96), np.linspace(-.8,.8,96))
    a,b = SPEC["face_ellipse"]
    c,d = SPEC["head_ellipse"]
    inner = (nx/a)**2+(ny/b)**2 <= 1
    ring = ((nx/c)**2+(ny/d)**2 <= 1)&~inner
    masks = [inner,ring&(nx < -.40),ring&(nx > .40),ring&(ny < -.42),ring&(ny > .42)]
    mx,my = (nx*w+cx).astype("float32"),(ny*h+cy).astype("float32")
    visible = (mx >= 0)&(mx <= width-1)&(my >= 0)&(my <= height-1)
    actual = [m&visible for m in masks]
    available = [i for i,(m,a) in enumerate(zip(masks,actual,strict=True))
                 if np.count_nonzero(a) >= SPEC["minimum_sector_visible_fraction"]*np.count_nonzero(m)]
    if 0 not in available or len(available)-1 < SPEC["sector_support"]:
        return None
    return mx,my,[actual[i] for i in available],available


def evaluate(path, receipt, detector):
    from app.detectors.video.decode import FrameTable, decode_frames

    require(sha(path)==receipt["sha256"], "Transfer delivery differs")
    table = FrameTable.from_json(receipt["clock"])
    require(hd.relative(table)==[Fraction(i,30) for i in range(120)],"Transfer delivery clock differs")
    anchors = sorted({i for a,b in receipt["truth"] for i in (a-9,a-6,a-3,b+3,b+6,b+9)})
    offsets = (-10,-9,-8,-7,0,7,8,9,10)
    edges = {i+d for span in receipt["truth"] for i in span for d in offsets}
    needed = sorted(set(anchors)|{i+d for i in edges for d in (-1,0)})
    pictures = {}
    for frame in decode_frames(path,table,needed,timeout_s=25):
        require(frame.pts_verified and frame.pts==table.pts[frame.index]
                and development.source.review.digest(frame.rgb)==receipt["rgb_sha256"][frame.index],"Transfer picture differs")
        pictures[frame.index] = frame.rgb
    require(set(pictures)==set(needed),"Incomplete source pixels")
    output = []
    for a,b in receipt["truth"]:
        box = development.face_rectangle(pictures,(a-9,a-6,a-3,b+3,b+6,b+9),detector)
        geo = geometry(pictures[a].shape,box) if box is not None else None
        if geo is None:
            output.append({"status":"geometry_unavailable"})
            continue
        metrics = development.pair_metrics(pictures,[i+d for i in (a,b) for d in offsets],geo[:3])
        boundary = []
        for edge in (a,b):
            residual = np.maximum(0., np.asarray(metrics[edge])-np.median([metrics[edge+d] for d in offsets if d],axis=0))
            enough_face = float(residual[0]) >= SPEC["face_excess_minimum"]
            support = sum(float(v) >= max(SPEC["sector_excess_minimum"],SPEC["sector_to_face_minimum"]*float(residual[0]))
                          for v in residual[1:])
            boundary.append({"edge":edge,"pts":table.pts[edge],"available":geo[3],"residual":residual.tolist(),
                             "central_transition":enough_face,"broad":enough_face and support>=SPEC["sector_support"]})
        output.append({"status":"assessed","central_transition":all(x["central_transition"] for x in boundary),
                       "broad":all(x["broad"] for x in boundary),"boundaries":boundary})
    return {"native_frame_reads":len(needed),"intervals":output}


def score(output,lock):
    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(1)
    media = read(output/"media.json")
    require(media["freeze_sha256"]==sha(output/"freeze.json"),"Ordinary transfer manifest differs")
    observed = [row for row in read(RTFS/"media.json")["rows"] if row["condition"] in ("two_swaps","tandem_two_swaps")]
    require([r["sha256"] for r in observed]==lock["rtfs_file_sha256"],"Swap transfer files differ")
    detector = load_yunet(development.ROOT/"models/weights/face_detection_yunet_2023mar.onnx")
    started = time.monotonic()
    with (output/"observations.jsonl").open("x") as stream:
        for folder,rows in ((output,media["rows"]),(RTFS,observed)):
            for row in rows:
                result = evaluate(folder/"media"/row["file"],row,detector)
                stream.write(json.dumps({k:row[k] for k in ("sha256","group","condition","truth")}|result,allow_nan=False)+"\n")
                print(row["condition"],row["group"],[(a["status"],a.get("broad",False)) for a in result["intervals"]],flush=True)
    write(output/"runtime.json",{"elapsed_s":time.monotonic()-started,"media_sha256":sha(output/"media.json"),
                                 "freeze_sha256":sha(output/"freeze.json"),"observations_sha256":sha(output/"observations.jsonl")})


def report(output):
    runtime = read(output/"runtime.json")
    require(all(runtime[f"{name}_sha256"]==sha(output/f"{name}.{extension}")
                for name,extension in (("media","json"),("freeze","json"),("observations","jsonl"))),"Frozen results changed")
    counts, source_groups, native_reads = {}, set(), 0
    with (output/"observations.jsonl").open() as stream:
        for row in map(json.loads,stream):
            condition = row["condition"]
            count = counts.setdefault(condition,Counter())
            count["files"] += 1
            native_reads += row["native_frame_reads"]
            for result in row["intervals"]:
                count.update(intervals=1,assessed=result["status"]=="assessed",
                             central_transition=result.get("central_transition",False),broad=result.get("broad",False))
                if result.get("broad") and condition in ("cutaway","tandem_cutaway"):
                    source_groups.add(row["group"])
    swaps = [counts[k] for k in ("two_swaps","tandem_two_swaps")]
    cuts = [counts[k] for k in ("cutaway","tandem_cutaway")]
    controls = [counts[k] for k in ("unchanged","jump","punch_in","gain")]
    thresholds = SPEC["gate"]
    measured = all(sum(row["assessed"] for row in groups) >= thresholds["assessment_minimum"]*sum(row["intervals"] for row in groups)
                   for groups in (swaps,cuts,controls))
    gate = {"cutaway_groups":len(source_groups)>=thresholds["cutaway_groups"],
            "swap_annotations":sum(row["broad"] for row in swaps)<=thresholds["swap_annotation_max"],
            "ordinary_processing":sum(row["broad"] for row in controls)<=thresholds["ordinary_processing_max"],
            "assessment":measured}
    require(sum(row["intervals"] for row in counts.values())==62,"Incomplete native edge comparison")
    write(output/"aggregate.json",{"protocol":SPEC,"runtime":runtime,"counts":counts,
                                    "ordinary_cutaway_groups_qualified":len(source_groups),
                                    "native_frame_reads":native_reads,"gate":gate,"advances":all(gate.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","prepare","score","report"))
    parser.add_argument("development",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    lock = plan(args.development)
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700,parents=True)
        write(args.output/"freeze.json",lock)
        return
    require(lock==read(args.output/"freeze.json"),"Frozen methodology or inputs changed")
    if args.phase == "prepare":
        prepare(args.output,lock)
    elif args.phase == "score":
        score(args.output,lock)
    else:
        report(args.output)


if __name__=="__main__":
    main()
