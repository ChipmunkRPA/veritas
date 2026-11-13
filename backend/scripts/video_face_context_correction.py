# SPDX-License-Identifier: Apache-2.0
"""Test bounded scale and brightness correction for source-bound video context."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from scripts import video_face_near_head_transfer as near
from scripts.video_common_sources_v89 import read, require, sha, write

dev = near.development
DATA = dev.ROOT/"backend/data/independent-evaluation"
GRID = DATA/"video-face-hd-retry-scene-challenge-2026-11-12"
CREMA = DATA/"video-face-near-head-transfer-2026-11-12-v2"
ARMS = ("raw", "scale", "brightness", "combined")
CALIBRATION_GROUPS = {1, 3, 5, 7}
LIMITS = ((2., .25), (2., .35), (2., .45), (3., .25), (3., .35), (3., .45))
SPEC = {
    "schema": "chipmunk.video-face-context-correction/1",
    "profile": [1280, 720, 30, 120, "h264", "yuv420p"],
    "calibration": "Four fixed GRID pairs, separate from four other GRID pairs and all CREMA-D/RTFS transfer sources.",
    "held": "Retained CREMA-D source-time cutaways and controls, RTFS source-time swaps, plus GRID and RTFS replay from saved Full timestamps.",
    "appearance": "Prior fixed central face and four near-head sectors. Central edge selection precedes correction. Reuse raw flank motion floor.",
    "fit_pixels": "Alternating 4-pixel blocks of near-head sectors. Validate improvement on the disjoint blocks and require central face improvement.",
    "scale": {"bounds": [.8, 1.24], "offset_bound": 8, "coarse_scales": [.82, .9, 1., 1.12, 1.22],
              "coarse_offsets": [-6, 0, 6], "refinements": [[.04, 2], [.015, 1]]},
    "brightness": {"gain": [.75, 1.30], "offset": [-24., 24.], "iqr_floor": 8.},
    "threshold_candidates": [list(value) for value in LIMITS],
    "calibration_rule": "From points retaining at least six of eight timed speaker cutaways choose fewest other control annotations, "
                        "then most cutaways, then the stricter validation-improvement requirement. Use the same point for all correction arms.",
    "gate": {"cutaway_groups": 4, "other_controls_max": 1, "swaps_max": 2, "minimum_assessment": .75},
}


def verify_run(folder, check_held=False):
    ledger = read(folder/"media.json")
    require(ledger["freeze_sha256"] == sha(folder/"freeze.json"), "Input source freeze differs")
    if check_held:
        require(read(folder/"runtime.json")["held_sha256"] == sha(folder/"held.jsonl"), "Full observations differ")
    return ledger


def plan():
    from app.detectors.video import decode

    grid = verify_run(GRID, True)
    crema = verify_run(CREMA)
    rtfs = verify_run(near.RTFS, True)
    require(len(grid["rows"])==61 and len(crema["rows"])==30 and len(rtfs["rows"])==64, "Missing source cohorts")
    previous = read(CREMA/"freeze.json")
    require(previous["rtfs_manifest_sha256"]==sha(near.RTFS/"media.json") and previous["rtfs_transfer_groups"]==8,
            "Previously established RTFS source separation differs")
    return {"protocol": SPEC, "sources": {folder.name: sha(folder/"media.json") for folder in (GRID, CREMA, near.RTFS)},
            "app_observations": {folder.name: sha(folder/"held.jsonl") for folder in (GRID, near.RTFS)},
            "bindings": {Path(m.__file__).name: sha(m.__file__) for m in (near, dev, decode)}|{"runner":sha(__file__)},
            "face_asset":sha(dev.ROOT/"models/weights/face_detection_yunet_2023mar.onnx"),
            "opencv":cv2.__version__}


def ordinary_gain(output, grid_rows):
    """Author gain controls only for the disjoint calibration pairs."""
    media = output/"media"
    media.mkdir(mode=0o700)
    result = []
    for row in grid_rows:
        if row["condition"] != "ordinary" or row["group"] not in CALIBRATION_GROUPS:
            continue
        source = GRID/"media"/row["file"]
        span = dev.source.PLACEMENTS[(row["group"]-1)%len(dev.source.PLACEMENTS)]
        pictures = load(source,row,range(120))
        tile = dev.source.SPEC["webcam_tile"]["xywh"]
        x,y,w,h = tile
        for index in range(*span):
            frame = pictures[index].copy()
            frame[y:y+h,x:x+w] = np.clip(frame[y:y+h,x:x+w].astype("float32")*1.12,0,255).astype("uint8")
            pictures[index] = frame
        path = media/f"gain-{row['group']}.mp4"
        hd = near.hd
        hd.authored(path,pictures=(pictures[i] for i in range(120)))
        result.append({**hd.describe(path,row["group"],"gain",[list(span)]),"authored_span":list(span)})
    require(len(result)==4,"Incomplete GRID gain controls")
    write(output/"media.json",{"freeze_sha256":sha(output/"freeze.json"),"rows":result})


def load(path,row,indices):
    from app.detectors.video.decode import FrameTable, decode_frames

    require(sha(path)==row["sha256"],"Video file differs from its commitment")
    table = FrameTable.from_json(row["clock"])
    require(near.hd.relative(table)==[Fraction(i,30) for i in range(120)],"Video clock differs")
    result = {}
    for frame in decode_frames(path,table,indices,timeout_s=25):
        require(frame.pts_verified and frame.pts==table.pts[frame.index]
                and dev.source.review.digest(frame.rgb)==row["rgb_sha256"][frame.index],"Source picture differs")
        result[frame.index] = frame.rgb
    require(set(result)==set(indices),"Requested native frames are missing")
    return result


def prepare_patches(pictures, geometry, indices):
    mx,my,masks,available = geometry
    # Extra sampled margin keeps valid content available to a bounded zoom correction.
    pad = 26
    yy,xx = np.mgrid[-pad:96+pad,-pad:96+pad]
    xx = (mx[0,0]+xx*(mx[0,-1]-mx[0,0])/95).astype("float32")
    yy = (my[0,0]+yy*(my[-1,0]-my[0,0])/95).astype("float32")
    height,width = next(iter(pictures.values())).shape[:2]
    valid = (xx>=1)&(xx<=width-2)&(yy>=1)&(yy<=height-2)
    result = {}
    for i in indices:
        patches = cv2.remap(pictures[i],xx,yy,cv2.INTER_LINEAR)
        result[i] = cv2.GaussianBlur(cv2.cvtColor(patches,cv2.COLOR_RGB2GRAY),(3,3),0).astype("float32")
    return result,valid,masks,available


PAD = 26
Y,X = np.mgrid[:96,:96].astype("float32")
TRAIN = ((X//4+Y//4)%2)==0


def warp(patch, valid, transform):
    scale,dx,dy = transform
    mx = ((X-47.5)*scale+47.5+PAD+dx).astype("float32")
    my = ((Y-47.5)*scale+47.5+PAD+dy).astype("float32")
    return cv2.remap(patch,mx,my,cv2.INTER_LINEAR),cv2.remap(valid,mx,my,cv2.INTER_NEAREST)>0


def brightness(source, target):
    lo,med,hi = np.quantile(source,(.25,.5,.75))
    l,m,h = np.quantile(target,(.25,.5,.75))
    gain = float(np.clip((h-l)/(hi-lo),.75, 1.3)) if hi-lo >= 8 else 1.
    return gain,float(np.clip(m-gain*med,-24,24))


def abs_regions(diff,masks):
    return np.asarray([np.mean(diff[mask]) for mask in masks],dtype=float)


def fit(left,right,valid,masks,geometry,illumination):
    left = left[PAD:PAD+96,PAD:PAD+96]
    original = right[PAD:PAD+96,PAD:PAD+96]
    valid = valid.astype("uint8")
    ring = np.logical_or.reduce(masks[1:])
    original_difference = abs(left-original)
    if np.count_nonzero(ring&TRAIN)<100 or np.count_nonzero(ring&~TRAIN)<100:
        return None

    def evaluate(transform):
        image,visible = warp(right,valid,transform)
        if any(np.count_nonzero(mask&visible)<.8*np.count_nonzero(mask) for mask in masks):
            return None
        training = ring&visible&TRAIN
        held = ring&visible&~TRAIN
        if min(np.count_nonzero(training),np.count_nonzero(held))<100:
            return None
        gain,offset = brightness(image[training],left[training]) if illumination else (1.,0.)
        diff = abs(left-np.clip(gain*image+offset,0,255))
        return float(np.mean(diff[training])),diff,held,visible,(gain,offset)

    def winner(proposals):
        best = None
        for transform in proposals:
            scale,dx,dy = transform
            if not (.8<=scale<=1.24 and abs(dx)<=8 and abs(dy)<=8):
                continue
            result = evaluate(transform)
            if result is None:
                continue
            objective = (result[0],abs(scale-1)+.01*(abs(dx)+abs(dy)))
            if best is None or objective<best[0]:
                best = objective,transform,result
        return best

    offsets = SPEC["scale"]["coarse_offsets"]
    initial = ((s,x,y) for s in SPEC["scale"]["coarse_scales"] for x in offsets for y in offsets)
    best = winner(initial if geometry else [(1.,0,0)])
    if best is None:
        return None
    for ds,delta in SPEC["scale"]["refinements"] if geometry else ():
        s,x,y = best[1]
        proposals = ((s+a*ds,x+b*delta,y+c*delta) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1))
        better = winner(proposals)
        if better is not None and better[0]<best[0]:
            best = better
    _,transform,(_,difference,held,visible,photometric) = best
    raw_held = float(np.mean(original_difference[held]))
    corrected_held = float(np.mean(difference[held]))
    center = masks[0]&visible
    center_gain = float(np.mean(original_difference[center])-np.mean(difference[center]))
    return {"transform":list(transform),"photometric":list(photometric),"held_raw":raw_held,
            "held_corrected":corrected_held,"central_improvement":center_gain,
            "held_regions":abs_regions(difference,[mask&visible&~TRAIN for mask in masks]).tolist()}


def measure(path,row,ranges,detector,radius):
    if not ranges:
        return [],0
    windows = []
    needed = set()
    for a,b in ranges:
        pairs = [([i for i in range(t-radius,t+radius+1) if 0<i<120 and (2*i<a+b if left else 2*i>=a+b)],
                  [i for i in (*range(t-radius-10,t-radius-6),*range(t+radius+7,t+radius+11)) if 0<i<120])
                 for t,left in ((a,True),(b,False))]
        anchors = [i for i in (a-9,a-6,a-3,b+3,b+6,b+9) if 0<=i<120]
        edges = {i for nearby,flank in pairs for i in (*nearby,*flank)}
        indices = {i+k for i in edges for k in (-1,0)}
        needed.update(indices|set(anchors))
        windows.append((pairs,anchors,indices))
    pictures = load(path,row,sorted(needed))
    report = []
    for target,(pairs,anchors,indices) in zip(ranges,windows,strict=True):
        box = dev.face_rectangle(pictures,anchors,detector)
        shape = next(iter(pictures.values())).shape
        geom = near.geometry(shape,box) if box is not None else None
        if geom is None or any(not search or len(ref)<4 for search,ref in pairs):
            report.append({"target":target,"status":"unassessed"})
            continue
        patches,valid,masks,available = prepare_patches(pictures,geom,indices)
        cropped = {i:p[PAD:PAD+96,PAD:PAD+96] for i,p in patches.items()}
        values = {i:abs_regions(abs(cropped[i]-cropped[i-1]),[m&~TRAIN for m in masks])
                  for search,ref in pairs for i in (*search,*ref)}
        output = []
        for center,(search,refs) in zip(target,pairs,strict=True):
            floor = np.median([values[i] for i in refs],axis=0)
            edge = max(search,key=lambda i:(values[i][0]-floor[0],-abs(i-center),-i))
            raw = np.maximum(0,values[edge]-floor).tolist()
            fitted = {}
            for arm in ARMS[1:]:
                correction = fit(patches[edge-1],patches[edge],valid,masks,arm!="brightness",arm!="scale")
                if correction is not None:
                    correction["residual"] = np.maximum(0,np.asarray(correction["held_regions"])-floor).tolist()
                fitted[arm] = correction
            output.append({"index":edge,"pts":row["clock"]["pts"][edge],"native_hashes":row["rgb_sha256"][edge-1:edge+1],
                           "sectors":available,"motion_floor":floor.tolist(),"raw":raw,"fitted":fitted})
        report.append({"target":target,"status":"assessed","boundaries":output})
    return report,len(needed)


def grid_targets(row):
    if row["condition"] in ("person_switch","tandem_person_switch","same_person_jump","same_person_reframe","ordinary"):
        return [row["authored_span"] or list(dev.source.PLACEMENTS[(row["group"]-1)%2])]
    return [row["authored_span"]] if row["condition"]=="gain" else []


def selection(details, row):
    pts = row["clock"]["pts"]
    record = next((details.get(key) for key in dev.SPEC["report_interval"] if details.get(key)),None)
    return [[pts.index(record["start_pts"]),pts.index(record["end_pts"])]] if record else []


def generate(output, phase):
    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(1)
    detector = load_yunet(dev.ROOT/"models/weights/face_detection_yunet_2023mar.onnx")
    grid = verify_run(GRID,True)["rows"]
    kind = "calibration" if phase=="calibrate" else "held"
    rows = []
    if phase=="calibrate":
        ordinary_gain(output,grid)
        rows.extend((GRID,m,"grid_timed",grid_targets(m),0) for m in grid if m["group"] in CALIBRATION_GROUPS and grid_targets(m))
        rows.extend((output,m,"grid_timed",grid_targets(m),0) for m in read(output/"media.json")["rows"])
    else:
        rows.extend((CREMA,m,"crema_timed",m["truth"],0) for m in verify_run(CREMA)["rows"])
        rtfs = verify_run(near.RTFS,True)["rows"]
        for m,run in zip(grid,map(json.loads,(GRID/"held.jsonl").read_text().splitlines()),strict=True):
            if m["group"] not in CALIBRATION_GROUPS and m["condition"] in dev.source.GRID_ROLES:
                require(m["sha256"]==run["sha256"],"GRID Full result differs")
                rows.append((GRID,m,"grid_app",selection(run["before"],m),8))
        for m,run in zip(rtfs,map(json.loads,(near.RTFS/"held.jsonl").read_text().splitlines()),strict=True):
            if m["condition"] in ("two_swaps","tandem_two_swaps"):
                require(m["sha256"]==run["sha256"],"RTFS Full result differs")
                rows.append((near.RTFS,m,"rtfs_timed",m["truth"],0))
                rows.append((near.RTFS,m,"rtfs_app",selection(run["baseline"],m),8))
    journal = output/f"{kind}.jsonl"
    began = time.monotonic()
    with journal.open("x") as stream:
        for folder,row,bank,targets,radius in rows:
            result,reads = measure(folder/"media"/row["file"],row,targets,detector,radius)
            truth = row.get("authored_span")
            stream.write(json.dumps({"bank":bank,"group":row["group"],"condition":row["condition"],"sha256":row["sha256"],
                                     "expected":[truth] if truth else row["truth"],"reads":reads,"results":result},allow_nan=False)+"\n")
            print(kind,bank,row["condition"],row["group"],len(result),flush=True)
    return {"journal_sha256":sha(journal),"runtime_s":time.monotonic()-began,"files":len(rows)}


def broad(residual):
    face,*ring = residual
    return face>=4 and sum(v>=max(8,.55*face) for v in ring)>=2


def disposition(result,arm,point):
    if result["status"]!="assessed":
        return False,0
    edge_states = []
    accepted = 0
    for edge in result["boundaries"]:
        correction = edge["fitted"].get(arm)
        applied = False
        if correction:
            improvement = correction["held_raw"]-correction["held_corrected"]
            applied = (correction["central_improvement"]>=2 and improvement>=max(point[0],point[1]*correction["held_raw"]))
        accepted += int(applied)
        edge_states.append(broad(correction["residual"] if applied else edge["raw"]))
    return all(edge_states),accepted


def calibration(output,metadata):
    journal = output/"calibration.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    options = []
    for point in LIMITS:
        qualified = Counter()
        for row in rows:
            for result in row["results"]:
                qualified[row["condition"]] += disposition(result,"combined",point)[0]
        cuts = sum(qualified[k] for k in ("person_switch","tandem_person_switch"))
        controls = sum(qualified.values())-cuts
        options.append({"point":list(point),"cutaways":cuts,"other_controls":controls})
    options.sort(key=lambda x:(x["cutaways"]<6,x["other_controls"],-x["cutaways"],-x["point"][1],-x["point"][0]))
    require(options[0]["cutaways"]>=6, "No correction point preserved six calibration cutaways")
    write(output/"calibration.json",{**metadata,"freeze_sha256":sha(output/"freeze.json"),"options":options,"point":options[0]["point"]})


def report(output, seal):
    calibration = read(output/"calibration.json")
    journal = output/"held.jsonl"
    require(seal["journal_sha256"]==sha(journal),"Held observations differ")
    data = {}
    reads = 0
    qualified = set()
    with journal.open() as stream:
        for row in map(json.loads,stream):
            counts = data.setdefault(row["bank"],{}).setdefault(row["condition"],Counter())
            counts["files"]+=1
            reads+=row["reads"]
            for result in row["results"]:
                counts["ranges"]+=1
                counts["assessed"]+=result["status"]=="assessed"
                for arm in ARMS:
                    outcome, corrected = disposition(result,arm,calibration["point"])
                    counts[f"{arm}_broad"]+=outcome
                    counts[f"{arm}_edges_corrected"]+=corrected
                    if arm=="combined" and outcome and row["bank"]=="crema_timed" and row["condition"] in ("cutaway","tandem_cutaway"):
                        qualified.add(row["group"])
                    if outcome and row["expected"] and result.get("boundaries"):
                        edges = [entry["index"] for entry in result["boundaries"]]
                        counts[f"{arm}_broad_exact_within_two"]+=any(
                            all(abs(actual-expected)<=2 for actual,expected in zip(edges,span,strict=True)) for span in row["expected"])
    gates = SPEC["gate"]
    crema = data["crema_timed"]
    swaps = list(data["rtfs_timed"].values())
    cuts = [crema[k] for k in ("cutaway","tandem_cutaway")]
    controls = [crema[k] for k in ("unchanged","jump","punch_in","gain")]
    measured = all(sum(row["assessed"] for row in group)>=gates["minimum_assessment"]*sum(row["ranges"] for row in group)
                   for group in (swaps,cuts,controls))
    gate = {"cutaway_groups":len(qualified)>=gates["cutaway_groups"],
            "other_controls":sum(row["combined_broad"] for row in controls)<=gates["other_controls_max"],
            "swap_annotations":sum(row["combined_broad"] for row in swaps)<=gates["swaps_max"],
            "assessment":measured}
    write(output/"aggregate.json",{"protocol":SPEC,"runtime":seal,"calibration":calibration,"counts":data,
                                    "native_frame_reads":reads,"cutaway_groups_qualified":len(qualified),
                                    "timed_gate":gate,"timed_gate_passed":all(gate.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","calibrate","held","report"))
    parser.add_argument("run",type=Path)
    args = parser.parse_args()
    output = args.run
    frozen = plan()
    if args.phase=="freeze":
        output.mkdir(mode=0o700,parents=True)
        write(output/"freeze.json",frozen)
        return
    require(read(output/"freeze.json")==frozen,"Source, protocol or executable changed")
    if args.phase=="calibrate":
        calibration(output,generate(output,args.phase))
        return
    fitted = read(output/"calibration.json")
    require(fitted["freeze_sha256"]==sha(output/"freeze.json")
            and fitted["journal_sha256"]==sha(output/"calibration.jsonl"),"Calibration changed")
    if args.phase=="held":
        write(output/"held-seal.json",generate(output,args.phase)|{"calibration_sha256":sha(output/"calibration.json")})
    else:
        seal = read(output/"held-seal.json")
        require(seal["calibration_sha256"]==sha(output/"calibration.json"),"Held point changed")
        report(output,seal)


if __name__=="__main__":
    main()
