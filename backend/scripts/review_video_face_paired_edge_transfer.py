# SPDX-License-Identifier: Apache-2.0
"""Independently recount frozen paired-edge locations and reopen their native pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SHA = re.compile(r"[a-f0-9]{64}")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
PANELS = {
    "selection_edits": "video-face-hd-sparse-retry-2026-11-12-v2",
    "selection_controls": "video-face-scale-context-holdout-2026-11-12-v2",
    "held_edits_and_cuts": "video-face-hd-retry-scene-challenge-2026-11-12",
    "held_two_edits": "video-face-two-returns-hd-2026-11-12",
}
COUNTS = dict(zip(PANELS, (48, 30, 61, 64), strict=True))
LIMITS = {"min_added": 3, "min_groups": 2, "max_new_ordinary_files": 1, "max_new_cut_intervals": 1,
          "max_wrong_on_edit": 1, "min_assessed_routed_edit_share": .75}
ORDINARY_CUTS = {"person_switch", "tandem_person_switch", "cross_recording_cuts"}
HEVC_POINT = .03176017255556951
HEVC_LIMITS = {"min_added": 2, "min_groups": 2, "min_assessed_routed_share": .60, "max_ordinary_ranges": 0,
               "max_wrong_edit_ranges": 0}
HEVC_DELIVERIES = {
    "publisher_and_cameras": {"camera": 3, **dict.fromkeys(("original", "recompressed", "light", "jump", "cut", "swap_a", "swap_b"),4)},
    "cutaways": dict.fromkeys(("unchanged", "cutaway", "tandem_cutaway", "jump", "punch_in", "gain"),5),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def matched(span, truth):
    left = max(span[0], truth[0])
    right = min(span[1], truth[1])
    common = max(0, right-left)
    span_size, truth_size = (span[1]-span[0]), (truth[1]-truth[0])
    return 3*common >= span_size+truth_size


def clock_range(span, valid_min=0, valid_max=119):
    return (isinstance(span, list) and len(span)==2 and all(type(i) is int for i in span)
            and valid_min <= span[0] < span[1] <= valid_max)


def candidate(value, coarse, guarded):
    a,b=value["frames"]
    c,d=coarse
    require(clock_range(value["frames"],5,115) and 5 <= b-a <= 16 and abs(a-c)<=12 and abs(b-d)<=12
            and 2*(min(b,d)-max(a,c))>=min(b-a,d-c) and math.isfinite(value["score"])
            and isinstance(value["pts"],list) and len(value["pts"])==2
            and all(type(pts) is int for pts in value["pts"]) and value["pts"][0]<value["pts"][1],
            "An observed pair is outside its fixed source search")
    before,during,after=value["anchors"]
    require(before["frame_index"]==a-1 and a<=during["frame_index"]<b and after["frame_index"]==b
            and value["pts"][0]<=during["pts"]<value["pts"][1]==after["pts"] and before["pts"]<value["pts"][0],
            "Selected source roles or timestamps differ")
    for anchor in (before,during,after):
        require(all(SHA.fullmatch(anchor[key]) for key in ("decoded_rgb_sha256","aligned_rgb_sha256")),
                "Invalid original-picture or crop commitment")
        box = anchor["bbox"]
        require(set(box)=={"x","y","w","h"} and all(type(v) in (float,int) and v>=0 and v==int(v) for v in box.values())
                and 48<=box["w"]<=1280-box["x"] and 48<=box["h"]<=720-box["y"], "Face crop is outside the source")
    require(len(value["edges"])==2,"Both native edges must be recorded")
    for edge in value["edges"]:
        center,sectors,ids=(edge[key] for key in ("central","sectors","sector_ids"))
        require(3.<=center<=255 and 2<=len(sectors)==len(set(ids))<=4 and set(ids)<={1,2,3,4}
                and all(0<=v<=255 for v in sectors) and edge["supported"] is True, "Missing near-head measurements")
        broad = sum(v>=max(6.,center/2.) for v in sectors)>=2
        require(edge["broad"] is broad and not (guarded and broad), "Outside-face gate changed")


def observations(data):
    lock=data["freeze"]
    rows={row["id"]:row for row in lock["rows"]}
    require(lock["schema"]=="chipmunk.video-face-paired-edge-transfer/1" and lock["method"]["gate"]==LIMITS
            and Counter(r["panel"] for r in rows.values())==COUNTS and list(rows)==list(range(203))
            and len({r["sha256"] for r in rows.values()})==203 and all(SHA.fullmatch(r["sha256"]) for r in rows.values()),
            "Source roster differs")
    require(all(r["coarse"] is None or (clock_range(r["coarse"]) and r["coarse"]==r["incumbent"][0])
                for r in rows.values()) and all(clock_range(span) for r in rows.values() for span in r["incumbent"]+r["truth"]),
            "Invalid baseline or truth ranges")
    stages={}
    freeze_pin=digest(encoded(lock))
    for name in ("selection","held"):
        stage=data[name]
        stage_rows=stage["observations"]
        require(stage["freeze_sha256"]==freeze_pin and sorted(o["id"] for o in stage_rows)==sorted(
            r["id"] for r in rows.values() if r["panel"].startswith(name)), "Stage roster or binding differs")
        for item in stage_rows:
            row=rows[item["id"]]
            coarse=row["coarse"]
            raw,guarded=item["raw"],item["guarded"]
            expected_status = "not_routed" if coarse is None else "no_supported_pair" if raw is None else "assessed"
            expected_reads = min(120,coarse[1]+18)-max(0,coarse[0]-18) if coarse else 0
            require(item["status"]==expected_status and item["native_reads"]==expected_reads
                    and (raw is None or coarse is not None) and (guarded is None or raw is not None),
                    "Unsupported pair or source read count")
            if raw:
                candidate(raw,coarse,False)
            if guarded:
                candidate(guarded,coarse,True)
                require(guarded["score"]<=raw["score"],"Guarded choice outranks unrestricted choice")
        stages[name]=stage_rows
    values=[item["guarded"]["score"] for item in stages["selection"] if item["guarded"] and not rows[item["id"]]["truth"]]
    point=max([.02,*values])
    require(data["point"]=={"point":point,"guarded_ordinary_candidates":len(values),"selection_sha256":digest(encoded(data["selection"]))},
            "Genuine-only selection point differs")
    return rows,stages,point


def audit(data):
    rows,stages,point=observations(data)
    stats,totals,gains=defaultdict(Counter),Counter(),set()
    for item in stages["held"]:
        row=rows[item["id"]]
        truth,old=row["truth"],row["incumbent"]
        value=item["guarded"]
        proposal=value["frames"] if value and value["score"]>point else None
        raw=item["raw"]
        previously={i for i,t in enumerate(truth) if any(matched(t,s) for s in old)}
        jointly={i for i,t in enumerate(truth) if proposal and matched(t,proposal)}
        newly=jointly-previously
        if newly:
            gains.add((row["panel"],row["group"]))
        cell=stats[row["panel"]+"/"+row["condition"]]
        cell.update({"files":1,"truth":len(truth),"routed":int(bool(row["coarse"])),
            "assessed":int(item["status"]=="assessed"),"ordinary_files_already_flagged":int(bool(not truth and old)),
            "raw_above_point":int(bool(raw and raw["score"]>point)),"guarded_above_point":int(bool(proposal)),
            "before_useful":len(previously),"combined_useful":len(jointly|previously),"additional_useful":len(newly)})
        totals["added"]+=len(newly)
        totals["edit_files"]+=bool(truth)
        totals["edit_files_routed"]+=bool(truth and row["coarse"])
        totals["edit_files_assessed"]+=bool(truth and item["status"]=="assessed")
        if proposal and newly and any(matched(proposal,old_span) for old_span in old[1:]):
            cell["overlaps_existing_refinement"]+=1
        if proposal and not truth:
            cut=row["condition"] in ORDINARY_CUTS
            cell["new_ordinary_file"]+=not old
            cell["extra_cut_interval"]+=cut
            totals["new_ordinary_files"]+=not old
            totals["new_cut_intervals"]+=cut
        if proposal and truth and not jointly:
            cell["wrong_edit_range"]+=1
            totals["wrong_on_edit"]+=1
    gate={"added":totals["added"]>=LIMITS["min_added"],"groups":len(gains)>=LIMITS["min_groups"],
          "ordinary_files":totals["new_ordinary_files"]<=LIMITS["max_new_ordinary_files"],
          "ordinary_cut_intervals":totals["new_cut_intervals"]<=LIMITS["max_new_cut_intervals"],
          "wrong_on_edit":totals["wrong_on_edit"]<=LIMITS["max_wrong_on_edit"],
          "assessment":totals["edit_files_assessed"]>=LIMITS["min_assessed_routed_edit_share"]*totals["edit_files_routed"]}
    result={"cells":dict(stats),"totals":dict(totals),"gained_groups":len(gains),"gate":gate,"passed":all(gate.values())}
    require(result==data["result"],"Held results do not reproduce")
    return result,rows,stages


def hevc(data, prior):
    lock = data["freeze"]
    require(lock["schema"]=="chipmunk.video-face-paired-edge-hevc-confirmation/1"
            and lock["point"]==HEVC_POINT and lock["gate"]==HEVC_LIMITS
            and lock["encoding"]=={"codec_name":"hevc","codec_tag_string":"hvc1","pix_fmt":"yuv420p","width":1280,
                                   "height":720,"r_frame_rate":"30/1","nb_frames":"120"}, "Confirmation method differs")
    require(lock["pins"]["original_experiment"]==digest(prior.read_bytes()), "Wrong H.264 method commitment")
    prior_data=json.loads(prior.read_bytes())
    audit(prior_data)
    require(prior_data["freeze"]["code"]==lock["candidate_code"] and prior_data["freeze"]["weights"]==lock["weights"],
            "The candidate code or weights changed across codec")
    counts=defaultdict(Counter)
    roster=defaultdict(set)
    rows=data["rows"]
    groups=set()
    for key in ("sha256","input_h264_sha256"):
        require(len({row[key] for row in rows})==len(rows)==61 and all(SHA.fullmatch(row[key]) for row in rows),
                "Confirmation has missing or repeated files")
    require(not ({row["input_h264_sha256"] for row in rows}&{row["sha256"] for row in prior_data["freeze"]["rows"]}),
            "Confirmation reused a source delivery")
    for row in rows:
        bank,condition,group=row["bank"],row["condition"],row["group"]
        require(group not in roster[(bank,condition)],"Repeated confirmation source within a condition")
        roster[bank,condition].add(group)
        old,coarse,raw,guarded=row["prior"],row["coarse"],row["raw"],row["guarded"]
        truth={"swap_a":[72,78],"swap_b":[87,97]}.get(condition) if bank=="publisher_and_cameras" else None
        require(row["truth"]==([truth] if truth else []) and all(clock_range(span) for span in old)
                and (coarse is None or old[0]==coarse) and (raw is None or coarse is not None) and (guarded is None or raw is not None),
                "Invalid frozen placement or comparator route")
        state="not_routed" if not coarse else "assessed" if raw else "no_supported_pair"
        reads=min(120,coarse[1]+18)-max(0,coarse[0]-18) if coarse else 0
        require(row["status"]==state and row["native_reads"]==reads,"Incorrect confirmation assessment state")
        if raw:
            candidate(raw,coarse,False)
        if guarded:
            candidate(guarded,coarse,True)
            require(guarded["score"]<=raw["score"],"Guarded confirmation choice outranks raw choice")
        chosen=guarded["frames"] if guarded and guarded["score"]>HEVC_POINT else None
        require(chosen==row["selected"],"Fixed confirmation point was not applied")
        old_match=bool(truth and any(matched(truth,span) for span in old))
        new_match=bool(truth and chosen and matched(truth,chosen))
        added=new_match and not old_match
        if added:
            groups.add(group)
        counts[f"{bank}/{condition}"].update({
            "files":1,"routed":int(bool(coarse)),"assessed":int(state=="assessed"),"ordinary_prior":int(bool(not truth and old)),
            "ordinary_added_ranges":int(bool(not truth and chosen)),"new_ordinary_files":int(bool(not truth and chosen and not old)),
            "wrong_edit_ranges":int(bool(truth and chosen and not new_match)),"useful_before":int(old_match),
            "useful_combined":int(old_match or new_match),"additional_useful":int(added),
            "exact_before":int(bool(truth and truth in old)),"exact_combined":int(bool(truth and (truth in old or truth==chosen))),
        })
    require({key:len(value) for key,value in roster.items()}=={(bank,condition):size
            for bank,conditions in HEVC_DELIVERIES.items() for condition,size in conditions.items()}, "Confirmation bank differs")
    edits=[row for row in rows if row["truth"]]
    routed=sum(bool(row["coarse"]) for row in edits)
    assessed=sum(row["status"]=="assessed" for row in edits)
    totals=Counter()
    for cell in counts.values():
        totals.update(cell)
    gate={"added":totals["additional_useful"]>=2,"groups":len(groups)>=2,"assessment":bool(routed and 5*assessed>=3*routed),
          "ordinary":not totals["ordinary_added_ranges"],"wrong_edit":not totals["wrong_edit_ranges"]}
    result={"counts":dict(counts),"gate":gate,"passed":all(gate.values()),"edits":len(edits),"routed_edits":routed,
            "assessed_edits":assessed,"added_groups":len(groups)}
    require(result==data["result"],"Confirmation counts do not reproduce")
    return result


def native(data, rows, stages, output=None):
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence.face_interval_binding import face_context_box

    cache={}
    verified=Counter()
    example_written=False
    point=data["point"]["point"]
    for panel,folder in PANELS.items():
        root=DATA/folder
        payload=(root/"media.json").read_bytes()
        require(digest(payload)==data["freeze"]["inputs"][panel]["manifest_sha256"],"Native source manifest changed")
        cache[panel]=(root,{row["sha256"]:row for row in json.loads(payload)["rows"]})
    for item in stages["held"]:
        row=rows[item["id"]]
        root,index=cache[row["panel"]]
        manifest=index[row["sha256"]]
        path=root/"media"/manifest["file"]
        require(digest(path.read_bytes())==row["sha256"],"Native source differs")
        verified["files"]+=1
        result=item["guarded"]
        if not result or result["score"]<=point:
            continue
        table=probe_frames(path,timeout_s=15)
        require(table.to_json()==manifest["clock"] and all((v-table.pts[0])*table.tb==Fraction(i,30)
                for i,v in enumerate(table.pts)),"Native video clock differs")
        anchors={r["frame_index"]:r for r in result["anchors"]}
        start,stop=result["frames"]
        require(result["pts"]==[table.pts[start],table.pts[stop]],"Saved endpoints differ from native timestamps")
        measured=set()
        useful=any(matched(result["frames"],span) and not any(matched(old,span) for old in row["incumbent"])
                   for span in row["truth"])
        for frame in decode_frames(path,table,sorted(anchors),timeout_s=20):
            anchor=anchors[frame.index]
            fingerprint=digest(frame.rgb.tobytes())
            require(frame.pts_verified and frame.pts==anchor["pts"]==table.pts[frame.index]
                    and fingerprint==anchor["decoded_rgb_sha256"]==manifest["rgb_sha256"][frame.index],"Saved native picture differs")
            measured.add(frame.index)
            verified["original_frames"]+=1
            box=face_context_box(anchor["bbox"],{"width":table.width,"height":table.height})
            x,y,w,h=(box[k] for k in ("x","y","w","h"))
            crop=frame.rgb[y:y+h,x:x+w]
            require(crop.shape==(h,w,3),"Source face crop cannot be reproduced")
            verified["context_crops"]+=1
            if output and useful and not example_written:
                from PIL import Image

                role=("before","during","after")[list(anchors).index(frame.index)]
                Image.fromarray(frame.rgb).save(output/f"{role}-frame.png")
                Image.fromarray(crop).save(output/f"{role}-crop.png")
        require(measured==anchors.keys(),"One or more native roles were not read")
        if output and useful and not example_written:
            (output/"timing.json").write_bytes(encoded({
                "source_sha256":row["sha256"],"start_frame":start,"end_frame_exclusive":stop,"truth":row["truth"],
                "time_base":table.time_base,"pts":result["pts"],"seconds":[table.seconds(start),table.seconds(stop)],
                "frame_indices":list(anchors),
            }))
            example_written=True
    return verified


def confirmation_native(data, folder, output=None):
    import cv2

    from app.detectors.image.faces import detect_faces, load_yunet
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.detectors.video.face_appearance import face_row
    from app.evidence.face_interval_binding import face_context_box

    manifest=folder/"media.json"
    require(digest(manifest.read_bytes())==data["media_sha256"],"Confirmation media ledger changed")
    media=json.loads(manifest.read_bytes())
    require(digest((folder/"freeze.json").read_bytes())==media["freeze_sha256"]
            and json.loads((folder/"freeze.json").read_bytes())==data["freeze"],"Wrong confirmation media freeze")
    index={row["sha256"]:row for row in media["rows"]}
    cv2.setNumThreads(1)
    weights=ROOT/"models/weights"
    require({name:digest((weights/name).read_bytes()) for name in data["freeze"]["weights"]}==data["freeze"]["weights"],
            "Confirmation crop weights changed")
    detector=load_yunet(weights/"face_detection_yunet_2023mar.onnx")
    aligner=cv2.FaceRecognizerSF.create(str(weights/"face_recognition_sface_2021dec.onnx"),"")
    verified=Counter()
    example_written=False
    for row in data["rows"]:
        binding=index[row["sha256"]]
        require(all(row[key]==binding[key] for key in ("input_h264_sha256","truth","group","condition","bank")),
                "Confirmation source provenance differs")
        path=folder/binding["file"]
        require(digest(path.read_bytes())==row["sha256"],"Confirmation source differs")
        table=probe_frames(path,timeout_s=15)
        require(table.to_json()==binding["clock"] and len(table)==120
                and all((pts-table.pts[0])*table.tb==Fraction(i,30) for i,pts in enumerate(table.pts)),
                "Confirmation native clock differs")
        verified.update({"files":1,"native_timestamps":len(table)})
        if not row["selected"]:
            continue
        start,stop=row["selected"]
        result=row["guarded"]
        require(result["pts"]==[table.pts[start],table.pts[stop]],"Confirmation source endpoints differ")
        anchors={item["frame_index"]:item for item in result["anchors"]}
        replayed=set()
        useful=any(matched(row["selected"],span) and not any(matched(old,span) for old in row["prior"]) for span in row["truth"])
        for frame in decode_frames(path,table,sorted(anchors),timeout_s=20):
            anchor=anchors[frame.index]
            fingerprint=digest(frame.rgb.tobytes())
            require(frame.pts_verified and frame.pts==anchor["pts"]==table.pts[frame.index]
                    and fingerprint==anchor["decoded_rgb_sha256"]==binding["rgb_sha256"][frame.index],"Confirmation picture differs")
            replayed.add(frame.index)
            bgr=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
            [face]=detect_faces(bgr,detector=detector,min_size=48)
            require({key:float(getattr(face,key)) for key in ("x","y","w","h")}==anchor["bbox"],"Confirmation face box differs")
            alignment=face_row(face.landmarks)
            alignment[0,:4]=face.box()
            alignment[0,14]=face.score
            aligned=cv2.cvtColor(aligner.alignCrop(bgr,alignment),cv2.COLOR_BGR2RGB)
            require(digest(aligned.tobytes())==anchor["aligned_rgb_sha256"],"Confirmation aligned face crop differs")
            box=face_context_box(anchor["bbox"],{"width":table.width,"height":table.height})
            x,y,w,h=(box[k] for k in ("x","y","w","h"))
            context=frame.rgb[y:y+h,x:x+w]
            require(context.shape==(h,w,3),"Confirmation source face crop cannot be reproduced")
            verified.update({"original_frames":1,"aligned_crops":1,"context_crops":1})
            if output and useful and not example_written:
                from PIL import Image

                role=("before","during","after")[list(anchors).index(frame.index)]
                Image.fromarray(frame.rgb).save(output/f"{role}-frame.png")
                Image.fromarray(context).save(output/f"{role}-crop.png")
        require(replayed==anchors.keys(),"One or more confirmation roles were not read")
        if output and useful and not example_written:
            (output/"timing.json").write_bytes(encoded({
                "source_sha256":row["sha256"],"start_frame":start,"end_frame_exclusive":stop,"truth":row["truth"],
                "time_base":table.time_base,"pts":result["pts"],"seconds":[table.seconds(start),table.seconds(stop)],
                "frame_indices":list(anchors),
            }))
            example_written=True
    return verified


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--example",type=Path,help="New private directory for a useful example's frames and face crops")
    parser.add_argument("--confirm-against",type=Path,help="Check the HEVC receipt against this H.264 receipt")
    parser.add_argument("--confirmation-media",type=Path,help="Reopen the native pictures from this local HEVC study")
    args=parser.parse_args()
    data=json.loads(args.receipt.read_bytes())
    if args.confirm_against:
        if args.native or (args.example and not args.confirmation_media):
            parser.error("Use --confirmation-media to reopen the HEVC source files")
        result=hevc(data,args.confirm_against)
        if args.confirmation_media:
            if args.example:
                args.example.mkdir(mode=0o700)
            result["native_replay"]=confirmation_native(data,args.confirmation_media,args.example)
        print(encoded(result).decode(),end="")
        raise SystemExit(0)
    if args.confirmation_media:
        parser.error("--confirmation-media requires --confirm-against")
    if args.example:
        if not args.native:
            parser.error("An example requires native media")
        args.example.mkdir(mode=0o700)
    result,rows,stages=audit(data)
    if args.native:
        result={**result,"native_replay":native(data,rows,stages,args.example)}
    print(encoded(result).decode(),end="")
