# SPDX-License-Identifier: Apache-2.0
"""Freeze, author and score a source-separated paired closure-timing panel."""

from __future__ import annotations

import argparse
import wave
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import build_grid_word_timed_controls as delivery
from scripts import grid_paired_closure_timing as candidate
from scripts import grid_reserve_neural_mouth as neural
from scripts import grid_reserve_neural_mouth_readout as previous
from scripts import grid_reserve_word_clocks as clocks
from scripts import review_grid_reserve_neural_mouth as closure_review
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts import video_lipsync_local_contrast_transfer as prior_av
from scripts import video_mouth_reconstruction_transfer as geometry
from scripts.video_common_sources_v89 import read, require, sha, write

ROLES = {
    "original": (0,0), "mouth_late_ordinary": (8,0), "mouth_early_ordinary": (-8,0),
    "audio_late_ordinary": (0,8), "coherent_late_ordinary": (8,8), "light_gain_ordinary": (0,0),
}
SPEC = {
    "schema": "chipmunk.grid-paired-closure-panel/1",
    "held": "First four eligible source groups from the qualified reserve outside both discovery components. "
            "Choose by original word clock and visible closure. Every retained group supplies both signed visual "
            "controls, audio-only, synchronized retime, brightness/gain, and original.",
    "match": "Known sign and lag error at most two native pictures. Report event times rather than estimating the entire edit.",
    "earlier": "Same-source earlier native recording and its marked visible closure. Timing evidence only.",
    "gate": "At least eight correct known shifts across three source groups, zero flags on original or coherent controls, "
            "at least 75 percent assessed cases, and comparable native crop and PCM replay.",
}


def video(path):
    digest = sha(path)
    table = probe_frames(path,timeout_s=15)
    require(not (table.rotation or table.truncated or table.interpolated)
            and [pts*table.tb for pts in table.pts]==[Fraction(i,25) for i in range(75)],"Exact native picture clock required")
    pixels = []
    for frame in decode_frames(path,table,None,timeout_s=25):
        require(frame.pts_verified and frame.pts==table.pts[frame.index],"Invalid native picture")
        pixels.append(frame.rgb)
    pcm, origin, dropped = audio.source_pcm(path,digest)
    require(len(pixels)==75 and len(pcm)==47648 and origin==0 and not dropped,"Complete paired source clocks required")
    return table,pixels,pcm


def motion(module,task,pixels):
    return [neural.observed(module,task,rgb)[1] for rgb in pixels]


def discovery(study,out):
    plan = read(study/"freeze.json")
    approved = read(study/"construction.json")
    replay = read(study/"independent-replay.json")
    require(replay["status"]=="verified" and replay["construction_sha256"]==sha(study/"construction.json"),
            "The original neural study must be independently verified")
    sources,_ = clocks.plan()
    out.mkdir(mode=0o700,parents=True)
    module,task = neural.mp_task()
    found = []
    try:
        for pin in plan["sources"]:
            source = Path(sources[pin["ordinal"]]["input"]["source_file"])
            _,rgb,pcm = video(source)
            reference = motion(module,task,rgb)
            first = pin["original_closure"]
            found.append({"role":"original","expected_pictures":0,"result":candidate.compare(reference,reference,pcm,pcm,first)})
            for item in approved["cases"]:
                if item["ordinal"]!=pin["ordinal"]:
                    continue
                file = study/"deliveries"/item["file"]
                require(sha(file)==item["proof"]["control_sha256"],"Discovery delivery differs")
                _,clip,sound = video(file)
                result = candidate.compare(reference,motion(module,task,clip),pcm,sound,first)
                found.append({"role":item["mode"],"expected_pictures":0 if item["mode"]=="source_time_neural" else 8,
                              "result":result})
    finally:
        task.close()
    write(out/"discovery.json",{"method_sha256":sha(Path(candidate.__file__)),"cases":found})


def frozen(qualification,study,out):
    old = read(study/"freeze.json")
    original_groups = {row["group"] for row in old["sources"]}
    qualified = qualification/"private-word-clocks.json"
    require(sha(qualified)==read(qualification/"qualification.json")["private_word_clocks_sha256"],"Reserve word clocks differ")
    discovery_data = read(out/"discovery.json")
    require(discovery_data["method_sha256"]==sha(Path(candidate.__file__)),"Repeat discovery for the exact prospective cue")
    inputs,plan_sha = clocks.plan()
    module,task = neural.mp_task()
    selection,availability = [],[]
    try:
        for row in read(qualified)["records"]:
            group = inputs[row["ordinal"]]["component_index"]
            if group in original_groups or group in {r["group"] for r in selection}:
                continue
            source = inputs[row["ordinal"]]["input"]
            _,rgb,_ = video(Path(source["source_file"]))
            observed = motion(module,task,rgb)
            point = None
            for word in row["words"]:
                if word["word"][0] not in neural.SPEC["bilabial_word_initials"]:
                    continue
                anchor = word["source_native_picture_indices"][0]
                point = neural.closure(observed,anchor)
                if point is not None and 12<=point<=57 and all(value is not None for value in observed[point-12:point+15]):
                    selection.append({"ordinal":row["ordinal"],"group":group,"event":point,"source_sha256":row["source_sha256"],
                                      "source_word_picture":anchor})
                    break
                point = None
            availability.append({"group":group,"qualified":point is not None})
            if len(selection)==4:
                break
    finally:
        task.close()
    require(len(selection)==4 and not {row["group"] for row in selection}&original_groups,"Need four source-disjoint held groups")
    prior = read(previous.GRID_TIMING/"point.json")
    fit = read(previous.GRID_TIMING/"head.json")
    require(prior["head_sha256"]==sha(previous.GRID_TIMING/"head.json")
            and prior["journal_sha256"]==sha(previous.GRID_TIMING/"point.jsonl")
            and fit["freeze_sha256"]==sha(previous.GRID_TIMING/"freeze.json"),"Prior timing calibration differs")
    files = [Path(__file__),Path(candidate.__file__),Path(neural.__file__),Path(previous.__file__),Path(prior_av.__file__)]
    write(out/"freeze.json",{"protocol":SPEC,"method":candidate.SPEC,"source_plan_sha256":plan_sha,
          "discovery_sha256":sha(out/"discovery.json"),"selection":selection,"intake":availability,"roles":ROLES,
          "bindings":{p.name:sha(p) for p in files},"prior_timing_point":prior["value"],
          "prior_timing_point_sha256":sha(previous.GRID_TIMING/"point.json"),
          "prior_timing_head_sha256":sha(previous.GRID_TIMING/"head.json")})


def lock(out):
    freeze = read(out/"freeze.json")
    files = [Path(__file__),Path(candidate.__file__),Path(neural.__file__),Path(previous.__file__),Path(prior_av.__file__)]
    require(freeze["protocol"]==SPEC and freeze["method"]==candidate.SPEC
            and freeze["bindings"]=={p.name:sha(p) for p in files}
            and freeze["discovery_sha256"]==sha(out/"discovery.json")
            and freeze["prior_timing_point_sha256"]==sha(previous.GRID_TIMING/"point.json")
            and freeze["prior_timing_head_sha256"]==sha(previous.GRID_TIMING/"head.json"),"Frozen study or earlier models changed")
    return freeze


def local_picture(originals,geo,index,visual_shift,light):
    target = originals[index]
    _,transform,center,shape = geo[index]
    if light:
        proposal = np.clip(target.astype(np.int16)+12,0,255).astype(np.uint8)
    else:
        old = index-visual_shift
        _,source_transform,_,_ = geo[old]
        warp = cv2.invertAffineTransform(transform)@np.vstack((source_transform,[0,0,1]))
        proposal = cv2.warpAffine(originals[old],warp,shape[::-1],flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    support = delivery.mouth_alpha(shape,transform,center)[...,None]
    output = np.uint8(np.rint(target.astype(np.float32)*(1-support)+proposal.astype(np.float32)*support))
    require(np.any(output!=target) and np.array_equal(output[support[...,0]==0],target[support[...,0]==0]),
            "Known local conventional edit changed pixels outside its support")
    return output


def save_pcm(path,pcm):
    with wave.open(str(path),"wb") as stream:
        stream.setparams((1,2,16000,len(pcm),"NONE","not compressed"))
        stream.writeframes(pcm.tobytes())


def author(out):
    freeze = lock(out)
    inputs,_ = clocks.plan()
    folder = out/"media"
    folder.mkdir(mode=0o700)
    cv2.setNumThreads(1)
    detector = load_yunet(geometry.WEIGHT)
    items = []
    for item in freeze["selection"]:
        ordinal,point = item["ordinal"],item["event"]
        source = Path(inputs[ordinal]["input"]["source_file"])
        require(sha(source)==item["source_sha256"],"Frozen original differs")
        _,rgb,original = video(source)
        all_geometry = {}
        for role,(vshift,ashift) in ROLES.items():
            if role=="original":
                items.append({"ordinal":ordinal,"role":role,"path":str(source),"sha256":sha(source),
                              "event":point,"span":None,"visual_shift":0,"audio_shift":0})
                continue
            begin,end = (point-11,point+6) if vshift<0 else (point-4,point+14)
            altered = list(rgb)
            pcm = original.copy()
            if vshift or role=="light_gain_ordinary":
                for frame in range(begin,end):
                    for i in (frame,frame-vshift):
                        if i not in all_geometry:
                            all_geometry[i] = geometry.aligned_face(rgb[i],detector)
                        require(all_geometry[i] is not None,"Fixed authoring geometry unavailable")
                    altered[frame] = local_picture(rgb,all_geometry,frame,vshift,role=="light_gain_ordinary")
            first,last = begin*640,end*640
            if ashift:
                pcm[first:last] = original[first-ashift*640:last-ashift*640]
            elif role=="light_gain_ordinary":
                pcm[first:last] = np.rint(.75*original[first:last]).astype("<i2")
            wav = folder/f"{ordinal:02}-{role}.wav"
            save_pcm(wav,pcm)
            dest = wav.with_suffix(".mkv")
            delivery.encode(dest,altered,wav)
            _,actual,sound = video(dest)
            require(all(np.array_equal(x,y) for x,y in zip(altered,actual,strict=True)) and np.array_equal(sound,pcm),
                    "Authored native picture or soundtrack did not replay exactly")
            changed = [i for i,(x,y) in enumerate(zip(rgb,actual,strict=True)) if not np.array_equal(x,y)]
            expected = list(range(begin,end)) if vshift or role=="light_gain_ordinary" else []
            require(changed==expected and np.array_equal(sound[:first],original[:first]) and np.array_equal(sound[last:],original[last:]),
                    "Changes exceed the declared native interval")
            items.append({"ordinal":ordinal,"role":role,"path":str(dest),"sha256":sha(dest),"event":point,
                          "span":[begin,end],"visual_shift":vshift,"audio_shift":ashift,
                          "changed_rgb_frames":len(changed),"changed_pcm":int(np.count_nonzero(sound!=original))})
        print("authored held group",ordinal,flush=True)
    require(len(items)==24,"The complete held comparison is required")
    write(out/"media.json",{"freeze_sha256":sha(out/"freeze.json"),"cases":items})


def signed_correct(result,expected):
    if result["status"]!="timing_difference" or not expected:
        return False
    return abs(Fraction(result["picture_minus_audio_shift_s"])-Fraction(expected,25))<=Fraction(2,25)


def prior_correct(flag,span,expected):
    return bool(expected and span and prior_av.qualifies(flag,span,-expected))


def score(out):
    freeze = lock(out)
    media = read(out/"media.json")
    require(media["freeze_sha256"]==sha(out/"freeze.json"),"Frozen authoring differs")
    fit = read(previous.GRID_TIMING/"head.json")
    cv2.setNumThreads(1)
    detector = load_yunet(geometry.WEIGHT)
    module,task = neural.mp_task()
    originals,results = {},[]
    try:
        for row in media["cases"]:
            path = Path(row["path"])
            require(sha(path)==row["sha256"],"Held file hash differs")
            table,frames,pcm = video(path)
            observed = motion(module,task,frames)
            ordinal,anchor = row["ordinal"],row["event"]
            if row["role"]=="original":
                originals[ordinal] = observed,pcm
            ref,ref_pcm = originals[ordinal]
            proposed = candidate.compare(ref,observed,ref_pcm,pcm,anchor)
            expected = row["visual_shift"]-row["audio_shift"]
            if row["span"]:
                observed_event = closure_review.best_event(ref,observed,anchor,*row["span"]) if row["visual_shift"] else None
            else:
                observed_event = None
            raw,grid,faces,patches,_,_ = previous.observe(path,row["sha256"],detector)
            require(np.array_equal(raw,pcm),"Current timing received different audio")
            baseline = prior_av.baseline.current(faces,patches,pcm.astype(np.float32)/32768,grid)
            comparison = prior_av.candidate(prior_av.visual_features(patches,grid.ticks),*prior_av.acoustic_features(pcm),fit)
            av_flag = comparison["best"] if comparison["best"] is not None and comparison["best"]["score"]>freeze["prior_timing_point"] else None
            independently_confirmed = (not row["visual_shift"] or bool(observed_event and abs(
                observed_event["difference_from_source_frames"]-row["visual_shift"])<=1))
            # Independently bind the selected waveform and native picture before presenting a signed result.
            if proposed["status"]!="unassessed":
                for label,signal in (("source",ref_pcm),("submitted",pcm)):
                    a,b = proposed["sound"][f"{label}_samples"]
                    require(audio.sha(signal[a:b].tobytes())==proposed["sound"][f"{label}_pcm_sha256"],"Audio evidence hash differs")
                i = proposed["visual"]["submitted_frame"]
                require(table.pts[i]*table.tb==Fraction(i,25),"Observed event differs from native picture clock")
                proposed["native_frame_sha256"] = audio.sha(frames[i].tobytes())
            results.append({"ordinal":ordinal,"role":row["role"],"expected_picture_minus_audio_ticks":expected,
                "independent_visual_closure":observed_event,"independently_confirmed":independently_confirmed,
                "proposal":proposed,"correct":independently_confirmed and signed_correct(proposed,expected),
                "current_timing":{"measured":baseline["measured_windows"],"flags":baseline["ranges"],
                                  "correct":independently_confirmed and any(
                                      prior_correct(flag,row["span"],expected) for flag in baseline["ranges"])},
                "prior_av":{"measured":comparison["measured"],"flag":av_flag,"correct":independently_confirmed and bool(
                    av_flag and prior_correct(av_flag,row["span"],expected))}})
            print(row["role"],proposed["status"],proposed.get("picture_minus_audio_shift_s"),flush=True)
    finally:
        task.close()
    by_role = {}
    for role in ROLES:
        batch = [r for r in results if r["role"]==role]
        by_role[role] = {"views":len(batch),"measured":sum(r["proposal"]["status"]!="unassessed" for r in batch),
           "flags":sum(r["proposal"]["status"]=="timing_difference" for r in batch),"correct":sum(r["correct"] for r in batch),
           "current_flags":sum(bool(r["current_timing"]["flags"]) for r in batch),
           "prior_flags":sum(bool(r["prior_av"]["flag"]) for r in batch)}
    gates = {"correct":sum(r["correct"] for r in results)>=8,
             "groups":len({r["ordinal"] for r in results if r["correct"]})>=3,
             "controls":not any(r["proposal"]["status"]=="timing_difference" for r in results if not r["expected_picture_minus_audio_ticks"]),
             "coverage":sum(r["proposal"]["status"]!="unassessed" for r in results)>=18}
    write(out/"readout.json",{"results":results,"summary":by_role,"gate":gates,"passes":all(gates.values()),
                              "freeze_sha256":sha(out/"freeze.json"),"media_sha256":sha(out/"media.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("discovery","freeze","author","score"))
    parser.add_argument("qualification",type=Path)
    parser.add_argument("study",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.phase=="discovery":
        discovery(args.study,args.output)
    elif args.phase=="freeze":
        frozen(args.qualification,args.study,args.output)
    elif args.phase=="author":
        author(args.output)
    else:
        score(args.output)


if __name__ == "__main__":
    main()
