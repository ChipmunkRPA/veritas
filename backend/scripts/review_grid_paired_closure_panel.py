# SPDX-License-Identifier: Apache-2.0
"""Independently replay timing arithmetic and local media evidence."""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import grid_paired_closure_panel as study
from scripts import grid_paired_closure_timing as method
from scripts.video_common_sources_v89 import read, require, sha, write


def replay(folder):
    freeze = study.lock(folder)
    media = read(folder/"media.json")
    result = read(folder/"readout.json")
    require(result["freeze_sha256"]==media["freeze_sha256"]==sha(folder/"freeze.json")
            and result["media_sha256"]==sha(folder/"media.json"),"Frozen native inputs differ")
    held_cases, scores = media["cases"],result["results"]
    require(len(held_cases)==len(scores)==24,"Need all held views")
    originals = {}
    native_count, pcm_count, verified_slices, measured_frames = 0,0,0,0
    example = None
    outcomes = {}
    for clip,observation in zip(held_cases,scores,strict=True):
        role,ordinal = clip["role"],clip["ordinal"]
        path = Path(clip["path"])
        require((role,ordinal)==(observation["role"],observation["ordinal"])
                and sha(path)==clip["sha256"],"Native media identity differs")
        table,pictures,pcm = study.video(path)
        native_count += len(pictures)
        pcm_count += len(pcm)
        if role=="original":
            originals[ordinal] = pictures,pcm
        ref,source_pcm = originals[ordinal]
        if clip["span"]:
            a,b = clip["span"]
            want = list(range(a,b)) if clip["visual_shift"] or role=="light_gain_ordinary" else []
            require([i for i,(l,r) in enumerate(zip(ref,pictures,strict=True)) if not np.array_equal(l,r)]==want
                    and len(want)==clip["changed_rgb_frames"],"Authored pictures changed outside their frozen interval")
            expected = source_pcm.copy()
            start,stop = a*640,b*640
            shift = clip["audio_shift"]*640
            if shift:
                expected[start:stop] = source_pcm[start-shift:stop-shift]
            if role=="light_gain_ordinary":
                expected[start:stop] = np.rint(.75*source_pcm[start:stop]).astype("<i2")
            require(np.array_equal(expected,pcm) and int(np.count_nonzero(pcm!=source_pcm))==clip["changed_pcm"],
                    "Exact source audio edit did not reproduce")
        known = clip["visual_shift"]-clip["audio_shift"]
        require(observation["expected_picture_minus_audio_ticks"]==known,"Signed test truth differs")
        proposed = observation["proposal"]
        if proposed["status"]=="unassessed":
            require(proposed["visual"] is None or proposed["sound"] is None,"Unavailable cue has no missing input")
            correct = False
        else:
            visual,sound = proposed["visual"],proposed["sound"]
            source_picture,target_picture = visual["source_frame"],visual["submitted_frame"]
            a0,b0 = sound["source_samples"]
            a1,b1 = sound["submitted_samples"]
            require(source_picture==clip["event"] and visual["offset_pictures"]==target_picture-source_picture
                    and table.pts[target_picture]*table.tb==Fraction(target_picture,25)
                    and hashlib.sha256(pictures[target_picture].tobytes()).hexdigest()==proposed["native_frame_sha256"],
                    "Reported mouth event differs from the native picture")
            require(0<=a0<b0<=len(source_pcm) and 0<=a1<b1<=len(pcm) and b0-a0==b1-a1
                    and sound["offset_samples"]==a1-a0,"Signed audio event has invalid source samples")
            require(all(hashlib.sha256(signal[a:b].tobytes()).hexdigest()==sound[f"{label}_pcm_sha256"]
                        for label,signal,a,b in (("source",source_pcm,a0,b0),("submitted",pcm,a1,b1))),
                    "Reported acoustic excerpts differ from the native waveform")
            verified_slices += 2
            measured_frames += 1
            time_picture = Fraction(target_picture,25)
            time_audio = Fraction(source_picture,25)+Fraction(a1-a0,16000)
            offset = time_picture-time_audio
            require(Fraction(proposed["picture_minus_audio_shift_s"])==offset
                    and list(map(Fraction,proposed["event_time_range_s"]))==sorted((time_picture,time_audio))
                    and (abs(offset)>=Fraction(method.SPEC["review_offset_minimum_seconds"]))==(proposed["status"]=="timing_difference"),
                    "Timing evidence arithmetic does not replay")
            correct = bool(known and abs(offset-Fraction(known,25))<=Fraction(2,25)
                           and proposed["status"]=="timing_difference" and observation["independently_confirmed"])
            if correct and example is None:
                example = (clip,proposed,ref,pictures,source_pcm,pcm)
        require(correct==observation["correct"],"The measured result differs from its frozen decision")
        outcomes[role] = outcomes.get(role,0)+correct
    require(all(outcomes[role]==result["summary"][role]["correct"] for role in outcomes),"Published counts disagree")
    receipt = {"status":"verified","frozen_held_groups":len(freeze["selection"]),"exact_native_pictures":native_count,
               "verified_pcm_samples":pcm_count,"reopened_event_pcm_slices":verified_slices,
               "verified_event_picture_anchors":measured_frames,"readout_sha256":sha(folder/"readout.json")}
    if example:
        item,found,ref,pictures,source_pcm,pcm = example
        out = folder/"private-example"
        out.mkdir(mode=0o700)
        source,target = found["visual"]["source_frame"],found["visual"]["submitted_frame"]
        for name,frame in (("reference-closure",ref[source]),("submitted-before",pictures[min(source,target)-1]),
                           ("submitted-closure",pictures[target]),("submitted-after",pictures[max(source,target)+1])):
            Image.fromarray(frame).save(out/f"{name}.png")
        for label,signal in (("source",source_pcm),("submitted",pcm)):
            a,b = found["sound"][f"{label}_samples"]
            study.save_pcm(out/f"{label}-speech.wav",signal[a:b])
        receipt["anonymous_example"] = {"role":item["role"],"source_picture_s":str(Fraction(source,25)),
            "submitted_picture_s":str(Fraction(target,25)),"range_s":found["event_time_range_s"],
            "signed_picture_minus_audio_s":found["picture_minus_audio_shift_s"],
            "aligned_speech_samples":found["sound"]["source_samples"][1]-found["sound"]["source_samples"][0]}
    write(folder/"independent-replay.json",receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    replay(parser.parse_args().folder)


if __name__ == "__main__":
    main()
