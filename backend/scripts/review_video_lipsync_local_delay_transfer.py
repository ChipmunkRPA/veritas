# SPDX-License-Identifier: Apache-2.0
"""Recompute local timing comparisons and exact media evidence without model calls."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio, source_ranges
from app.detectors.video.decode import decode_frames, probe_frames

NAMES = ("current", "short_aperture", "short_joint")
ROLES = ("before", "during", "after")
RATE, HOP = 16000, 640


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return list(map(json.loads, path.open()))


def source_pcm(source, digest):
    decoded = decode_audio(source, source_sha256=digest, max_seconds=8)
    records = source_ranges(decoded.mapping, 0, len(decoded.pcm))
    by_time = {}
    pcm = np.rint(decoded.pcm * 32768).astype("<i2")
    for row in records:
        a, b = row["source_start_pts"], row["source_end_pts"]
        for t, v in zip(range(a,b), pcm[row["start_sample"]:row["end_sample"]], strict=True):
            by_time.setdefault(t, int(v))
    first, end = min(by_time), max(by_time) + 1
    assert len(by_time) == end-first
    return np.array([by_time[t] for t in range(first,end)], dtype="<i2"), Fraction(first,RATE), len(pcm)-len(by_time)


def rebuilt(pcm, case, start, end, delta, fade):
    if case == "genuine":
        return pcm
    a, b = start * HOP, end * HOP
    changed = pcm.copy()
    if case == "gain":
        changed[a:b] = np.rint(.7 * pcm[a:b]).astype("<i2")
        return changed
    lag = delta * HOP * (1 if case.endswith("late") else -1)
    if case.startswith("global"):
        for i in range(len(changed)):
            src = i-lag
            changed[i] = pcm[src] if 0 <= src < len(pcm) else 0
        return changed
    for i in range(a,b):
        factor = min(i-a+1, b-i, fade) / fade
        changed[i] = round((1-factor) * int(pcm[i]) + factor * int(pcm[i-lag]))
    assert np.array_equal(changed[:a], pcm[:a]) and np.array_equal(changed[b:], pcm[b:])
    return changed


def overlaps(a, b):
    left,right=(Fraction(str(t)) for t in a)
    start,end=(Fraction(str(t)) for t in b)
    common=max(0,min(right,end)-max(left,start))
    union=max(right,end)-min(left,start)
    return common / union


def correct(row, truth, sign, point):
    delta = row["relative_lag_ticks"]
    return (delta is None or delta * sign > 0) and overlaps(row["extent_ticks"], truth) >= point


def bracket(trial, original, wave, origin, media):
    if (best := trial["best"]) is None:
        assert trial["evidence"] is None
        return 0, 0
    assert trial["compared_brackets"] > 0
    before,middle,after=best["lag_correlations"]
    assert all(len(row) == 13 and all(math.isfinite(value) and abs(value) <= 1.00001 for value in row)
               for row in (before,middle,after))
    def highest(signal):
        return max(range(len(signal)),key=signal.__getitem__)
    k=highest(middle)
    r=highest([min(a,b) for a,b in zip(before,after,strict=True)])
    assert abs(highest(before)-highest(after)) <= 1 and min(before[r],after[r],middle[k]) >= .15
    assert abs(k-r) >= 3 and k-r == best["relative_lag_ticks"]
    assert (r-6,k-6) == (best["flank_lag_ticks"],best["changed_lag_ticks"])
    flank=min(before[r]-before[k],after[r]-after[k])
    assert flank >= .015 and math.isclose(min(middle[k]-middle[r],2*flank),best["score"],abs_tol=1e-12)
    shifts={kind:best[f"{kind}_lag_ticks"] for kind in ("flank","changed")}
    combined_frames = 0
    for role in ROLES:
        lo,hi=best["roles"][role]
        saved=trial["evidence"][role]
        assert hi-lo in (12,16,20)
        assert list(map(Fraction,saved["video_ticks_native_time"])) == [origin+Fraction(lo,25),origin+Fraction(hi,25)]
        assert Fraction(saved["native_video_time_base"]) == original.tb
        assert [row["tick"] for row in saved["frames"]] == list(range(lo,hi))
        for row in saved["frames"]:
            i,t=row["frame_index"],row["tick"]
            assert row["pts"] == original.pts[i]
            assert abs(row["pts"]*original.tb-(origin+Fraction(t,25))) <= Fraction(1,25)
            if i in media:
                assert media[i] == row, "Saved RGB or crop identity differs for a shared picture"
            else:
                media[i] = {**row, "tick": t}
            combined_frames += 1
        for name,shift in shifts.items():
            audio=saved["audio"][name]
            a,b=(lo+shift)*HOP,(hi+shift)*HOP
            assert 0<=a<b<=len(wave) and audio["samples"] == [a,b]
            assert list(map(Fraction,audio["native_time"])) == [origin+Fraction(a,RATE),origin+Fraction(b,RATE)]
            assert audio["pcm_sha256"] == sha(wave[a:b].tobytes())
    return combined_frames, 6


def verify_pictures(source, table, wanted):
    seen = set()
    for frame in decode_frames(source,table,sorted(wanted),timeout_s=25):
        row=wanted[frame.index]
        assert frame.pts_verified and frame.pts == row["pts"] and sha(frame.rgb.tobytes()) == row["decoded_rgb_sha256"]
        x0,y0,x1,y1=row["mouth_box_xyxy"]
        assert 0<=x0<x1<=table.width and 0<=y0<y1<=table.height
        patch=cv2.resize(cv2.cvtColor(frame.rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY),(32,24),interpolation=cv2.INTER_AREA).astype("<f4")
        assert sha(patch.tobytes()) == row["mouth_patch_sha256"]
        seen.add(frame.index)
    assert seen == set(wanted)
    return len(seen)


def audit(folder):
    freeze,cal,aggregate=(read(folder / f"{name}.json") for name in ("freeze","calibration","aggregate"))
    dev,held=(jsonl(folder / f"{name}.jsonl") for name in ("development","held"))
    assert aggregate["freeze_sha256"] == sha((folder/"freeze.json").read_bytes())
    assert cal["development_sha256"] == sha((folder/"development.jsonl").read_bytes())
    assert aggregate["held_journal_sha256"] == sha((folder/"held.jsonl").read_bytes())
    for arm in NAMES[1:]:
        negative=[c["arms"][arm]["best"]["score"] for row in dev for c in row["cases"]
                  if c["edit_extent_ticks"] is None and c["arms"][arm]["best"] is not None]
        assert cal["points"][arm] == max(.1,max(negative,default=0)+.025)
    totals = {arm: Counter() for arm in NAMES}
    unique_pictures=frame_records=audio_ranges=inputs=0
    for row in held:
        pin=next(r for r in freeze["rows"] if r["ordinal"]==row["group"])
        source=folder/pin["file"]
        assert row["source_sha256"] == pin["sha256"] == sha(source.read_bytes())
        table=probe_frames(source,timeout_s=20)
        pcm,origin,overlaps_removed=source_pcm(source,pin["sha256"])
        assert origin==Fraction(row["origin_seconds"]) and overlaps_removed==row["removed_overlapping_samples"]
        (start,end)=next(case["edit_extent_ticks"] for case in row["cases"] if case["case"]=="local_late")
        assert end-start==freeze["edit_ticks"]
        media = {}
        for case in row["cases"]:
            label=case["case"]
            wave=rebuilt(pcm,label,start,end,freeze["delta_ticks"],freeze["fade_samples"])
            assert sha(wave.tobytes()) == case["input_pcm_sha256"]
            inputs+=1
            truth, sign=case["edit_extent_ticks"],case["known_relative_lag_ticks"]
            for name in NAMES:
                if name=="current":
                    ranges=case["current"]["ranges"]
                    measured=bool(case["current"]["measured_windows"])
                else:
                    trial=case["arms"][name]
                    measured=bool(trial["compared_brackets"])
                    selected=trial["best"]
                    num,ar=bracket(trial,table,wave,origin,media)
                    frame_records+=num
                    audio_ranges+=ar
                    ranges=[selected] if selected and selected["score"]>cal["points"][name] else []
                if not measured:
                    key="unassessed"
                elif not ranges:
                    key="measured_no_range"
                elif not truth:
                    key="false_range"
                else:
                    key="correct_range" if any(correct(x,truth,sign,Fraction(freeze["minimum_exact_iou"])) for x in ranges) else "wrong_range"
                totals[name][key]+=1
        unique_pictures+=verify_pictures(source,table,media)
    assert totals == aggregate["totals"]
    return {"status":"verified","held_pcm_inputs_rebuilt":inputs,"native_mouth_picture_records":frame_records,
            "distinct_exact_source_pictures_and_crops_replayed":unique_pictures,
            "bound_pcm_ranges":audio_ranges,"aggregate_sha256":sha((folder/"aggregate.json").read_bytes())}


if __name__ == "__main__":
    root=Path(sys.argv[1])
    result=audit(root)
    with (root/"independent-audit.json").open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps(result,sort_keys=True))
