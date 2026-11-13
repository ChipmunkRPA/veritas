# SPDX-License-Identifier: Apache-2.0
"""Reobserve a short audiovisual release in the retained GRID speech-conditioned edits."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import grid_mouth_motion_readout_v158 as archive
from scripts.grid_fresh_release_labels import audio_only
from scripts.video_common_sources_v89 import read, require, sha, write

FIRST, LAST = 27, 47
TICK = 640
ROLES = (
    "original",
    "musetalk_source_time",
    "musetalk_earlier_320ms",
    "resample_musetalk",
    "brightness_musetalk",
    "genuine_frame_retime",
    "audio_delay_320ms",
)
SPEC = {
    "speech": "Select a unique salient original PCM burst in the fixed usable edit neighborhood with the existing audio-only transient method. Candidate span is frame times 29 through 34 inclusive and 0.65 minimum salience margin. No transcript, renderer target or mouth measurement enters acoustic selection.",
    "visual": "Mute the clip and find closure/reopening events from saved observed inner lip gaps, using the median eye-normalized aperture. Require <=.075 then <=.055 then >.055, and >=.032 growth over the next four pictures. Report all candidates. Source and output must each have one visual event in the fixed usable edit neighborhood and matched events may differ by no more than one frame.",
    "binding": "Require original event and delivered event each within 80 ms of the independently measured acoustic burst midpoint. Require both ordinary source-time region deliveries retain the same event. Verified exact original PCM and native pictures through the existing v149/v157 readbacks.",
    "wrong_event_controls": "Require an exact native earlier-picture retime to move the detected event by eight frames within one-frame error and not admit a different source-time event. Independently search the shifted soundtrack for its burst in the corresponding later range; it must move 320 ms within 15 ms and fail alignment with unchanged pictures.",
    "scope": "One local stop-like event, not independently labeled phoneme identity or full-clip/perceived synchrony. Previously exposed pilot sources, separately frozen source-timing qualification. No detector score used.",
}


def inputs():
    _, exact = archive.sealed_plan()
    return {
        "protocol": SPEC,
        "method_sha256": sha(Path(__file__)),
        "pilot_source_binding": exact,
        "historical_readout_sha256": sha(archive.PUBLIC),
        "acoustic_rule_sha256": sha(Path(__file__).with_name("grid_fresh_release_labels.py")),
    }


def decode(case):
    require(
        case["review"]["pcm_exact"] and case["review"]["audio_origin_exact"],
        "Expected verified native source sound",
    )
    path = Path(case["file"])
    expected = case["expected"]
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ],
        timeout=20,
    )
    require(
        len(raw) == expected["audio_samples"] * 2
        and hashlib.sha256(raw).hexdigest() == expected["pcm_s16_sha256"],
        "Native original PCM differs from archived construction",
    )
    return np.frombuffer(raw, dtype="<i2")


def acoustic(pcm, delay_frames=0):
    candidates = [
        r
        for r in audio_only(pcm, (FIRST + LAST) * TICK // 2 + delay_frames * TICK)["nearby"]
        if (FIRST + 2 + delay_frames) * TICK <= r["center_sample"] <= (LAST - 13 + delay_frames) * TICK
    ]
    if not candidates:
        return {"status": "no_unique_burst"}
    if len(candidates) > 1 and candidates[0]["salience"] - candidates[1]["salience"] < 0.65:
        return {"status": "ambiguous_bursts"}
    return {"status": "observed", "sample": candidates[0]["center_sample"]}


def visual(by_frame):
    found = []
    for i in range(FIRST + 1, LAST - 2):
        near = [by_frame.get(j) for j in range(i - 1, i + 5)]
        if any(v is None for v in near):
            continue
        if near[0] <= 0.075 and near[1] <= 0.055 < near[2] and max(near[2:]) - near[1] >= 0.032:
            found.append(i)
    return found


def supported(candidates, acoustic_record):
    local = [i for i in candidates if FIRST + 2 <= i <= LAST - 6]
    return (
        len(local) == 1
        and acoustic_record["status"] == "observed"
        and abs((local[0] * TICK + TICK // 2) - acoustic_record["sample"]) <= 1280
    )


def qualify(source):
    cases, observations, _, audit = archive.observation_metadata(source)
    by_name = {r["condition"]: r for r in cases}
    descriptors = defaultdict(dict)
    for row in observations:
        if not row["repeat"] and row["status"] == "geometry_available":
            index = row["frame_index"]
            require(
                Fraction(row["source_time_seconds"]) == Fraction(index, 25),
                "Unexpected native observer clock",
            )
            descriptors[row["case_number"]][index] = float(
                np.median([float.fromhex(v) for v in row["descriptor_hex"][:3]])
            )
    events = {name: visual(descriptors[by_name[name]["case_number"]]) for name in ROLES}
    genuine_pcm = decode(by_name["original"])
    shifted_pcm = decode(by_name["audio_delay_320ms"])
    sound, shifted = acoustic(genuine_pcm), acoustic(shifted_pcm, 8)
    for name in ROLES[:-1]:
        require(
            by_name[name]["expected"]["pcm_s16_sha256"] == by_name["original"]["expected"]["pcm_s16_sha256"],
            "Visual delivery changes the frozen soundtrack",
        )
    original = events["original"]
    good_original = supported(original, sound)
    close = next((i for i in original if FIRST + 2 <= i <= LAST - 6), None) if good_original else None
    ordinary = (
        all(
            supported(events[name], sound) and any(abs(i - close) <= 1 for i in events[name])
            for name in ("resample_musetalk", "brightness_musetalk")
        )
        if good_original
        else False
    )
    retime = (
        good_original
        and any(abs(i - close - 8) <= 1 for i in events["genuine_frame_retime"])
        and not supported(events["genuine_frame_retime"], sound)
    )
    shifted_control = (
        good_original
        and shifted["status"] == "observed"
        and abs(shifted["sample"] - sound["sample"] - 8 * TICK) <= 240
        and not supported(events["audio_delay_320ms"], shifted)
    )
    candidate = (
        good_original
        and supported(events["musetalk_source_time"], sound)
        and any(abs(i - close) <= 1 for i in events["musetalk_source_time"])
    )
    earlier_is_control = (
        good_original
        and supported(events["musetalk_earlier_320ms"], sound)
        and any(abs(i - close) <= 1 for i in events["musetalk_earlier_320ms"])
    )
    return {
        "group": source["component_index"],
        "native_clip": source["ordinal"],
        "source_commitments": audit,
        "acoustic": sound,
        "shifted_acoustic": shifted,
        "visual_last_closed": events,
        "genuine_audio_visual_event": good_original,
        "ordinary_preserved_event": ordinary,
        "picture_retime_control": retime,
        "soundtrack_delay_control": shifted_control,
        "redraw_retained_event": candidate,
        "earlier_neural_also_matches": earlier_is_control,
        "qualified": bool(ordinary and retime and shifted_control and candidate and not earlier_is_control),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "run"))
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    locked = inputs()
    freeze = args.folder / "freeze.json"
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        write(freeze, locked)
        return
    require(locked == read(freeze), "Changed short-event protocol or archived inputs")
    sources, _ = archive.sealed_plan()
    require(
        len(sources) == 16 and len({s["component_index"] for s in sources}) == 8,
        "Expected the complete 16-case roster",
    )
    rows = [qualify(source) for source in sources]
    counters = Counter()
    for row in rows:
        counters.update({k: v for k, v in row.items() if type(v) is bool})
        counters[row["acoustic"]["status"]] += 1
    write(args.folder / "held.json", {"freeze_sha256": sha(freeze), "summary": counters, "rows": rows})
    print(dict(counters))


if __name__ == "__main__":
    main()
