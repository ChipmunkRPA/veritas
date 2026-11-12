"""Verify exact native and graph input clocks, edge contrasts and decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from array import array
from pathlib import Path

RAW, EDGE = "spectra_max", "spectra_edge_contrast"
RATE, WIDTH = 16_000, 48_000
F32 = struct.Struct("<f")


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rounded(value):
    return F32.unpack(F32.pack(value))[0]


def input_hash(raw):
    pcm = array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    scale = 32768.
    coefficient = rounded(.97)
    audio = array("f", [pcm[0]/scale])
    audio.extend(rounded(value/scale - rounded(coefficient*previous/scale)) for previous, value in zip(pcm, pcm[1:]))
    audio.extend(audio[:64_600-len(audio)])
    if sys.byteorder != "little":
        audio.byteswap()
    require(len(audio) == 64_600, "Unexpected model input length")
    return digest(audio.tobytes())


def ranges(raw, observed):
    starts = range(0, len(raw)//2-WIDTH+1, RATE)
    require([r["start"] for r in observed] == list(starts), "Unexpected context clocks")
    for window in observed:
        start, end = window["start"], window["end"]
        require(end-start == WIDTH, "Context is not three seconds")
        selected = raw[2*start:2*end]
        require(digest(selected) == window["pcm_sha256"] and input_hash(selected) == window["input_sha256"],
                "Selected PCM or exact model input changed")
    scores = [item["score"] for item in observed]
    i = scores.index(max(scores))
    gaps = [scores[i]-scores[i+1] for i in range(len(scores)-1)]
    j = max(range(len(gaps)), key=lambda i: abs(gaps[i]))
    shift = 0 if gaps[j] >= 0 else WIDTH
    return {RAW: (scores[i], [i*RATE, i*RATE+WIDTH]), EDGE: (abs(gaps[j]), [j*RATE+shift, (j+1)*RATE+shift])}


def useful(span, truth):
    if truth is None:
        return False
    common = len(range(max(span[0], truth[0]), min(span[1], truth[1])))
    return common/(span[1]-span[0]+truth[1]-truth[0]-common) >= .30 and common/(truth[1]-truth[0]) >= .50


def check_tables(rows, outcomes):
    for panel, result in outcomes.items():
        group = [r for r in rows if r["panel"] == panel]
        for arm, values in result["fixed"].items():
            flagged = [row for row in group if row["checks"][arm]["flag"]]
            require(sum(row["truth"] is None for row in flagged) == values["genuine_false_flags"]
                    and sum(row["checks"][arm]["located"] for row in flagged) == values["useful"], "Fixed comparison table changed")
        for budget, arms in result["equal_false_groups"].items():
            for arm, values in arms.items():
                negatives = {}
                for row in group:
                    if row["truth"] is None:
                        value = row["checks"][arm]["score"]
                        negatives[row["group"]] = max(value, negatives.get(row["group"], float("-inf")))
                cut = sorted(negatives.values())[-1-int(budget)]
                flagged = [row for row in group if row["checks"][arm]["score"] > cut]
                require(len({row["group"] for row in flagged if row["truth"] is None}) == values["genuine_false_groups"]
                        and sum(row["checks"][arm]["located"] for row in flagged) == values["useful"], "Equal-alarm table changed")


def verify(run, parent, meeting):
    result = read(run / "result.json")
    for name, pin in result["commitments"].items():
        require(digest((run / f"{name}.json").read_bytes()) == pin, "Changed analysis receipt")
    for folder, key in ((parent, "parent"), (meeting, "meeting")):
        for name, pin in result[key].items():
            require(digest((folder / name).read_bytes()) == pin, "Changed source parent commitment")
    source = {row["id"]: row for row in read(parent / "inputs.json")["rows"]}
    rooms = {row["id"]: row for row in read(meeting / "roster.json")}
    selection, held = (read(run / f"{name}.json") for name in ("selection", "confirmation"))
    require(selection["freeze_sha256"] == result["commitments"]["freeze"]
            and held["selection_sha256"] == result["commitments"]["selection"], "Held score role changed")
    require({r["id"] for r in selection["controls"]} == {r["id"] for r in source.values()
            if r["role"] == "confidence_cal" and r["condition"] == "speech_minus5db" and not r["label"]}, "Point control roster differs")
    measured = []
    for record in selection["controls"]:
        old = source[record["id"]]
        raw = (parent / old["path"]).read_bytes()
        require(digest(raw) == old["pcm_sha256"] == record["pcm_sha256"], "Point source differs")
        measured.append(ranges(raw, record["windows"]))
    points = {arm: max(view[arm][0] for view in measured) for arm in (RAW, EDGE)}
    require(points == selection["points"], "Point arithmetic differs")
    for record in held["rows"]:
        identifier = record["id"]
        if record["panel"] == "matched":
            old = source[identifier]
            require(old["role"] == "confirmation" and old["condition"] == "speech_minus5db", "Unexpected matched case")
            raw = (parent / old["path"]).read_bytes()
            expected = old["pcm_sha256"]
        else:
            require(record["panel"] == "ami", "Unknown case role")
            old = rooms[identifier]
            require(old["panel"] == "confirmation" and old["condition"] == "room", "Unexpected AMI case")
            raw = (meeting / "pcm" / f"{identifier:03}.pcm").read_bytes()
            expected = old["source_pcm_sha256"]
        require(digest(raw) == expected == record["pcm_sha256"] and old["truth"] == record["truth"], "Held source changed")
        for arm, (score, span) in ranges(raw, record["windows"]).items():
            check = record["checks"][arm]
            require(check == {"score": score, "range": span, "flag": score > points[arm], "located": useful(span, record["truth"])},
                    "Fixed source decision or location changed")
    total = sum(len(row["windows"]) for row in selection["controls"]+held["rows"])
    require(total == result["forwards"], "Graph input denominator changed")
    check_tables(held["rows"], result["outcomes"])
    print(f"Independently reproduced {total} graph inputs, native context ranges, exclusive edges and both operating comparisons.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "parent", "meeting"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.run, args.parent, args.meeting)
