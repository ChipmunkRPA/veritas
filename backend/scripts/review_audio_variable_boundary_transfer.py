"""Check fresh noisy speech sources, exact clocks and reported comparisons."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from review_audio_enrolled_target_transfer import pcm
from review_audio_original_pcm_boundary_transfer import RATE, counts, digest, read, require, sha, useful

ARMS = ("current", "boundary", "variable")
WIDTHS = (6400, 10240, 12800, 16640)


def source_windows(run, rows):
    groups = defaultdict(list)
    for row in rows:
        files = (row["id"], f"{row['id']}-foreground", f"{row['id']}-background")
        samples = [(run / "pcm" / f"{name}.pcm").read_bytes() for name in files]
        require(all(len(raw) == 192000 and digest(raw) == row[pin] for raw, pin in
                    zip(samples, ("pcm_sha256", "foreground_sha256", "background_sha256"), strict=True)), "Exact native source files differ")
        require(all(x == y+z for x, y, z in zip(*(pcm(raw) for raw in samples), strict=True)), "Exact input does not equal summed source components")
        groups[row["group"]].append((row, samples))
    cuts = defaultdict(list)
    hosts, other, donors, source_hashes, extracted = set(), set(), set(), set(), set()
    for items in groups.values():
        first, buffers = items[0]
        room = first["room"]
        origin = first["background"]["capture_start_sample"]
        require(not any(a < origin+96000 and origin < b for a, b in cuts[room]), "Room captures overlap")
        cuts[room].append((origin, origin+96000))
        if first["edit"] is None:
            require(len(items) == 1 and not any(buffers[1]), "Isolated room contains foreground")
            continue
        a, b = first["edit"]
        require({r["arm"] for r, _ in items} == {"ordinary_same_host", "ordinary_other_voice", "synthetic"}
                and len(items) == 3 and 0 <= a < b <= 96000 and b-a in WIDTHS, "Matched source episode differs")
        hosts.add(first["host"]["canonical_speaker"])
        host_range = first["host"]["host_range"]
        require((host_range["start_sample"], host_range["end_sample"]) == (64000, 160000), "Fresh host extends beyond retained post-prefix range")
        for row, data in items:
            arm, donor = row["arm"], row["donor"]
            require(row["edit"] == [a, b] and row["truth"] == ([a, b] if arm == "synthetic" else None) and row["background"] == first["background"],
                    "Source clock or matched background differs")
            require(buffers[2] == data[2] and all(x[:2*a] == y[:2*a] and x[2*b:] == y[2*b:] for x, y in zip(data[:2], buffers[:2], strict=True)),
                    "Matched files differ beyond the documented edit")
            segment = donor["cut"]
            require(segment["end_sample"]-segment["start_sample"] == b-a == segment["source_end_pts"]-segment["source_start_pts"]
                    and segment["time_base"] == "1/16000" and donor["pcm_sha256"] not in extracted, "Donor range or exact source reuse differs")
            extracted.add(donor["pcm_sha256"])
            if arm == "synthetic":
                require(donor["source_sha256"] not in source_hashes, "Repeated generated source file")
                source_hashes.add(donor["source_sha256"])
                donors.add(donor["canonical_speaker"])
            else:
                require(segment["start_sample"] >= 160000 and segment["end_sample"] <= 192000, "Prior genuine source prefix reused")
                if arm == "ordinary_other_voice":
                    other.add(donor["canonical_speaker"])
    require(len(hosts) == len(other) == len(donors) == 8 and not hosts & other, "Foreground speaker groups overlap")


def actual_overlap(rows, annotations):
    with ZipFile(annotations) as archive:
        overlaps, members = {}, set()
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        for room in sorted({r["room"] for r in rows}):
            roster = next(node for node in meetings.iter() if node.get("observation") == room)
            people = {p.get("global_name") for p in roster if p.get("global_name")}
            require(len(people) == 4 and not members & people, "Recorded room participants repeat")
            members.update(people)
            both = Counter()
            for channel in "ABCD":
                track = ElementTree.fromstring(archive.read(f"words/{room}.{channel}.words.xml"))
                present = set()
                for word in track:
                    start, stop = word.get("starttime"), word.get("endtime")
                    if word.tag == "w" and not word.get("punc") and start is not None and stop is not None:
                        present.update(range(round(float(start)*100), round(float(stop)*100)))
                both.update(present)
            overlaps[room] = {tick for tick, count in both.items() if count >= 2}
        exact = []
        for row in rows:
            if row["truth"] is not None:
                source = row["background"]["capture_start_sample"]
                start, end = ((source+n)//160 for n in row["truth"])
                measured = 10*sum(tick in overlaps[row["room"]] for tick in range(start, end))
                require(measured == row["background"]["edit_overlap_ms"] >= max(240, 5*(end-start)), "Exact background interference differs")
                exact.append(measured)
    return exact


def scored(run, rows):
    lock, measured, reported, decisions = (read(run / f"{name}.json") for name in ("freeze", "current", "result", "decisions"))
    require(all(sha(run / f"{name}.json") == pin for name, pin in reported["commitments"].items())
            and measured["freeze_sha256"] == sha(run / "freeze.json"), "Readback commitments differ")
    for row, observation, decision in zip(rows, measured["records"], decisions, strict=True):
        require(row == {k: v for k, v in decision.items() if k != "responses"}
                and (observation["id"], observation["pcm_sha256"]) == (row["id"], row["pcm_sha256"]), "Decision source binding differs")
        original = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
        require([w["start"] for w in observation["windows"]] == [RATE, 3*RATE], "Native window grid differs")
        for window in observation["windows"]:
            a, b = window["start"], window["end"]
            require(b-a == 2*RATE and digest(original[2*a:2*b]) == window["pcm_sha256"], "Native model window differs")
        for arm in ARMS[1:]:
            _, (a, b) = decision["responses"][arm]
            require(a % 1280 == 0 and 3840 <= a < b <= 96000-2880 and b-a in ((12800,) if arm == "boundary" else WIDTHS),
                    "Reported span exceeds its native candidate support")
    for arm in ARMS:
        require(counts(decisions, arm, lock["points"][arm]) == reported["fixed"][arm], "Fixed comparison differs")
        points = defaultdict(list)
        for row in decisions:
            value = row["responses"][arm][0]
            if row["truth"] is None and value is not None:
                points[row["group"]].append(value)
        ranked = sorted((max(v) for v in points.values()), reverse=True)
        for budget, table in reported["equal_false_groups"].items():
            require(counts(decisions, arm, ranked[int(budget)]) == table[arm], "Equal false group comparison differs")
        ordinary = {kind: sum(row["arm"] == kind and row["responses"][arm][0] is not None
                              and row["responses"][arm][0] > lock["points"][arm] and useful(row["responses"][arm][1], row["edit"])
                              for row in decisions) for kind in ("ordinary_same_host", "ordinary_other_voice")}
        require(ordinary == reported["fixed_located_genuine_splices"][arm], "Ordinary splice comparison differs")
    print({arm: counts(decisions, arm, float("-inf"))["useful_ranges"] for arm in ARMS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    roster = read(args.run / "roster.json")
    require(len(roster) == 26 and {r["background"]["room"] for r in roster} == {"ES2012b", "IS1001b"}, "Unexpected held roster")
    source_windows(args.run, roster)
    overlap = actual_overlap(roster, args.annotations)
    if not args.source_only:
        scored(args.run, roster)
    print(f"Verified 26 original mixes and exact clocks. Eight edited ranges have {min(overlap)}–{max(overlap)} ms of simultaneous background speech.")
