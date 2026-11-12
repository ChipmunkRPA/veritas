"""Verify long room-speech graph inputs, edit support and operating comparisons."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from review_audio_spectra_context_transfer import RATE, digest, ranges, read, require, useful

ARMS = ("current_two_second", "spectra_max", "spectra_edge_contrast")
PROVIDER = ("current", "spectra")


def statistics(rows, arm, point):
    predicted = [r for r in rows if r["checks"][arm]["score"] is not None and r["checks"][arm]["score"] > point]
    return len({r["group"] for r in predicted if r["truth"] is None}), sum(r["checks"][arm]["located"] for r in predicted)


def local_overlap(rows, annotations, digest_expected):
    with annotations.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == digest_expected, "AMI annotation release differs")
    output = Counter()
    with ZipFile(annotations) as archive:
        names = {r["background"]["room"] for r in rows}
        metadata = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        groups = [{p.get("global_name") for p in room if p.get("global_name")}
                  for room in metadata.iter() if room.get("observation") in names]
        require(len(names) == len(groups) and all(len(group) == 4 for group in groups)
                and len(set().union(*groups)) == 4*len(names), "Room participants overlap")
        for r in rows:
            if r["truth"] is None:
                continue
            metadata = r["background"]
            start, end = ((metadata["capture_start_sample"]+x)//160 for x in r["truth"])
            words = Counter()
            for speaker in "ABCD":
                track = ElementTree.fromstring(archive.read(f"words/{metadata['room']}.{speaker}.words.xml"))
                selected = set()
                for word in track:
                    first, last = word.get("starttime"), word.get("endtime")
                    if word.tag == "w" and not word.get("punc") and first is not None and last is not None:
                        selected.update(range(max(start, round(float(first)*100)), min(end, round(float(last)*100))))
                words.update(selected)
            output["edits_with_any_annotated_interference"] += bool(words)
            output["edits_with_concurrent_annotated_interferers"] += any(value >= 2 for value in words.values())
            output["edit_milliseconds_with_any_annotated_interference"] += 10*len(words)
            output["edit_milliseconds_with_concurrent_annotated_interferers"] += 10*sum(value >= 2 for value in words.values())
    return dict(output)


def verify(run, annotations):
    result = read(run / "result.json")
    for name, pin in result["commitments"].items():
        require(digest((run / f"{name}.json").read_bytes()) == pin, f"Changed {name} receipt")
    frozen, roster = (read(run / f"{name}.json") for name in ("freeze", "roster"))
    require(frozen["roster_sha256"] == result["commitments"]["roster"], "Freeze is disconnected from source roster")
    points = read(run / "points.json")["points"]
    held = {r["id"]: r for r in read(run / "decisions.json")}
    require(len(held) == 29 and len(roster) == 43, "Unexpected source denominator")
    ancestry = {}
    for role in ("selection", "confirmation"):
        cases = [row for row in roster if row["role"] == role]
        ancestry[role] = {row[field] for row in cases for field in ("host_group", "ordinary_group", "donor_group") if row.get(field)}
    require(not ancestry["selection"] & ancestry["confirmation"] and Counter(
        row["family"] for row in held.values() if row["truth"] is not None) == {"clone": 3, "expressive": 3, "stock": 3},
        "Foreground ancestry or generator cohort differs")
    compare, siblings = defaultdict(list), defaultdict(list)
    graph, admitted = 0, 0
    for role in ("selection", "confirmation"):
        providers = [read(run / f"{role}-{arm}.json") for arm in PROVIDER]
        require(all(obj["freeze_sha256"] == result["commitments"]["freeze"] for obj in providers), "Score freeze changed")
        selected = [r for r in roster if r["role"] == role]
        for row, current, spectra in zip(selected, *(obj["rows"] for obj in providers), strict=True):
            path = run / "pcm" / f"{row['id']}.pcm"
            raw = path.read_bytes()
            require(digest(raw) == row["pcm_sha256"] and len(raw) == 2*row["background"]["samples"]
                    and all((obj["id"], obj["pcm_sha256"]) == (row["id"], row["pcm_sha256"]) for obj in (current, spectra)),
                    "Exact source input differs")
            graph += len(spectra["windows"])
            observations = ranges(raw, spectra["windows"])
            expected = range(RATE, len(raw)//2-2*RATE+1, 2*RATE)
            require([w["start"] for w in current["windows"]] == list(expected), "Current context clocks differ")
            for w in current["windows"]:
                require(w["end"]-w["start"] == 2*RATE and digest(raw[2*w["start"]:2*w["end"]]) == w["pcm_sha256"],
                        "Current context PCM differs")
            admitted += sum(w["score"] is not None for w in current["windows"])
            winner = max((w for w in current["windows"] if w["score"] is not None), key=lambda w: w["score"], default=None)
            observations[ARMS[0]] = (None, None) if winner is None else (winner["score"], [winner["start"], winner["end"]])
            siblings[row["group"]].append((row, raw))
            for arm, (score, span) in observations.items():
                if role == "selection":
                    require(row["truth"] is None, "Point selection contains an edited case")
                    if score is not None:
                        compare[arm].append(score)
                else:
                    stored = held[row["id"]]["checks"][arm]
                    require((score, span) == (stored["score"], stored["range"])
                            and stored["flag"] is (score is not None and score > points[arm])
                            and stored["located"] is (span is not None and useful(span, row["truth"])), "Decision range differs")
    for pairs in siblings.values():
        base, reference = pairs[0]
        if len(pairs) > 1:
            start, end = base["edit_range"]
            require(all(row["edit_range"] == [start, end] and raw[:2*start] == reference[:2*start]
                        and raw[2*end:] == reference[2*end:] for row, raw in pairs), "Matched edits changed audio outside the exact support")
    require({arm: max(values) for arm, values in compare.items()} == points, "Fixed points differ")
    for row in held.values():
        checked = row["checks"]
        flagged = [checked[name] for name in (ARMS[0], ARMS[2]) if checked[name]["flag"]]
        require(checked["incumbent_or_contrast"] == {"flag": bool(flagged), "located": any(x["located"] for x in flagged),
                                                   "range": [x["range"] for x in flagged]}, "Fixed union differs")
    for label, records in result["fixed"].items():
        subset = [r for r in held.values() if label in {"all", r["background"]["room"], f"{r['background']['samples']//RATE}s"}]
        for arm, saved in records.items():
            flags = [r for r in subset if r["checks"][arm]["flag"]]
            require(len({r["group"] for r in flags if r["truth"] is None}) == saved["genuine_false_groups"]
                    and sum(r["checks"][arm]["located"] for r in flags) == saved["useful"], "Fixed outcomes differ")
    for budget, records in result["equal_false_groups"].items():
        for arm, saved in records.items():
            group = defaultdict(list)
            for row in held.values():
                value = row["checks"][arm]["score"]
                if row["truth"] is None and value is not None:
                    group[row["group"]].append(value)
            cutoff = sorted((max(v) for v in group.values()), reverse=True)[int(budget)]
            require(statistics(held.values(), arm, cutoff) == (saved["genuine_false_groups"], saved["useful"]), "Equal-alarm results differ")
    support = local_overlap(roster, annotations, frozen["annotation_sha256"])
    print(f"Replayed 43 exact waveforms, {graph} graph inputs, {admitted} admitted current windows, matched edit support and decisions.")
    print(support)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "annotations"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    verify(args.run, args.annotations)
