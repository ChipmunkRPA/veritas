"""Independently audit speaker enrollment, source support and acoustic outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from sys import byteorder
from xml.etree import ElementTree
from zipfile import ZipFile

from review_audio_spectra_context_transfer import RATE, digest, read, require

SOURCE_SHA256 = "8b667675a803bd03851c27251e08a63b040b2bd29474462f9f0a4e3af9108ea6"
METRICS = ("snr_gain_db", "retained_target_db")
PARTS = ("whole", "edit", "outside")
ARMS = ("correct", "sham")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pcm(raw):
    values = array("h", raw)
    if byteorder == "big":
        values.byteswap()
    return values


def check_speaker_roster(roster, frozen, sources):
    require(sha(sources) == SOURCE_SHA256, "Unrecognized source qualifications")
    source = {row["raw_sha256"]: row for row in read(sources)["foregrounds"] if row["kind"] == "genuine"}
    groups = defaultdict(set)
    references = set(frozen["generic"])
    for key in references:
        row = source[key]
        require((row["role"], row["license"]) == ("confidence_cal", "CC-BY-4.0"), "Generic rights or role changed")
        groups["generic"].add(row["canonical_speaker"])
    require(len(references) == len(groups["generic"]) == 4, "Generic speakers are repeated")
    for row in roster:
        if row["arm"] == "background_only":
            require(all(row[key] is None for key in ("target_reference", "sham_reference", "host_sha256", "edit", "truth")),
                    "Isolated room received a target enrollment or edit")
            continue
        keys = tuple(row[key] for key in ("host_sha256", "ordinary_sha256", "target_reference", "sham_reference"))
        host, other, target, sham = (source[key] for key in keys)
        refs = [host, other, target, sham]
        identity, ordinary, positive, negative = (item["canonical_speaker"] for item in refs)
        role = {"selection": "fusion_fit", "confirmation": "selection"}[row["role"]]
        require(len(set(keys)) == 4 and len({identity, ordinary, negative}) == 3 and identity == positive
                and target["text_identity"] != host["text_identity"]
                and all(item["role"] == role and item["license"] == "CC-BY-4.0" for item in refs),
                "Enrollment, sham identity or project source role changed")
        groups[row["role"]].update((identity, ordinary, negative))
        references.update(keys[2:])
    require(len(references) == 21 and references == set(frozen["references"]), "Frozen reference inventory changed")
    require(sum(map(len, groups.values())) == len(set.union(*groups.values())), "Enrollment roles share a speaker")


def check_room_overlap(rows, lengths, annotations, annotation_pin):
    require(sha(annotations) == annotation_pin, "Unrecognized AMI annotation release")
    rooms = {row["background"]["meeting"] for row in rows}
    with ZipFile(annotations) as archive:
        metadata = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = [set(p.get("global_name") for p in room if p.get("global_name"))
                        for room in metadata.iter() if room.get("observation") in rooms]
        require(len(rooms) == len(participants) == 4 and all(len(group) == 4 for group in participants)
                and len(set.union(*participants)) == 16, "Meeting participant roles overlap")
        tracks = {}
        for room in sorted(rooms):
            recorded = Counter()
            for speaker in "ABCD":
                track = ElementTree.fromstring(archive.read(f"words/{room}.{speaker}.words.xml"))
                active = set()
                for word in track:
                    start, end = word.get("starttime"), word.get("endtime")
                    if word.tag == "w" and not word.get("punc") and start is not None and end is not None:
                        active.update(range(round(float(start)*100), round(float(end)*100)))
                recorded.update(active)
            tracks[room] = {tick for tick, count in recorded.items() if count >= 2}
        synthetic = []
        for row, length in zip(rows, lengths, strict=True):
            meta = row["background"]
            active = tracks[meta["meeting"]]
            origin = meta["capture_start_sample"]//160
            ms = 10*sum(t in active for t in range(origin, origin+length//160))
            require(ms == meta["overlap_ms"] >= 1200, "Recorded room overlap differs")
            if row["edit"] is not None:
                start, end = (origin+x//160 for x in row["edit"])
                local = 10*sum(t in active for t in range(start, end))
                require(local == meta["edit_overlap_ms"] >= 400, "Overlap during the known edit differs")
                if row["arm"] == "synthetic":
                    synthetic.append(local)
    return {"edits": len(synthetic), "overlap_ms": sorted(synthetic), "overlap_total_ms": sum(synthetic), "edit_total_ms": 800*len(synthetic)}


def check_waveforms(run, roster):
    episode, used = defaultdict(list), defaultdict(list)
    lengths = []
    for row in roster:
        signals = {}
        for key in ("original", "foreground", "background"):
            raw = (run / "pcm" / f"{row['id']}-{key}.pcm").read_bytes()
            require(digest(raw) == row[f"{key}_pcm_sha256"], "Waveform receipt changed")
            signals[key] = raw
        mixture, first, second = (pcm(signals[key]) for key in ("original", "foreground", "background"))
        length = len(mixture)
        require(length in {6*RATE, 8*RATE} and all(m == f+b for m, f, b in zip(mixture, first, second, strict=True)),
                "Exact source components do not sum to original samples")
        lengths.append(length)
        episode[row["group"]].append((row, signals))
    for group in episode.values():
        reference, signals = group[0]
        meta = reference["background"]
        start = meta["capture_start_sample"]
        end = start+len(signals["original"])//2
        used[meta["meeting"]].append((start, end, reference["role"]))
        if reference["arm"] == "background_only":
            require(len(group) == 1 and not any(signals["foreground"]), "Isolated room includes foreground")
            continue
        a, b = reference["edit"]
        require(0 <= a < b <= len(signals["original"])//2 and b-a == 12800, "Edit clock differs")
        expected = {"ordinary_same_host", "ordinary_other_voice"}
        if reference["role"] == "confirmation":
            expected.add("synthetic")
        require(set(row["arm"] for row, _ in group) == expected and len(group) == len(expected), "Matched controls missing")
        for row, found in group:
            require(row["edit"] == [a, b] and row["truth"] == ([a, b] if row["arm"] == "synthetic" else None)
                    and row["background"] == meta and found["background"] == signals["background"]
                    and all(found[key][:2*a] == signals[key][:2*a] and found[key][2*b:] == signals[key][2*b:]
                            for key in ("original", "foreground")), "Edit support or matched background differs")
    for cuts in used.values():
        ordered = sorted(cuts)
        require(len({role for _, _, role in ordered}) == 1 and all(first[1] <= second[0] for first, second in zip(ordered, ordered[1:])),
                "Repeated or cross-role AMI capture interval")
    return lengths


def check_outcomes(rows, measures, result):
    cells, status = defaultdict(list), Counter()
    for row, proof in zip(rows, measures, strict=True):
        require(row["id"] == proof["id"], "Measurement does not bind to the correct input")
        status[proof["status"]] += 1
        if row["edit"] is None:
            require(proof["status"] == "no_target_enrollment" and "arms" not in proof, "Unavailable room was scored")
        else:
            require(proof["status"] == "measured", "Eligible acoustic comparison was skipped")
            cells[row["role"], row["background"]["meeting"], row["arm"]].append(proof["arms"])
    confirmed = {}
    for saved in result["outcomes"]:
        key = (saved["role"], saved["meeting"], saved["arm"])
        items = cells.pop(key)
        expected = {arm: {part: {metric: median(item[arm][part][metric] for item in items) for metric in METRICS}
                          for part in PARTS} for arm in ARMS}
        require(len(items) == saved["views"] and expected == saved["median"], "Incorrect published acoustic medians")
        if saved["role"] == "confirmation":
            confirmed[saved["meeting"], saved["arm"]] = items, expected
    require(not cells and status == result["scope"] == {"measured": 39, "no_target_enrollment": 4}
            and result["neural_authenticity_scores_measured"] == 0, "Acoustic result scope differs")
    same_host = [item for (_, arm), (rows, _) in confirmed.items() if arm == "ordinary_same_host" for item in rows]
    wins = sum(item["correct"]["edit"]["snr_gain_db"] > item["sham"]["edit"]["snr_gain_db"] for item in same_host)
    gate = {"right_reference_beats_sham": wins, "same_host_cases": len(same_host), "two_thirds_specific": wins*3 >= len(same_host)*2,
            "each_room_genuine_median_snr_gain": all(expected["correct"]["edit"]["snr_gain_db"] >= .5
                                                  for (_, arm), (_, expected) in confirmed.items() if arm == "ordinary_same_host"),
            "each_room_edit_preserved": all(expected["correct"]["edit"]["retained_target_db"] >= -3
                                           for (_, arm), (_, expected) in confirmed.items() if arm == "synthetic")}
    gate["qualifies"] = all(gate[key] for key in ("two_thirds_specific", "each_room_genuine_median_snr_gain", "each_room_edit_preserved"))
    require(gate == result["acoustic_gate"], "Frozen acoustic gate did not replay")


def verify(run, annotations, sources, parent):
    result, frozen, roster, measures = (read(run / f"{name}.json") for name in ("result", "freeze", "roster", "measurements"))
    require({name: sha(run / f"{name}.json") for name in ("freeze", "roster", "measurements")} == result["commitments"]
            and frozen["roster_sha256"] == result["commitments"]["roster"] and sha(parent) == frozen["parent_roster_sha256"],
            "Freeze, parent source or measurement receipt differs")
    previous = read(parent)
    require(len(roster) == len({row["id"] for row in roster}) == len(previous) == 43, "Unexpected case denominator")
    for row, earlier in zip(roster, previous, strict=True):
        mapping = {"edit": "edit_range", "generated_sha256": "donor_sha256"}
        keys = ("id", "group", "role", "arm", "family", "truth", "edit", "generated_sha256", "host_sha256", "ordinary_sha256")
        require(all(row[key] == earlier.get(mapping.get(key, key)) for key in keys), "Episode or qualified donor changed")
    edited = [row for row in previous if row["truth"] is not None]
    require(len({row["donor_group"] for row in edited}) == len({row["host_group"] for row in edited}) == 9
            and Counter(row["family"] for row in edited) == {"clone": 3, "expressive": 3, "stock": 3}, "Held source groups changed")
    check_speaker_roster(roster, frozen, sources)
    for key, pin in frozen["references"].items():
        raw = (run / "references" / f"{key}.pcm").read_bytes()
        require(len(raw) == 4*RATE*2 and digest(raw) == pin, "Exact enrollment PCM differs")
    lengths = check_waveforms(run, roster)
    overlap = check_room_overlap(roster, lengths, annotations, frozen["annotations_sha256"])
    check_outcomes(roster, measures, result)
    return {"waveforms": len(roster), "references": len(frozen["references"]), "overlapping_words_during_synthetic_edit": overlap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run", "annotations", "sources", "parent"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.annotations, args.sources, args.parent), indent=2))
