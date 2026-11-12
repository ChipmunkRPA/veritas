"""Independently replay the paired call-processing confirmation from local files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from review_audio_communication_transfer import check, choice, digest, load, locate
from review_audio_enrolled_target_transfer import pcm
from review_audio_remote_neighbor_replication import exact_sources
from review_audio_variable_boundary_transfer import actual_overlap

ARMS = ("original", "communication", "both")
WINDOWS = ((16000, 48000), (48000, 80000))


def declared_speakers(rows):
    witnesses = (row[key] for row in rows for key in ("host", "donor") if row.get(key))
    voices, originals = set(), set()
    for item in witnesses:
        voices.add(item["canonical_speaker"])
        originals.add(item["source_sha256"])
    return voices, originals


def room_members(annotations, rows):
    rooms = {row["room"] for row in rows}
    with ZipFile(annotations) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
    members = [p.get("global_name") for m in meetings.iter() if m.get("observation") in rooms
               for p in m if p.get("global_name")]
    check(len(members) == len(set(members)) == 16 and len(rooms) == 4, "Declared meeting participants repeat")
    return set(members)


def native_scores(run, rows, score_rows):
    cases = []
    forwards = Counter()
    for row, proof in zip(rows, score_rows, strict=True):
        check(proof["id"] == row["id"] and set(proof["windows"]) == set(ARMS[:2]), "Missing or unrelated score record")
        original = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
        path = run / "deliveries" / row["file"]
        delivered = path.with_suffix(".pcm").read_bytes()
        check(len(delivered) == len(original) == 192000 and digest(delivered) == row["delivered_pcm_sha256"]
              and digest(path.read_bytes()) == row["encoded_sha256"], "Processed input hash or timeline length differs")
        windows = proof["windows"]
        for arm, raw in zip(ARMS[:2], (original, delivered), strict=True):
            for window, (start, stop) in zip(windows[arm], WINDOWS, strict=True):
                check((window["start"], window["end"]) == (start, stop)
                      and window["pcm_sha256"] == digest(raw[start * 2:stop * 2]), "Model window differs from native samples")
                forwards[arm] += window["score"] is not None
        both = [{"start": start, "end": stop, "score": min(a["score"], b["score"])}
                for a, b, (start, stop) in zip(windows["original"], windows["communication"], WINDOWS, strict=True)
                if a["score"] is not None and b["score"] is not None]
        cases.append((row, {**{arm: choice(windows[arm]) for arm in ARMS[:2]}, "both": choice(both)}))
    return cases, forwards


def mixture_support(run, rows):
    snr = []
    for row in rows:
        prefix = run / "pcm" / row["id"]
        original = pcm(prefix.with_suffix(".pcm").read_bytes())
        check(max(map(abs, original)) < 32767, "Original mixture is clipped")
        if row["arm"] != "untouched":
            continue
        energies = [sum(sample ** 2 for sample in pcm(prefix.with_name(prefix.name + f"-{part}.pcm").read_bytes()))
                    for part in ("foreground", "background")]
        snr.append(10 * math.log10(energies[0] / energies[1]))
    check(len(snr) == 12 and all(abs(value + 5) < .01 for value in snr), "Exact mixture SNR differs from the frozen source scope")
    return [round(min(snr), 4), round(max(snr), 4)]


def tally(cases, arm, cutoff):
    counts = dict.fromkeys(("ai_flags", "useful_ranges", "unassessed", "background_false_flags"), 0)
    false, useful = set(), {}
    for row, winners in cases:
        winner = winners[arm]
        if winner is None:
            counts["unassessed"] += 1
        elif winner["score"] > cutoff:
            if row["truth"] is None:
                false.add(row["group"])
                counts["background_false_flags"] += row["arm"] == "background_only"
            else:
                counts["ai_flags"] += 1
                if locate(winner, row["truth"]):
                    counts["useful_ranges"] += 1
                    useful[row["id"]] = (winner["start"], winner["end"])
    return {**counts, "false_groups": len(false)}, false, useful


def verify_tables(cases, cutoff, report):
    checked = 0
    for group, expected in report["tables"].items():
        subset = [(row, selected) for row, selected in cases if group == "all" or row["family"] == group or
                  (group == "interior" and row["coverage"] in {"interior", "room"}) or (group == "edge" and row["coverage"] != "interior")]
        check(Counter(row["arm"] for row, _ in subset) == expected["inputs"], "Subset counts differ")
        for arm in ARMS:
            fixed, _, _ = tally(subset, arm, cutoff)
            check(fixed == expected["fixed"][arm], "Fixed outcome differs")
            peak = {}
            for row, selected in subset:
                candidate = selected[arm]
                if row["truth"] is None and candidate is not None:
                    g = row["group"]
                    peak[g] = max(peak.get(g, float("-inf")), candidate["score"])
            points = sorted(peak.values(), reverse=True)
            for n in range(3):
                value = tally(subset, arm, points[n])[0] if n < len(points) else None
                check(value == expected["equal_false_groups"][str(n)][arm], "Equal-alarm readback differs")
            checked += 4
    _, original_false, original_useful = tally(cases, "original", cutoff)
    _, paired_false, paired_useful = tally(cases, "both", cutoff)
    retained = sum(paired_useful.get(key) == interval for key, interval in original_useful.items())
    removed, added = len(original_false - paired_false), len(paired_false - original_false)
    check(retained == report["exact_useful_original_locations_retained"] and removed == report["original_false_groups_removed"]
          and added == report["new_false_groups"] and bool(paired_useful and retained == len(original_useful) and removed and not added)
          == report["passes_advancement"], "Paired comparison advancement differs")
    return {"comparison_cells_verified": checked, "useful_original_locations_retained": retained,
            "false_groups_removed": removed, "new_false_groups": added}


def replay(run, prior, discovery, annotations):
    freeze, scores, report = (load(run / f"{name}.json") for name in ("freeze", "scores", "aggregate"))
    first = load(discovery / "freeze.json")
    check(digest((discovery / "freeze.json").read_bytes()) == freeze["source_bindings"]["discovery_freeze"]
          and report["fixed_point"] == first["cutoff"]
          and all(freeze[key] == first[key] for key in ("cutoff", "chain", "codec", "windows", "model_assets")), "The fixed rule changed")
    check(digest((prior / "roster.json").read_bytes()) == first["sources"]["roster.json"], "Discovery source receipt changed")
    check(all(digest((run / f"{name}.json").read_bytes()) == pin for name, pin in report["commitments"].items())
          and scores["freeze_sha256"] == report["commitments"]["freeze"]
          and freeze["roster_sha256"] == report["commitments"]["roster"] and digest(annotations.read_bytes()) == freeze["annotation_sha256"],
          "A sealed result, roster, or annotation was modified")
    rows, old_rows = load(run / "roster.json"), load(prior / "roster.json")
    check(len(rows) == len(old_rows) == 52 and Counter(r["coverage"] for r in rows if r["truth"]) == {"interior": 8, "early": 2, "late": 2},
          "The study denominator or time strata differ")
    exact_sources(run, rows)
    overlap = actual_overlap(rows, annotations)
    voices, originals = declared_speakers(rows)
    old_voices, old_originals = declared_speakers(old_rows)
    check(len(voices) == len(originals) == 36 and not voices & old_voices and not originals & old_originals
          and not room_members(annotations, rows) & room_members(annotations, old_rows), "Discovery and confirmation sources are not disjoint")
    cases, forwards = native_scores(run, rows, scores["records"])
    check(forwards == report["new_classifier_windows"], "Forward counts differ")
    return {"paired_files_verified": len(rows), "native_model_window_bindings": len(rows) * 4, "source_components_verified": len(rows) * 3,
            "disjoint_foreground_voices": len(voices), "disjoint_meeting_participants": 16, "overlapping_competing_words_ms": [min(overlap), max(overlap)],
            "untouched_mixture_snr_db": mixture_support(run, rows),
            **verify_tables(cases, first["cutoff"], report)}


def public_rows(run):
    groups = {}
    for row, record in zip(load(run / "roster.json"), load(run / "scores.json")["records"], strict=True):
        group = groups.setdefault(row["group"], len(groups) + 1)
        yield {**{key: row[key] for key in ("arm", "coverage", "family", "truth")}, "group": group,
               "scores": {arm: [window["score"] for window in record["windows"][arm]] for arm in ARMS[:2]}}


def replay_public(report):
    cases = []
    for i, row in enumerate(report["anonymous_windows"], 1):
        windows = {arm: [{"start": start, "end": stop, "score": value}
                         for (start, stop), value in zip(WINDOWS, row["scores"][arm], strict=True)] for arm in ARMS[:2]}
        both = [{**a, "score": min(a["score"], b["score"])} for a, b in zip(*windows.values(), strict=True)
                if a["score"] is not None and b["score"] is not None]
        cases.append((row | {"id": i}, {arm: choice(w) for arm, w in (windows | {"both": both}).items()}))
    check(len(cases) == 52 and len({row["group"] for row, _ in cases}) == 16, "Public panel denominator differs")
    return verify_tables(cases, report["fixed_point"], report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.public is not None and not args.paths and args.export is None:
        outcome = replay_public(load(args.public))
    elif len(args.paths) == 4 and args.public is None:
        outcome = replay(*args.paths)
        if args.export is not None:
            run = args.paths[0]
            report = load(run / "aggregate.json") | {"independent_native_replay": outcome, "anonymous_windows": list(public_rows(run))}
            replay_public(report)
            with args.export.open("x") as stream:
                stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        parser.error("Provide run, prior, discovery and annotations paths, or use --public RECEIPT")
    print(json.dumps(outcome, sort_keys=True))
