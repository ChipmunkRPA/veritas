"""Compare overlap-conditioned abstention on separate voices and natural rooms."""

from __future__ import annotations

import argparse
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from scripts import audio_harmonic_overlap as harmonic
from scripts import audio_spectra_background_transfer as voice
from scripts.audio_short_edit_cases_v172 import exact_source, sources
from scripts.prepare_natural_overlap_transfer import track_words

sha, read, write = voice.sha, voice.read, voice.write_new
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
EXPOSED = {
    "ES2002a", "ES2003c", "ES2004a", "ES2005a", "ES2006d", "ES2009d", "ES2012b", "ES2013a",
    "ES2015d", "IS1001b", "IS1003a", "IS1006a", "IS1009a",
}
DEV_ROOMS = ("ES2012b", "IS1001b")
QUOTAS = {"clone": 3, "expressive": 3, "stock": 3}
ACOUSTIC_POINTS_MS = (60, 80, 120, 180, 240, 320, 480, 640)
POLICIES = ("overlap_only", "independent_disagreement", "both")
RULE = {
    "development_maximal_minimal_false_each_room": 1, "minimum_high_marked_confirmation": 4,
    "minimum_high_marked_each_room": 1, "maximal_minimal_false_confirmation": 2,
    "minimum_corrected_genuine_ai": 3, "minimum_fraction_genuine_ai_corrected": .30,
    "minimum_corrected_genuine_groups": 2, "maximum_synthetic_ai_lost": 1,
    "minimum_fraction_synthetic_ai_preserved": .90, "maximum_clean_synthetic_ai_lost": 0,
    "minimum_genuine_assessed_in_each_mixed_room": 6,
}


def room_activity(archive, room):
    tracks = [track_words(archive.read(f"words/{room}.{speaker}.words.xml")) for speaker in "ABCD"]
    length = min(90_000, max(end for track in tracks for _, end in track))
    activity = np.zeros((4, length), dtype=bool)
    for index, track in enumerate(tracks):
        for start, stop in track:
            activity[index, start:stop] = True
    return activity.sum(axis=0)


def windows(archive, room, exclude=()):
    occupancy = room_activity(archive, room)
    result = {"high": [], "minimal": []}
    for start in range(400, len(occupancy) - 300, 400):
        if any(abs(start * 160 - prior) < harmonic.SAMPLES for prior in exclude):
            continue
        view = occupancy[start:start + 300]
        if (view >= 1).sum() < 245:
            continue
        overlap = int((view >= 2).sum())
        if overlap >= 120:
            result["high"].append(start)
        elif overlap <= 10:
            result["minimal"].append(start)
    return {key: sorted(starts, key=lambda time: sha(f"chipmunk/harmonic-overlap/window/1:{room}:{time}".encode()))
            for key, starts in result.items()}


def fresh_rooms(archive):
    document = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
    participants = {node.get("observation"): {p.get("global_name") for p in node if p.get("global_name")}
                    for node in document.iter() if node.get("observation")}
    exposed = set().union(*(participants[name] for name in EXPOSED))
    selected = {}
    ordered = sorted(participants, key=lambda room: sha(f"chipmunk/harmonic-overlap/room/1:{room}".encode()))
    for site in ("ES", "IS"):
        for room in ordered:
            people = participants[room]
            if not room.startswith(site) or len(people) != 4 or people & exposed:
                continue
            try:
                available = windows(archive, room)
            except KeyError:
                continue
            if len(available["high"]) >= 8 and len(available["minimal"]) >= 4:
                selected[room] = available
                exposed.update(people)
                break
    if len(selected) != 2:
        raise ValueError("The annotation roster has no qualifying participant-disjoint pair")
    return selected


def room_cut(path, begin, occupancy):
    start_sample = begin * 160
    with wave.open(str(path), "rb") as audio:
        if audio.getparams()[:3] != (1, 2, harmonic.RATE):
            raise ValueError("The native room recording has a different sample format")
        audio.setpos(start_sample)
        raw = audio.readframes(harmonic.SAMPLES)
    if len(raw) != harmonic.SAMPLES * 2:
        raise ValueError("The native room cut is truncated")
    edges = np.flatnonzero(np.diff(np.r_[False, occupancy[begin:begin + 300] >= 2, False]))
    overlap = [[int(a) * 160, int(b) * 160] for a, b in zip(edges[::2], edges[1::2], strict=True)]
    return raw, {"source_start_sample": start_sample, "source_end_sample": start_sample + harmonic.SAMPLES,
                 "time_base": "1/16000", "annotated_overlap": overlap}


def foregrounds():
    receipt = read(voice.PANEL / "audio-short-edit-materialization-complete-v169c.json")
    manifests = {}
    for role in ("development", "holdout"):
        path = sources.MEDIA / role / "private-manifest.json"
        if sha(path) != receipt["roles"][role]["private_manifest_sha256"]:
            raise ValueError("The commercial-rights foreground manifest changed")
        manifests[role] = read(path)["foregrounds"]
    prior = manifests["holdout"]
    for field in ("voice", "group_id", "original_sha256", "prefix_pcm_sha256"):
        if {row[field] for row in prior} & {row[field] for row in manifests["development"]}:
            raise ValueError(f"Development and confirmation have a declared shared {field}")
    eligible = [row for row in manifests["development"] if row["status"] == "qualified"]
    selected = [row for row in eligible if row["lane"] == "genuine"]
    if len(selected) != 14 or len({row["voice"] for row in selected}) != 7:
        raise ValueError("Expected two genuine utterances per reader from seven groups")
    for family, count in QUOTAS.items():
        candidates = sorted((row for row in eligible if row["lane"] == family),
                            key=lambda row: sha(f"chipmunk/harmonic-overlap/foreground/1:{row['slot']}".encode()))
        chosen = candidates[:count]
        if len({row["voice"] for row in chosen}) != count:
            raise ValueError(f"Confirmation {family} voices are incomplete")
        selected.extend(chosen)
    return [(exact_source(row, "development"), row) for row in selected], receipt["roles"]["development"]["private_manifest_sha256"]


def prepare(folder, recovery, ami, assets):
    published = read(voice.PANEL / "audio-spectra-recovery-comparison-2026-11-11.json")
    inputs = recovery / "comparison"
    lock = read(inputs / "freeze.json")
    if (sha(inputs / "freeze.json") != published["freeze_sha256"]
            or sha(inputs / "roster.json") != lock["roster_sha256"]
            or sha(inputs / "confirm.json") != published["confirmation_sha256"]):
        raise ValueError("The exact earlier exploratory comparison is required for point selection")
    annotations = recovery / "ami/ami_public_manual_1.6.2.zip"
    if sha(annotations) != voice.ANNOTATION_SHA256:
        raise ValueError("AMI manual annotation archive differs")
    hold, manifest_sha = foregrounds()
    pinned = voice.pinned_assets(assets)
    rows, room_binding = [], {}
    (folder / "pcm").mkdir(parents=True)

    def save(raw, role, condition, family, group, **extra):
        index = len(rows)
        baseline = extra.pop("baseline", None)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, "role": role, "condition": condition, "family": family, "group": group,
                     "pcm_sha256": sha(raw), "baseline": baseline, **extra})

    previous = {row["id"]: row for row in read(inputs / "confirm.json")["rows"]}
    for row in read(inputs / "roster.json"):
        if row["id"] not in previous:
            continue
        archived = previous[row["id"]]
        if archived["pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("Archived inference was recorded on a different waveform")
        save(voice.exact(inputs, row), "development", row["condition"], row["family"], row["group"],
             baseline={key: archived[key] for key in ("decision", "raw")}, independent=archived["candidate"])
    background = defaultdict(list)
    with ZipFile(annotations) as archive:
        selected = fresh_rooms(archive)
        for stage, rooms in (("development", DEV_ROOMS), ("confirmation", selected)):
            for room in rooms:
                path = (recovery / "ami" if stage == "development" else ami) / f"{room}.Array1-01.wav"
                media_hash = sha(path)
                occupancy = room_activity(archive, room)
                old_cuts = [row for row in lock["prospective_backgrounds"] if row["meeting"] == room]
                if stage == "development" and (len(old_cuts) != 1 or media_hash != old_cuts[0]["archive_sha256"]):
                    raise ValueError("The two archived development microphone recordings changed")
                available = selected[room] if stage == "confirmation" else windows(
                    archive, room, [row["original_start_sample"] for row in old_cuts])
                if len(available["high"]) < 4 or len(available["minimal"]) < 4:
                    raise ValueError("Incomplete annotation-only room windows")
                room_binding[room] = {"archive_sha256": media_hash, "role": stage, "official_url": f"{MIRROR}/{room}/audio/{path.name}"}
                for condition in ("high", "minimal"):
                    for begin in available[condition][:4]:
                        raw, meta = room_cut(path, begin, occupancy)
                        # Native development audio sets only the acoustic operating point.
                        baseline = {"decision": "unassessed", "raw": None} if stage == "development" else None
                        save(raw, stage, "native", "genuine", room, room=room, label=condition, baseline=baseline, **meta)
                if stage == "confirmation":
                    for begin in available["high"][4:8]:
                        raw, meta = room_cut(path, begin, occupancy)
                        save(raw, stage, "isolated_background", "background", room, room=room, **meta)
                        background[room].append((raw, meta, sha(raw)))
    for index, (clean, reference) in enumerate(hold):
        family, group = reference["lane"], reference["group_id"]
        save(clean, "confirmation", "clean", family, group, origin_sha256=reference["original_sha256"])
        for room_index, (room, clips) in enumerate(background.items()):
            raw, meta, digest = clips[(index + room_index) % len(clips)]
            mixture, scale, gain = voice.mix(clean, raw)
            save(mixture, "confirmation", "mixed", family, group, room=room, background_sha256=digest,
                 clean_pcm_sha256=sha(clean), background_start=meta["source_start_sample"],
                 background_annotated_overlap=meta["annotated_overlap"], scale=scale, common_gain=gain)
    write(folder / "roster.json", rows)
    binding = {
        "schema": "chipmunk.audio-harmonic-overlap-comparison/1", "runner_sha256": sha(Path(voice.__file__)),
        "comparison_code_sha256": sha(Path(__file__)), "harmonic_code_sha256": sha(Path(harmonic.__file__)),
        "roster_sha256": sha(folder / "roster.json"), "rule": voice.RULE, "advance_rule": RULE,
        "acoustic_points_ms": list(ACOUSTIC_POINTS_MS), "method": harmonic.METHOD,
        "incumbent_assets": voice.incumbent.ASSETS, "incumbent_calibration_sha256": sha(voice.incumbent.CALIBRATION),
        "independent_assets": pinned, "parent_comparison": published["confirmation_sha256"],
        "annotations_sha256": voice.ANNOTATION_SHA256, "confirmation_foreground_manifest_sha256": manifest_sha,
        "rooms": room_binding, "planned": dict(Counter(row["role"] for row in rows)),
    }
    write(folder / "freeze.json", binding)
    print(binding["planned"], flush=True)


def frozen(folder):
    lock, rows = voice.frozen(folder)
    if (lock["schema"] != "chipmunk.audio-harmonic-overlap-comparison/1" or lock["advance_rule"] != RULE
            or lock["acoustic_points_ms"] != list(ACOUSTIC_POINTS_MS) or lock["method"] != harmonic.METHOD
            or lock["comparison_code_sha256"] != sha(Path(__file__)) or lock["harmonic_code_sha256"] != sha(Path(harmonic.__file__))):
        raise ValueError("The overlap comparison method changed after freeze")
    return lock, rows


def duration_ms(spans):
    return sum(stop - start for start, stop in spans) // 16


def is_synthetic(row):
    return row["family"] not in {"genuine", "background"}


def fit(folder):
    _, rows = frozen(folder)
    measured = []
    for row in rows:
        if row["role"] != "development":
            continue
        spans = harmonic.measure(voice.exact(folder, row))
        measured.append({**row, "spans": spans, "duration_ms": duration_ms(spans)})
    native = [row for row in measured if row["condition"] == "native"]
    acoustic = []
    for point in ACOUSTIC_POINTS_MS:
        flags = [row for row in native if row["duration_ms"] >= point]
        false = Counter(row["room"] for row in flags if row["label"] == "minimal")
        high = sum(row["label"] == "high" for row in flags)
        if max(false.values(), default=0) <= RULE["development_maximal_minimal_false_each_room"]:
            acoustic.append((high, -sum(false.values()), point))
    acoustic_point = max(acoustic)[-1] if acoustic else None
    flagged = [row for row in measured if row["baseline"]["decision"] == "ai"]
    outcomes = {}
    for policy in ("independent_disagreement", "both"):
        thresholds = sorted({row["independent"] for row in flagged})
        viable = []
        for point in thresholds:
            removed = [row for row in flagged if row["independent"] < point and
                       (policy != "both" or (acoustic_point is not None and row["duration_ms"] >= acoustic_point))]
            false_corrected = len({row["group"] for row in removed if not is_synthetic(row)})
            true_lost = sum(is_synthetic(row) for row in removed)
            clean_lost = sum(is_synthetic(row) and row["condition"] == "clean" for row in removed)
            if true_lost <= RULE["maximum_synthetic_ai_lost"] and not clean_lost and false_corrected:
                viable.append((false_corrected, -true_lost, -point))
        outcomes[policy] = -max(viable)[-1] if viable else None
    write(folder / "fit.json", {"freeze_sha256": sha(folder / "freeze.json"), "acoustic_point_ms": acoustic_point,
                               "independent_points": outcomes, "native_windows": len(native),
                               "development_ai_directions": len(flagged),
                               "acoustic_operating_points_considered": acoustic})
    write(folder / "fit-spans-private.json", measured)
    print(read(folder / "fit.json"), flush=True)


def point_fit(folder):
    points = read(folder / "fit.json")
    if points["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("Selection receipt does not bind the fixed input plan")
    return points


def confirmation(folder, assets):
    _, rows = frozen(folder)
    point_fit(folder)
    baseline = read(folder / "baseline.json")
    if baseline["freeze_sha256"] != sha(folder / "freeze.json"):
        raise ValueError("The current detector responses do not match the confirmation freeze")
    previous = {row["id"]: row for row in baseline["rows"]}
    voice.incumbent.network_free()
    session = voice.start_session(assets)
    measured, forwards = [], 0
    for row in rows:
        if row["role"] != "confirmation":
            continue
        raw = voice.exact(folder, row)
        reference = previous.pop(row["id"])
        if reference["pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("The current detector scored a different waveform")
        spans = harmonic.measure(raw)
        independent, graph = None, None
        if reference["decision"] == "ai":
            independent, graph = voice.candidate_score(session, raw)
            if forwards == 0:
                if voice.candidate_score(session, raw) != (independent, graph):
                    raise ValueError("The independent score does not reproduce exactly")
                forwards += 1
            forwards += 1
        measured.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "reference": reference["decision"],
                         "independent": independent, "model_input_sha256": graph, "spans": spans})
    if previous:
        raise ValueError("Unused incumbent observations")
    write(folder / "confirmation.json", {"freeze_sha256": sha(folder / "freeze.json"), "fit_sha256": sha(folder / "fit.json"),
          "incumbent_sha256": sha(folder / "baseline.json"), "rows": measured, "independent_forwards": forwards})
    print(f"Verified {len(measured)} exact confirmation views and {forwards} independent forwards", flush=True)


def assess(row, got, point, points):
    cue = point is not None and duration_ms(got["spans"]) >= point
    veto = {"overlap_only": cue}
    for policy in ("independent_disagreement", "both"):
        threshold = points[policy]
        veto[policy] = (threshold is not None and got["independent"] is not None and got["independent"] < threshold
                        and (policy != "both" or cue))
    return cue, {name: result and got["reference"] == "ai" for name, result in veto.items()}


def readout(folder):
    lock, rows = frozen(folder)
    points = point_fit(folder)
    measured = read(folder / "confirmation.json")
    if (measured["freeze_sha256"] != sha(folder / "freeze.json") or measured["fit_sha256"] != sha(folder / "fit.json")
            or measured["incumbent_sha256"] != sha(folder / "baseline.json")):
        raise ValueError("Confirmation outputs are not bound to the frozen inputs and points")
    archive = {r["pcm_sha256"]: voice.exact(folder, r) for r in rows}
    report, acoustic, effects, evidence = defaultdict(Counter), Counter(), defaultdict(list), []
    overlap_intersection = marked_samples = annotated_samples = 0
    for row, got in zip((r for r in rows if r["role"] == "confirmation"), measured["rows"], strict=True):
        if (got["id"], got["pcm_sha256"]) != (row["id"], row["pcm_sha256"]):
            raise ValueError("Confirmation order and waveform digest differ")
        if row["condition"] == "mixed":
            remixed, scale, gain = voice.mix(archive[row["clean_pcm_sha256"]], archive[row["background_sha256"]])
            if (sha(remixed), scale, gain) != (row["pcm_sha256"], row["scale"], row["common_gain"]):
                raise ValueError("Exact counterpart mixture reconstruction differs")
        cue, veto = assess(row, got, points["acoustic_point_ms"], points["independent_points"])
        native = row["condition"] == "native"
        stats = report[(row["condition"], row.get("room"), row["family"])]
        stats.update(planned=1, assessed=got["reference"] != "unassessed", ai=got["reference"] == "ai",
                     original=got["reference"] == "original", inconclusive=got["reference"] == "inconclusive", cue=cue)
        if native:
            acoustic[(row["room"], row["label"], "planned")] += 1
            acoustic[(row["room"], row["label"], "marked")] += cue
            spans, expected = got["spans"], row["annotated_overlap"]
            marked_samples += sum(stop - start for start, stop in spans)
            annotated_samples += sum(stop - start for start, stop in expected)
            overlap_intersection += sum(max(0, min(b, d) - max(a, c)) for a, b in spans for c, d in expected)
        for name, withheld in veto.items():
            stats[f"{name}_withheld"] += withheld
            if withheld:
                effects[name].append(row)
        if cue or any(veto.values()):
            evidence.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "time_base": "1/16000",
                             "inspected": [0, harmonic.SAMPLES], "observed_overlap_cue": got["spans"], "withheld": veto})
    rooms = [room for room, info in lock["rooms"].items() if info["role"] == "confirmation"]
    high = [acoustic[(room, "high", "marked")] for room in rooms]
    minimal = [acoustic[(room, "minimal", "marked")] for room in rooms]
    genuine_false = sum(v["ai"] for (_, _, family), v in report.items() if family in {"genuine", "background"})
    synthetic_ai = sum(v["ai"] for (_, _, family), v in report.items() if family not in {"genuine", "background"})
    base = {"false": genuine_false, "synthetic_ai": synthetic_ai}
    acoustic_pass = (sum(high) >= RULE["minimum_high_marked_confirmation"] and min(high) >= RULE["minimum_high_marked_each_room"]
                     and sum(minimal) <= RULE["maximal_minimal_false_confirmation"])
    assessed_pass = all(report[("mixed", room, "genuine")]["assessed"] >= RULE["minimum_genuine_assessed_in_each_mixed_room"] for room in rooms)
    policies = {}
    for name in POLICIES:
        removed_false = [row for row in effects[name] if not is_synthetic(row)]
        removed_true = [row for row in effects[name] if is_synthetic(row)]
        false_groups = len({row["group"] for row in removed_false})
        clean_loss = sum(row["condition"] == "clean" for row in removed_true)
        eligible = (len(removed_false) >= RULE["minimum_corrected_genuine_ai"]
                    and len(removed_false) >= math.ceil(genuine_false * RULE["minimum_fraction_genuine_ai_corrected"])
                    and false_groups >= RULE["minimum_corrected_genuine_groups"]
                    and len(removed_true) <= RULE["maximum_synthetic_ai_lost"]
                    and synthetic_ai - len(removed_true) >= math.ceil(synthetic_ai * RULE["minimum_fraction_synthetic_ai_preserved"])
                    and clean_loss <= RULE["maximum_clean_synthetic_ai_lost"])
        policies[name] = {"false_withheld": len(removed_false), "false_groups": false_groups, "true_withheld": len(removed_true),
                          "clean_true_withheld": clean_loss, "passes_decision": eligible,
                          "advance": assessed_pass and eligible and (acoustic_pass or name == "independent_disagreement")}
    write(folder / "evidence-private.json", evidence)
    aggregate = {"schema": "chipmunk.audio-harmonic-overlap-comparison-result/1", "freeze": lock,
                 "freeze_sha256": sha(folder / "freeze.json"), "fit_sha256": sha(folder / "fit.json"),
                 "confirmation_sha256": sha(folder / "confirmation.json"), "baseline_sha256": measured["incumbent_sha256"],
                 "fit": points, "baseline": base, "policies": policies, "acoustic_pass": acoustic_pass,
                 "genuine_coverage_pass": assessed_pass, "independent_forwards": measured["independent_forwards"],
                 "evidence_sha256": sha(folder / "evidence-private.json"), "evidence_rows": len(evidence),
                 "room_acoustic": [{"room": room, "high_marked": h, "minimal_marked": m, "planned_each": 4}
                                   for room, h, m in zip(rooms, high, minimal, strict=True)],
                 "native_localization": {"shared_samples": overlap_intersection, "cue_samples": marked_samples,
                                         "annotated_samples": annotated_samples},
                 "cells": [{"condition": condition, "room": room, "family": family, **value}
                           for (condition, room, family), value in report.items()]}
    write(folder / "aggregate.json", aggregate)
    print({"baseline": base, "policies": policies, "acoustic_pass": acoustic_pass, "genuine_coverage_pass": assessed_pass}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "baseline", "confirm", "readout"))
    parser.add_argument("folder", type=Path)
    for option in ("recovery", "ami", "assets"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        prepare(args.folder, args.recovery, args.ami, args.assets)
    elif args.stage == "fit":
        fit(args.folder)
    elif args.stage == "baseline":
        frozen(args.folder)
        point_fit(args.folder)
        voice.baseline(args.folder)
    elif args.stage == "confirm":
        confirmation(args.folder, args.assets)
    else:
        readout(args.folder)
