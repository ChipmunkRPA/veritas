"""Prospective six- and eight-second room-speech comparison for context edges."""

from __future__ import annotations

import argparse
import math
import time
import wave
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from app.detectors.audio import calibration, decode
from app.detectors.audio.content import window_in_speech
from scripts import audio_partial_residual_inputs as inputs
from scripts import audio_partial_residual_transfer as voice
from scripts import audio_short_edit_cases_v172 as donor_sources
from scripts import audio_spectra_background_transfer as spectra
from scripts import audio_spectra_context_transfer as edge
from scripts import prepare_natural_overlap_transfer as ami

ROOT = Path(__file__).resolve().parents[2]
BASE, ARCHIVE = inputs.BASE, donor_sources.roster.ARCHIVE
PREVIOUS = BASE / "audio-overlap-window-transfer-2026-11-12-final"
PREVIOUS_EDGE = BASE / "audio-spectra-context-2026-11-12"
ANNOTATION = BASE / "spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
RATE = 16_000
ROLES = {"selection": ("fusion_fit", ("ES2004a", "IS1009a"), 6),
         "confirmation": ("selection", ("ES2007c", "IS1004c"), 9)}
RUNTIME_ARMS = ("spectra", "current")
ARMS = ("current_two_second", edge.RAW, edge.CONTRAST)
METHOD = {
    "roles": "Six point-setting genuine episodes from v81 fusion_fit and nine held episodes from v81 selection. Three separately qualified v169 holdout donors per clone, expressive, stock lane, excluding all foreground groups in the previous Spectra AMI panel. Four separate AMI meetings with distinct recorded participants. All archives had prior project use.",
    "geometry": "Six- and eight-second host segments begin at 4 seconds of original decoded v81 source. Place one .8-second edit on a fixed 80 ms candidate grid, with 20 ms fades and matched native host RMS. Same-host source comes from the first four seconds. Other-voice and synthetic donors contribute fixed eligible snippets from separately verified recordings. Source order and eligibility use no model response.",
    "background": "One exact annotated Array1-01 cut per episode, split by meeting role. Require >80% annotated words of any speaker and >=1.2 s simultaneous word activity. Choose before scoring by salted source order on nonoverlapping 8-second grid; exclude recorded prior native room cuts. Mix at -5 dB relative to unedited foreground with one peak-safe gain. Additional held control per room reuses no mixed cut.",
    "arms": "Retained Spectra graph, unchanged three-second contexts and one-second exclusive edge contrast. Actual current model scores native two-second windows beginning at odd seconds (1,3 for six seconds; 1,3,5 for eight). Apply the incumbent speech activity rule. Neither route fits a head or maps to customer probability.",
    "points": "For each arm use the maximum genuine point-setting clip score, then apply unchanged to held. Also report previous unchanged points. Fixed union adds one positive if either current or exclusive edge flags and retains both reported locations. Separate retrospective held false-group budgets 0,1,2 diagnose ranking.",
    "gate": "At least two net useful locations from two generator families in both held rooms combined, with gains in each room, at most one extra genuine false group and no false isolated room cut. Locate at .30 IoU and .50 edit coverage. A success permits a wider prospective trial only.",
}
sha, digest, require, write, read = edge.sha, edge.digest, edge.require, edge.write, edge.read


def order(value):
    return digest(f"long-spectra-room/1:{value}".encode())


def room_inputs():
    v291 = BASE / "audio-ami-native-stereo-v291"
    v291_data = read(v291 / "acquired.json")["files"]
    harmonic = BASE / "audio-harmonic-overlap-2026-11-11/run"
    old_rooms = read(harmonic / "freeze.json")["rooms"]
    spatial = BASE / "audio-spatial-interference-2026-11-11/run/roster.json"
    excluded = defaultdict(list)
    for r in read(v291 / "qualified.json")["cases"]:
        excluded[r["session"]].append(tuple(r["source_range"][f"{key}_sample"] for key in ("start", "end")))
    for r in read(harmonic / "roster.json"):
        if "source_start_sample" in r:
            excluded[r["room"]].append((r["source_start_sample"], r["source_end_sample"]))
    for r in read(spatial):
        if "native_start" in r:
            excluded[r["room"]].append((r["native_start"], r["native_start"]+3*RATE))
    require(sha(ANNOTATION) == ami.ANNOTATION_SHA256, "Pinned annotation bytes changed")
    specs = {}
    with ZipFile(ANNOTATION) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {x.get("observation"): {p.get("global_name") for p in x if p.get("global_name")} for x in meetings.iter()}
        people = set()
        for role, (_, rooms, _) in ROLES.items():
            for room in rooms:
                require(len(participants[room]) == 4 and not participants[room] & people, "Meeting participants overlap")
                people.update(participants[room])
                name = f"{room}.Array1-01.wav"
                if name in v291_data:
                    path = ARCHIVE / "audio-ami-native-stereo-v291/raw" / name
                    expected = v291_data[name]["sha256"]
                else:
                    path = harmonic.parent / "ami" / name
                    expected = old_rooms[room]["archive_sha256"]
                require(sha(path) == expected, "AMI original WAV differs")
                tracks = [ami.track_words(archive.read(f"words/{room}.{speaker}.words.xml")) for speaker in "ABCD"]
                stop = max(b for voice_track in tracks for _, b in voice_track)
                activity = np.zeros((4, stop), dtype=np.int8)
                for index, voice_track in enumerate(tracks):
                    for a, b in voice_track:
                        activity[index, a:b] = 1
                specs[room] = (path, expected, activity.sum(axis=0), role, tuple(sorted(set(excluded[room]))))
    return specs


def cut_room(spec, room, seconds, used):
    path, pin, activity, role, excluded = spec
    amount = RATE*seconds
    options = []
    with wave.open(str(path), "rb") as file:
        channels, width, rate = file.getparams()[:3]
        require(channels in {1, 2} and width == 2 and rate == RATE, "Source AMI waveform clock changed")
        for tick in range(800, min(len(activity)-seconds*100, file.getnframes()//160-seconds*100), 800):
            start = tick*160
            if any(start < b and a < start+amount for a, b in (*excluded, *used[room])):
                continue
            segment = activity[tick:tick+seconds*100]
            total, overlap = np.count_nonzero(segment), np.count_nonzero(segment >= 2)
            if total > seconds*80 and overlap >= 120:
                options.append((order((room, tick)), start, int(total)*10, int(overlap)*10))
        require(bool(options), f"No unused overlapping cut for {room}")
        _, start, speech, overlap = min(options)
        file.setpos(start)
        raw = np.frombuffer(file.readframes(amount), "<i2").reshape(-1, channels)
    require(len(raw) == amount and (channels == 1 or np.array_equal(raw[:, 0], raw[:, 1])), "Physical microphone storage differs")
    used[room].append((start, start+amount))
    samples = raw[:, 0].astype(np.float64)
    return samples, {"room": room, "role": role, "capture_sha256": pin, "capture_start_sample": start,
                     "samples": amount, "capture_pcm_sha256": digest(raw[:, 0].tobytes()), "speech_ms": speech, "overlap_ms": overlap}


def chosen_donors():
    path = donor_sources.sources.MEDIA / "holdout/private-manifest.json"
    receipt = read(ROOT / "docs/evaluation-panels/audio-short-edit-materialization-complete-v169c.json")
    require(sha(path) == receipt["roles"]["holdout"]["private_manifest_sha256"], "Qualified donor roster changed")
    prior = read(PREVIOUS / "roster.json")
    excluded = {row[k] for row in prior for k in ("source_group", "host_group")}
    rights = {"clone": "MIT (publisher PLN scenario declaration)", "stock": "CC-BY-SA-4.0", "expressive": "CC-BY-4.0"}
    picked = []
    for family, terms in rights.items():
        groups = {}
        for item in sorted(read(path)["foregrounds"], key=lambda r: order(r["original_sha256"])):
            if item["lane"] == family and item["status"] == "qualified" and item["group_id"] not in excluded:
                require(item["license"] == terms, "Unqualified donor license")
                groups.setdefault(item["group_id"], item)
        require(len(groups) >= 3, "Not enough disjoint generator voices")
        picked.extend(list(groups.values())[:3])
    return sorted(picked, key=lambda row: order(row["group_id"])), sha(path)


def pcm(values):
    rounded = np.rint(values)
    require(np.isfinite(rounded).all() and max(abs(rounded)) <= 31_130, "Unexpected mix clipping")
    return rounded.astype("<i2").tobytes()


def data():
    catalog, _, _, _ = inputs.catalogues()
    spec = room_inputs()
    donors, donor_hash = chosen_donors()
    blocked = defaultdict(list)
    output = []

    @cache
    def voice_pcm(source_sha):
        row = catalog[source_sha]
        samples = decode.decode_audio(Path(row["source_path"]), source_sha256=source_sha)
        require(samples.mapping["pcm_sha256"] == row["pcm_sha256"] and len(samples.pcm) == row["samples"], "Full retained host differs")
        return samples.pcm.astype(np.float64)*32768

    length = 12_800
    prior_sources = read(BASE / "audio-partial-residual-2026-11-12/inputs-freeze.json")["pairs"]
    recent = {row[key] for row in prior_sources if row["role"] in {"confidence_cal", "confirmation"} for key in ("host", "ordinary", "synthetic")}
    for role, (old_role, rooms, count) in ROLES.items():
        hosts = [r for r in catalog.values() if r["kind"] == "genuine" and r["role"] == old_role
                 and r["samples"] >= 12*RATE and r["raw_sha256"] not in recent]
        require(all(r["license"] == "CC-BY-4.0" for r in hosts), "Long host license changed")
        hosts.sort(key=lambda r: order(r["raw_sha256"]))
        used = set()
        for index in range(count):
            name = f"{role}-{index:02}"
            seconds = 6 if index//2 % 2 == 0 else 8
            room = rooms[index % len(rooms)]
            noise, background = cut_room(spec[room], room, seconds, blocked)
            target_positions = inputs.positions(seconds*RATE, length, name+"-target")
            for host in hosts:
                if host["study_group"] in used:
                    continue
                raw = voice_pcm(host["raw_sha256"])
                recording = raw[4*RATE:(4+seconds)*RATE]
                first = next((a for a in inputs.positions(4*RATE, length, name+"-same") if inputs.signal_eligible(raw[a:a+length])), None)
                target = next((a for a in target_positions if inputs.signal_eligible(recording[a:a+length])), None)
                if first is not None and target is not None:
                    break
            require(first is not None and target is not None and host["study_group"] not in used, "No qualified held host")
            used.add(host["study_group"])
            other = next(r for r in reversed(hosts) if r["study_group"] != host["study_group"])
            other_raw = voice_pcm(other["raw_sha256"])
            other_start = next(a for a in inputs.positions(4*RATE, length, name+"-other") if inputs.signal_eligible(other_raw[a:a+length]))
            variants = {"ordinary_same_host": inputs.splice(recording, raw, target, first, length),
                        "ordinary_other_voice": inputs.splice(recording, other_raw, target, other_start, length)}
            donor, donor_start = None, None
            if role == "confirmation":
                donor = donors[index]
                exact = np.frombuffer(donor_sources.exact_source(donor, "holdout"), "<i2").astype(np.float64)
                donor_start = next(a for a in inputs.positions(len(exact), length, name+"-AI") if inputs.signal_eligible(exact[a:a+length]))
                variants["synthetic"] = inputs.splice(recording, exact, target, donor_start, length)
            noise *= np.linalg.norm(recording)*10**.25/np.linalg.norm(noise)
            gain = min(1., 31_129/max(np.max(abs(a)+abs(noise)) for a in variants.values()))
            residual = np.rint(gain*noise)
            for arm, value in variants.items():
                payload = pcm(np.rint(gain*value)+residual)
                output.append(({"id": f"{name}-{arm}", "role": role, "group": name, "arm": arm,
                                "family": donor["lane"] if donor else None, "truth": [target, target+length] if arm == "synthetic" else None,
                                "host_sha256": host["raw_sha256"], "host_group": host["study_group"], "source_start": 4*RATE,
                                "ordinary_sha256": other["raw_sha256"], "ordinary_group": other["study_group"],
                                "donor_sha256": donor["original_sha256"] if donor else None,
                                "donor_group": donor["group_id"] if donor else None, "donor_start": donor_start,
                                "same_host_start": first, "other_voice_start": other_start, "edit_range": [target, target+length],
                                "gain": gain, "background": background, "pcm_sha256": digest(payload)}, payload))
        for room in rooms:
            samples, background = cut_room(spec[room], room, 8, blocked)
            gain = min(1., 31_129/max(abs(samples)))
            payload = pcm(gain*samples)
            output.append(({"id": f"{role}-native-{room}", "role": role, "group": f"native-{room}", "arm": "background_only",
                            "truth": None, "family": None, "gain": gain, "background": background, "pcm_sha256": digest(payload)}, payload))
    require(len(output) == len({row["pcm_sha256"] for row, _ in output}) == 43, "Duplicate or missing panel input")
    return output, donor_hash


def freeze(args):
    views, donor_hash = data()
    rows = [row for row, _ in views]
    for key in ("host_group", "ordinary_group", "donor_group"):
        require(not {r.get(key) for r in rows if r["role"] == "selection" and r.get(key)} &
                {r.get(key) for r in rows if r["role"] == "confirmation" and r.get(key)}, "Role speaker ancestry overlaps")
    args.run.mkdir(mode=0o700)
    (args.run / "pcm").mkdir()
    for row, payload in views:
        (args.run / "pcm" / f"{row['id']}.pcm").write_bytes(payload)
    write(args.run / "roster.json", rows)
    write(args.run / "freeze.json", {"schema": "chipmunk.spectra-long-room-holdout/1", "method": METHOD,
          "code_sha256": sha(Path(__file__)), "roster_sha256": sha(args.run / "roster.json"), "donor_manifest_sha256": donor_hash,
          "annotation_sha256": sha(ANNOTATION), "spectra_assets": spectra.pinned_assets(args.weights), "current_assets": voice.ASSETS,
          "previous_points_sha256": [sha(PREVIOUS_EDGE / "selection.json"), sha(PREVIOUS / "points.json")],
          "planned": dict(Counter(row["role"] for row in rows))})
    print("Frozen 43 exact inputs from four participant-disjoint rooms", flush=True)


def raw_input(run, row):
    payload = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
    require(digest(payload) == row["pcm_sha256"] and len(payload) == 2*row["background"]["samples"], "Frozen native clip changed")
    return payload


def score_spectra(args, rows):
    import onnxruntime as ort

    ort.disable_telemetry_events()
    session = spectra.start_session(args.weights)
    for row in rows:
        raw = raw_input(args.run, row)
        windows = []
        for start in range(0, len(raw)//2-3*RATE+1, RATE):
            end = start+3*RATE
            selected = raw[2*start:2*end]
            value, commitment = spectra.candidate_score(session, selected)
            windows.append({"start": start, "end": end, "score": value, "input_sha256": commitment, "pcm_sha256": digest(selected)})
        yield {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "windows": windows}


def score_current(args, rows):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    require(all(sha(args.voice / name) == digest for name, digest in voice.ASSETS.items()), "Incumbent assets changed")
    model = load_voice({"weights": str(args.voice), "assets": voice.ASSETS})
    activity = SpeechActivity(args.voice / "silero_vad.jit")
    for row in rows:
        raw = raw_input(args.run, row)
        values = np.frombuffer(raw, "<i2").astype(np.float32)/32768
        activity_spans = activity.spans(values)
        windows = []
        for start in range(RATE, len(values)-2*RATE+1, 2*RATE):
            end = start+2*RATE
            score = None
            if window_in_speech(activity_spans, start/RATE, end/RATE):
                measured, = model.predict([values[start:end]])
                require(math.isfinite(measured), "Invalid incumbent response")
                score = calibration.logit(measured)
            windows.append({"start": start, "end": end, "score": score, "pcm_sha256": digest(raw[2*start:2*end])})
        yield {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "windows": windows}


def score(args, rows):
    name = f"{args.role}-{args.arm}"
    if args.role == "confirmation":
        require(read(args.run / "points.json")["freeze_sha256"] == sha(args.run / "freeze.json"), "Select fixed points first")
    chosen = [row for row in rows if row["role"] == args.role]
    evaluator = score_spectra if args.arm == "spectra" else score_current
    started = time.monotonic()
    measured = []
    for row in evaluator(args, chosen):
        measured.append(row)
        if len(measured) % 8 == 0:
            print(name, len(measured), flush=True)
    write(args.run / f"{name}.json", {"freeze_sha256": sha(args.run / "freeze.json"), "rows": measured,
                                       "seconds": round(time.monotonic()-started, 3)})


def observations(args, rows, role):
    ledgers = [read(args.run / f"{role}-{arm}.json") for arm in RUNTIME_ARMS]
    require(all(x["freeze_sha256"] == sha(args.run / "freeze.json") for x in ledgers), "Role score commitment changed")
    chosen = [r for r in rows if r["role"] == role]
    for row, experimental, original in zip(chosen, *(x["rows"] for x in ledgers), strict=True):
        require(all((x["id"], x["pcm_sha256"]) == (row["id"], row["pcm_sha256"]) for x in (experimental, original)), "Wrong input")
        report = edge.best_values(experimental["windows"])
        winner = max((x for x in original["windows"] if x["score"] is not None), key=lambda x: x["score"], default=None)
        report[ARMS[0]] = (winner["score"], [winner["start"], winner["end"]]) if winner else (None, None)
        yield row, report


def points(args, rows):
    selected = list(observations(args, rows, "selection"))
    require(all(r["truth"] is None for r, _ in selected), "Point setting includes synthetic truth")
    limit = {name: max(obs[name][0] for _, obs in selected if obs[name][0] is not None) for name in ARMS}
    write(args.run / "points.json", {"freeze_sha256": sha(args.run / "freeze.json"), "points": limit,
          "selection_sha256": {name: sha(args.run / f"selection-{name}.json") for name in RUNTIME_ARMS}})


def report(args, rows):
    point = read(args.run / "points.json")
    require(point["freeze_sha256"] == sha(args.run / "freeze.json") and all(
        sha(args.run / f"selection-{k}.json") == v for k, v in point["selection_sha256"].items()), "Selection commitment changed")
    points = point["points"]
    old_paths = (PREVIOUS_EDGE / "selection.json", PREVIOUS / "points.json")
    require(read(args.run / "freeze.json")["previous_points_sha256"] == [sha(path) for path in old_paths], "Previous points changed")
    prior = read(old_paths[0])["points"] | {ARMS[0]: read(old_paths[1])["points"]["short_odd"]["point"]}
    held = []
    for row, response in observations(args, rows, "confirmation"):
        checks = {name: {"score": value, "range": span, "flag": value is not None and value > points[name],
                         "located": span is not None and voice.useful_range(*span, row["truth"]),
                         "prior_flag": value is not None and value > prior[name]} for name, (value, span) in response.items()}
        combined = [checks[k] for k in (ARMS[0], edge.CONTRAST) if checks[k]["flag"]]
        checks[edge.UNION] = {"flag": bool(combined), "located": any(x["located"] for x in combined), "range": [x["range"] for x in combined]}
        held.append(row | {"checks": checks})
    arms = (*ARMS, edge.UNION)
    groups = {"all": held}
    groups |= {room: [r for r in held if r["background"]["room"] == room] for room in ROLES["confirmation"][1]}
    groups |= {f"{duration}s": [r for r in held if r["background"]["samples"] == duration*RATE] for duration in (6, 8)}
    fixed = {label: {name: edge.bound.counts(group, name) for name in arms} for label, group in groups.items()}
    equal = {str(n): {name: edge.bound.counts(held, name, n) for name in ARMS} for n in (0, 1, 2)}
    previous = {name: {"useful": sum(r["checks"][name]["prior_flag"] and r["checks"][name]["located"] for r in held),
                       "genuine_false_groups": len({r["group"] for r in held if r["truth"] is None and r["checks"][name]["prior_flag"]})} for name in ARMS}
    gains = [r for r in held if r["checks"][edge.UNION]["flag"] and r["checks"][edge.UNION]["located"]
             and not (r["checks"][ARMS[0]]["flag"] and r["checks"][ARMS[0]]["located"])]
    gate = {"new_useful": len(gains), "families": sorted({r["family"] for r in gains}),
            "rooms": sorted({r["background"]["room"] for r in gains}),
            "extra_false_groups": fixed["all"][edge.UNION]["genuine_false_groups"]-fixed["all"][ARMS[0]]["genuine_false_groups"],
            "background_flags": fixed["all"][edge.UNION]["background_only_flags"]}
    gate["passes"] = (gate["new_useful"] >= 2 and len(gate["families"]) >= 2 and len(gate["rooms"]) == 2
                      and gate["extra_false_groups"] <= 1 and gate["background_flags"] == 0)
    reachable = Counter(f"{r['background']['samples']//RATE}s" for r in held if r["truth"] is not None and any(
        voice.useful_range(a, a+RATE, r["truth"]) for a in range(0, r["background"]["samples"], RATE)))
    write(args.run / "decisions.json", held)
    write(args.run / "result.json", {"method": METHOD, "fixed": fixed, "equal_false_groups": equal, "previous_points": previous,
          "matched_pair_ranking": {name: edge.bound.pair_ranking(held, name, voice.HUMAN) for name in ARMS},
          "geometry_reachable": dict(reachable), "gate": gate,
          "commitments": {name: sha(args.run / f"{name}.json") for name in ("freeze", "roster", "points", "decisions",
                            "selection-spectra", "selection-current", "confirmation-spectra", "confirmation-current")}})
    print(gate, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "points", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    parser.add_argument("--role", choices=tuple(ROLES))
    parser.add_argument("--arm", choices=RUNTIME_ARMS)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
        return
    frozen = read(args.run / "freeze.json")
    require(frozen["code_sha256"] == sha(Path(__file__)) and frozen["roster_sha256"] == sha(args.run / "roster.json"), "Frozen roster changed")
    rows = read(args.run / "roster.json")
    if args.phase == "audit":
        expected, _ = data()
        require([r for r, _ in expected] == rows and all(raw == raw_input(args.run, row) for row, raw in expected), "Exact source replay failed")
        print(f"Reproduced all {len(rows)} native mixed and control waveforms.")
    else:
        {"score": score, "points": points, "report": report}[args.phase](args, rows)


if __name__ == "__main__":
    main()
