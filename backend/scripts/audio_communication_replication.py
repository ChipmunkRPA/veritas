"""Confirm the frozen paired call-processing check on fresh source groups."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts import audio_communication_transfer as trial

FRESH = trial.source
QUALIFIED = FRESH.qualified
BASE, ROOT = FRESH.BASE, FRESH.ROOT
RATE, SAMPLES, WINDOWS, ARMS = trial.RATE, trial.SAMPLES, trial.WINDOWS, trial.ARMS
read, write, sha, digest, require = trial.read, trial.write, trial.sha, trial.digest, trial.require
DISCOVERY = BASE / "audio-communication-transfer-2026-11-12"
EARLIER = BASE / "audio-remote-neighbor-replication-2026-11-12"
ANNOTATIONS = trial.OLD.ANNOTATION
SCHEMA = "chipmunk.audio-communication-replication/1"
ROOMS = ("ES2007c", "IS1004c", "ES2012b", "IS1001b")
FAMILIES = (("clone", "expressive", "stock"), ("expressive", "stock", "clone"),
            ("stock", "clone", "expressive"), ("clone", "expressive", "stock"))
WIDTHS = (10240, 12800, 15360)
HISTORY = ("audio-variable-boundary-2026-11-12", "audio-short-native-response-2026-11-12",
           "audio-remote-neighbor-final-2026-11-12", "audio-measured-spatial-transfer-final-2026-11-12")
METHOD = {
    "scope": "Twelve new groups with an unedited host, an ordinary same-reader splice, an ordinary other-reader splice and a generated splice. Add four background-only examples. Discovery and confirmation exclude shared declared foreground speakers, original source bytes, exact donor fragments and meeting participants. Retained publisher archives and some expressive and stock voice profiles have earlier unrelated project exposure.",
    "geometry": "Freeze exact six-second mono mixtures with one 0.64 s edge edit, one 0.80 s interior edit and one 0.96 s interior edit per meeting. Two edges are early and two late. Match levels and 20 ms fades. The recorded competing voices overlap at least half of each exact edit, at -5 dB whole-clip SNR relative to the untouched host. Keep all samples outside each edit unchanged within its group.",
    "transform": "Use the original FFmpeg bandpass, compressor and voice Opus chain. Both deliveries must map to the same six-second 16 kHz clock.",
    "decision": "Reapply both fixed two-second native windows, original activity admission and original strict point. Paired score is the lower score on each shared admitted interval. A useful range reaches at least .30 temporal IoU and .50 edit coverage. False alarms count distinct genuine source groups. Advance only if the paired check retains every useful original location, finds at least one useful location, and removes at least one genuine false group without introducing new ones. No customer confidence change.",
}


def participants(annotations, sessions):
    with ZipFile(annotations) as archive:
        root = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
    members = {meeting.get("observation"): {p.get("global_name") for p in meeting if p.get("global_name")}
               for meeting in root.iter() if meeting.get("observation") in sessions}
    require(set(members) == set(sessions) and all(len(v) == 4 for v in members.values()), "Expected four declared speakers per meeting")
    group = set.union(*members.values())
    require(len(group) == len(sessions) * 4, "The retained meetings share participants")
    return group


def discovery_point(meta):
    published = meta.read("published_discovery", ROOT / "docs/evaluation-panels/audio-communication-transfer-2026-11-12.json")
    frozen = meta.read("discovery_freeze", DISCOVERY / "freeze.json")
    require(published["commitments"]["freeze"] == meta.bindings["discovery_freeze"], "Published discovery freeze changed")
    require(frozen["chain"] == trial.CHAIN and frozen["codec"] == list(trial.CODEC)
            and frozen["windows"] == [list(x) for x in WINDOWS] and frozen["model_assets"] == trial.OLD.voice.ASSETS
            and frozen["cutoff"] == published["fixed_point"], "Discovery processing, model or point differs")
    earlier = meta.read("discovery_source_groups", EARLIER / "roster.json")
    require(meta.bindings["discovery_source_groups"] == frozen["sources"]["roster.json"], "Discovery source roster differs")
    return frozen["cutoff"], earlier


def source_roles(meta):
    point, discovery = discovery_point(meta)
    ids, contexts, seen, _ = QUALIFIED.history(meta)
    used, _ = FRESH.histories(meta, seen, ids)
    previous = meta.source("earlier_short_room_roster", "audio-spectra-long-room-final-2026-11-12/roster.json")
    native = trial.OLD.room_inputs()
    rooms = {name: (path, pin, activity, list(past)) for name, (path, pin, activity, _, past) in native.items() if name in ROOMS}
    rooms.update(FRESH.source.room_specs(meta))
    require(set(rooms) == set(ROOMS) and not participants(ANNOTATIONS, ROOMS) & participants(ANNOTATIONS, FRESH.SESSIONS),
            "Fresh background sessions or participants cross the discovery role")
    blocked = defaultdict(set)
    earlier = list(discovery) + list(previous)
    for name in HISTORY:
        earlier.extend(meta.source(f"used_{name}", f"{name}/roster.json"))
    for row in earlier:
        for role in ("host", "donor"):
            if item := row.get(role):
                used.add(item["canonical_speaker"])
                for key in ("source_sha256", "decoder_input_sha256", "pcm_sha256"):
                    if key in item:
                        seen.add(f"{'pcm' if key == 'pcm_sha256' else 'bytes'}:{item[key]}")
        if bg := row.get("background"):
            name = bg.get("room", bg.get("meeting"))
            if name in rooms and "capture_start_sample" in bg:
                start = bg["capture_start_sample"]
                blocked[name].add((start, start + bg.get("samples", SAMPLES)))
    for name, spec in rooms.items():
        spec[3][:] = sorted(set(spec[3]) | blocked[name])
    return point, discovery, ids, contexts, seen, used, rooms


def render():
    meta = QUALIFIED.sources.Metadata()
    point, discovery, ids, contexts, seen, used, rooms = source_roles(meta)
    genuine = FRESH.genuine_rows(meta, ids, seen, used)
    donors = {"clone": FRESH.clone_rows(meta, ids, seen), "expressive": FRESH.source.donors(meta, ids, seen),
              "stock": FRESH.cue.recent.stock_rows(meta, seen, ids)}
    discovery_voices = {item["canonical_speaker"] for row in discovery for role in ("host", "donor") if (item := row.get(role))}
    prior_exposed = (used & {row["voice"] for family in ("expressive", "stock") for row in donors[family]}) - discovery_voices
    # The retained pools have only one entirely unused profile in each family.
    # Keep exact prior recordings excluded and allow profile reuse outside the discovery study.
    used.difference_update(prior_exposed)
    data, text, cuts = [], set(), set()
    with ExitStack() as stack, TemporaryDirectory() as folder:
        reader = QUALIFIED.source_reader.SourceReader(stack, meta.bindings)
        path = Path(folder) / "native"
        for i, (room, families) in enumerate(zip(ROOMS, FAMILIES, strict=True)):
            for j, (family, width) in enumerate(zip(families, WIDTHS, strict=True)):
                coverage = "interior" if j else "early" if i % 2 == 0 else "late"
                key = f"call-{i}-{j}-{coverage}"
                host, same, same_at, h = FRESH.genuine(genuine, reader, path, used, seen, text, width, key, host=True)
                _, other, other_at, o = FRESH.genuine(genuine, reader, path, used, seen, text, width, key + "-other")
                synth, synth_at, s = FRESH.generated(donors[family], reader, path, used, seen, text, contexts, width, key)
                s["profile_exposed_in_other_studies"] = s["canonical_speaker"] in prior_exposed
                start, background, witness = FRESH.capture(rooms[room], room, FRESH.anchors(host, width, key), width)
                background *= np.linalg.norm(host * 32768) * 10 ** .25 / np.linalg.norm(background)
                samples = [("untouched", host * 32768, None)]
                for arm, source, offset, pin in (("ordinary_same_host", same, same_at, h), ("ordinary_other_voice", other, other_at, o),
                                                  ("synthetic", synth, synth_at, s)):
                    fragment = QUALIFIED.pcm_digest(source[offset:offset + width].astype(np.float32))
                    require(fragment not in cuts and f"pcm:{fragment}" not in seen, "Donor fragment crossed roles")
                    cuts.add(fragment)
                    insert = trial.OLD.inputs.splice(host, source, start, offset, width) * 32768
                    samples.append((arm, insert, pin | {"pcm_sha256": fragment}))
                gain = min(1., 31_120 / max(float(np.max(abs(pcm) + abs(background))) for _, pcm, _ in samples))
                for arm, pcm, pin in samples:
                    FRESH.source.emit(data, f"{key}-{arm}", key, arm, [start, start + width], witness, h, pin,
                                      np.rint(gain * pcm), np.rint(gain * background))
                    data[-1][0].update(family=family, coverage=coverage)
            _, background, witness = FRESH.capture(rooms[room], room, [32000], 12800)
            background = np.rint(background * min(1., 31_120 / np.max(abs(background))))
            key = f"room-{i}"
            FRESH.source.emit(data, key, key, "background_only", None, witness, None, None, np.zeros(SAMPLES), background)
            data[-1][0].update(family=None, coverage="room")
    require(len(data) == 52 and len(cuts) == 36 and len({r["pcm_sha256"] for r, *_ in data}) == len(data)
            and Counter(r["family"] for r, *_ in data if r["truth"]) == dict.fromkeys(FRESH.FAMILIES, 4),
            "Fresh rendered denominator differs")
    require(not {r["pcm_sha256"] for r, *_ in data} & {r["pcm_sha256"] for r in discovery}, "Original discovery waveform repeats")
    return data, point, meta.bindings


def prepare(run):
    rows, point, source_bindings = render()
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir()
    (run / "deliveries").mkdir()
    recordings = []
    for index, (row, raw, foreground, background) in enumerate(rows):
        for suffix, pcm in (("", raw), ("-foreground", foreground), ("-background", background)):
            (run / "pcm" / f"{row['id']}{suffix}.pcm").write_bytes(pcm)
        path = run / "deliveries" / f"{index:02d}.opus"
        encoded, delivered = trial.transform(raw, path)
        require(delivered != raw, "Call processing must change the PCM")
        path.with_suffix(".pcm").write_bytes(delivered)
        recordings.append(row | {"file": path.name, "encoded_sha256": encoded, "delivered_pcm_sha256": digest(delivered)})
    write(run / "roster.json", recordings)
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "cutoff": point, "chain": trial.CHAIN,
                                "codec": trial.CODEC, "windows": WINDOWS, "model_assets": trial.OLD.voice.ASSETS,
                                "roster_sha256": sha(run / "roster.json"), "runner_sha256": sha(Path(__file__)),
                                "transform_sha256": sha(Path(trial.__file__)), "annotation_sha256": sha(ANNOTATIONS),
                                "source_bindings": source_bindings})
    print(f"Froze {len(recordings)} original and processed pairs on twelve new foreground groups", flush=True)


def panel(run):
    lock = read(run / "freeze.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["transform_sha256"] == sha(Path(trial.__file__))
            and lock["roster_sha256"] == sha(run / "roster.json"), "Frozen sources or implementation changed")
    for row in read(run / "roster.json"):
        path = run / "deliveries" / row["file"]
        original = (run / "pcm" / f"{row['id']}.pcm").read_bytes()
        delivered = path.with_suffix(".pcm").read_bytes()
        require(len(original) == len(delivered) == 2 * SAMPLES and digest(original) == row["pcm_sha256"]
                and digest(delivered) == row["delivered_pcm_sha256"] and sha(path) == row["encoded_sha256"], "Delivery commitment changed")
        yield row, original, delivered, path


def score(run, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    trial.OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in trial.OLD.voice.ASSETS.items()), "Pinned voice assets changed")
    model = load_voice({"weights": str(weights), "assets": trial.OLD.voice.ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    records = []
    for i, (row, original, delivered, _) in enumerate(panel(run), 1):
        responses = {}
        for arm, pcm in zip(ARMS[:2], (original, delivered), strict=True):
            values = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
            spans = activity.spans(values)
            windows = []
            for a, b in WINDOWS:
                value = None
                if trial.OLD.window_in_speech(spans, a / RATE, b / RATE):
                    measured, = model.predict([values[a:b]])
                    require(math.isfinite(measured), "Invalid voice response")
                    value = trial.OLD.calibration.logit(measured)
                windows.append({"start": a, "end": b, "score": value, "pcm_sha256": digest(pcm[2 * a:2 * b])})
            responses[arm] = windows
        records.append({"id": row["id"], "windows": responses})
        if i % 13 == 0:
            print(f"Scored {i}/52 original and processed pairs", flush=True)
    write(run / "scores.json", {"freeze_sha256": sha(run / "freeze.json"), "records": records})


def report(run, audit=False):
    scores = read(run / "scores.json")
    require(scores["freeze_sha256"] == sha(run / "freeze.json"), "Voice responses refer to a different freeze")
    point = read(run / "freeze.json")["cutoff"]
    results = []
    for proof, (row, original, delivered, path) in zip(scores["records"], panel(run), strict=True):
        require(proof["id"] == row["id"], "Voice scores assigned to a different file")
        a, b = (proof["windows"][arm] for arm in ARMS[:2])
        trial.validate_windows(a, original)
        trial.validate_windows(b, delivered)
        if audit:
            audio = decode_audio(path, source_sha256=sha(path), max_seconds=7)
            require((audio.pcm * 32768).astype("<i2").tobytes() == delivered, "Processed original did not re-decode exactly")
        paired = [{**w, "score": min(v["score"], w["score"]) if None not in (v["score"], w["score"]) else None}
                  for v, w in zip(a, b, strict=True)]
        results.append(row | {"responses": dict(zip(ARMS, map(trial.best, (a, b, paired)), strict=True))})
    subsets = {"all": results, "interior": [r for r in results if r["coverage"] in {"interior", "room"}],
               "edge": [r for r in results if r["coverage"] != "interior"]}
    subsets.update({family: [r for r in results if r["family"] == family] for family in FRESH.FAMILIES})
    counts = {name: {"inputs": dict(Counter(r["arm"] for r in rows)),
                     "fixed": {arm: FRESH.OLD.performance(rows, arm, point) for arm in ARMS},
                     "equal_false_groups": {str(n): {arm: FRESH.OLD.equal_alarm(rows, arm, n) for arm in ARMS} for n in range(3)}}
              for name, rows in subsets.items()}
    useful, false = {}, {}
    for arm in ARMS:
        flags = [r for r in results if r["responses"][arm][0] is not None and r["responses"][arm][0] > point]
        useful[arm] = {r["id"]: r["responses"][arm][1] for r in flags
                       if r["truth"] and trial.OLD.voice.useful_range(*r["responses"][arm][1], r["truth"])}
        false[arm] = {r["group"] for r in flags if not r["truth"]}
    retained = sum(useful["both"].get(key) == span for key, span in useful["original"].items())
    removed, added = len(false["original"] - false["both"]), len(false["both"] - false["original"])
    summary = {"schema": SCHEMA, "fixed_point": point, "tables": counts, "exact_useful_original_locations_retained": retained,
               "original_false_groups_removed": removed, "new_false_groups": added,
               "passes_advancement": bool(useful["both"] and retained == len(useful["original"]) and removed and not added),
               "new_classifier_windows": {arm: sum(w["score"] is not None for r in scores["records"] for w in r["windows"][arm]) for arm in ARMS[:2]},
               "commitments": {name: sha(run / f"{name}.json") for name in ("freeze", "roster", "scores")}}
    if audit:
        rerendered, _, ancestry = render()
        require(ancestry == read(run / "freeze.json")["source_bindings"], "Rerendered ancestry changed")
        for old, (row, *parts) in zip(results, rerendered, strict=True):
            require(all(old[key] == value for key, value in row.items()), "Rendered source record changed")
            for suffix, payload in zip(("", "-foreground", "-background"), parts, strict=True):
                require((run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() == payload, "Rerendered component samples changed")
        require(summary == read(run / "aggregate.json"), "Frozen results changed")
        print("Rebuilt all 52 original waveforms and re-decoded all 52 processed deliveries", flush=True)
    else:
        write(run / "aggregate.json", summary)
        print(counts["all"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run)
    elif args.phase == "score":
        score(args.run, args.weights)
    else:
        report(args.run, args.phase == "audit")


if __name__ == "__main__":
    main()
