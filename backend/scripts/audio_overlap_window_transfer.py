"""Test overlapping shorter voice windows on known short edits under room speech."""

from __future__ import annotations

import argparse
import json
import resource
import time
import wave
from collections import Counter
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from app.detectors.audio import calibration
from app.detectors.audio.content import window_in_speech
from app.detectors.video.audio import voice_windows
from scripts import audio_disagreement_transfer as prior
from scripts import audio_paired_nuisance_transfer as bind
from scripts import audio_short_edit_cases_v172 as edits
from scripts import prepare_natural_overlap_transfer as ami
from scripts.prepare_audio_specialists_v15 import ASSETS

RATE, TRACK, SHORT = 16_000, 96_000, 32_000
PARTS = {"selection": "development", "confirmation": "holdout"}
ARMS = ("current", "long_window", "short_even", "short_odd", "overlap", "screening")
FAMILIES = ("clone", "expressive", "stock")
TREATMENTS = {"ordinary": "genuine_to_genuine_fade20", "partial_ai": "genuine_to_ai_fade20"}
SOURCE_RECEIPT = bind.PANEL / "audio-short-edit-cases-result-v172.json"
ROOM_RECEIPT = bind.PANEL / "audio-stream-presence-transfer-2026-11-11.json"
OLD_ROOMS = bind.BASE / "audio-disagreement-transfer-2026-11-12/freeze.json"
METHOD = {
    "sources": "Use matched clean 20 ms faded 0.8-second genuine and AI inserts, plus a second exact utterance of the genuine host.",
    "assembly": "Place the intact 3 s edited center at a fixed salted offset between 0.75 and 2.25 s. Split the other 3 s host utterance around it.",
    "background": "Use separate six-second recorded AMI scenes at -5 dB, one selection room and two held rooms; match gain across both edit kinds and deliveries.",
    "incumbent": "Production Full profile 4 s windows, 2 s hop, speech admission and file-mean calibration. Six seconds is outside the current whole-file calibration scope.",
    "long_window": "Use the strongest admitted four-second incumbent raw score under the same experimental calibration rule as the short arms.",
    "short_even": "Admitted 2 s windows beginning at 0, 2 and 4 seconds.",
    "short_odd": "Admitted 2 s windows beginning at 1 and 3 seconds.",
    "overlap": "Use the highest lower raw log-odds score among adjacent short windows, with half of the one-second intersection voiced.",
    "operating_points": "For each experimental arm strictly exceed the maximum assessed genuine track score across clean and mixed selection controls.",
    "screening": "When overlap evidence is assessable, use it to retain, withhold or add an AI flag only on incumbent AI or inconclusive results. Preserve original and unassessed.",
    "advance": "At least two usefully localized noisy synthetic edits from two families, no more than one extra noisy false flag and no extra clean false flags. "
               "A useful location reaches 0.30 IoU and 0.50 recall on the sample clock. Advance to wider research only.",
}
read, write, digest, require = bind.read, bind.write, bind.digest, bind.require


def source_pairs(role, quota):
    published = read(SOURCE_RECEIPT)["roles"][role]
    journal = edits.TRIAL / role / "bind-00/output/private-cases.jsonl"
    manifest = edits.sources.MEDIA / role / "private-manifest.json"
    require(digest(journal) == published["private_case_journal_sha256"]
            and digest(manifest) == read(prior.MATERIAL)["roles"][role]["private_manifest_sha256"], "Source receipt differs")
    all_inputs = read(manifest)["foregrounds"]
    by_slot = {row["slot"]: row for row in all_inputs}
    pairs = {}
    with journal.open() as stream:
        for row in map(json.loads, stream):
            if row["condition"] == "clean" and row["arm"] in TREATMENTS.values() and row["status"] == "constructed":
                pairs.setdefault(row["comparison_id"], {})[row["arm"]] = row
    chosen, host_groups = [], set()
    for family in FAMILIES:
        n = 0
        for key in sorted(pairs, key=lambda value: ami.salt(f"overlap-window/1:{role}:{value}")):
            pair = pairs[key]
            control, edit = (pair[value] for value in TREATMENTS.values())
            if edit["lane"] != family or edit["genuine_host_group"] in host_groups:
                continue
            host = by_slot[edit["genuine_host_slot"]]
            options = [row for row in all_inputs if row["group_id"] == host["group_id"] and row["slot"] != host["slot"]
                       and row["status"] == "qualified" and row["prefix_pcm_sha256"] != host["prefix_pcm_sha256"]]
            if not options:
                continue
            alternate = min(options, key=lambda row: row["slot"])
            offset = (3 + int(ami.salt(f"overlap-window-offset/1:{role}:{key}")[:8], 16) % 7) * RATE // 4
            chosen.append((control, edit, alternate, offset))
            host_groups.add(host["group_id"])
            n += 1
            if n == quota:
                break
        require(n == quota, "Too few independent hosts with both matched inserts and a second utterance")
    return chosen, {"v172_journal": digest(journal), "v169_source": digest(manifest)}


def cuts(args):
    earlier = ami.make_plan(args.annotations)
    retained = read(ROOM_RECEIPT)["freeze"]["native_sources"]
    excluded = read(OLD_ROOMS)["rooms"]
    output = []
    with ZipFile(args.annotations) as archive:
        for i, room in enumerate(earlier["rooms"]):
            meeting = room["meeting"]
            path = args.rooms / f"{meeting}.wav"
            source_hash = digest(path)
            require(retained[i] == {"group": i, "sha256": source_hash}, "Pinned AMI native capture changed")
            tracks = [ami.track_words(archive.read(f"words/{meeting}.{person}.words.xml")) for person in "ABCD"]
            stop = max(b for track in tracks for _, b in track)
            active = np.zeros((4, stop), dtype=bool)
            for index, words in enumerate(tracks):
                for a, b in words:
                    active[index, a:b] = True
            blocked = [case["begin_centiseconds"] for case in room["cases"]]
            blocked.extend(row["start_sample"] // 160 for row in excluded if row["meeting"] == meeting)
            choices = []
            for start in range(1200, stop - 600, 600):
                if any(start < item + 300 and start + 600 > item for item in blocked):
                    continue
                count = active[:, start:start+600].sum(axis=0)
                speech, overlap = int(np.count_nonzero(count)), int(np.count_nonzero(count >= 2))
                if speech >= 500 and overlap >= 100:
                    choices.append((start, speech, overlap))
            require(bool(choices), "A disjoint loud room scene is unavailable")
            start, speech, overlap = min(choices, key=lambda row: ami.salt(f"overlap-window-room/1:{meeting}:{row[0]}"))
            with wave.open(str(path), "rb") as stream:
                require(stream.getparams()[:3] == (1, 2, RATE), "Room waveform clock changed")
                stream.setpos(start * 160)
                raw = stream.readframes(TRACK)
            require(len(raw) == 2 * TRACK, "Native noise source is truncated")
            output.append((raw, {"room_index": i, "meeting": meeting, "start_sample": start * 160, "source_sha256": source_hash,
                                 "speech_ms": 10 * speech, "overlap_ms": 10 * overlap, "room_pcm_sha256": prior.spectra.sha(raw)}))
    return output


def retained_case(row):
    path = Path(row["file"])
    require(path.parent == edits.MEDIA / row["role"] and not path.is_symlink() and digest(path) == row["wav_sha256"],
            "Committed edited WAV differs")
    raw = edits.read_pcm(path)
    require(prior.spectra.sha(raw) == row["pcm_s16le_sha256"], "Committed edit PCM differs")
    return raw


def raw_tracks(pair):
    a, b, extra, offset = pair
    prefix = edits.exact_source(extra, a["role"])
    centers = [retained_case(row) for row in (a, b)]
    left, right = edits.controls.TARGET_SAMPLES
    require(centers[0][:2*left] == centers[1][:2*left] and centers[0][2*right:] == centers[1][2*right:], "Matched centers differ outside their insert")
    return [prefix[:2*offset] + center + prefix[2*offset:] for center in centers]


def mix(pair, raw_background):
    host, synthetic = [np.frombuffer(raw, "<i2").astype(np.float64) for raw in pair]
    noise = np.frombuffer(raw_background, "<i2").astype(np.float64)
    scaled = noise * (np.linalg.norm(host) / np.linalg.norm(noise)) * 10**.25
    peak = max(np.max(np.abs(signal) + np.abs(scaled)) for signal in (host, synthetic))
    gain = min(1., 31129 / peak)
    return [np.rint(gain * signal).astype("<i2").tobytes() for signal in (host, synthetic, host+scaled, synthetic+scaled)]


def plan(args):
    cuts_by_room = cuts(args)
    roles = {panel: source_pairs(role, 2) for panel, role in PARTS.items()}
    source_groups = [{row[key] for control, edit, extra, _ in selected for row, key in (
                     (control, "genuine_donor_group"), (edit, "genuine_host_group"), (edit, "source_group"), (extra, "group_id"))}
                     for selected, _ in roles.values()]
    require(not source_groups[0] & source_groups[1], "Cross-role foreground groups overlap")
    metadata, tracks = [], []
    for panel, (sources, _) in roles.items():
        for ordinal, pair in enumerate(sources):
            control, edit, extra, offset = pair
            room = 0 if panel == "selection" else (ordinal + ordinal // 2) % 2 + 1
            background, _ = cuts_by_room[room]
            truth = [n + offset for n in edit["label"]["ai_support_samples"]]
            for index, raw in enumerate(mix(raw_tracks(pair), background)):
                is_ai = index % 2 == 1
                metadata.append({"id": len(tracks), "panel": panel, "family": edit["lane"], "kind": "partial_ai" if is_ai else "ordinary",
                                 "condition": "clean" if index < 2 else "room", "room_index": None if index < 2 else room,
                                 "source_group": edit["source_group"], "host_group": edit["genuine_host_group"],
                                 "source_pcm_sha256": prior.spectra.sha(raw), "truth": truth if is_ai else None})
                tracks.append(raw)
    lock = {"schema": "chipmunk.audio-overlap-window-transfer/1", "method": METHOD, "implementation_sha256": digest(Path(__file__)),
            "source_receipts": {panel: rec for panel, (_, rec) in roles.items()}, "room_source_receipt_sha256": digest(ROOM_RECEIPT),
            "excluded_earlier_intervals_sha256": digest(OLD_ROOMS), "rooms": [meta for _, meta in cuts_by_room],
            "annotations_sha256": digest(args.annotations), "voices": {panel: len(rows) for panel, (rows, _) in roles.items()},
            "assets": ASSETS, "calibration_sha256": calibration.SHA256, "planned": dict(Counter(row["panel"] for row in metadata))}
    return lock, metadata, tracks


def freeze(args):
    lock, metadata, tracks = plan(args)
    args.output.mkdir(mode=0o700)
    folder = args.output / "pcm"
    folder.mkdir()
    for row, raw in zip(metadata, tracks, strict=True):
        (folder / f"{row['id']:03}.pcm").write_bytes(raw)
    write(args.output / "roster.json", metadata)
    write(args.output / "freeze.json", {**lock, "roster_sha256": digest(args.output / "roster.json")})


def checked(args):
    folder = args.output
    lock = read(folder / "freeze.json")
    require(lock["implementation_sha256"] == digest(Path(__file__)) and lock["roster_sha256"] == digest(folder / "roster.json")
            and lock["calibration_sha256"] == digest(calibration.PATH), "Study definition or current calibration changed")
    return lock, read(folder / "roster.json")


def readout(short, spans):
    def best(options):
        return max(options, key=lambda item: item[0], default=None)

    pairs = [(min(a["score"], b["score"]), [b["start"], a["end"]]) for a, b in zip(short[:-1], short[1:], strict=True)
             if a["score"] is not None and b["score"] is not None and window_in_speech(spans, b["start"] / RATE, a["end"] / RATE)]
    return {"short_even": best((row["score"], [row["start"], row["end"]]) for row in short[::2] if row["score"] is not None),
            "short_odd": best((row["score"], [row["start"], row["end"]]) for row in short[1::2] if row["score"] is not None),
            "overlap": best(pairs)}


def score(args, rows, lock):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    prior.spectra.incumbent.network_free()
    model = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    result, calls, started = [], Counter(), time.monotonic()

    def track(*_):
        calls["model"] += 1

    hook = model.register_forward_pre_hook(track)
    try:
        for row in rows:
            raw = (args.output / "pcm" / f"{row['id']:03}.pcm").read_bytes()
            require(len(raw) == 2 * TRACK and prior.spectra.sha(raw) == row["source_pcm_sha256"], "Exact six-second track differs")
            wave = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            spans = activity.spans(wave)
            full = voice_windows(wave, 2., keep=partial(window_in_speech, spans))
            old_values = model.predict([x for _, x in full]) if full else []
            baseline = "unassessed"
            if full:
                assessment = calibration.evaluate(float(np.mean(old_values)), wave, full, model, Deadline(120))
                require(not assessment["scope_supported"], "Six-second current voice calibration scope changed")
                baseline = assessment["decision"]
            records = [{"start": round(start * RATE), "end": round(start * RATE) + len(samples), "raw": float(value),
                        "pcm_sha256": prior.spectra.sha(raw[2*round(start*RATE):2*(round(start*RATE)+len(samples))])}
                       for (start, samples), value in zip(full, old_values, strict=True)]
            short = []
            for start in range(0, TRACK-SHORT+1, RATE):
                end = start + SHORT
                value = None
                if window_in_speech(spans, start / RATE, end / RATE):
                    probability, = model.predict([wave[start:end]])
                    value = calibration.logit(probability)
                short.append({"start": start, "end": end, "score": value, "pcm_sha256": prior.spectra.sha(raw[2*start:2*end])})
            longest = max(records, key=lambda win: win["raw"], default=None)
            full_feature = (calibration.logit(longest["raw"]), [longest["start"], longest["end"]]) if longest else None
            result.append({"id": row["id"], "input_sha256": row["source_pcm_sha256"], "speech_spans": spans, "current": baseline,
                           "full": records, "short": short, "features": {"long_window": full_feature, **readout(short, spans)}})
            if len(result) % 6 == 0:
                print(args.panel, len(result), "files", calls["model"], "voice forwards", flush=True)
    finally:
        hook.remove()
    write(args.output / f"{args.panel}.json", {"freeze_sha256": digest(args.output / "freeze.json"), "rows": result,
          "classifier_forwards": calls["model"], "gate_forwards": gate.calls, "elapsed_s": time.monotonic() - started,
          "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def records(args, panel, rows):
    selected = [row for row in rows if row["panel"] == panel]
    stored = read(args.output / f"{panel}.json")
    require(stored["freeze_sha256"] == digest(args.output / "freeze.json"), "Observation freeze changed")
    pairs = list(zip(selected, stored["rows"], strict=True))
    require(all((row["id"], row["source_pcm_sha256"]) == (obs["id"], obs["input_sha256"]) for row, obs in pairs), "Input ledger mismatch")
    return pairs


def fit(pairs):
    points = {}
    for arm in ARMS[1:-1]:
        controls = [(obs["features"][arm][0], row["host_group"]) for row, obs in pairs
                    if row["kind"] == "ordinary" and obs["features"][arm] is not None]
        require(len({group for _, group in controls}) >= 4, "Insufficient independent short-window calibration support")
        points[arm] = {"point": max(score for score, _ in controls), "genuine_assessed": len(controls),
                       "genuine_groups": len({group for _, group in controls})}
    return points


def decisions(obs, points):
    current = obs["current"]
    current_region = max(obs["full"], key=lambda win: win["raw"], default=None)
    values = {"current": (current, [current_region["start"], current_region["end"]] if current == "ai" else None)}
    for arm in ARMS[1:-1]:
        view = obs["features"][arm]
        values[arm] = ("unassessed", None) if view is None else ("ai", view[1]) if view[0] > points[arm]["point"] else ("inconclusive", None)
    overlap = values["overlap"]
    values["screening"] = overlap if current in ("ai", "inconclusive") and overlap[0] != "unassessed" else values["current"]
    return values


def useful(region, truth):
    left, right = region
    a, b = truth
    shared = max(0, min(right, b) - max(left, a))
    return 2 * shared >= b-a and 10 * shared >= 3 * (right-left + b-a - shared)


def summarize(pairs, points):
    output = {}
    for row, obs in pairs:
        counts = output.setdefault(row["condition"], {arm: Counter() for arm in ARMS})
        for arm, (status, region) in decisions(obs, points).items():
            label = row["kind"]
            count = counts[arm]
            count[f"{label}_{status}"] += 1
            if label == "partial_ai" and status == "ai" and useful(region, row["truth"]):
                count["useful_location"] += 1
                count[f"useful_{row['family']}"] += 1
    return output


def qualifies(held):
    mixed, clean = held["room"], held["clean"]
    return {"noisy_useful_locations": mixed["screening"].get("useful_location", 0) >= 2,
            "noisy_two_families": sum(mixed["screening"].get(f"useful_{family}", 0) > 0 for family in FAMILIES) >= 2,
            "noisy_false_limits": mixed["screening"].get("ordinary_ai", 0) <= mixed["long_window"].get("ordinary_ai", 0) + 1,
            "clean_false_limits": clean["screening"].get("ordinary_ai", 0) <= clean["long_window"].get("ordinary_ai", 0)}


def run(args):
    if args.stage == "freeze":
        freeze(args)
        return
    lock, rows = checked(args)
    point_path = args.output / "points.json"
    freeze_sha = digest(args.output / "freeze.json")
    selection_sha = args.output / "selection.json"
    if args.stage == "fit":
        write(point_path, {"freeze_sha256": freeze_sha, "selection_sha256": digest(selection_sha), "points": fit(records(args, "selection", rows))})
        return
    if args.stage == "report" or args.panel == "confirmation":
        calibrated = read(point_path)
        require(calibrated["freeze_sha256"] == freeze_sha and calibrated["selection_sha256"] == digest(selection_sha),
                "Held score requires frozen selection points")
    if args.stage == "score":
        score(args, [row for row in rows if row["panel"] == args.panel], lock)
    else:
        outcome = {panel: summarize(records(args, panel, rows), calibrated["points"]) for panel in PARTS}
        criteria = qualifies(outcome["confirmation"])
        write(args.output / "result.json", {"results": outcome, "advances": all(criteria.values()), "criteria": criteria,
              "freeze_sha256": freeze_sha, "points_sha256": digest(point_path),
              "ledgers_sha256": {panel: digest(args.output / f"{panel}.json") for panel in PARTS}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "fit", "report"))
    for name in ("output", "annotations", "rooms", "weights"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--panel", choices=PARTS, default="selection")
    run(parser.parse_args())
