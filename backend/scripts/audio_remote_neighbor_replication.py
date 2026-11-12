"""Replicate the frozen original-audio neighbor cue with coverage controls."""

from __future__ import annotations

import argparse
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from app.detectors.audio import decode
from scripts import audio_enrolled_target_transfer as rooms
from scripts import audio_remote_neighbor_transfer as cue

BASE, ROOT, RATE = cue.BASE, cue.ROOT, cue.RATE
source, qualified, OLD = cue.source, cue.qualified, cue.OLD
DISCOVERY = BASE / "audio-remote-neighbor-final-2026-11-12"
SESSIONS = ("ES2004a", "IS1009a", "ES2010d", "IS1007a")
FAMILIES = ("clone", "expressive", "stock")
WIDTHS = (10240, 12800, 15360)
SCHEMA = "chipmunk.audio-remote-neighbor-replication/1"
read, write, sha, digest, require = cue.read, cue.write, cue.sha, cue.digest, cue.require
METHOD = {
    "cohort": "Twelve new paired host and synthetic source groups, four from each of the PLN clone, ElevenLabs v4 expressive, and Kokoro stock families. Every host has a different ordinary reader, an ordinary same-reader cut, and an unedited delivery. Add four genuine room-only controls. Never reuse recorded earlier input bytes, PCM, scripts, or recent trial speaker roles. Older source banks and all room sessions have prior project exposure.",
    "geometry": "Six-second original unaltered mono inputs. Four rooms each contain one 0.64-second edge edit, one 0.80-second interior edit, and one 0.96-second interior edit. Alternate before 1 second and after 5 seconds for edge; every interior truth interval is wholly inside [1,3) or [3,5) seconds. Exact room captures exclude the bound prior intervals. Require two simultaneous annotated background voices for at least half the edit, at least 1.2 s overall, and at least 80% annotated speech overall. Mix at -5 dB using equal gain and 20 ms fades.",
    "comparators": "Reuse the byte-pinned WavLM remote-context implementation and both earlier fixed points. Use the original two-second incumbent windows, speech admission, and score. Measure the cue once on the complete original clip. For shared time coverage, rank only 0.8-second candidate ranges contained in the union of the actual admitted incumbent windows. The cue still sees its complete original recording when computing its features. Report fixed, equal-false-group, full-span, matched-time, interior and edge results and unavailable cases.",
    "rule": "Useful localization needs .30 temporal IoU and at least .50 edit coverage. Count a genuine group once if any of its three matched controls or its isolated room clip flags. Both former points remain unchanged. Retrospective false-group budgets 0/1/2 compare ranking. No customer confidence calibration."
}


def rank(value):
    return digest(f"{SCHEMA}:{value}".encode())


def check_discovery():
    receipt = read(DISCOVERY / "result.json")
    require(all(sha(DISCOVERY / f"{name}.json") == receipt["commitments"][name] for name in ("freeze", "roster", "points"))
            and read(DISCOVERY / "freeze.json")["runner_sha256"] == sha(Path(cue.__file__)), "Discovery implementation or points changed")
    point = read(cue.POINT / "points.json")
    return {"current": point["values"]["points"]["current"], "neighbor": read(DISCOVERY / "points.json")["neighbor"]}


def histories(meta, seen, ids):
    _, recent, _ = cue.recent.prior_inputs(meta, seen, ids)
    specs, blocked = rooms.word_room_sources()
    for name in ("audio-enrolled-target-2026-11-12", "audio-same-speaker-enrollment-2026-11-12", cue.SHORT.name, cue.VARIABLE.name, DISCOVERY.name):
        folder = BASE / name
        for row in meta.read(f"held_history_{name}", folder / "roster.json"):
            for key in ("host", "donor"):
                if witness := row.get(key):
                    recent.add(witness["canonical_speaker"])
                    seen.add(f"bytes:{witness['source_sha256']}")
            if bg := row.get("background"):
                room = bg.get("room", bg.get("meeting"))
                if room in SESSIONS:
                    start = bg["capture_start_sample"]
                    extent = bg.get("samples", 96000)
                    if name == "audio-enrolled-target-2026-11-12":
                        extent = (folder / "pcm" / f"{row['id']}-original.pcm").stat().st_size//2
                    blocked[room].append((start, start+extent))
    for i, (relative, duration) in enumerate((("audio-overlap-window-transfer-2026-11-12-final/freeze.json", 6), ("audio-disagreement-transfer-2026-11-12/freeze.json", 3))):
        for bg in meta.source(f"related_room_history_{i}", relative)["rooms"]:
            if bg["meeting"] in SESSIONS:
                start = bg["start_sample"]
                blocked[bg["meeting"]].append((start, start+duration*RATE))
    return recent, {room: (*specs[room][:3], blocked[room]) for room in SESSIONS}


def genuine_rows(meta, ids, seen, used):
    files = qualified.sources.assets(meta, "librispeech", qualified.PARQUET, "audio-v8/source-planning/libri-acquisition-plan.json")
    result = []
    for item in files.values():
        with pq.ParquetFile(item["file"]) as table:
            for i in range(table.num_row_groups):
                rows = table.read_row_group(i, columns=["id", "speaker_id", "text"], use_threads=False).to_pylist()
                for offset, row in enumerate(rows):
                    voice, identity = f"librispeech:{row['speaker_id']}", f"libri:{row['id']}"
                    texts = qualified.sources.text_keys(row["text"])
                    if voice in used or row["id"] in ids or identity in ids or not 20 <= len(row["text"].split()) <= 44 or texts & seen:
                        continue
                    result.append({"lane": "genuine", "id": identity, "voice": voice, "texts": sorted(texts), "license": item["license"],
                                   "locator": qualified.sources.parquet_locator(item, i, offset)})
    return sorted(result, key=lambda row: rank(row["id"]))


def clone_rows(meta, ids, seen):
    files = qualified.sources.assets(meta, "pln", qualified.ARCHIVES)
    candidates = read(BASE / qualified.CLONES)
    result = []
    for row in candidates:
        if (row["archive"] not in files or not row["archive"].startswith("PLN-") or row["audio_header_status"] != "ready"
                or not 2.8 <= row["seconds"] <= 12 or row["id"] in ids or row["novelty"]["archive_member_seen"] or set(row["text_identity"]) & seen):
            continue
        item = files[row["archive"]]
        require(item["sha256"] == row["archive_sha256"] and row["license"] == qualified.TAR_LICENSE, "PLN source terms or archive differ")
        result.append(qualified.sources.candidate("clone", row["id"], row["canonical_speaker"], row["transcript"], qualified.sources.tar_locator(item, row),
                                                 row["license"], context=row["source_group"], donor=row["canonical_speaker"],
                                                 native=[row["channels"], row["sample_rate"], row["header_frames"]]))
    return sorted(result, key=lambda row: rank(row["id"]))


def exact(raw, path, history):
    source_pin = digest(raw)
    if f"bytes:{source_pin}" in history:
        return None
    path.write_bytes(raw)
    audio = decode.decode_audio(path, source_sha256=source_pin)
    pcm = {audio.mapping["pcm_sha256"], *(qualified.pcm_digest(audio.pcm[:n]) for n in (47_648, 48_000, 64_000) if len(audio.pcm) >= n)}
    if any(f"pcm:{pin}" in history for pin in pcm):
        return None
    return source_pin, audio


def genuine(candidates, loader, path, used, seen, text, width, key, host=False):
    for row in candidates:
        if (int(rank(("reader-role", row["voice"]))[0], 16) < 8) != host or row["voice"] in used or set(row["texts"]) & text:
            continue
        raw = loader.original(row)
        if qualified.genuine_frames(raw) < 8*RATE:
            continue
        found = exact(raw, path, seen)
        if found is None:
            continue
        pin, decoded = found
        base, reserve = decoded.pcm[:6*RATE].astype(np.float64), decoded.pcm[6*RATE:8*RATE].astype(np.float64)
        donor = source.options(reserve, width, key+"-reserve")
        if not donor or (host and not anchors(base, width, key)):
            continue
        offset = donor[0]
        witness = {"source_sha256": pin, "canonical_speaker": row["voice"], "license": row["license"],
                   "archive_sha256": row["locator"]["archive_sha256"], "cut": source.clock(decoded.mapping, 6*RATE+offset, 6*RATE+offset+width)}
        if host:
            witness["host_range"] = source.clock(decoded.mapping, 0, 6*RATE)
        used.add(row["voice"])
        text.update(row["texts"])
        return base, reserve, offset, witness
    raise ValueError(f"No exact independent genuine source for {key}")


def generated(candidates, loader, path, used, seen, text, contexts, width, key):
    for row in candidates:
        if row["voice"] in used or row["context"] in contexts or set(row["texts"]) & text:
            continue
        raw = loader.original(row)
        pin = digest(raw)
        if f"bytes:{pin}" in seen:
            continue
        native = {}
        if row["lane"] == "clone":
            _, raw, native = qualified.clone_channel(raw, row, loader)
        found = exact(raw, path, seen)
        if found is None:
            continue
        selected_pin, decoded = found
        values = decoded.pcm.astype(np.float64)
        options = source.options(values, width, key+"-synthetic")
        if not options:
            continue
        offset = options[0]
        used.add(row["voice"])
        text.update(row["texts"])
        contexts.add(row["context"])
        witness = {"source_sha256": pin, "decoder_input_sha256": selected_pin, "canonical_speaker": row["voice"], "license": row["license"],
                   "archive_sha256": row["locator"]["archive_sha256"], "cut": source.clock(decoded.mapping, offset, offset+width), "family": row["lane"], **native}
        if row["lane"] in ("clone", "expressive"):
            catalog = loader.clone if row["lane"] == "clone" else loader.expressive
            witness["sidecar_sha256"] = catalog[row["id"]]["metadata_sha256"]
        return values, offset, witness
    raise ValueError(f"No unused exact synthesis source for {key}")


def anchors(base, width, key):
    allowed = source.options(base, width, key+"-target")
    if key.endswith("-early"):
        return [a for a in allowed if a+width <= RATE]
    if key.endswith("-late"):
        return [a for a in allowed if a >= 5*RATE]
    return [a for a in allowed if any(s <= a and a+width <= s+2*RATE for s in (RATE, 3*RATE))]


def capture(spec, room, positions, width):
    path, pin, timeline, used = spec
    with wave.open(str(path), "rb") as file:
        channels, size, rate = file.getparams()[:3]
        require(channels in {1, 2} and size == 2 and rate == RATE, "Native AMI capture clock differs")
        candidates = []
        for tick in range(800, min(len(timeline), file.getnframes()//160)-600, 200):
            a = tick*160
            if any(a < b and x < a+96000 for x, b in used):
                continue
            active = timeline[tick:tick+600]
            overlap = active >= 2
            if np.count_nonzero(active) >= 480 and overlap.sum() >= 120:
                candidates.append((rank((room, tick)), a, overlap))
        candidates.sort(key=lambda row: row[0])
        for position in positions:
            local, width_ticks = position//160, width//160
            for _, offset, overlap in candidates:
                overlap_local = int(overlap[local:local+width_ticks].sum())
                if overlap_local*2 < width_ticks:
                    continue
                file.setpos(offset)
                values = np.frombuffer(file.readframes(96000), "<i2").reshape(-1, channels)
                require(len(values) == 96000 and (channels == 1 or np.array_equal(values[:, 0], values[:, 1])), "Native mono microphone samples differ")
                used.append((offset, offset+96000))
                data = values[:, 0]
                return position, data.astype(np.float64), {"room": room, "samples": 96000, "capture_sha256": pin, "capture_start_sample": offset,
                              "capture_pcm_sha256": digest(data.tobytes()), "overlap_ms": 10*int(overlap.sum()), "edit_overlap_ms": 10*overlap_local}
    raise ValueError(f"No native two-talker capture interval for {room}")


def render():
    meta = qualified.sources.Metadata()
    ids, _, seen, _ = qualified.history(meta)
    used, specs = histories(meta, seen, ids)
    clean = genuine_rows(meta, ids, seen, used)
    donors = {"expressive": source.donors(meta, ids, seen), "stock": cue.recent.stock_rows(meta, seen, ids), "clone": clone_rows(meta, ids, seen)}
    output, extracts, text, contexts = [], set(), set(), set()
    with ExitStack() as stack, TemporaryDirectory() as folder:
        loader = qualified.source_reader.SourceReader(stack, meta.bindings)
        path = Path(folder) / "audio"
        for i, room in enumerate(SESSIONS):
            for j, width in enumerate(WIDTHS):
                coverage = "interior" if j else "early" if i % 2 == 0 else "late"
                key = f"replicate-{i}-{j}-{coverage}"
                family = FAMILIES[(i+j) % len(FAMILIES)]
                host, own, own_start, host_proof = genuine(clean, loader, path, used, seen, text, width, key, host=True)
                _, other, other_start, other_proof = genuine(clean, loader, path, used, seen, text, width, key+"-other")
                ai, ai_start, ai_proof = generated(donors[family], loader, path, used, seen, text, contexts, width, key)
                a, noise, background = capture(specs[room], room, anchors(host, width, key), width)
                noise *= np.linalg.norm(host*32768)*10**.25/np.linalg.norm(noise)
                matched = [("untouched", host*32768, None)]
                for arm, pcm, start, proof in (("ordinary_same_host", own, own_start, host_proof), ("ordinary_other_voice", other, other_start, other_proof),
                                               ("synthetic", ai, ai_start, ai_proof)):
                    pin = qualified.pcm_digest(pcm[start:start+width].astype(np.float32))
                    require(pin not in extracts and f"pcm:{pin}" not in seen, "Selected source fragment repeats a recorded exact input")
                    extracts.add(pin)
                    mixed = OLD.old.inputs.splice(host, pcm, a, start, width)*32768
                    matched.append((arm, mixed, proof | {"pcm_sha256": pin}))
                gain = min(1., 31_120/max(float(np.max(abs(pcm)+abs(noise))) for _, pcm, _ in matched))
                for arm, pcm, proof in matched:
                    source.emit(output, f"{key}-{arm}", key, arm, [a, a+width], background, host_proof, proof, np.rint(gain*pcm), np.rint(gain*noise))
                    output[-1][0].update({"coverage": coverage, "family": family, "asset": "fresh"})
    for room in SESSIONS:
        _, noise, background = capture(specs[room], room, [32000], 12800)
        noise = np.rint(noise*min(1., 31_120/np.max(abs(noise))))
        source.emit(output, f"room-{room}", f"room-{room}", "background_only", None, background, None, None, np.zeros(len(noise)), noise)
        output[-1][0].update({"coverage": "room", "family": None, "asset": "fresh"})
    require(len(output) == 52 and len(extracts) == 36 and Counter(r["family"] for r, *_ in output if r["truth"]) == dict.fromkeys(FAMILIES, 4),
            "Held denominator or family balance differs")
    return output, meta.bindings


def freeze(args):
    points = check_discovery()
    rows, ancestry = render()
    require(len({row["pcm_sha256"] for row, *_ in rows}) == len(rows), "Some held originals repeat")
    args.run.mkdir(mode=0o700)
    (args.run / "pcm").mkdir()
    for row, *parts in rows:
        for suffix, payload in zip(("", "-foreground", "-background"), parts, strict=True):
            (args.run / "pcm" / f"{row['id']}{suffix}.pcm").write_bytes(payload)
    write(args.run / "roster.json", [row for row, *_ in rows])
    write(args.run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "points": points, "ancestry": ancestry,
          "runner_sha256": sha(Path(__file__)), "cue_sha256": sha(Path(cue.__file__)), "discovery_points": sha(DISCOVERY / "points.json"),
          "current_points": sha(cue.POINT / "points.json"), "roster_sha256": sha(args.run / "roster.json"),
          "wavlm_assets": cue.wavlm.assets(args.wavlm), "current_assets": OLD.old.voice.ASSETS})
    print("Frozen 12 genuinely new edited source groups, 40 genuine views and both existing points", flush=True)


def time_restricted(support, windows):
    admitted = []
    for w in windows:
        if w["score"] is None:
            continue
        a, b = w["start"], w["end"]
        if admitted and admitted[-1][1] >= a:
            admitted[-1][1] = max(admitted[-1][1], b)
        else:
            admitted.append([a,b])
    ranked = []
    for a in range(0, 96000-cue.RANGE+1, cue.STEP):
        b = a+cue.RANGE
        if not any(x <= a < b <= y for x,y in admitted):
            continue
        scores = [r["score"] for r in support if a <= r["center"] < b]
        if len(scores) >= 24:
            ranked.append((float(np.median(scores)), [a,b]))
    return list(max(ranked, key=lambda row: row[0], default=(None,None)))


def tables(found, points):
    subsets = {"all": found, "interior": [r for r in found if r["coverage"] in ("interior", "room")],
               "edge": [r for r in found if r["coverage"] in ("early", "late", "room")]}
    subsets.update({family: [r for r in found if r["family"] == family] for family in FAMILIES})
    arms = ("current", "neighbor", "same_time")
    return {group: {"counts": dict(Counter(r["arm"] for r in rows)),
                    "fixed": {arm: OLD.performance(rows, arm, points["current" if arm == "current" else "neighbor"]) for arm in arms},
                    "equal_false_groups": {str(n): {arm: OLD.equal_alarm(rows, arm, n) for arm in arms} for n in (0, 1, 2)},
                    "locations_before_cutoff": {arm: OLD.performance(rows, arm, -np.inf)["useful_ranges"] for arm in arms}}
            for group, rows in subsets.items()}


def report(args, rows, lock, audit):
    found = []
    recordings, current = (read(args.run / f"{name}.json") for name in ("neighbor", "current"))
    require(recordings["freeze"] == current["freeze"] == sha(args.run / "freeze.json"), "Model calls used different source freezes")
    for row, a, b in zip(rows, recordings["records"], current["records"], strict=True):
        data = OLD.old.raw_input(args.run, row)
        require((row["id"], row["pcm_sha256"]) == (a["id"], a["pcm_sha256"]) == (b["id"], b["pcm_sha256"]), "Exact model source differs")
        windows = b["windows"]
        require([w["start"] for w in windows] == [RATE, 3*RATE] and all(w["end"]-w["start"] == 2*RATE
                and digest(data[2*w["start"]:2*w["end"]]) == w["pcm_sha256"] for w in windows), "Current model range source differs")
        old = max((w for w in windows if w["score"] is not None), key=lambda w: w["score"], default=None)
        base = [old["score"], [old["start"], old["end"]]] if old else [None, None]
        found.append(row | {"responses": {"current": base, "neighbor": cue.native_ranges(a["support"]), "same_time": time_restricted(a["support"], windows)}})
    summary = {"schema": SCHEMA, "method": METHOD, "results": tables(found, lock["points"]),
               "new_model_calls": {"neighbor": len(recordings["records"]), "current": sum(w["score"] is not None for r in current["records"] for w in r["windows"])},
               "commitments": {name: sha(args.run / f"{name}.json") for name in ("freeze", "roster", "neighbor", "current")}}
    if audit:
        regenerated, _ = render()
        require([r for r, *_ in regenerated] == rows and summary == read(args.run / "result.json") and found == read(args.run / "decisions.json"),
                "Source or decision replay differs")
        for row, *parts in regenerated:
            require(all((args.run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() == pcm for suffix, pcm in zip(("", "-foreground", "-background"), parts, strict=True)),
                    "Fresh source PCM differs")
        print("Reproduced 52 exact original waveforms, both frozen points, and common-time results", flush=True)
    else:
        write(args.run / "decisions.json", found)
        write(args.run / "result.json", summary)
        print(summary["results"]["all"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "neighbor", "current", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--wavlm", type=Path, default=ROOT / "models/research/wavlm-v13")
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
        return
    lock, rows = read(args.run / "freeze.json"), read(args.run / "roster.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(args.run / "roster.json")
            and lock["cue_sha256"] == sha(Path(cue.__file__)) and lock["points"] == check_discovery(), "Frozen methodology differs")
    if args.phase == "neighbor":
        write(args.run / "neighbor.json", {"freeze": sha(args.run / "freeze.json"), "records": cue.measure(args, rows)})
    elif args.phase == "current":
        OLD.old.spectra.incumbent.network_free()
        write(args.run / "current.json", {"freeze": sha(args.run / "freeze.json"), "records": list(OLD.old.score_current(args, rows))})
    else:
        report(args, rows, lock, args.phase == "audit")


if __name__ == "__main__":
    main()
