"""Compare native short detector inputs on original noisy partial edits."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from app.detectors.audio import decode
from scripts import audio_same_speaker_clone_sources as qualified
from scripts import audio_variable_boundary_transfer as source

BASE, RATE, ROOT = source.BASE, source.RATE, source.ROOT
OLD = source.prior
LONG, SHORT, STRIDE = 32000, 19200, 9600
ARMS = ("current", "short", "cross_scale")
ROOMS = ("ES2007c", "IS1004c")
SCHEMA = "chipmunk.audio-short-native-response-transfer/1"
FAMILIES = ("expressive", "stock")
WIDTHS = (10240, 12800, 15360)
POINT = BASE / "audio-original-pcm-boundary-2026-11-12"
HISTORY = ("audio-spectra-long-room-final-2026-11-12", "audio-enrolled-target-2026-11-12", "audio-variable-boundary-2026-11-12")
read, write, sha, digest, require = source.read, source.write, source.sha, source.digest, source.require
METHOD = {
    "input": "Compare unaltered original mono native windows of 2.00 seconds at 1 and 3 seconds with the same installed speech detector on native 1.20-second windows every 0.60 seconds. Both receive their usual inference preprocessing and half-window speech admission. Record exact source ranges and input hashes. No denoising, separation, enrollment, geometry cue, or extra inference weights.",
    "points": "Use the 12 exact genuine earlier point-control PCM files, retaining prior exact 2.00-second responses. Fit no model. Use separate max-genuine points per arm. Cross-scale response is the mean of separate genuine-reference median/IQR-standardized short score and the strongest eligible long score overlapping at least 0.60 seconds. Retain the short range. Freeze all points before held inference.",
    "held": "Six new source episodes spanning four roles of previously qualified LibriSpeech donors, two synthesis families, and two retained AMI rooms. Each room has .64, .80, and .96-second edits. Each episode includes distinct true same-reader and other-reader controls, matched fades/level, identical samples outside the splice, and fresh exact cuts with competing recorded words in at least half the edit. Two isolated room controls are genuine. Voice source banks and rooms have prior project exposure, and selected synthesized byte/PCM inputs and room cuts exclude recorded earlier inputs.",
    "decision": "Report AI flags, exact useful ranges, genuine false groups, isolated-room flags, and unassessed cases. A useful location needs 0.30 IoU and 0.50 edit coverage. Equal false budgets 0, 1, and 2 are descriptive. Advance only for useful gains across both generator families or separately confirmed cohorts with comparable false alarms, including isolated room controls."
}


def order(key):
    return digest(f"{SCHEMA}:{key}".encode())


def stock_rows(meta, seen, used):
    assets = qualified.sources.assets(meta, "stock", "audio-v8/reserved-sources/archives", "audio-v8/source-planning/confirmation-acquisition-plan.json")
    asset = assets["kokoro-directed-test.parquet"]
    require(asset["license"] == "CC-BY-SA-4.0", "Stock recording grant differs")
    results = []
    fields = ["id", "rendition_id", "duration_s", "language", "transcript", "final_text", "tts_engine", "tts_voice", "license", "audio"]
    with pq.ParquetFile(asset["file"]) as parquet:
        for group in range(parquet.num_row_groups):
            entries = parquet.read_row_group(group, columns=fields, use_threads=False).to_pylist()
            for offset, item in enumerate(entries):
                if item["language"] != "en" or item["duration_s"] < 2.8 or item["rendition_id"] in used:
                    continue
                require(item["tts_engine"] == "kokoro-82m" and item["license"] == "cc-by-sa-4.0", "Stock publisher labels differ")
                pin = digest(item["audio"]["bytes"])
                text = qualified.sources.text_keys(item["transcript"]) | qualified.sources.text_keys(item["final_text"])
                if f"bytes:{pin}" in seen or text & seen:
                    continue
                results.append({"id": item["rendition_id"], "lane": "stock", "voice": f"kokoro-82m:{item['tts_voice']}", "license": asset["license"],
                                "locator": qualified.sources.parquet_locator(asset, group, offset), "raw_sha256": pin, "texts": sorted(text),
                                "context": f"stock-script:{item['id']}"})
    return sorted(results, key=lambda r: order(r["raw_sha256"]))


def prior_inputs(meta, seen, ids):
    voices = set()
    used_exact = set()
    blocked = {room: [] for room in ROOMS}
    for i, name in enumerate(HISTORY):
        for row in meta.source(f"prior_noisy_{i}", f"{name}/roster.json"):
            if bg := row.get("background"):
                room = bg.get("room", bg.get("meeting"))
                if room in blocked:
                    offset = bg["capture_start_sample"]
                    blocked[room].append((offset, offset+bg.get("samples", 128000)))
            used_exact.update(v for key in ("host_sha256", "ordinary_sha256", "donor_sha256", "generated_sha256") if (v := row.get(key)))
            for key in ("host", "donor"):
                if value := row.get(key):
                    voices.add(value["canonical_speaker"])
                    used_exact.add(value["source_sha256"])
    for row in meta.source("previous_same_roles", "audio-same-speaker-clone-sources-final-2026-11-12/sources.json"):
        if row["status"] == "qualified":
            for key in ("host", "enrollment", "clone"):
                voices.add(row[key]["voice"])
                used_exact.add(row[key]["original_sha256"])
    seen.update(f"bytes:{value}" for value in used_exact)
    catalogue = meta.source("source_catalogue", "audio-complete-sources-v81/manifest.json")["foregrounds"]
    voices.update(r["canonical_speaker"] for r in catalogue if r["raw_sha256"] in used_exact)
    for row in catalogue:
        if row["raw_sha256"] in used_exact:
            ids.add(row["id"])
    return catalogue, voices, blocked


def genuine(catalogue, used, roles, length, name, host=False):
    for row in catalogue:
        identity = row["canonical_speaker"]
        if row["role"] not in roles or identity in used:
            continue
        decoded = decode.decode_audio(Path(row["source_path"]), source_sha256=row["raw_sha256"])
        require(decoded.mapping["pcm_sha256"] == row["pcm_sha256"], "Genuine source decode differs")
        reserve = decoded.pcm[10*RATE:12*RATE].astype(np.float64)
        starts = source.options(reserve, length, name)
        base = decoded.pcm[4*RATE:10*RATE].astype(np.float64)
        if not starts or (host and not source.options(base, length, name)):
            continue
        start = starts[0]
        proof = {"source_sha256": row["raw_sha256"], "canonical_speaker": identity, "license": "CC-BY-4.0",
                 "cut": source.clock(decoded.mapping, 10*RATE+start, 10*RATE+start+length)}
        if host:
            proof["host_range"] = source.clock(decoded.mapping, 4*RATE, 10*RATE)
        used.add(identity)
        return base, reserve, start, proof
    raise ValueError("No disjoint qualified genuine reader with usable post-prefix audio")


def generated(candidates, loader, tmp, used, used_text, seen, length, name):
    for row in candidates:
        if row["voice"] in used or set(row["texts"]) & used_text:
            continue
        path = tmp / "native-audio"
        path.write_bytes(loader.original(row))
        decoded = decode.decode_audio(path, source_sha256=row["raw_sha256"])
        values = decoded.pcm.astype(np.float64)
        starts = source.options(values, length, name)
        checks = {f"pcm:{decoded.mapping['pcm_sha256']}", *(f"pcm:{qualified.pcm_digest(decoded.pcm[:n])}" for n in (47_648, 48_000, 64_000) if len(values) >= n)}
        if not starts or checks & seen:
            continue
        start = starts[0]
        used.add(row["voice"])
        used_text.update(row["texts"])
        proof = {"source_sha256": row["raw_sha256"], "canonical_speaker": row["voice"], "license": row["license"],
                 "cut": source.clock(decoded.mapping, start, start+length), "family": row["lane"], "archive_sha256": row["locator"]["archive_sha256"]}
        if row["lane"] == "expressive":
            proof["sidecar_sha256"] = loader.expressive[row["id"]]["metadata_sha256"]
        return values, start, proof
    raise ValueError("No unused exact generated source in the held family")


def render():
    meta = qualified.sources.Metadata()
    ids, _, seen, _ = qualified.history(meta)
    entries, used, blocked = prior_inputs(meta, seen, ids)
    candidates = {"expressive": source.donors(meta, ids, seen), "stock": stock_rows(meta, seen, ids)}
    sources = sorted((row for row in entries if row["kind"] == "genuine" and row["license"] == "CC-BY-4.0" and row["samples"] >= 12*RATE),
                     key=lambda row: order(row["raw_sha256"]))
    rooms = {room: (path, pin, active, list(excluded)+blocked[room]) for room, (path, pin, active, _, excluded) in OLD.old.room_inputs().items() if room in ROOMS}
    require(set(rooms) == set(ROOMS), "Expected two participant-disjoint native meeting sessions")
    output, extracted, used_text = [], set(), set()
    with ExitStack() as stack, TemporaryDirectory() as folder:
        loader = qualified.source_reader.SourceReader(stack, meta.bindings)
        for i, (room, width) in enumerate((r, w) for r in ROOMS for w in WIDTHS):
            name = f"held-{i:02d}"
            family = FAMILIES[i % 2]
            host, same, same_at, host_proof = genuine(sources, used, {"confirmation", "selection"}, width, name+"-host", host=True)
            _, other, other_at, other_proof = genuine(sources, used, {"confidence_cal", "fusion_fit"}, width, name+"-ordinary")
            synth, ai_at, ai_proof = generated(candidates[family], loader, Path(folder), used, used_text, seen, width, name+"-synthetic")
            a, noise, background = source.room_cut(rooms[room], room, source.options(host, width, name+"-target"), width)
            noise *= np.linalg.norm(host*32768)*10**.25/np.linalg.norm(noise)
            variants = (("ordinary_same_host", same, same_at, host_proof), ("ordinary_other_voice", other, other_at, other_proof), ("synthetic", synth, ai_at, ai_proof))
            mixed = []
            for arm, signal, start, proof in variants:
                pin = qualified.pcm_digest(signal[start:start+width].astype(np.float32))
                require(pin not in extracted and f"pcm:{pin}" not in seen, "Exact donor extract reuses earlier PCM")
                extracted.add(pin)
                inserted = OLD.old.inputs.splice(host, signal, a, start, width)*32768
                mixed.append((arm, inserted, proof | {"pcm_sha256": pin}))
            gain = min(1., 31_120/max(float(np.max(abs(signal)+abs(noise))) for _, signal, _ in mixed))
            residual = np.rint(gain*noise)
            for arm, signal, proof in mixed:
                source.emit(output, f"{name}-{arm}", name, arm, [a, a+width], background, host_proof, proof, np.rint(gain*signal), residual)
    for room in ROOMS:
        _, noise, background = source.room_cut(rooms[room], room, [32_000], 12_800)
        noise = np.rint(noise*min(1., 31_120/np.max(abs(noise))))
        source.emit(output, f"room-{room}", f"room-{room}", "background_only", None, background, None, None, np.zeros(len(noise)), noise)
    require(len(output) == 20 and len(extracted) == 18, "Held exact denominator differs")
    return output, meta.bindings


def point_sources():
    rows = [r for r in read(POINT / "roster.json") if r["panel"] == "point"]
    lock = read(POINT / "points.json")
    prior = read(POINT / "point.json")
    require(lock["point_sha256"] == sha(POINT / "point.json") and len(rows) == len(prior) == 12, "Genuine-only point receipt changed")
    require(all(r["truth"] is None and (r["id"], r["pcm_sha256"]) == (p["id"], p["pcm_sha256"]) for r, p in zip(rows, prior, strict=True)), "Point PCM source differs")
    return rows, prior


def freeze(run):
    rows, ancestry = render()
    points, _ = point_sources()
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir()
    for row, main, fg, bg in rows:
        for suffix, data in (("", main), ("-foreground", fg), ("-background", bg)):
            (run / "pcm" / f"{row['id']}{suffix}.pcm").write_bytes(data)
    write(run / "roster.json", [row for row, *_ in rows])
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)), "roster_sha256": sha(run / "roster.json"),
          "point_sources": [r["pcm_sha256"] for r in points], "prior_point_sha256": sha(POINT / "points.json"), "ancestry": ancestry,
          "source_helper_sha256": sha(Path(source.__file__)), "current_assets": OLD.old.voice.ASSETS})
    print("Frozen 20 exact held mixes, 12 genuine controls, and native window grids", flush=True)


def measure(args, rows, point):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    OLD.old.spectra.incumbent.network_free()
    model = load_voice({"weights": str(args.voice), "assets": OLD.old.voice.ASSETS})
    vad = SpeechActivity(args.voice / "silero_vad.jit")
    saved = dict(zip((r["id"] for r in rows), point, strict=True)) if point else {}
    counts = Counter()
    result = []
    for row in rows:
        original = OLD.raw(row) if point else OLD.old.raw_input(args.run, row)
        pcm = np.frombuffer(original, "<i2").astype(np.float32)/32768
        spans = vad.spans(pcm)
        record = {"id": row["id"], "pcm_sha256": row["pcm_sha256"]}
        for arm, width, starts in (("current", LONG, [RATE, 3*RATE]), ("short", SHORT, range(0, len(pcm)-SHORT+1, STRIDE))):
            if arm == "current" and point:
                record[arm] = saved[row["id"]]["current"]
                continue
            windows = []
            for a in starts:
                b = a+width
                raw = None
                if OLD.old.window_in_speech(spans, a/RATE, b/RATE):
                    probability, = model.predict([pcm[a:b]])
                    require(math.isfinite(probability), "Current raw score is invalid")
                    raw = OLD.old.calibration.logit(probability)
                    counts[arm] += 1
                windows.append({"start": a, "end": b, "pcm_sha256": digest(original[2*a:2*b]), "score": raw})
            record[arm] = windows
        result.append(record)
        if len(result) % 4 == 0:
            print(f"{len(result)}/{len(rows)} inputs, {sum(counts.values())} forwards", flush=True)
    return result, dict(counts)


def local_pairs(record):
    for short in record["short"]:
        if short["score"] is None:
            continue
        supports = [win["score"] for win in record["current"] if win["score"] is not None
                    and min(win["end"], short["end"])-max(win["start"], short["start"]) >= STRIDE]
        if supports:
            yield short, max(supports)


def responses(record, scalers):
    outcomes = {arm: [(w["score"], [w["start"], w["end"]]) for w in record[arm] if w["score"] is not None] for arm in ARMS[:2]}
    mixed = [(((long-scalers[0][0])/scalers[0][1]+(short["score"]-scalers[1][0])/scalers[1][1])/2, [short["start"], short["end"]])
             for short, long in local_pairs(record)]
    outcomes["cross_scale"] = mixed
    return {arm: list(max(values, key=lambda x: x[0], default=(None, None))) for arm, values in outcomes.items()}


def points(records):
    values = np.array([(long, short["score"]) for record in records for short, long in local_pairs(record)])
    scalers = [[float(mid), max(1e-6, float(hi-lo))] for lo, mid, hi in np.percentile(values, [25, 50, 75], axis=0).T]
    outputs = [responses(record, scalers) for record in records]
    return {"scalers": scalers, "points": {arm: max(x[arm][0] for x in outputs if x[arm][0] is not None) for arm in ARMS}}


def check_native(args, rows, records, point):
    for row, record in zip(rows, records, strict=True):
        raw = OLD.raw(row) if point else OLD.old.raw_input(args.run, row)
        require((row["id"], row["pcm_sha256"]) == (record["id"], record["pcm_sha256"]), "Response source differs")
        for arm, width, starts in (("current", LONG, [RATE, 3*RATE]), ("short", SHORT, range(0, len(raw)//2-SHORT+1, STRIDE))):
            require([w["start"] for w in record[arm]] == list(starts), "Native input clock differs")
            for w in record[arm]:
                a, b = w["start"], w["end"]
                require(b-a == width and w["pcm_sha256"] == digest(raw[2*a:2*b]), "Native input bytes differ")


def readout(args, rows, point_rows, audit):
    lock, held, point = (read(args.run / f"{name}.json") for name in ("points", "held", "point"))
    check_native(args, rows, held["records"], False)
    check_native(args, point_rows, point["records"], True)
    require(lock["values"] == points(point["records"]) and lock["point_sha256"] == sha(args.run / "point.json")
            and held["points_sha256"] == sha(args.run / "points.json"), "Genuine-only calibration points differ")
    found = [row | {"responses": responses(record, lock["values"]["scalers"])} for row, record in zip(rows, held["records"], strict=True)]
    subsets = {"held": found}
    subsets |= {family: [r for r in found if r["group"] in {x["group"] for x in found if x["truth"] and x["donor"]["family"] == family}] for family in FAMILIES}
    reports = {key: {"fixed": {arm: OLD.performance(items, arm, lock["values"]["points"][arm]) for arm in ARMS},
                     "equal_false_groups": {str(n): {arm: OLD.equal_alarm(items, arm, n) for arm in ARMS} for n in (0, 1, 2)}} for key, items in subsets.items()}
    receipt = {"schema": SCHEMA, "method": METHOD, "results": reports, "views": dict(Counter(r["arm"] for r in rows)),
               "new_model_calls": {"point": point["forwards"], "held": held["forwards"]},
               "commitments": {key: sha(args.run / f"{key}.json") for key in ("freeze", "roster", "point", "points", "held")}}
    if audit:
        rerendered, _ = render()
        require([r for r, *_ in rerendered] == rows and read(args.run / "decisions.json") == found and read(args.run / "result.json") == receipt,
                "Native source, windows or decision replay differs")
        for row, *packets in rerendered:
            require(all((args.run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() == raw for suffix, raw in zip(("", "-foreground", "-background"), packets, strict=True)),
                    "Exact rendered source samples differ")
        print("Reproduced 20 exact held waveforms, both native window grids and all decisions", flush=True)
    else:
        write(args.run / "decisions.json", found)
        write(args.run / "result.json", receipt)
        print(reports["held"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "point", "held", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.run)
        return
    lock, rows = read(args.run / "freeze.json"), read(args.run / "roster.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(args.run / "roster.json")
            and lock["source_helper_sha256"] == sha(Path(source.__file__)) and lock["prior_point_sha256"] == sha(POINT / "points.json"), "Native study freeze differs")
    point_rows, prior = point_sources()
    if args.phase == "point":
        records, calls = measure(args, point_rows, prior)
        write(args.run / "point.json", {"records": records, "forwards": calls})
        write(args.run / "points.json", {"values": points(records), "point_sha256": sha(args.run / "point.json"), "freeze_sha256": sha(args.run / "freeze.json")})
    elif args.phase == "held":
        seal = read(args.run / "points.json")
        require(seal["point_sha256"] == sha(args.run / "point.json") and seal["freeze_sha256"] == sha(args.run / "freeze.json"), "Freeze genuine points before held scoring")
        records, calls = measure(args, rows, None)
        write(args.run / "held.json", {"records": records, "forwards": calls, "points_sha256": sha(args.run / "points.json")})
    else:
        readout(args, rows, point_rows, args.phase == "audit")


if __name__ == "__main__":
    main()
