"""Test original-audio paired boundaries across edit durations and separate rooms."""

from __future__ import annotations

import argparse
import wave
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from app.detectors.audio import decode
from scripts import audio_original_pcm_boundary_transfer as prior
from scripts import audio_same_speaker_clone_sources as qualified

BASE, ROOT, RATE, STEP = prior.BASE, prior.ROOT, prior.RATE, prior.STEP
POINT = BASE / "audio-original-pcm-boundary-2026-11-12"
ROOM_BANK = BASE / "spectra-recovery-2026-11-11"
ROOMS = ("ES2012b", "IS1001b")
WIDTHS = (6400, 10240, 12800, 16640)
ARMS = ("current", "boundary", "variable")
SCHEMA = "chipmunk.audio-variable-boundary-transfer/1"
read, write, sha, digest, require = prior.read, prior.write, prior.sha, prior.digest, prior.require
METHOD = {
    "sources": "Eight held episodes from distinct LibriSpeech hosts, different-reader genuine controls, and unused exact English ElevenLabs v4 archived recordings with eight declared donor voices and unique texts. Use only samples after the previously analyzed genuine four-second prefixes. Balance four edit durations and two AMI rooms. Two isolated room controls remain genuine. No previously scored exact archive recording is eligible as generated donor in the bound source history.",
    "rooms": "Fresh nonoverlapping six-second intervals from two native AMI rooms absent from the earlier paired-boundary panel. Earlier studies used both rooms. Require at least 80% speech, at least 1.2 s of concurrent annotated words overall, and at least max(240 ms, half the edit length) with two annotated background voices inside each edit. Use -5 dB whole-clip level and matched 20 ms fades/gain/PCM outside exact splice.",
    "candidates": "Apply the original paired endpoint score unchanged, at .40, .64, .80 and 1.04-second intervals. Scale each duration by its own median and interquartile range from the same 12 already measured genuine point controls. Use the strongest standardized interval. Replay the earlier fixed .80-second arm and current native two-second model, with their earlier fixed points unchanged.",
    "outcomes": "Freeze the maximum variable response on the same 12 genuine controls before any held inference. Count a genuine host group once if either matched ordinary splice flags. Separately show ordinary-splice localized cues and synthetic localizations at fixed and retrospective equal false-group budgets 0, 1 and 2. Useful requires 0.30 time IoU and 0.50 edit coverage. Variable-duration advantage requires additional useful locations at the same false groups and no isolated-room flag before a new prospective trial.",
}


def order(key):
    return digest(f"{SCHEMA}:{key}".encode())


def clock(mapping, a, b):
    ranges = decode.source_ranges(mapping, a, b)
    require(len(ranges) == 1 and ranges[0]["source_end_pts"]-ranges[0]["source_start_pts"] == b-a, "Continuous native source clock required")
    return ranges[0]


def options(pcm, width, key):
    scaled = pcm * 32768
    return [a for a in prior.old.inputs.positions(len(pcm), width, key) if prior.old.inputs.signal_eligible(scaled[a:a+width])]


def room_specs(meta):
    receipt = meta.source("earlier_spectra_rooms", "spectra-recovery-2026-11-11/comparison/freeze.json")
    harmonic = meta.source("earlier_harmonic_rooms", "audio-harmonic-overlap-2026-11-11/run/roster.json")
    annotations = prior.old.ANNOTATION
    require(sha(annotations) == prior.old.ami.ANNOTATION_SHA256, "AMI annotation release changed")
    plan = prior.old.ami.make_plan(annotations)
    result = {}
    with ZipFile(annotations) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        prior_rooms = {r["room"] for r in read(POINT / "roster.json")}
        people = {m.get("observation"): {p.get("global_name") for p in m if p.get("global_name")} for m in meetings.iter() if m.get("observation")}
        require(len(set.union(*(people[r] for r in ROOMS))) == 8 and not set(ROOMS) & prior_rooms
                and not set.union(*(people[r] for r in ROOMS)) & set.union(*(people[r] for r in prior_rooms)), "Rooms or participants overlap earlier boundary trial")
        for item in receipt["prospective_backgrounds"]:
            room = item["meeting"]
            path = ROOM_BANK / "ami" / f"{room}.Array1-01.wav"
            require(sha(path) == item["archive_sha256"], "Native AMI microphone changed")
            tracks = [prior.old.ami.track_words(archive.read(f"words/{room}.{voice}.words.xml")) for voice in "ABCD"]
            end = max(b for ranges in tracks for _, b in ranges)
            active = np.zeros((4, end), dtype=bool)
            for i, ranges in enumerate(tracks):
                for a, b in ranges:
                    active[i, a:b] = True
            blocked = [(r["source_start_sample"], r["source_end_sample"]) for r in harmonic if r.get("room") == room and "source_start_sample" in r]
            blocked += [(r["begin_centiseconds"]*160, (r["begin_centiseconds"]+300)*160)
                        for earlier in plan["rooms"] if earlier["meeting"] == room for r in earlier["cases"]]
            start = item["original_start_sample"]
            blocked.append((start, start+3*RATE))
            result[room] = path, item["archive_sha256"], active.sum(axis=0), blocked
    require(set(result) == set(ROOMS), "Expected two qualified native rooms")
    return result


def room_cut(spec, room, anchors, width):
    path, pin, active, blocked = spec
    count = 6*RATE
    with wave.open(str(path), "rb") as stream:
        require(stream.getparams()[:3] == (1, 2, RATE), "Native AMI microphone format changed")
        limit = min(len(active), stream.getnframes()//160)-600
        choices = []
        for tick in range(800, limit, 200):
            start = tick*160
            if any(start < end and other < start+count for other, end in blocked):
                continue
            activity = active[tick:tick+600]
            overlap = activity >= 2
            if np.count_nonzero(activity) >= 480 and overlap.sum() >= 120:
                choices.append((order((room, tick)), start, overlap))
        for anchor in anchors:
            local_start = anchor//160
            expected = max(24, width//320)
            admissible = [(rank, start, overlap) for rank, start, overlap in choices
                          if int(overlap[local_start:local_start+width//160].sum()) >= expected]
            if admissible:
                _, start, overlap = min(admissible, key=lambda x: x[0])
                stream.setpos(start)
                data = stream.readframes(count)
                require(len(data) == 2*count, "Native AMI microphone window is incomplete")
                blocked.append((start, start+count))
                return anchor, np.frombuffer(data, "<i2").astype(np.float64), {
                    "room": room, "capture_sha256": pin, "samples": count, "capture_start_sample": start,
                    "overlap_ms": 10*int(overlap.sum()), "edit_overlap_ms": 10*int(overlap[local_start:local_start+width//160].sum()),
                    "capture_pcm_sha256": digest(data)}
    raise ValueError(f"No eligible overlapping six-second capture in {room}")


def donors(meta, ids, seen):
    folder = "audio-v8/reserved-sources/archives"
    files = qualified.sources.assets(meta, "stock_expressive", folder, "audio-v8/source-planning/confirmation-acquisition-plan.json")
    meta.source("clone_candidates", qualified.CLONES)
    rows = meta.source("expressive_candidates", "audio-expressive-inventory-v66/candidates.json")
    result = []
    for row in rows:
        if (row["id"] in ids or row["duration_estimate_s"] < 2.8 or row["novelty"]["recorded_bytes_seen"]
                or row["novelty"]["declared_text_seen"] or f"bytes:{row['sha256']}" in seen or set(row["text_identity"]) & seen):
            continue
        file = files[row["archive"]]
        require(file["license"] == row["license"] == "CC-BY-4.0" and row["generator"] == "ElevenLabs-v4", "Expressive source terms differ")
        result.append(qualified.sources.candidate("expressive", row["id"], row["canonical_speaker"], row["transcript"],
                                                 qualified.sources.tar_locator(file, row), row["license"], context=row["challenge_group"],
                                                 raw_sha256=row["sha256"], subset=row["source_subset"]))
    return sorted(result, key=lambda r: order(r["raw_sha256"]))


def render():
    meta = qualified.sources.Metadata()
    ids, _, seen, _ = qualified.history(meta)
    earlier_sources = meta.source("prior_same_speaker_roles", "audio-same-speaker-clone-sources-final-2026-11-12/sources.json")
    used_voices = {r["host"]["voice"] for r in earlier_sources if r["status"] == "qualified"}
    old_roster = meta.source("prior_long_roles", "audio-spectra-long-room-final-2026-11-12/roster.json")
    used_pins = {r[key] for r in old_roster for key in ("host_sha256", "ordinary_sha256") if r.get(key)}
    catalogue = meta.source("held_genuine_catalogue", "audio-complete-sources-v81/manifest.json")
    used_voices |= {r["canonical_speaker"] for r in catalogue["foregrounds"] if r["raw_sha256"] in used_pins}
    subjects = sorted([r for r in catalogue["foregrounds"] if r["kind"] == "genuine" and r["license"] == "CC-BY-4.0" and r["samples"] >= 12*RATE],
                      key=lambda r: order(r["raw_sha256"]))
    generated = donors(meta, ids, seen)
    rooms = room_specs(meta)
    generated_voices, text, audio, output = set(), set(), set(), []
    with ExitStack() as stack, TemporaryDirectory() as folder:
        loader = qualified.source_reader.SourceReader(stack, meta.bindings)
        for i, (room, width) in enumerate((r, w) for r in ROOMS for w in WIDTHS):
            name = f"held-{i:02d}"
            for role in ("confirmation", "confidence_cal"):
                for row in subjects:
                    if row["role"] != role or row["canonical_speaker"] in used_voices:
                        continue
                    parsed = decode.decode_audio(Path(row["source_path"]), source_sha256=row["raw_sha256"])
                    require(parsed.mapping["pcm_sha256"] == row["pcm_sha256"], "Genuine decode differs from qualified source")
                    reserve = parsed.pcm[10*RATE:12*RATE].astype(np.float64)
                    reserve_starts = options(reserve, width, name+role)
                    base = parsed.pcm[4*RATE:10*RATE].astype(np.float64)
                    if not reserve_starts or (role == "confirmation" and not options(base, width, name)):
                        continue
                    used_voices.add(row["canonical_speaker"])
                    start = reserve_starts[0]
                    proof = {"source_sha256": row["raw_sha256"], "canonical_speaker": row["canonical_speaker"], "license": row["license"],
                             "cut": clock(parsed.mapping, 10*RATE+start, 10*RATE+start+width)}
                    if role == "confirmation":
                        host, same, same_at = base, reserve, start
                        host_proof = proof | {"host_range": clock(parsed.mapping, 4*RATE, 10*RATE)}
                    else:
                        other, other_at, other_proof = reserve, start, proof
                    break
                else:
                    raise ValueError("Insufficient role-disjoint genuine extracts")
            for donor in generated:
                if donor["voice"] in generated_voices or set(donor["texts"]) & text:
                    continue
                asset = Path(folder) / "source.mp3"
                asset.write_bytes(loader.original(donor))
                decoded = decode.decode_audio(asset, source_sha256=donor["raw_sha256"])
                synth = decoded.pcm.astype(np.float64)
                starts = options(synth, width, name+"-synthetic")
                previous = {f"pcm:{decoded.mapping['pcm_sha256']}", *(f"pcm:{qualified.pcm_digest(decoded.pcm[:n])}" for n in (47_648, 48_000, 64_000) if len(synth) >= n)}
                if not starts or previous & seen:
                    continue
                ai_at = starts[0]
                donor_proof = {"source_sha256": donor["raw_sha256"], "canonical_speaker": donor["voice"], "license": donor["license"],
                               "sidecar_sha256": loader.expressive[donor["id"]]["metadata_sha256"], "subset": donor["subset"],
                               "cut": clock(decoded.mapping, ai_at, ai_at+width)}
                generated_voices.add(donor["voice"])
                text.update(donor["texts"])
                break
            else:
                raise ValueError("Insufficient unused exact generation sources")
            a, noise, background = room_cut(rooms[room], room, options(host, width, name), width)
            noise *= np.linalg.norm(host*32768)*10**.25/np.linalg.norm(noise)
            variants = (("ordinary_same_host", same, same_at, host_proof), ("ordinary_other_voice", other, other_at, other_proof), ("synthetic", synth, ai_at, donor_proof))
            matched = []
            for arm, values, offset, proof in variants:
                exact = qualified.pcm_digest(values[offset:offset+width].astype(np.float32))
                require(exact not in audio and f"pcm:{exact}" not in seen, "An exact speech extract was reused")
                audio.add(exact)
                signal = prior.old.inputs.splice(host, values, a, offset, width)*32768
                matched.append((arm, signal, proof | {"pcm_sha256": exact}))
            gain = min(1., 31_120/max(float(np.max(abs(signal)+abs(noise))) for _, signal, _ in matched))
            residual = np.rint(gain*noise)
            for arm, signal, proof in matched:
                emit(output, f"{name}-{arm}", name, arm, [a, a+width], background, host_proof, proof, np.rint(gain*signal), residual)
    for room in ROOMS:
        _, noise, background = room_cut(rooms[room], room, [32_000], 12800)
        noise = np.rint(noise*min(1., 31_120/np.max(abs(noise))))
        emit(output, f"room-{room}", f"room-{room}", "background_only", None, background, None, None, np.zeros(len(noise)), noise)
    require(len(output) == 26 and len(generated_voices) == 8, "Held input denominator changed")
    return output, meta.bindings


def emit(output, name, group, arm, edit, background, host, donor, foreground, residual):
    main, first, second = (prior.old.pcm(x) for x in (foreground+residual, foreground, residual))
    output.append(({"id": name, "group": group, "arm": arm, "edit": edit, "truth": edit if arm == "synthetic" else None,
                    "room": background["room"], "background": background, "host": host, "donor": donor,
                    "pcm_sha256": digest(main), "foreground_sha256": digest(first), "background_sha256": digest(second)}, main, first, second))


def variable_windows(data):
    fixed = prior.boundary_cells(np.frombuffer(data, "<i2").astype(np.float32)/32768)
    points = {span[key]: value for span in fixed for key, value in zip(("start", "end"), span["endpoints"], strict=True)}
    return {str(width): [(min(value, points[a+width]), [a, a+width]) for a, value in sorted(points.items()) if a+width in points] for width in WIDTHS}


def point_values():
    roster = [row for row in read(POINT / "roster.json") if row["panel"] == "point"]
    receipt = read(POINT / "points.json")
    require(len(roster) == 12 and all(row["truth"] is None for row in roster) and receipt["point_sha256"] == sha(POINT / "point.json"), "Genuine-only point receipts differ")
    candidates = [variable_windows(prior.raw(row)) for row in roster]
    factors = {}
    for width in map(str, WIDTHS):
        lower, med, upper = np.percentile([score for options in candidates for score, _ in options[width]], [25, 50, 75])
        factors[width] = [float(med), max(1e-6, float(upper-lower))]
    return receipt["values"]["points"] | {"variable": max(best(row, factors)[0] for row in candidates)}, factors


def best(options, factors):
    return max((((score-factors[w][0])/factors[w][1], span) for w, rows in options.items() for score, span in rows), key=lambda x: x[0])


def freeze(run):
    output, ancestry = render()
    points, factors = point_values()
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir()
    for row, main, fg, bg in output:
        for suffix, data in (("", main), ("-foreground", fg), ("-background", bg)):
            (run / "pcm" / f"{row['id']}{suffix}.pcm").write_bytes(data)
    write(run / "roster.json", [row for row, *_ in output])
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)),
          "previous_method_sha256": sha(Path(prior.__file__)), "prior_points_sha256": sha(POINT / "points.json"),
          "roster_sha256": sha(run / "roster.json"), "points": {arm: points[arm] for arm in ARMS}, "scalers": factors, "ancestry": ancestry})
    print("Frozen 26 exact held waveforms and genuine-only cutoffs for three arms", flush=True)


def score(args, rows):
    prior.old.spectra.incumbent.network_free()
    records = list(prior.old.score_current(args, rows))
    write(args.run / "current.json", {"freeze_sha256": sha(args.run / "freeze.json"), "records": records})
    print(f"Measured {sum(w['score'] is not None for r in records for w in r['windows'])} native incumbent windows", flush=True)


def result(args, rows, lock, audit):
    ledger = read(args.run / "current.json")
    require(ledger["freeze_sha256"] == sha(args.run / "freeze.json"), "Model window ledger uses another freeze")
    found = []
    for row, record in zip(rows, ledger["records"], strict=True):
        raw = prior.old.raw_input(args.run, row)
        require(row["id"] == record["id"] and row["pcm_sha256"] == record["pcm_sha256"], "Original-PCM record differs")
        require([w["start"] for w in record["windows"]] == [RATE, 3*RATE], "Native current window clock differs")
        for w in record["windows"]:
            require(w["end"]-w["start"] == 2*RATE and w["pcm_sha256"] == digest(raw[2*w["start"]:2*w["end"]]), "Current model source window differs")
        windows = variable_windows(raw)
        old = max((w for w in record["windows"] if w["score"] is not None), key=lambda w: w["score"], default=None)
        current = [None, None] if old is None else [old["score"], [old["start"], old["end"]]]
        fixed = max(windows[str(prior.EDIT)], key=lambda x: x[0])
        found.append(row | {"responses": {"current": current, "boundary": list(fixed), "variable": list(best(windows, lock["scalers"]))}})
    fixed = {arm: prior.performance(found, arm, lock["points"][arm]) for arm in ARMS}
    equal = {str(n): {arm: prior.equal_alarm(found, arm, n) for arm in ARMS} for n in (0, 1, 2)}
    genuine = {arm: {kind: sum(r["arm"] == kind and r["responses"][arm][0] is not None and r["responses"][arm][0] > lock["points"][arm]
                                    and prior.old.voice.useful_range(*r["responses"][arm][1], r["edit"]) for r in found)
                     for kind in ("ordinary_same_host", "ordinary_other_voice")} for arm in ARMS}
    aggregate = {"schema": SCHEMA, "method": METHOD, "fixed": fixed, "equal_false_groups": equal, "fixed_located_genuine_splices": genuine,
                 "counts": dict(Counter(row["arm"] for row in rows)), "commitments": {name: sha(args.run / f"{name}.json") for name in ("freeze", "roster", "current")}}
    if audit:
        replay, _ = render()
        require([row for row, *_ in replay] == rows and aggregate == read(args.run / "result.json") and found == read(args.run / "decisions.json"), "Native replay differs")
        require(all(all((args.run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() == data for suffix, data in zip(("", "-foreground", "-background"), buffers, strict=True))
                    for row, *buffers in replay), "Replay source PCM differs")
        print("Reproduced 26 original mixes, exact stems, source clocks, and held decisions", flush=True)
    else:
        write(args.run / "decisions.json", found)
        write(args.run / "result.json", aggregate)
        print(fixed, equal, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.run)
        return
    lock, rows = read(args.run / "freeze.json"), read(args.run / "roster.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(args.run / "roster.json")
            and lock["previous_method_sha256"] == sha(Path(prior.__file__)) and lock["prior_points_sha256"] == sha(POINT / "points.json"), "Variable-boundary freeze changed")
    if args.phase == "score":
        score(args, rows)
    else:
        result(args, rows, lock, args.phase == "audit")


if __name__ == "__main__":
    main()
