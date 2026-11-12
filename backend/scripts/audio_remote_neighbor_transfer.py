"""Compare an independent content-matched speech cue on original noisy PCM."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import audio_short_native_response_transfer as recent
from scripts import audio_wavlm_features_v13 as wavlm

BASE, ROOT, RATE = recent.BASE, recent.ROOT, recent.RATE
OLD, source, qualified = recent.OLD, recent.source, recent.qualified
POINT = recent.POINT
SHORT = BASE / "audio-short-native-response-2026-11-12"
VARIABLE = BASE / "audio-variable-boundary-2026-11-12"
LAYERS = (4, 12)
RANGE, STEP, REMOTE, BUCKET = 12800, 3200, 12800, 3200
ARMS = ("current", "neighbor")
SCHEMA = "chipmunk.audio-remote-neighbor-transfer/1"
read, write, sha, digest, require = recent.read, recent.write, recent.sha, recent.digest, recent.require
METHOD = {
    "scope": "Single original 6-second PCM waveform. Frozen WavLM layer 12 supplies acoustic-context matches and layer 4 supplies the mismatch response. These features do not prove lexical phone or speaker identity. No alteration, stems, known edit range, or earlier recording enters inference.",
    "frames": "Native convolution stride 320 and receptive field 400 samples, frame centers at 200+320i. Center each layer across the clip and normalize every frame. Require the unchanged speech gate to admit the query center and each neighbor center. Look at least 0.80 seconds away and select the highest deep cosine match in each disjoint 0.20-second bucket. Keep four buckets with cosine at least .20.",
    "response": "Subtract the median early cosine distance among four neighbors from the query's median early cosine distance to those neighbors. Propose .80-second ranges every .20 seconds when at least 60% of the centers have supported matches. Retain the maximum median response, otherwise unassessed. This is a fixed content-conditioned inconsistency cue, not a boundary response or model-fit authenticity probability.",
    "roles": "Use 12 exact genuine controls for the new maximum-genuine point and the earlier exact native current operating point. Separate two previously scored original-PCM transfer panels from four freshly rendered held episodes spanning expressive and stock families in two AMI rooms. Include both ordinary counterparts and isolated rooms. All source roles, ranges and points freeze before held cue measurements.",
    "gate": "Compare useful locations at fixed and retrospective 0/1/2 false genuine group budgets, unassessed inputs and isolated-room flags. A useful range covers at least half the edit and reaches .30 temporal IoU. A supplemental fixed union retains both intervals and does not change any model. Original-only support remains experimental if gain or ordinary-control specificity fails."
}


def fresh_sources():
    meta = qualified.sources.Metadata()
    ids, _, seen, _ = qualified.history(meta)
    catalogue, used, _ = recent.prior_inputs(meta, seen, ids)
    rooms = source.room_specs(meta)
    for folder in (SHORT, VARIABLE):
        for row in meta.read(f"previous_exact_{folder.name}", folder / "roster.json"):
            if row["room"] in rooms:
                start = row["background"]["capture_start_sample"]
                rooms[row["room"]][3].append((start, start+96000))
            for key in ("host", "donor"):
                if value := row.get(key):
                    used.add(value["canonical_speaker"])
                    seen.add(f"bytes:{value['source_sha256']}")
    donors = {"expressive": source.donors(meta, ids, seen), "stock": recent.stock_rows(meta, seen, ids)}
    pool = sorted((r for r in catalogue if r["kind"] == "genuine" and r["license"] == "CC-BY-4.0" and r["samples"] >= 12*RATE),
                  key=lambda row: digest(f"remote-neighbor:{row['raw_sha256']}".encode()))
    output, used_text, exact_cuts = [], set(), set()
    with ExitStack() as stack, TemporaryDirectory() as folder:
        loader = qualified.source_reader.SourceReader(stack, meta.bindings)
        for i, (room, width) in enumerate((r, w) for r in source.ROOMS for w in (10240, 15360)):
            name = f"fresh-{i:02d}"
            family = recent.FAMILIES[(i+i//2) % 2]
            host, same, same_at, host_proof = recent.genuine(pool, used, {"confirmation", "selection"}, width, name+"-host", host=True)
            _, other, other_at, other_proof = recent.genuine(pool, used, {"confidence_cal", "fusion_fit"}, width, name+"-other")
            ai, ai_at, ai_proof = recent.generated(donors[family], loader, Path(folder), used, used_text, seen, width, name+"-synthetic")
            start, noise, background = source.room_cut(rooms[room], room, source.options(host, width, name+"-target"), width)
            noise *= np.linalg.norm(host*32768)*10**.25/np.linalg.norm(noise)
            pieces = []
            for arm, signal, at, proof in (("ordinary_same_host", same, same_at, host_proof), ("ordinary_other_voice", other, other_at, other_proof),
                                            ("synthetic", ai, ai_at, ai_proof)):
                pin = qualified.pcm_digest(signal[at:at+width].astype(np.float32))
                require(pin not in exact_cuts and f"pcm:{pin}" not in seen, "Fresh donor PCM overlaps prior source")
                exact_cuts.add(pin)
                piece = OLD.old.inputs.splice(host, signal, start, at, width)*32768
                pieces.append((arm, piece, proof | {"pcm_sha256": pin}))
            gain = min(1., 31_120/max(float(np.max(abs(piece)+abs(noise))) for _, piece, _ in pieces))
            for arm, piece, proof in pieces:
                source.emit(output, f"{name}-{arm}", name, arm, [start, start+width], background, host_proof, proof, np.rint(gain*piece), np.rint(gain*noise))
    for room in source.ROOMS:
        _, noise, background = source.room_cut(rooms[room], room, [32_000], 12_800)
        noise = np.rint(noise*min(1., 31_120/np.max(abs(noise))))
        source.emit(output, f"fresh-room-{room}", f"fresh-room-{room}", "background_only", None, background, None, None, np.zeros(len(noise)), noise)
    require(len(output) == 14 and len(exact_cuts) == 12, "Fresh held source construction incomplete")
    return output, meta.bindings


def older_rows():
    points, observations = recent.point_sources()
    results = [row | {"cohort": "point", "asset": "point"} for row in points]
    for name, folder, ledger in (("earlier_expressive", VARIABLE, "current"), ("earlier_two_family", SHORT, "held")):
        receipt = read(folder / "result.json")
        require(receipt["commitments"]["roster"] == sha(folder / "roster.json") and receipt["commitments"][ledger] == sha(folder / f"{ledger}.json"), "Earlier native source or score receipt differs")
        results.extend(row | {"cohort": name, "asset": name, "group": name+"-"+row["group"]} for row in read(folder / "roster.json"))
    return results, observations


def raw(run, row):
    if row["asset"] == "point":
        return OLD.raw(row | {"asset": "same"})
    folder = {"earlier_expressive": VARIABLE, "earlier_two_family": SHORT, "fresh": run}[row["asset"]]
    data = OLD.old.raw_input(folder, row)
    require(len(data) == 192000, "Six-second native input changed")
    return data


def freeze(args):
    fresh, ancestry = fresh_sources()
    previous, _ = older_rows()
    rows = previous + [row | {"cohort": "fresh", "asset": "fresh"} for row, *_ in fresh]
    require(len(rows) == len({row["pcm_sha256"] for row in rows}) == 72, "Frozen inputs are not distinct")
    args.run.mkdir(mode=0o700)
    (args.run / "pcm").mkdir()
    for row, *data in fresh:
        for suffix, pcm in zip(("", "-foreground", "-background"), data, strict=True):
            (args.run / "pcm" / f"{row['id']}{suffix}.pcm").write_bytes(pcm)
    write(args.run / "roster.json", rows)
    write(args.run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "ancestry": ancestry, "wavlm_assets": wavlm.assets(args.wavlm),
          "current_assets": OLD.old.voice.ASSETS, "runner_sha256": sha(Path(__file__)), "roster_sha256": sha(args.run / "roster.json"),
          "previous_commitments": [sha(path / "result.json") for path in (POINT, VARIABLE, SHORT)]})
    print("Frozen 12 genuine points, 46 exposed transfer views and 14 fresh held views", flush=True)


def normalize(values):
    values = values-np.median(values, axis=0)
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def neighbors(early, deep, voiced):
    centers = 200+np.arange(len(deep))*320
    groups = centers//BUCKET
    e, d = normalize(early), normalize(deep)
    deep_similarity, early_distance = d @ d.T, 1-np.clip(e @ e.T, -1, 1)
    within = np.abs(centers[:, None]-centers) >= REMOTE
    candidates = np.where(within & voiced[:, None] & voiced[None, :] & (deep_similarity >= .20), deep_similarity, -np.inf)
    output = []
    for index, similarity in enumerate(candidates):
        ranking = np.argsort(-similarity, kind="stable")
        picked, used = [], set()
        for other in ranking:
            if not np.isfinite(similarity[other]):
                break
            bucket = int(groups[other])
            if bucket not in used:
                picked.append(int(other))
                used.add(bucket)
            if len(picked) == 4:
                break
        if len(picked) == 4:
            within_neighbor = early_distance[np.ix_(picked, picked)][np.triu_indices(4, 1)]
            score = float(np.median(early_distance[index, picked])-np.median(within_neighbor))
            output.append({"center": int(centers[index]), "score": score, "neighbors": [int(centers[other]) for other in picked]})
    return output


def native_ranges(frames):
    output = []
    for a in range(0, 96000-RANGE+1, STEP):
        selected = [row["score"] for row in frames if a <= row["center"] < a+RANGE]
        if len(selected) >= 24:
            output.append((float(np.median(selected)), [a, a+RANGE]))
    return list(max(output, key=lambda row: row[0], default=(None, None)))


def measure(args, rows):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity

    OLD.old.spectra.incumbent.network_free()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = wavlm.load(args.wavlm)
    require(sha(args.voice / "silero_vad.jit") == OLD.old.voice.ASSETS["silero_vad.jit"], "Speech gate asset differs")
    vad = SpeechActivity(args.voice / "silero_vad.jit")
    result = []
    for row in rows:
        values = np.frombuffer(raw(args.run, row), "<i2").astype(np.float32)/32768
        require(values.shape == (96000,), "Expected exact unmodified six-second waveform")
        centers = (200+np.arange(299)*320)/RATE
        admitted = np.zeros(299, bool)
        for a, b in vad.spans(values):
            admitted |= (centers >= a) & (centers < b)
        matrix = torch.from_numpy(values.copy())[None]
        with torch.inference_mode():
            states = model(matrix, attention_mask=torch.ones_like(matrix, dtype=torch.long), output_hidden_states=True, return_dict=True).hidden_states
        early, deep = (states[layer][0].cpu().numpy() for layer in LAYERS)
        require(len(states) == 13 and early.shape == deep.shape == (299, 768) and np.isfinite(early).all() and np.isfinite(deep).all(),
                "Native WavLM frame clock differs")
        proof = neighbors(early, deep, admitted)
        result.append({"id": row["id"], "asset": row["asset"], "pcm_sha256": row["pcm_sha256"], "support": proof})
    print(f"Measured {len(result)} exact full-clip encoder forwards", flush=True)
    return result


def current_ledgers(args):
    return {"earlier_expressive": read(VARIABLE / "current.json")["records"], "earlier_two_family": read(SHORT / "held.json")["records"],
            "fresh": read(args.run / "fresh-current.json")["records"]}


def union_fixed(rows, points):
    counts, false = Counter(), set()
    for row in rows:
        checked = [row["responses"][arm] for arm in ARMS if row["responses"][arm][0] is not None]
        hits = [row["responses"][arm][1] for arm in ARMS if row["responses"][arm][0] is not None and row["responses"][arm][0] > points[arm]]
        counts["unassessed"] += not checked
        if hits and row["truth"] is None:
            false.add(row["group"])
            counts["background_false_flags"] += row["arm"] == "background_only"
        if hits and row["truth"] is not None:
            counts["ai_flags"] += 1
            counts["useful_ranges"] += any(OLD.old.voice.useful_range(*span, row["truth"]) for span in hits)
    return {key: counts[key] for key in ("ai_flags", "useful_ranges", "unassessed", "background_false_flags")} | {"false_groups": len(false)}


def report(args, rows, audit):
    point = read(args.run / "point.json")
    require(len(point) == 12 and all(row["asset"] == "point" for row in point), "Genuine point denominator differs")
    values = [native_ranges(row["support"])[0] for row in point]
    lock = read(args.run / "points.json")
    require(lock["point_sha256"] == sha(args.run / "point.json") and lock["neighbor"] == max(v for v in values if v is not None), "Genuine point calibration differs")
    old_point = read(POINT / "points.json")["values"]["points"]["current"]
    current = {(asset, row["id"]): row for asset, records in current_ledgers(args).items() for row in records}
    held = read(args.run / "held.json")
    require(held["points_sha256"] == sha(args.run / "points.json"), "Held encoder used a different freeze")
    outcomes = []
    for row, proof in zip((row for row in rows if row["cohort"] != "point"), held["rows"], strict=True):
        key = row["asset"], row["id"]
        data = raw(args.run, row)
        incumbent = current[key]
        require(key == (proof["asset"], proof["id"]) and row["pcm_sha256"] == proof["pcm_sha256"] == incumbent["pcm_sha256"], "Exact score source changed")
        windows = incumbent.get("current", incumbent.get("windows"))
        require([w["start"] for w in windows] == [RATE, 3*RATE] and all(w["end"]-w["start"] == 2*RATE
                and digest(data[2*w["start"]:2*w["end"]]) == w["pcm_sha256"] for w in windows), "Exact incumbent native clocks changed")
        for frame in proof["support"]:
            clock = [frame["center"], *frame["neighbors"]]
            require(len(clock) == 5 and all(200 <= p <= 95560 and (p-200) % 320 == 0 for p in clock)
                    and all(abs(x-clock[0]) >= REMOTE for x in clock[1:]) and len({x//BUCKET for x in clock[1:]}) == 4, "Remote frame support clock differs")
        best = max((w for w in windows if w["score"] is not None), key=lambda w: w["score"], default=None)
        incumbent_value = [None, None] if best is None else [best["score"], [best["start"], best["end"]]]
        outcomes.append(row | {"responses": {"current": incumbent_value, "neighbor": native_ranges(proof["support"])}})
    cutoffs = {"current": old_point, "neighbor": lock["neighbor"]}
    tables = {}
    for cohort in ("earlier_expressive", "earlier_two_family", "fresh"):
        subset = [r for r in outcomes if r["cohort"] == cohort]
        tables[cohort] = {"counts": dict(Counter(r["arm"] for r in subset)), "fixed": {arm: OLD.performance(subset, arm, cutoffs[arm]) for arm in ARMS},
                          "fixed_union": union_fixed(subset, cutoffs),
                          "equal_false_groups": {str(n): {arm: OLD.equal_alarm(subset, arm, n) for arm in ARMS} for n in (0, 1, 2)},
                          "locations_before_cutoff": {arm: OLD.performance(subset, arm, -np.inf)["useful_ranges"] for arm in ARMS}}
    payload = {"schema": SCHEMA, "method": METHOD, "result": tables, "new_model_calls": {"wavlm": len(point)+len(held["rows"]),
               "current": sum(w["score"] is not None for r in current_ledgers(args)["fresh"] for w in r["windows"])},
               "commitments": {name: sha(args.run / f"{name}.json") for name in ("freeze", "roster", "point", "points", "held", "fresh-current")}}
    if audit:
        reconstructed, _ = fresh_sources()
        require(all(all((args.run / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes() == samples for suffix, samples in zip(("", "-foreground", "-background"), data, strict=True))
                    for row, *data in reconstructed), "Fresh original and component replay differs")
        require(payload == read(args.run / "result.json") and outcomes == read(args.run / "decisions.json"), "Frozen scores or outcome replay differs")
        print("Reproduced all fresh original waveforms and 72 native source/decision bindings", flush=True)
    else:
        write(args.run / "decisions.json", outcomes)
        write(args.run / "result.json", payload)
        print(tables, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "point", "held", "current", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--wavlm", type=Path, default=ROOT / "models/research/wavlm-v13")
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
        return
    lock, rows = read(args.run / "freeze.json"), read(args.run / "roster.json")
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["roster_sha256"] == sha(args.run / "roster.json")
            and lock["previous_commitments"] == [sha(path / "result.json") for path in (POINT, VARIABLE, SHORT)], "Study freeze changed")
    if args.phase == "point":
        result = measure(args, [row for row in rows if row["cohort"] == "point"])
        write(args.run / "point.json", result)
        values = [native_ranges(row["support"])[0] for row in result]
        require(any(v is not None for v in values), "No assessable genuine points")
        write(args.run / "points.json", {"neighbor": max(v for v in values if v is not None), "assessed": sum(v is not None for v in values),
              "point_sha256": sha(args.run / "point.json"), "freeze_sha256": sha(args.run / "freeze.json")})
    elif args.phase in {"held", "current"}:
        point = read(args.run / "points.json")
        require(point["point_sha256"] == sha(args.run / "point.json") and point["freeze_sha256"] == sha(args.run / "freeze.json"), "Seal genuine controls before held scoring")
        if args.phase == "held":
            result = measure(args, [row for row in rows if row["cohort"] != "point"])
            write(args.run / "held.json", {"rows": result, "points_sha256": sha(args.run / "points.json")})
        else:
            OLD.old.spectra.incumbent.network_free()
            write(args.run / "fresh-current.json", {"records": list(OLD.old.score_current(args, [row for row in rows if row["cohort"] == "fresh"]))})
    else:
        report(args, rows, args.phase == "audit")


if __name__ == "__main__":
    main()
