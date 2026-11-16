# SPDX-License-Identifier: MIT
"""Rebuild and score more source-disjoint speech pairs without refitting any points."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts import audio_clean_teacher_transfer as experiment
from scripts import audio_short_edit_input_views as earlier

BASE = experiment.base
SOURCE = experiment.SOURCE
UNION = earlier.union
SINE = UNION.duration.previous
SHARDS = earlier.SHARDS
LEVELS = (-5, -10)
LIMIT = 24
SCHEMA = "chipmunk.audio-clean-teacher-fresh/1"
KNOWN = earlier.duration.previous.PANELS / "audio-short-edit-input-views-2026-11-15.json.gz"


def _excluded(phoneme, earlier_run):
    plan, _ = BASE.load_plan(phoneme)
    retained = experiment.read(earlier.PANEL)
    previous = experiment.read(KNOWN)
    old_pairs = previous["plan"]["fresh_source"]["pairs"]
    newest = experiment.read(earlier_run / "freeze.json")
    known = earlier.excluded_readers(retained)
    known.update(row["reader"] for row in [*old_pairs, *newest["pairs"]])
    return known, {row["reader_sha256"] for row in plan["sources"]}


def _mix(pair, noise, db):
    import numpy as np

    original, changed = (signal.astype(np.float64) for signal in pair)
    gain = np.linalg.norm(original) / np.linalg.norm(noise) * 10 ** (-db / 20)
    interferer = noise * gain
    peak = max(float(np.abs(original + interferer).max()), float(np.abs(changed + interferer).max()))
    scale = min(1., 32000 / peak)
    same = np.rint(interferer * scale).astype(np.int32)
    separate = [np.rint(source * scale).astype(np.int32) for source in (original, changed)]
    delivered = [(source + same).astype("<i2") for source in separate]
    assert all(np.array_equal(upload.astype(np.int32) - source, same)
               for upload, source in zip(delivered, separate, strict=True))
    return delivered, SOURCE.digest(same.astype("<i4").tobytes())


def build(args):
    import numpy as np

    excluded, hashed = _excluded(args.phoneme, args.earlier_run)
    roster, shards = {}, []
    listing = UNION.SINE.source_listing(args.cache)
    for shard in SHARDS:
        choices, pin = SINE.select_shard(args.cache, shard, listing[f"data/split-{shard}.tar.gz"], excluded)
        for speaker, item in choices.items():
            if BASE.sources.order(speaker) in hashed:
                continue
            if speaker not in roster or item[0] < roster[speaker][0]:
                roster[speaker] = item
        shards.append(pin)
        print(f"Read source prefix {shard}; {len(roster)} eligible readers so far.", flush=True)
    selected = sorted(roster.values(), key=lambda entry: entry[0])[:LIMIT]
    assert len(selected) >= 12
    head_freeze = experiment.read(args.heads / "freeze.json")
    room_plan = experiment.read(args.retained / "freeze.json")["rooms"]
    args.output.mkdir(mode=0o700)
    for subfolder in ("sine", "background", "pcm"):
        (args.output / subfolder).mkdir()
    pairs, noise = [], {}
    for i, (_, row, waves) in enumerate(selected):
        pairs.append(row | {"id": i, "room": room_plan[i % len(room_plan)]["room"]})
        for role, raw in waves.items():
            (args.output / "sine" / f"{i:03}-{role}.wav").write_bytes(raw)
    for room in room_plan:
        name = room["room"]
        data = (args.retained / "background" / f"{name}.pcm").read_bytes()
        assert SOURCE.digest(data) == room["pcm_sha256"]
        (args.output / "background" / f"{name}.pcm").write_bytes(data)
        noise[name] = np.frombuffer(data, dtype="<i2")

    def save(samples):
        raw = samples.tobytes()
        digest = SOURCE.digest(raw)
        (args.output / "pcm" / f"{digest}.pcm").write_bytes(raw)
        return digest

    originals = defaultdict(list)
    for row, samples in SINE.deliveries(args.output, {"pairs": pairs, "backgrounds": room_plan}):
        if row["condition"] == "dry":
            originals[row["group"]].append(samples)
    cases = []
    for pair in pairs:
        number, room = pair["id"], pair["room"]
        for db in LEVELS:
            mixtures, exact_noise = _mix(originals[number], noise[room], db)
            for role, samples in zip(("untouched", "edited/sine"), mixtures, strict=True):
                cases.append({"group": number, "room": room, "panel": f"fresh_{db}dB",
                              "kind": role, "pcm_sha256": save(samples), "transmitted_room_sha256": exact_noise,
                              "truth": [pair["truth"]] if role.startswith("edited") else []})
    for i, room in enumerate(room_plan):
        name = room["room"]
        cases.append({"group": -1 - i, "room": name, "panel": "room_only",
                      "kind": "background", "pcm_sha256": save(noise[name]), "truth": []})
    release = {"schema": SCHEMA, "shards": shards, "room_plan": room_plan, "pairs": pairs, "cases": cases,
               "levels_db": LEVELS, "excluded_readers": len(excluded), "heads": head_freeze,
               "runner_sha256": SOURCE.path_sha(Path(__file__)), "prior_roster_sha256": SOURCE.path_sha(KNOWN)}
    experiment.save(args.output / "freeze.json", release)
    print(f"Frozen {len(pairs)} matched pairs at both levels and {len(room_plan)} isolated rooms.", flush=True)


def score(args):
    import torch

    from app.detectors.audio.short_voice_review import scan
    from scripts.run_audio_specialists_v15 import load_voice

    freeze = experiment.read(args.media / "freeze.json")
    assert freeze["heads"] == experiment.read(args.heads / "freeze.json")
    assert freeze["runner_sha256"] == SOURCE.path_sha(Path(__file__))
    heads = [experiment._head(args.heads, args.heads / f"{family}-{arm}.json")
             for family in BASE.sources.FAMILIES for arm in experiment.ARMS]
    activity = experiment.runtime(args.weights / "silero_vad.jit")
    earlier_app = experiment.read(experiment.panels.PRIMARY)["incumbent"]
    voice = load_voice({"weights": args.weights, "assets": earlier_app["assets"]})
    encode = experiment.features.frozen_encoder(args.encoder)

    class Deadline:
        def check(self, _stage):
            pass

    observed = []
    with torch.inference_mode():
        for row in freeze["cases"]:
            wave = BASE.pcm(args.media, row)
            span = activity.spans(wave)
            raw = encode(wave)
            incumbent = scan(wave, span, voice, Deadline(), codec="pcm_s16le")
            witness = incumbent["selected"]
            readings = {}
            for seal, model, center, scale in heads:
                local, edge = model(torch.from_numpy(((raw - center[:, None]) / scale[:, None])[None]))[0].numpy()
                readings[f"{seal['trained_family']}/{seal['arm']}"] = BASE.best_interval(local, edge, span)
            observed.append(row | {"current": {"assessed": bool(incumbent["responses"]),
                                               "range": [witness["start_sample"], witness["end_sample"]] if witness else None},
                                   "candidates": readings})
            if len(observed) % 20 == 0:
                print(f"Scored {len(observed)} newly delivered mixtures.", flush=True)
    result = {"freeze": freeze, "heads": [entry[0] for entry in heads], "rows": observed}
    result["results"] = experiment.recount(result)
    experiment.save(args.output, result)
    print(json.dumps(result["results"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    builder = commands.add_parser("build")
    for name in ("phoneme", "earlier_run", "retained", "cache", "heads", "output"):
        builder.add_argument(name, type=Path)
    scorer = commands.add_parser("score")
    for name in ("media", "heads", "weights", "encoder", "output"):
        scorer.add_argument(name, type=Path)
    args = parser.parse_args()
    (build if args.phase == "build" else score)(args)


if __name__ == "__main__":
    main()
