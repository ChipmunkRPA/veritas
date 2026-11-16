# SPDX-License-Identifier: MIT
"""Check the frozen speech-edit heads on held SINE shards and matched controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from scripts import audio_clean_teacher_fresh as source
from scripts import audio_edit_direction_transfer as check

SOURCE, base, prior = check.SOURCE, check.base, check.prior
SHARDS = (30, 31)
LEVELS = (None, -5, -10)
METHOD = {
    "source": "Take all eligible new declared readers, up to 24 by the previous source order, from the first 192 MiB "
              "of two previously unused publisher shards. Exclude earlier source panels by declared reader and "
              "primary prompts by their stored reader hashes. Do not use scores to select inputs.",
    "controls": "Use untouched speech and the same-word round-trip resampling control at the publisher edit clock. "
                "Attach the labeled synthetic insertion at its native position. Keep the dry version and add identically "
                "scaled AMI speech to the three versions at nominal -5 and -10 dB relative to the original. "
                "Replay the transmitted background alone at each gain.",
    "points": "Keep every fitted parameter and operating point from the primary comparison. Evaluations use only "
              "the submitted mixture. Room recordings had earlier project exposure.",
}


def ordinary_edit(original, truth):
    import numpy as np
    from scipy.signal import resample_poly

    a, b = truth
    crop = original[a:b].astype(np.float64)
    filtered = resample_poly(resample_poly(crop, 3, 4), 4, 3)[:len(crop)]
    fade = (1 - np.cos(np.linspace(0, np.pi, 320))) / 2
    gain = np.ones(len(crop))
    gain[:len(fade)] = fade
    gain[-len(fade):] = fade[::-1]
    ordinary = original.copy()
    ordinary[a:b] = np.clip(np.rint(crop + gain * (filtered - crop)), -32768, 32767).astype("<i2")
    assert not np.array_equal(ordinary[a:b], original[a:b])
    return ordinary


def mix(foreground, room, db):
    import numpy as np

    foreground = np.asarray(foreground, dtype=np.float64)
    if db is None:
        return foreground.astype("<i2"), None
    noise = room * (np.linalg.norm(foreground[0]) / np.linalg.norm(room) * 10 ** (-db / 20))
    peak = max(float(np.abs(foreground + noise).max()), float(np.abs(noise).max()))
    gain = min(1., 32000 / peak)
    isolated = np.rint(noise * gain).astype(np.int32)
    stems = np.rint(foreground * gain).astype(np.int32)
    transmitted = (stems + isolated).astype("<i2")
    assert np.array_equal(transmitted.astype(np.int32) - stems, np.broadcast_to(isolated, stems.shape))
    assert np.abs(isolated).max() <= 32767
    return transmitted, isolated.astype("<i2")


def build(args):
    import numpy as np

    excluded, hashed = source._excluded(args.phoneme, args.previous)
    excluded.update(row["reader"] for row in prior.read(args.additional / "freeze.json")["pairs"])
    listing = source.UNION.SINE.source_listing(args.cache)
    offers, pins = {}, []
    for shard in SHARDS:
        rows, pin = source.SINE.select_shard(args.cache, shard, listing[f"data/split-{shard}.tar.gz"], excluded)
        for reader, offer in rows.items():
            if base.sources.order(reader) not in hashed and (reader not in offers or offer[0] < offers[reader][0]):
                offers[reader] = offer
        pins.append(pin)
        print(f"Read shard {shard}. {len(offers)} eligible new readers so far.", flush=True)
    selected = sorted(offers.values(), key=lambda row: row[0])[:24]
    assert selected
    rooms = prior.read(args.previous / "freeze.json")["room_plan"]
    args.output.mkdir(mode=0o700)
    for child in ("sine", "background", "pcm"):
        (args.output / child).mkdir()
    pairs, noise = [], {}
    for index, (_, row, raw) in enumerate(selected):
        pairs.append(row | {"id": index, "room": rooms[index % len(rooms)]["room"]})
        for kind, wave in raw.items():
            (args.output / "sine" / f"{index:03}-{kind}.wav").write_bytes(wave)
    for room in rooms:
        name = room["room"]
        raw = (args.previous / "background" / f"{name}.pcm").read_bytes()
        assert len(raw) == 2 * base.SAMPLES and SOURCE.digest(raw) == room["pcm_sha256"]
        (args.output / "background" / f"{name}.pcm").write_bytes(raw)
        noise[name] = np.frombuffer(raw, dtype="<i2")

    clean = defaultdict(dict)
    for row, samples in source.SINE.deliveries(args.output, {"pairs": pairs, "backgrounds": rooms}):
        if row["condition"] == "dry":
            clean[row["group"]][row["kind"]] = samples

    def keep(samples):
        raw = samples.astype("<i2", copy=False).tobytes()
        assert len(raw) == 2 * base.SAMPLES
        digest = SOURCE.digest(raw)
        (args.output / "pcm" / f"{digest}.pcm").write_bytes(raw)
        return digest

    cases = []
    for pair in pairs:
        index, room, truth = (pair[key] for key in ("id", "room", "truth"))
        original = clean[index]["untouched"]
        controls = ordinary_edit(original, truth)
        assert np.array_equal(original[:truth[0]], controls[:truth[0]]) and np.array_equal(original[truth[1]:], controls[truth[1]:])
        for db in LEVELS:
            rendered, background = mix([original, controls, clean[index]["edited/sine"]], noise[room], db)
            panel = "new_dry" if db is None else f"new_{db}dB"
            isolated = keep(background) if background is not None else None
            labels = (("untouched", []), ("resampled", []), ("edited/sine", [truth]))
            cases.extend({"group": index, "room": room, "panel": panel, "kind": kind, "truth": region,
                          "pcm_sha256": keep(samples), "shared_background_sha256": isolated}
                         for (kind, region), samples in zip(labels, rendered, strict=True))
            if isolated:
                cases.append({"group": index, "room": room, "panel": panel, "kind": "background", "truth": [],
                              "pcm_sha256": isolated, "shared_background_sha256": isolated})
    freeze = {"method": METHOD, "code_sha256": SOURCE.path_sha(Path(__file__)), "pairs": pairs, "shards": pins,
              "room_plan": rooms, "cases": cases, "excluded_readers": len(excluded),
              "fit_commitments": {name: SOURCE.path_sha(args.fit / name) for name in ("freeze.json", "heads.json")}}
    prior.save(args.output / "freeze.json", freeze)
    print(f"Frozen {len(cases)} submissions from {len(pairs)} previously unused readers.", flush=True)


def measure(args):
    from app.detectors.audio.short_voice_review import scan
    from scripts.run_audio_specialists_v15 import load_voice

    freeze = prior.read(args.media / "freeze.json")
    assert freeze["method"] == METHOD and freeze["code_sha256"] == SOURCE.path_sha(Path(__file__))
    for name, digest in freeze["fit_commitments"].items():
        assert SOURCE.path_sha(args.fit / name) == digest
    activity = prior.runtime(args.weights / "silero_vad.jit")
    serving = prior.read(check.panels.PRIMARY)["incumbent"]
    voice = load_voice({"weights": args.weights, "assets": serving["assets"]})
    encode = check.features.frozen_encoder(args.encoder)
    seals = prior.read(args.fit / "heads.json")
    heads = [check.load_head(args.fit, seal) for seal in seals]

    class Deadline:
        def check(self, _stage):
            pass

    observed, cached = [], {}
    for row in freeze["cases"]:
        digest = row["pcm_sha256"]
        if digest not in cached:
            wave = base.pcm(args.media, row)
            spans = activity.spans(wave)
            schedule, encoded = check.eligible(spans), encode(wave)
            existing = scan(wave, spans, voice, Deadline(), codec="pcm_s16le")
            hit = existing["selected"]
            current = {"assessed": bool(existing["responses"]), "range": [hit["start_sample"], hit["end_sample"]] if hit else None}
            responses = {f"{head[0]['trained_family']}/{head[0]['feature']}_{rule}": value
                         for head in heads for rule, value in check.measure(head, encoded, schedule).items()}
            cached[digest] = {"current": current, "candidates": responses}
        observed.append(row | cached[digest])
        if len(observed) % 20 == 0:
            print(f"Measured {len(observed)} new submissions.", flush=True)
    result = {"freeze": prior.read(args.fit / "freeze.json"), "heads": seals, "rows": observed,
              "source_manifest": freeze}
    result["comparison"] = check.compare(result)
    prior.save(args.output, result)
    print(f"Recounted {len(observed)} submissions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name, action, paths in (
        ("build", build, ("phoneme", "previous", "additional", "cache", "fit", "output")),
        ("measure", measure, ("media", "fit", "weights", "encoder", "output")),
    ):
        command = phases.add_parser(name)
        for role in paths:
            command.add_argument(role, type=Path)
        command.set_defaults(action=action)
    args = parser.parse_args()
    args.action(args)
