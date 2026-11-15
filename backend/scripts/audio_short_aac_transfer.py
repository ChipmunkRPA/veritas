# SPDX-License-Identifier: MIT
"""Measure the two-second voice review on exact six-second AAC-LC carriers."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import review_audio_short_codec_transfer as replay

PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PRIOR = PANELS / "audio-short-codec-transfer-2026-11-12.json"
CONFIRM = PANELS / "audio-short-mp3-confirmation-2026-11-12.json"
RANDOM = PANELS / "audio-random-clock-stock-2026-11-13.json"
SCHEMA = "chipmunk.audio-short-aac-transfer/1"
ROLES = ("calibration", "held", "confirmation", "random")
EXPECTED = {"calibration": (40, 16, 0), "held": (50, 14, 12), "confirmation": (52, 16, 12), "random": (83, 11, 32)}
RATE, NATIVE_RATE = 16_000, 48_000
SAMPLES, NATIVE_SAMPLES = 6 * RATE, 6 * NATIVE_RATE
METHOD = {
    "carrier": "One mono AAC-LC M4A track encoded by FFmpeg at 48 kHz and a nominal 64 kb/s. "
               "Require exactly six seconds on a zero-based continuous source clock and exclude decoder padding.",
    "points": "Independent four-second and two-second points from the same 16 genuine calibration source groups, "
              "at one and two flagged group budgets with strict excess. Freeze before held inference.",
    "held": "Replay two source-disjoint earlier carrier panels and the separate random-stock clock roster. "
            "Keep controls and sources paired and preserve hard edit clocks, with separate 0 dB and -5 dB outcomes.",
    "comparison": "Report useful intervals, wrong intervals, missed edits, ordinary source groups, and isolated backgrounds. "
                  "Count the two-second supplement as an addition to the four-second review. "
                  "Useful means at least half of the edit and 0.30 temporal intersection over union.",
}
require, read, write = replay.require, replay.load, replay.dump


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witness(counts):
    return len(counts), len({row["group"] for row in counts}), sum(row["truth"] is not None for row in counts)


def carrier(path, pin):
    from app.detectors.audio import decode

    presentation = decode.aac48_presentation(path)
    require(presentation == {"sample_rate": NATIVE_RATE, "channels": 1, "time_base": "1/48000",
                             "presentation_end_exclusive_pts": NATIVE_SAMPLES}, "AAC native clock differs")
    measured = decode.decode_audio(path, source_sha256=pin, max_seconds=6)
    clock = measured.mapping["timeline"]
    require(clock == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}],
            "AAC samples are discontinuous or end outside the presentation")
    return (measured.pcm * 32768).astype("<i2").tobytes()


def inputs(random_path):
    import numpy as np

    from scripts import audio_short_codec_transfer as codec
    from scripts import audio_short_mp3_confirmation as confirmation

    prior = read(PRIOR)
    wav = read(replay.BASE)
    replay.inspect(prior, wav)
    require(sha(codec.grouped.HELD / "sources.json") == wav["commitments"]["held_sources"], "Grouped WAV source freeze differs")
    regular = list(codec.inputs(codec.grouped.HELD))
    earlier = prior["calibration"] + prior["held"]
    for (role, ident, source, pcm), row in zip(regular, earlier, strict=True):
        require(ident == row["id"] and row["id"].startswith(role + "-")
                and (row["truth"], row["condition"], row["family"]) == (source["truth"], source["arm"], source.get("family")),
                "Previously verified original recording differs")
        yield role, {key: row[key] for key in ("id", "group", "truth", "condition", "family")}, pcm

    confirmed = read(CONFIRM)
    replay.confirm_inspect(confirmed)
    for (row, pcm), expected in zip(confirmation.source_cases(), confirmed["held"], strict=True):
        fields = ("id", "group", "truth", "condition", "family")
        record = {key: row[key] for key in fields}
        require(record == {key: expected[key] for key in fields}, "Confirmation source role differs")
        yield "confirmation", record, pcm

    roster = read(RANDOM)
    freeze = random_path / "placements-freeze.json"
    require(sha(freeze) == roster["native_freeze_sha256"], "Random-clock source manifest differs")
    frozen = read(freeze)["native_manifest_rows"]
    with np.load(random_path / "placements-native.npz", allow_pickle=False) as bank:
        native = bank["native"]
    require(native.shape == (83, SAMPLES) and native.dtype == np.dtype("<i2"), "Random-clock PCM geometry differs")
    for index, (item, bound, signal) in enumerate(zip(roster["rows"], frozen, native, strict=True)):
        pcm = signal.tobytes()
        require(all(item[key] == value for key, value in bound.items()) and codec.digest(pcm) == item["pcm_sha256"],
                "Random-clock source role or PCM differs")
        yield "random", {"id": f"random-{index:02d}", "group": f"random-{item['group']}", "truth": item["edit_clock"],
                         "condition": "background_only" if item["snr"] is None else f"{item['snr']}dB",
                         "family": item["kind"] if item["edit_clock"] else None, "variant": item["kind"]}, pcm


def prepare(run, random_path):
    from scripts.audio_clean_speech_aac_transfer import encode48

    run.mkdir(mode=0o700)
    grouped = {role: [] for role in ROLES}
    for role, metadata, original in inputs(random_path):
        path = run / f"{role}-{metadata['id']}.m4a"
        encode48(original, path, "aac48_mono")
        pin = sha(path)
        output = carrier(path, pin)
        require(len(output) == 2 * SAMPLES and output != original, "Unexpected AAC output extent or content")
        grouped[role].append(metadata | {"path": path.name, "encoded_sha256": pin,
                                         "source_sha256": hashlib.sha256(original).hexdigest(),
                                         "decoded_sha256": hashlib.sha256(output).hexdigest()})
        if len(grouped[role]) % 20 == 0:
            print(f"Verified {len(grouped[role])} {role} AAC deliveries.", flush=True)
    require(all(witness(grouped[role]) == EXPECTED[role] for role in ROLES), "Frozen delivery denominators differ")
    require(sum(len({row["source_sha256"] for row in rows}) for rows in grouped.values())
            == len({row["source_sha256"] for rows in grouped.values() for row in rows}), "Across-role PCM overlap")
    write(run / "inputs.json", grouped)
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)),
                                "inputs_sha256": sha(run / "inputs.json"),
                                "sources": {path.name: sha(path) for path in (PRIOR, CONFIRM, RANDOM)}})


def points(calibration):
    return {name: {str(budget): replay.calibrate(calibration, "aac", name, budget) for budget in replay.BUDGETS}
            for name in replay.ARMS}


def combined(rows, cuts, budget):
    result = {}
    full = replay.held_metrics(rows, "aac", "full", cuts["full"][budget])
    short = replay.held_metrics(rows, "aac", "combined", cuts["combined"][budget])
    added = Counter()
    for row in rows:
        score, span = replay.strongest(row["scores"]["aac"], "combined")
        if score is None or score <= cuts["combined"][budget]:
            continue
        earlier, old_span = replay.strongest(row["scores"]["aac"], "full")
        old_flag = earlier is not None and earlier > cuts["full"][budget]
        if row["truth"] is None:
            added["ordinary_files"] += not old_flag
            added["isolated_backgrounds"] += not old_flag and row["condition"] == "background_only"
        else:
            start, end = row["truth"]
            length = end - start
            new_intersection = max(0, min(span[1], end) - max(span[0], start))
            old_intersection = max(0, min(old_span[1], end) - max(old_span[0], start)) if old_flag else 0
            useful = 2 * new_intersection >= length and 10 * new_intersection >= 3 * (length + span[1] - span[0] - new_intersection)
            old_useful = old_flag and 2 * old_intersection >= length and 10 * old_intersection >= 3 * (
                length + old_span[1] - old_span[0] - old_intersection)
            added["useful_deliveries"] += useful and not old_useful
    added["ordinary_groups"] = len(set(short["false_groups"]) - set(full["false_groups"]))
    result.update(full=full, two_seconds=short, added_if_combined=dict(added))
    return result


def outcomes(measured, cutoffs):
    def metrics(rows):
        return {str(budget): combined(rows, cutoffs, str(budget)) for budget in replay.BUDGETS}

    result = {role: metrics(measured[role]) for role in ROLES[1:]}
    result["random_conditions"] = {condition: metrics([row for row in measured["random"] if row["condition"] == condition])
                                   for condition in ("0dB", "-5dB", "background_only")}
    return result


def score(run, weights):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts import audio_edge_window_transfer as windows
    from scripts.run_audio_specialists_v15 import load_voice

    assets = windows.native.transfer.OLD.voice.ASSETS
    lock = read(run / "freeze.json")
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["inputs_sha256"] == sha(run / "inputs.json")
            and lock["runner_sha256"] == sha(Path(__file__)) and all(sha(weights / name) == pin for name, pin in assets.items()),
            "Inference freeze or assets differ")
    windows.native.transfer.OLD.spectra.incumbent.network_free()
    torch.set_num_threads(3)
    speech = SpeechActivity(weights / "silero_vad.jit")
    model = load_voice({"weights": str(weights), "assets": assets})
    inputs_by_role = read(run / "inputs.json")
    require(set(inputs_by_role) == set(ROLES), "Unknown study role")
    measured = {}
    for role in ROLES:
        readings = []
        for index, row in enumerate(inputs_by_role[role], 1):
            pcm = carrier(run / row["path"], row["encoded_sha256"])
            require(hashlib.sha256(pcm).hexdigest() == row["decoded_sha256"], "Analyzed PCM changed after freeze")
            samples = windows.measure(pcm, model, speech, {})
            readings.append(row | {"scores": {"aac": [item["score"] for item in samples]},
                                   "window_sha256": [item["pcm_sha256"] for item in samples]})
            if index % 10 == 0:
                print(f"Scored {index} of {len(inputs_by_role[role])} in {role}.", flush=True)
        measured[role] = readings
        write(run / f"{role}.json", readings)
        if role == "calibration":
            cutoffs = points(readings)
            write(run / "points.json", cutoffs)
            print("Froze separate AAC points from the calibration groups.", flush=True)
    write(run / "results.json", {"freeze": lock, "assets": assets, "points": cutoffs, "rows": measured,
                                 "outcomes": outcomes(measured, cutoffs)})


def verify(report):
    import math

    require(report["freeze"]["schema"] == SCHEMA and report["freeze"]["method"] == METHOD
            and report["freeze"]["sources"] == {path.name: sha(path) for path in (PRIOR, CONFIRM, RANDOM)}, "Study inputs differ")
    rows = report["rows"]
    require(all(witness(rows[role]) == EXPECTED[role] for role in ROLES), "Frozen delivery denominators differ")
    present = {role: [{key: value for key, value in row.items() if key not in {"scores", "window_sha256"}} for row in items]
               for role, items in rows.items()}
    inputs_bytes = (json.dumps(present, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(inputs_bytes).hexdigest() == report["freeze"]["inputs_sha256"], "Source clock ledger changed")
    for role, readings in rows.items():
        for row in readings:
            require(len(row["window_sha256"]) == len(row["scores"]["aac"]) == len(replay.GRID)
                    and all(value is None or math.isfinite(value) for value in row["scores"]["aac"])
                    and (role != "calibration" or row["truth"] is None), "Invalid calibration labels or response values")
    cutoffs = points(rows["calibration"])
    measured = outcomes(rows, cutoffs)
    require(cutoffs == report["points"] and measured == report["outcomes"], "AAC decisions do not reproduce")
    return measured


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    for name, args in (("prepare", ("run", "random_media")), ("score", ("run", "weights")), ("review", ("receipt",))):
        command = phases.add_parser(name)
        for field in args:
            command.add_argument(field, type=Path)
    args = cli.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.random_media)
    elif args.phase == "score":
        score(args.run, args.weights)
    else:
        with gzip.open(args.receipt, "rt", encoding="utf-8") as stored:
            print(json.dumps(verify(json.load(stored)), indent=2, sort_keys=True))
