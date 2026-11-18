# SPDX-License-Identifier: Apache-2.0
"""Compare local speech-activity gaps with pinned noisy video inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

from scripts import video_shared_sound_panel as baseline
from scripts.video_stereo_covariance_panel import (
    comparisons,
    delivery_file,
    file_hash,
    read,
    save,
    validate_pictures,
)

SCHEMA = "chipmunk.video-speech-activity-gap/1"
VAD_SHA = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
ARMS = ("original", "enhanced", "both")
BUDGETS = (0, 1, 2)


def activity(model, pcm):
    import numpy as np
    import torch

    from app.bench.video_speech_activity_gap import BLOCK

    with torch.inference_mode():
        model.reset_states()
        try:
            return np.array([model(torch.from_numpy(block.copy()), 16000).item()
                             for block in pcm[:len(pcm) // BLOCK * BLOCK].reshape(-1, BLOCK)])
        finally:
            model.reset_states()


def pick(row, arm, point):
    result = row["candidate"][arm]
    value = result["reading"]
    return (value["span"] if value and value["margin_millionths"] > point else None), result["assessed"]


def calibrate(rows):
    decisions = {}
    for carrier in baseline.CARRIERS:
        bank = [row for row in rows if row["carrier"] == carrier]
        options = []
        for arm in ARMS:
            maxima = {row["group"]: 0 for row in bank}
            for row in bank:
                if row["truth"] is None and (value := row["candidate"][arm]["reading"]):
                    group = row["group"]
                    maxima[group] = max(maxima[group], value["margin_millionths"])
            controls = sorted(maxima.values(), reverse=True)
            for budget in BUDGETS:
                point = max(150_000, controls[budget])
                result = comparisons(bank, partial(pick, arm=arm, point=point))
                utility = sum((3 if scope.endswith("/-5") else 2) * counts["correct"]
                              - counts["misplaced"] - 2 * counts["ordinary_flags"] - counts["ordinary_groups"]
                              for scope, counts in result.items())
                options.append({"arm": arm, "budget": budget, "point": point, "utility": utility})
        decision = max(options, key=lambda row: (row["utility"], -row["budget"], -ARMS.index(row["arm"])))
        decisions[carrier] = {"selected": decision, "options": options}
    return decisions


def summarize(rows, decisions, previous):
    def selected(row):
        choice = decisions[row["carrier"]]["selected"]
        return pick(row, choice["arm"], choice["point"])

    def stereo(row):
        choice = previous[row["carrier"]]
        value = row["stereo"][choice["arm"]]
        return (value["span"] if value and value["margin_millidb"] > choice["point"] else None), row["stereo_assessed"]

    alternatives = {}
    for carrier, details in decisions.items():
        bank = [row for row in rows if row["carrier"] == carrier]
        alternatives[carrier] = [item | {"counts": comparisons(bank, partial(pick, arm=item["arm"], point=item["point"]))}
                                 for item in details["options"]]
    return {"selected": comparisons(rows, selected), "near_silence": comparisons(rows, lambda row: (row["mono"], None)),
            "prior_stereo": comparisons(rows, stereo), "alternatives": alternatives}


def check_split(source, frozen, pins):
    if source["split"] != ("held" if frozen else "development"):
        raise ValueError("Held inference requires a development freeze")
    if frozen and (frozen["assets"] != pins or any(
        {row[key] for row in source["sources"]} & {row[key] for row in frozen["sources"]}
        for key in ("speaker", "room_sha256", "sha256")
    )):
        raise ValueError("Held comparison must keep the frozen implementation and disjoint sources")


def measure(args):
    import torch

    from app.bench import video_speech_activity_gap as gap
    from app.detectors.audio.decode import decode_audio
    from app.evidence.video_viseme_binding import presented_samples
    from scripts import audio_noise_view_v4 as reducer

    torch.set_num_threads(1)
    bank = read(args.bank)
    frozen = read(args.freeze) if args.freeze else None
    if file_hash(args.vad) != VAD_SHA:
        raise ValueError("Speech activity weights differ from the MIT release")
    pictures = read(args.frames)
    if bank.get("mouth_sha256", file_hash(args.frames)) != file_hash(args.frames):
        raise ValueError("Mouth observations differ from their verified measurement")
    validate_pictures(bank, pictures)
    pins = {"method": file_hash(Path(gap.__file__)), "runner": file_hash(Path(__file__)), "vad": VAD_SHA,
            "reducer": reducer.PINS, "reducer_code": file_hash(Path(reducer.__file__))}
    check_split(bank, frozen, pins)
    model = torch.jit.load(str(args.vad), map_location="cpu").eval()
    enhance_model = reducer.load(args.enhancer)
    rows = []
    for number, row in enumerate(bank["rows"], 1):
        path = delivery_file(args.media, row["delivery"])
        checksum = file_hash(path)
        if checksum != row["input_sha256"]:
            raise ValueError(f"Unexpected original video bytes for {path.name}")
        decoded = decode_audio(path, source_sha256=checksum, retain_stereo=True)
        count = presented_samples(decoded.mapping)
        stereo = decoded.stereo_pcm
        if (count != row["samples"] or stereo is None or len(stereo) < count
                or hashlib.sha256(stereo[:count].astype("<i2", copy=False).tobytes()).hexdigest() != row["stereo_pcm_sha256"]):
            raise ValueError(f"Unexpected stereo sample clock or PCM for {path.name}")
        pcm = stereo[:count].mean(axis=1, dtype="float32") / 32768
        reduced = reducer.enhance(pcm, enhance_model)
        candidate = gap.scan(activity(model, pcm), activity(model, reduced), pictures[row["group"]])
        row_fields = ("group", "carrier", "snr_db", "role", "truth", "samples", "input_sha256", "stereo_pcm_sha256", "delivery")
        rows.append({key: row[key] for key in row_fields} | {"candidate": candidate, "stereo": row["with_picture"],
                     "stereo_assessed": row["audio_eligible"], "mono": row["mono_baseline"]})
        if number % 28 == 0:
            print(f"Verified and measured {number}/{len(bank['rows'])} deliveries", flush=True)
    decisions = frozen["decisions"] if frozen else calibrate(rows)
    previous = frozen["previous"] if frozen else bank["choices"]
    result = summarize(rows, decisions, previous)
    save(args.output, {"schema": SCHEMA, "split": bank["split"], "assets": pins, "sources": bank["sources"], "rows": rows,
                       "source_receipt_sha256": file_hash(args.bank), "mouth_sha256": file_hash(args.frames),
                       "freeze_sha256": file_hash(args.freeze) if frozen else None, "previous": previous,
                       "decisions": decisions, "summary": result})
    print(json.dumps(result["selected"], sort_keys=True), flush=True)


def verify(path, freeze=None):
    document = read(path)
    frozen = read(freeze) if freeze else None
    if document["schema"] != SCHEMA:
        raise ValueError("Unexpected evaluation receipt")
    check_split(document, frozen, document["assets"])
    if freeze:
        verify(freeze)
        if (document["freeze_sha256"] != file_hash(freeze) or document["decisions"] != frozen["decisions"]
                or document["previous"] != frozen["previous"]):
            raise ValueError("Held conditions differ from the development freeze")
    elif document["decisions"] != calibrate(document["rows"]):
        raise ValueError("Development operating settings differ from the receipt")
    if document["summary"] != summarize(document["rows"], document["decisions"], document["previous"]):
        raise ValueError("Comparison counts differ from the retained ranges")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest="mode", required=True)
    run = cli.add_parser("measure")
    for name in ("bank", "media", "frames", "vad", "enhancer", "output"):
        run.add_argument(name, type=Path)
    recount = cli.add_parser("verify")
    recount.add_argument("receipt", type=Path)
    for sub in (run, recount):
        sub.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.mode == "measure":
        measure(args)
    else:
        count = verify(args.receipt, args.freeze)["summary"]
        print(json.dumps({name: value for name, value in count.items() if name != "alternatives"}, sort_keys=True))


if __name__ == "__main__":
    main()
