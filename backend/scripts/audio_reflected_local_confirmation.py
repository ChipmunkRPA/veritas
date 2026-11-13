# SPDX-License-Identifier: Apache-2.0
"""Transfer the fixed 1.2-second speech point to different exact recordings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import audio_reflected_local_transfer as prior

io, require = prior.io, prior.require
SCHEMA = "chipmunk.audio-reflected-local-confirmation/1"
BASE = io.PUBLIC / "audio-reflected-local-transfer-2026-11-12.json"
PANEL = io.PUBLIC / "audio-dense-evidence-confirmation-2026-11-12.json"


def inputs():
    reference, dataset = io.read(BASE), io.read(PANEL)
    require(reference["freeze"] == prior.frozen(), "Original reflection procedure changed")
    previous, _ = prior.sources.inputs()
    original_groups = {
        row["source"]["group"] for rows in previous.values() for row, _ in rows if row["source"]
    }
    original_sources = {
        row["source"]["source_pcm_sha256"] for rows in previous.values() for row, _ in rows if row["source"]
    }
    groups = Counter()
    selected = []
    for row, item in zip(dataset["roster"], dataset["responses"], strict=True):
        require(row["source_pcm_sha256"] not in original_sources, "Reused original foreground samples")
        require(
            item["id"] == row["id"]
            and [tuple(w["span"]) for w in item["windows"]]
            == sorted(set(map(tuple, sum(prior.clock.schedules(row["samples"]), [])))),
            "Original dense schedule changed",
        )
        role = "seen_speaker" if row["group"] in original_groups else "distinct_speaker"
        groups[role, row["scene"], row["label"]] += 1
        source = {key: row[key] for key in ("label", "group", "family", "edits")}
        selected.append((row | {"source": source, "prior_exposure": role}, item["windows"]))
    require(len(selected) == 60, "Exact prior dense panel required")
    expected = {
        (scene, label): count
        for scene in ("gain_control", "traffic")
        for label, count in (("genuine", 6), ("partial", 4))
    }
    require(
        {
            (scene, label): count
            for (role, scene, label), count in groups.items()
            if role == "distinct_speaker"
        }
        == expected,
        "Distinct speakers differ",
    )
    freeze = {
        "schema": SCHEMA,
        "runner": io.sha(Path(__file__)),
        "prior": io.sha(BASE),
        "panel": io.sha(PANEL),
        "point": reference["point"],
        "voice_assets": dataset["freeze"]["voice_assets"],
    }
    return freeze, selected


def outcome(held, readings, point, folder=None):
    result = {"all": prior.outcomes(held, readings, point, folder)}
    for role in ("seen_speaker", "distinct_speaker"):
        matches = [
            (row, value)
            for row, value in zip(held, readings, strict=True)
            if row[0]["prior_exposure"] == role
        ]
        rows, values = zip(*matches, strict=True)
        result[role] = prior.outcomes(rows, values, point)
    return result


def run(folder, native):
    import torch

    from scripts.audio_reduced_background_localization import assets_for_scores

    lock, held = inputs()
    folder.mkdir(mode=0o700)
    io.save(folder / "freeze.json", lock)
    torch.set_num_threads(2)
    voice, activity = assets_for_scores(lock)
    measured = prior.measure(held, native, voice, activity, "held")
    io.save(folder / "readings.json", measured)
    result = outcome(held, measured, float.fromhex(lock["point"]["cutoff"]))
    io.save(folder / "receipt.json", {"freeze": lock, "readings": measured, "result": result})


def review(path, native):
    lock, held = inputs()
    record = io.read(path)
    require(record["freeze"] == lock, "Frozen transfer or point differs")
    result = outcome(held, record["readings"], float.fromhex(lock["point"]["cutoff"]), native)
    require(record["result"] == result, "Transfer results differ")
    print(io.encoded(result).decode(), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    if args.phase == "run" and args.native is None:
        parser.error("run requires --native")
    (run if args.phase == "run" else review)(args.path, args.native)
