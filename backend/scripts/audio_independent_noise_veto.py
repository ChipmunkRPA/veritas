"""Evaluate an independent voice veto on disjoint recorded rooms and deliveries."""

from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration
from app.detectors.audio.decode import decode_audio
from scripts import audio_spectra_background_transfer as spectra
from scripts.audio_spectra_background_transfer import PANEL, read, sha, write_new

SCHEMA = "chipmunk.audio-independent-noise-veto/1"
RULE = {
    "selection": "Highest saved independent score preserving every clean AI flag and all but one noisy AI flag "
                 "among synthetic foregrounds in the earlier two-room comparison.",
    "policy": "Only an incumbent AI direction can be withheld. Withhold below the frozen independent score. "
              "Return inconclusive after withholding. No AI flags are added.",
    "minimum_selection_clean": 6,
    "minimum_selection_noisy": 8,
    "center_minimum_corrected": 3,
    "center_minimum_groups": 2,
    "center_required_rooms": 2,
    "center_maximum_lost": 2,
    "offset_maximum_lost": 1,
    "delivery_maximum_lost_per_codec": 1,
    "clean_maximum_lost": 0,
}
SOURCE_ARTIFACTS = {
    "selection": ("audio-spectra-recovery-comparison", {"freeze": "freeze_sha256", "confirm": "confirmation_sha256"}),
    "held": ("audio-spatial-interference", {"freeze": "freeze_sha256", "scores": "scores_sha256"}),
    "delivery": ("audio-phase-corroboration", {"freeze": "freeze_sha256", "scores": "scores_sha256"}),
}


def receipts(selection, held, delivery):
    result = {}
    for role, folder in (("selection", selection), ("held", held), ("delivery", delivery)):
        name, keys = SOURCE_ARTIFACTS[role]
        public = read(PANEL / f"{name}-2026-11-11.json")
        binds = {name: sha(folder / f"{name}.json") for name in keys}
        if any(binds[name] != public[key] for name, key in keys.items()):
            raise ValueError(f"The published {role} commitments differ")
        result[role] = binds
    return result


def selection_and_views(selection, held, delivery):
    binds = receipts(selection, held, delivery)
    frozen = {role: read(folder / "freeze.json") for role, folder in
              (("selection", selection), ("held", held), ("delivery", delivery))}
    for role, folder in (("selection", selection), ("held", held)):
        if frozen[role]["roster_sha256"] != sha(folder / "roster.json"):
            raise ValueError(f"The bound {role} roster differs")
    if (frozen["selection"]["incumbent_calibration_sha256"] != calibration.SHA256
            or frozen["held"]["calibration_sha256"] != calibration.SHA256
            or frozen["selection"]["candidate_assets"] != spectra.ASSETS
            or frozen["delivery"]["parent"]["scores_sha256"] != binds["held"]["scores"]):
        raise ValueError("Source calibration, held delivery or independent checkpoint changed")

    selected = [row for row in read(selection / "roster.json") if row["role"] == "prospective"]
    selected_scores = read(selection / "confirm.json")["rows"]
    held_rows = read(held / "roster.json")
    held_scores = read(held / "scores.json")["rows"]
    held_clean = [row for row in held_rows if row["role"] == "transfer" and row["placement"] == "clean"]
    held_ids = {row["group"] for row in held_clean}
    select_ids = {row["group"] for row in selected if row["condition"] == "clean"}
    held_digests = {row["mono_sha256"] for row in held_clean}
    select_digests = {row[key] for row in selected if row["condition"] == "clean"
                      for key in ("pcm_sha256", "source_pcm_sha256")}
    room_keys = {room["meeting"] for room in frozen["selection"]["prospective_backgrounds"]}
    if (held_ids & select_ids or held_digests & select_digests or room_keys & frozen["held"]["publisher_mics"].keys()
            or (len(select_ids), len(held_ids), len(room_keys), len(frozen["held"]["publisher_mics"])) != (17, 16, 2, 2)):
        raise ValueError("The two roles are not source-disjoint under their recorded identifiers")
    chosen = []
    for row, score in zip(selected, selected_scores, strict=True):
        if any(row[key] != score[key] for key in ("id", "group", "family", "pcm_sha256")):
            raise ValueError("Selection scores are not tied to their source")
        if score["decision"] == "ai":
            if not isinstance(score["candidate"], float) or not math.isfinite(score["candidate"]):
                raise ValueError("Missing independent selection score")
            chosen.append({"condition": row["condition"], "family": row["family"], "candidate": score["candidate"]})

    native = {}
    for row, score in zip(held_rows, held_scores, strict=True):
        if any(row[key] != score[key] for key in ("id", "mono_sha256")):
            raise ValueError("Archived incumbent scores do not bind held PCM")
        if row["role"] == "transfer":
            native[row["id"]] = {**{key: row[key] for key in ("id", "family", "group", "placement", "mode", "room")},
                                  "pcm_sha256": row["mono_sha256"], "baseline": score["baseline"]["decision"],
                                  "delivery": "native"}
    views = list(native.values())
    delivery_rows = read(delivery / "scores.json")["observed"]
    seen = set()
    for row in delivery_rows:
        if row["view"] != "parent":
            source = native[row["id"]]
            pair = row["id"], row["view"]
            if pair in seen or row["view"] not in {"opus", "restored"}:
                raise ValueError("Duplicate or unsupported transformed delivery")
            seen.add(pair)
            views.append({**source, "pcm_sha256": row["sha256"], "baseline": row["baseline"]["decision"],
                          "delivery": row["view"], "file_sha256": row["delivery_sha256"],
                          "timeline": row["timeline"]})
    expected = {(i, name) for i, row in native.items() if row["placement"] == "clean" or
                (row["placement"] == "center" and row["mode"] == "simultaneous") for name in ("opus", "restored")}
    if len(native) != 116 or len(seen) != 96 or seen != expected:
        raise ValueError("Incomplete held room or delivery roster")
    return chosen, views, binds


def operating_point(chosen):
    control = sorted(row["candidate"] for row in chosen if row["family"] != "genuine" and row["condition"] == "clean")
    noisy = sorted(row["candidate"] for row in chosen if row["family"] != "genuine" and row["condition"] != "clean")
    if len(control) < RULE["minimum_selection_clean"] or len(noisy) < RULE["minimum_selection_noisy"]:
        raise ValueError("Insufficient positive selection controls")
    return min(control[0], noisy[1])


def pcm(held, delivery, row):
    i, codec = row["id"], row["delivery"]
    if codec == "native":
        data = (held / "mono" / f"{i:03}.pcm").read_bytes()
    else:
        extension = "opus" if codec == "opus" else "wav"
        source = delivery / "delivery" / f"{i:03}-{codec}.{extension}"
        if sha(source) != row["file_sha256"]:
            raise ValueError("Delivery file hash changed")
        decoded = decode_audio(source, source_sha256=row["file_sha256"], max_seconds=3)
        if decoded.mapping["timeline"] != row["timeline"]:
            raise ValueError("Delivery source clock changed")
        data = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
    if len(data) != 96_000 or sha(data) != row["pcm_sha256"]:
        raise ValueError("Held model input PCM changed")
    return data


def freeze(run, selection, held, delivery, assets):
    chosen, views, binds = selection_and_views(selection, held, delivery)
    point = operating_point(chosen)
    record = {"schema": SCHEMA, "rule": RULE, "source_bindings": binds,
              "runner_sha256": sha(Path(__file__)), "checkpoint_assets": spectra.pinned_assets(assets),
              "incumbent_calibration_sha256": calibration.SHA256, "point": point,
              "selection_ai_controls": len(chosen), "held": views}
    run.mkdir(parents=True)
    write_new(run / "freeze.json", record)
    print(f"Frozen {len(views)} held views and operating point from {len(chosen)} selection AI flags.", flush=True)


def score(run, selection, held, delivery, assets):
    lock = read(run / "freeze.json")
    chosen, views, binds = selection_and_views(selection, held, delivery)
    if (lock["runner_sha256"] != sha(Path(__file__)) or lock["rule"] != RULE or lock["held"] != views
            or lock["point"] != operating_point(chosen) or lock["source_bindings"] != binds
            or lock["checkpoint_assets"] != spectra.ASSETS):
        raise ValueError("Frozen rules or sources changed")
    spectra.incumbent.network_free()
    start = time.monotonic()
    session = spectra.start_session(assets)
    results = []
    for row in views:
        waveform = pcm(held, delivery, row)
        value, digest = spectra.candidate_score(session, waveform) if row["baseline"] == "ai" else (None, None)
        results.append({"id": row["id"], "delivery": row["delivery"], "candidate": value, "model_input_sha256": digest})
    first = next(row for row in views if row["baseline"] == "ai")
    expected = next(row for row in results if (row["id"], row["delivery"]) == (first["id"], first["delivery"]))
    repeat = spectra.candidate_score(session, pcm(held, delivery, first))
    if repeat != (expected["candidate"], expected["model_input_sha256"]):
        raise ValueError("Deterministic CPU repeat differs")
    receipt = {"freeze_sha256": sha(run / "freeze.json"), "rows": results,
               "forwards": 1 + sum(row["candidate"] is not None for row in results),
               "elapsed_s": round(time.monotonic() - start, 3), "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_new(run / "scores.json", receipt)
    print(f"{receipt['forwards']} forwards completed.", flush=True)


def tally(views, scores, point):
    cells = defaultdict(Counter)
    groups, rooms = set(), set()
    withheld_clean = 0
    for row, score in zip(views, scores, strict=True):
        if (row["id"], row["delivery"]) != (score["id"], score["delivery"]):
            raise ValueError("A held response differs from the frozen input order")
        origin = "genuine" if row["family"] == "genuine" else "synthetic"
        value = score["candidate"]
        if (value is not None) != (row["baseline"] == "ai") or value is not None and not math.isfinite(value):
            raise ValueError("The independent response is absent or outside its operating scope")
        withheld = value is not None and value < point
        if withheld and row["delivery"] == "native" and row["placement"] == "center" and origin == "genuine":
            groups.add(row["group"])
            rooms.add(row["room"])
        withheld_clean += int(withheld and row["placement"] == "clean")
        key = f'{row["delivery"]}/{row["placement"]}/{origin}'
        cells[key].update({"planned": 1, f'current_{row["baseline"]}': 1, "withheld": int(withheld),
                           "remaining_ai": int(row["baseline"] == "ai" and not withheld)})
    checks = {
        "center_genuine_cleared": cells["native/center/genuine"]["withheld"] >= RULE["center_minimum_corrected"],
        "center_genuine_groups": len(groups) >= RULE["center_minimum_groups"],
        "center_rooms": len(rooms) == RULE["center_required_rooms"],
        "center_synthetic_retained": cells["native/center/synthetic"]["withheld"] <= RULE["center_maximum_lost"],
        "offset_synthetic_retained": cells["native/offset/synthetic"]["withheld"] <= RULE["offset_maximum_lost"],
        "clean_retained": withheld_clean <= RULE["clean_maximum_lost"],
        "delivery_synthetic_retained": all(cells[f"{codec}/center/synthetic"]["withheld"] <= RULE["delivery_maximum_lost_per_codec"]
                                            for codec in ("opus", "restored")),
    }
    return {"checks": checks, "advance": all(checks.values()), "center_corrected_groups": len(groups),
            "center_corrected_rooms": len(rooms), "cells": {key: dict(value) for key, value in cells.items()}}


def report(run, selection, held, delivery):
    lock, measured = (read(run / f"{name}.json") for name in ("freeze", "scores"))
    chosen, views, bindings = selection_and_views(selection, held, delivery)
    if measured["freeze_sha256"] != sha(run / "freeze.json") or lock["source_bindings"] != bindings:
        raise ValueError("The measured run or source receipts differ")
    point = operating_point(chosen)
    if point != lock["point"] or views != lock["held"]:
        raise ValueError("The frozen selection or held inputs differ")
    result = {"schema": SCHEMA, "rule": RULE, "source_bindings": bindings, "point": point,
              "freeze_sha256": measured["freeze_sha256"], "scores_sha256": sha(run / "scores.json"),
              "selection": {key: dict(value) for key, value in summarize_selection(chosen, point).items()},
              "held_views": len(views), "forwards": measured["forwards"], "elapsed_s": measured["elapsed_s"],
              "peak_rss_bytes": measured["rss_bytes"], **tally(views, measured["rows"], point)}
    write_new(run / "aggregate.json", result)
    print(result["checks"], flush=True)


def summarize_selection(chosen, point):
    cells = defaultdict(Counter)
    for row in chosen:
        origin = "genuine" if row["family"] == "genuine" else "synthetic"
        kind = "clean" if row["condition"] == "clean" else "noisy"
        cells[f"{kind}/{origin}"].update({"current_ai": 1, "withheld": int(row["candidate"] < point)})
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "report"))
    parser.add_argument("run", type=Path)
    for option in ("selection", "held", "delivery", "assets"):
        parser.add_argument(f"--{option}", required=option != "assets", type=Path)
    args = parser.parse_args()
    locations = (args.run, args.selection, args.held, args.delivery)
    if args.stage == "report":
        report(*locations)
    else:
        if args.assets is None:
            parser.error("--assets is required for freeze and score")
        (freeze if args.stage == "freeze" else score)(*locations, args.assets)


if __name__ == "__main__":
    main()
