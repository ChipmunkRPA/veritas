"""Confirm the frozen stricter MP3 short-window point on separate source groups."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts import audio_communication_serving_grid as cohorts
from scripts import audio_short_codec_transfer as codec
from scripts import review_audio_short_codec_transfer as review

ROOT, RATE, SAMPLES = codec.ROOT, codec.RATE, codec.SAMPLES
DEFAULT_RUN = codec.windows.BASE / "audio-short-mp3-confirmation-2026-11-12"
SCHEMA = "chipmunk.audio-short-mp3-confirmation/1"
READ, WRITE, SHA, REQUIRE = codec.read, codec.write, codec.sha, codec.require


def source_cases():
    prior = list(codec.inputs(codec.grouped.HELD))
    seen = {codec.digest(pcm) for _, _, _, pcm in prior}
    genuine_seen = {row["host"]["canonical_speaker"] for _, _, row, _ in prior if row.get("host")}
    donors_seen = {row["donor"]["canonical_speaker"] for _, _, row, _ in prior if row.get("donor")}
    groups = {}
    for number, (row, pcm, _, _) in enumerate(cohorts.cases("confirmation")):
        digest = codec.digest(pcm)
        REQUIRE(digest not in seen and (not row.get("host") or row["host"]["canonical_speaker"] not in genuine_seen),
                "Confirmation audio or host speaker overlaps the codec transfer")
        group = groups.setdefault(row["group"], f"group-{len(groups):02d}")
        reused_voice = row["arm"] == "synthetic" and row["donor"]["canonical_speaker"] in donors_seen
        yield {"id": f"case-{number:02d}", "group": group, "truth": row["truth"], "condition": row["arm"],
               "family": row["family"], "prior_generator_profile": bool(reused_voice), "original": {"sha256": digest}}, pcm


def frozen(run, rows):
    receipt = READ(codec.RELEASE)
    REQUIRE(len(rows) == 52 and len({row["group"] for row in rows}) == 16
            and sum(row["truth"] is not None for row in rows) == 12, "Confirmation denominator changed")
    return {
        "schema": SCHEMA, "inputs": {"sha256": SHA(run / "inputs.json")},
        "earlier_codec_receipt": {"sha256": SHA(codec.RELEASE)},
        "source_receipt": {"sha256": SHA(cohorts.public_receipt("confirmation"))},
        "runner": {"sha256": SHA(Path(__file__))},
        "assets": codec.windows.native.transfer.OLD.voice.ASSETS,
        "groups": 16, "cases": 52, "cases_by_condition": dict(Counter(row["condition"] for row in rows)),
        "generator_profiles_previously_used": sum(row["prior_generator_profile"] for row in rows),
        "cutoffs": {arm: receipt["points"]["mp3"][arm]["1"] for arm in codec.ARMS},
        "comparison": {"useful_location_iou": .30, "useful_location_edit_recall": .50,
                       "minimum_new_useful_groups": 2, "maximum_new_false_groups": 1},
    }


def prepare(run):
    run.mkdir(mode=0o700)
    records = []
    for row, pcm in source_cases():
        path = run / (row["id"] + ".mp3")
        codec.recipe("mp3", path, pcm)
        decoded, clock = codec.decode_delivered(path, "mp3")
        row["delivery"] = {"path": path.name, "bytes": {"sha256": SHA(path)},
                           "pcm": {"sha256": codec.digest(decoded)}, "clock": clock}
        records.append(row)
    WRITE(run / "inputs.json", records)
    WRITE(run / "freeze.json", frozen(run, records))
    print("Sealed 52 MP3 deliveries with frozen stricter points and 16 groups.", flush=True)


def score(run, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    rows = READ(run / "inputs.json")
    lock = READ(run / "freeze.json")
    REQUIRE(lock == frozen(run, rows), "The prior points, source panel, or scorer changed after freezing")
    codec.windows.native.transfer.OLD.spectra.incumbent.network_free()
    REQUIRE(all(SHA(weights / name) == pin for name, pin in lock["assets"].items()), "Pinned model assets changed")
    activity = SpeechActivity(weights / "silero_vad.jit")
    model = load_voice({"weights": str(weights), "assets": lock["assets"]})
    measured = []
    for idx, row in enumerate(rows, 1):
        source = row["delivery"]
        path = run / source["path"]
        REQUIRE(SHA(path) == source["bytes"]["sha256"], "MP3 carrier changed")
        pcm, clock = codec.decode_delivered(path, "mp3")
        REQUIRE(codec.digest(pcm) == source["pcm"]["sha256"] and clock == source["clock"], "Decoded samples or clock changed")
        measured.append(row | {"windows": codec.windows.measure(pcm, model, activity, {})})
        if idx % 10 == 0:
            print(f"Scored {idx} exact MP3 files", flush=True)
    WRITE(run / "scores.json", measured)
    ledger = [{**row, "scores": {"mp3": [window["score"] for window in row["windows"]]}} for row in measured]
    metrics = {arm: review.held_metrics(ledger, "mp3", arm, lock["cutoffs"][arm]) for arm in codec.ARMS}
    WRITE(run / "results.json", {"freeze": {"sha256": SHA(run / "freeze.json")}, "metrics": metrics})
    print(metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--weights", type=Path, default=codec.windows.native.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run)
    else:
        score(args.run, args.weights)


if __name__ == "__main__":
    main()
