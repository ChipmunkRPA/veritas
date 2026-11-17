# SPDX-License-Identifier: Apache-2.0
"""Score both prepared voice views with the frozen six-second speech head."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts import audio_focused_context_transfer as event
from scripts import audio_long_context_fit as model
from scripts import audio_long_context_transfer as heads
from scripts import audio_overlap_reconstruction as study
from scripts import audio_overlap_reconstruction_views as views
from scripts.audio_separation_third_voice_pilot import signal_quality


def strongest(candidates):
    return [max((reading for reading in column if reading is not None),
                key=lambda reading: (reading["score"], -reading["range"][0]), default=None)
            for column in zip(*candidates, strict=True)]


def score(args):
    from scripts import audio_clean_teacher_transfer as teacher
    from scripts import audio_short_edit_ssl_transfer as features

    model.start_cpu()
    plan = event.read(args.plan)
    rows = [row for row in plan["rows"] if row["split"] == args.split]
    fit_record, predictors = heads.fitted_heads(args.fit)
    if tuple(model.WIDTHS) != study.WIDTHS or model.file_sha(args.speech) != fit_record["speech_sha256"]:
        raise ValueError("Expected the frozen speech activity detector and window widths")
    expected = {"plan_sha256": model.file_sha(args.plan), "fit_sha256": model.file_sha(args.fit / "freeze.json"),
                "encoder": fit_record["encoder"], "speech_sha256": fit_record["speech_sha256"],
                "view_code_sha256": model.file_sha(Path(views.__file__)), "scoring_sha256": model.file_sha(Path(__file__))}
    if args.split != "calibration" and (args.calibration is None or event.read(args.calibration)["freeze"] != expected):
        raise ValueError("Test partitions require previously frozen operating points for these exact inputs")
    predict = predictors["short"]
    encode = features.frozen_encoder(args.encoder, variable_context=True)
    activity = teacher.runtime(args.speech)

    def reading(pcm):
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        peak = float(np.max(np.abs(pcm)))
        if pcm.shape != (views.SAMPLES,) or not np.isfinite(pcm).all():
            raise ValueError("Invalid separated waveform on the measured clock")
        if peak >= 1:
            pcm *= .99 / peak
        spans = activity.spans(pcm)
        if not spans:
            return [None] * len(study.WIDTHS)
        return heads.common_reading(predict, encode(pcm), 0, spans)

    journal = args.views / f"{args.split}.json"
    committed, processed, result = {}, {}, []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        digest = row["view_id"]
        bank = args.data / plan["banks"][row["bank"]]
        mixture = views.open_pcm(bank, row)
        if digest not in processed:
            deadline = time.monotonic() + 240
            while digest not in committed:
                if journal.exists():
                    current = event.read(journal)
                    if (current["construction"]["plan_sha256"] != expected["plan_sha256"]
                            or current["construction"]["code_sha256"] != expected["view_code_sha256"]
                            or current["construction"]["assets"] != views.ASSETS):
                        raise ValueError("Waveform views were generated with a different plan, code or weights")
                    committed = current["waveforms"]
                if digest in committed:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the voice estimates journal")
                time.sleep(.5)
            path = args.views / f"{digest}.npz"
            if (committed[digest]["file_sha256"] != model.file_sha(path)
                    or committed[digest]["mixture_sha256"] != hashlib.sha256(mixture.tobytes()).hexdigest()):
                raise ValueError("Separated views differ from their retained source commitments")
            with np.load(path, allow_pickle=False) as packets:
                if np.max(np.abs(packets["mixture_phase"].sum(axis=0) - mixture)) > 2e-6:
                    raise ValueError("Mixture views must reconstitute the decoded source")
                readings = {method: strongest([reading(stream) for stream in packets[method]]) for method in study.METHODS}
            raw = row["raw_readings"] if "raw_readings" in row else reading(mixture)
            processed[digest] = raw, readings
        raw, readings = processed[digest]
        sample = row | {"raw_readings": raw, "readings": readings, "view_sha256": committed[digest]["file_sha256"]}
        if clean := row["clean_reference"]:
            a = (clean["samples"] - views.SAMPLES) // 2
            target = views.open_pcm(bank, clean | {"scope": [a, a + views.SAMPLES]})
            if target.shape != mixture.shape or (clean["samples"] - views.SAMPLES) // 2 != row["scope"][0]:
                raise ValueError("The controlled reference used a different speech clock")
            sample["clean_readings"] = reading(target)
            with np.load(args.views / f"{digest}.npz", allow_pickle=False) as packets:
                sample["target_si_sdr_db"] = {"raw": round(signal_quality(mixture, target), 6)} | {
                    method: round(max(signal_quality(track, target) for track in packets[method]), 6) for method in study.METHODS}
        result.append(sample)
        if index == 1 or index % 12 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} {args.split} recordings in {time.monotonic() - started:.1f} seconds", flush=True)
    output = {"freeze": expected, "rows": result, "view_runtime": current["construction"]["runtime"]}
    if args.split == "calibration":
        points = {name: model.calibrate([row | {"readings": row["readings"][name]} for row in result]) for name in study.METHODS}
        output["points"] = {"raw": plan["raw_points"]} | points
        print(json.dumps(output["points"], sort_keys=True), flush=True)
    event.save(args.output, output)


def collect(args):
    cal, held, matched = [event.read(path) for path in (args.calibration, args.held, args.matched)]
    if cal["freeze"] != held["freeze"] or cal["freeze"] != matched["freeze"]:
        raise ValueError("The compared input plans or runtime check are different")
    rows = held["rows"] + matched["rows"]
    document = {"schema": study.SCHEMA, "design": study.DESIGN, "calibration": cal["rows"], "rows": rows,
                "points": cal["points"], "freeze": cal["freeze"], "view_runtime": cal["view_runtime"],
                "view_assets": {"revision": views.REVISION, "sha256": views.ASSETS},
                "summary": study.summarize(rows, cal["points"]), "reference_diagnostics": study.reference_diagnostics(rows, cal["points"])}
    print(json.dumps(study.verify(document), sort_keys=True), flush=True)
    event.save(args.output, document)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    measure = modes.add_parser("score")
    for name in ("plan", "data", "views", "fit", "encoder", "speech", "output"):
        measure.add_argument(name, type=Path)
    measure.add_argument("split", choices=("calibration", "held", "matched"))
    measure.add_argument("--calibration", type=Path)
    combine = modes.add_parser("collect")
    for name in ("calibration", "held", "matched", "output"):
        combine.add_argument(name, type=Path)
    args = parser.parse_args()
    (score if args.mode == "score" else collect)(args)
