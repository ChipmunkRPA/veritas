"""Recount the 48 kHz speech comparison and verify its exact deliveries without models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

from app.bench import independent
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import audio_two_second_carrier_transfer as study


def export(receipt, output):
    from scripts.audio_two_second_transfer import prior

    assert len(receipt["rows"]) == 96 and study.recount(receipt["rows"]) == receipt["counts"]
    inputs, truth, baseline, combined = [], [], [], []
    for row in receipt["rows"]:
        case = {"id": row["id"], "input_sha256": row["file_sha256"]}
        kind = row["kind"]
        assert row["native_ticks"] == 3 * row["samples"] and row["decoded_samples"] >= row["samples"]
        inputs.append(case | {
            "modality": "audio", "split": "confirmation", "source_sha256": row["pcm_sha256"],
            "group": f"reader/{row['group']}" if kind != "background" else f"ami/{row['room']}",
            "lineage": [f"reader/{row['group']}", f"ami/{row['room']}"],
            "condition": f"{row['role']}/{row['condition']}/{kind if kind == 'background' else 'mixture'}",
            "label": int(kind == "edited"), "family": row["family"] or kind,
            "license": {"id": "CC BY 4.0 (LlamaPartialSpoof, LibriSpeech, AMI)",
                        "evaluation_permitted": True, "redistribution_permitted": True},
            "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                   "target_units": sum(b-a for a, b in row["truth"])},
        })
        truth.append(case | {"extent": [row["samples"]], "regions": row["truth"]})
        old, joined, _ = study.ranges(row)
        current = any(w["direction"] != "unassessed" for w in row["current_windows"])
        short = any(excerpt["responses"] for excerpt in row["excerpts"])
        for results, spans, assessed in ((baseline, old, current), (combined, joined, current or short)):
            if assessed:
                results.append(case | {"status": "ok", "response": int(bool(spans)),
                                       "localization": {"status": "ok", "regions": [list(span) for span in spans]}})
            else:
                reason = "No speech window was assessed."
                results.append(case | {"status": "not_applicable", "response": None, "reason": reason,
                                       "localization": {"status": "not_applicable", "reason": reason}})
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": study.binding.RATE, "cases": inputs}
    identity = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    targets = identity | {"schema": TRUTH_SCHEMA, "cases": truth}
    output.mkdir(mode=0o700)
    prior.save(output / "manifest.json", manifest)
    prior.save(output / "truth.json", targets)
    for name, rows in (("baseline", baseline), ("combined", combined)):
        response = identity | {"system": f"{receipt['carrier']}/{name}", "method_kind": "weighted", "score_kind": "raw",
                               "response_units": "A source range was flagged (0 or 1)", "predictions": rows,
                               "implementation_sha256": independent.digest([receipt["plan"]["code"], name]),
                               "weight_sha256": sorted(receipt["model_sha256"].values())}
        events = count_events(manifest, targets, response, minimum_iou=.30, minimum_recall=.50)
        for role in study.ROLE:
            assert sum(key["matched_events"] for key in events["strata"] if key["condition"].startswith(role + "/")) == (
                receipt["counts"][role][f"useful_{name}"])
        prior.save(output / f"{name}.json", response)
        prior.save(output / f"{name}-events.json", events)


def verify_native(receipt, roots):
    carrier = receipt["carrier"]
    padding = 0
    with tempfile.TemporaryDirectory() as work:
        for row in receipt["rows"]:
            source = roots[row["role"]] / "pcm" / f"{row['pcm_sha256']}.pcm"
            assert study.digest(source.read_bytes()) == row["pcm_sha256"]
            path = Path(work) / f"delivery.{carrier}"
            assert study.encode(source, path) == row["file_sha256"]
            probe = subprocess.check_output(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,time_base,duration_ts:frame=pts,nb_samples", "-of", "json", str(path)], timeout=20)
            recording = json.loads(probe)
            stream, = recording["streams"]
            assert (stream["sample_rate"], stream["time_base"], stream["duration_ts"]) == ("48000", "1/48000", row["native_ticks"])
            clock = 0
            for frame in recording["frames"]:
                assert frame["pts"] == clock
                clock += frame["nb_samples"]
            assert 0 <= clock - row["native_ticks"] < (1024 if carrier == "m4a" else 1)
            padding = max(padding, clock - row["native_ticks"])
            pcm = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-copyts", "-i", str(path),
                "-map", "0:a:0", "-af", "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000",
                "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], timeout=20)
            integers = array("h")
            integers.frombytes(pcm)
            if sys.byteorder == "big":
                integers.byteswap()
            floats = array("f", (value / 32768. for value in integers))
            if sys.byteorder == "big":
                floats.byteswap()
            assert len(floats) == row["decoded_samples"] and study.digest(floats.tobytes()) == row["decoded_pcm_sha256"]
            path.unlink()
    return {"carrier": carrier, "files": len(receipt["rows"]), "max_native_padding_ticks": padding}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    result = study.previous.prior.read(args.receipt)
    export(result, args.output)
    if args.native:
        print(json.dumps(verify_native(result, dict(zip(study.ROLE, args.native, strict=True)))))
