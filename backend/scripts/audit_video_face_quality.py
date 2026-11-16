# SPDX-License-Identifier: MIT
"""Replay quantization arithmetic and the fixed negative camera transfer."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

from scripts.audit_video_face_temporal import cadence, public_roster
from scripts.video_face_quality_calibration import bind, compare_weights, evaluate
from scripts.video_face_temporal_fit import read, save, sha

TRACE_PREFIX = r"^\[trace_headers @ 0x[0-9a-fA-F]+\] "
PACKET = re.compile(TRACE_PREFIX + r"Packet: .*?\bpts (-?\d+),.*?$", re.M)


def read_checks(path, recorded):
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
                            "-show_entries", "stream=time_base:frame=pts,pict_type", "-of", "json", str(path)],
                           check=True, capture_output=True, timeout=30, text=True)
    clock = json.loads(probe.stdout)
    stream, = clock["streams"]
    assert stream["time_base"] == recorded["time_base"] and sha(path) == recorded["sha256"]
    visible = [(int(row["pts"]), row["pict_type"]) for row in clock["frames"]]

    traced = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-protocol_whitelist", "file",
                             "-i", str(path), "-map", "0:v:0", "-an", "-c:v", "copy", "-bsf:v", "trace_headers",
                             "-f", "null", "-"], check=True, capture_output=True, timeout=30, text=True).stderr
    parts = PACKET.split(traced)
    seed = re.findall(r"pic_init_qp_minus26\s+[01]+\s+=\s+(-?\d+)", traced)
    assert seed and len(set(seed)) == 1
    qp = {}
    for tick, section in zip(parts[1::2], parts[2::2], strict=True):
        value, = re.findall(r"slice_qp_delta\s+[01]+\s+=\s+(-?\d+)", section)
        assert int(tick) not in qp
        qp[int(tick)] = 26 + int(seed[0]) + int(value)
    assert set(qp) == {tick for tick, _ in visible} and len(qp) == len(visible) == 100
    signature = [[tick, kind, qp[tick]] for tick, kind in visible]
    quantization = recorded["quantization"]
    assert signature == quantization["frames"] and median(qp[tick] for tick, kind in visible if kind in "BP") == (
        quantization["inter_picture_median"]
    )
    return len(signature)


def audit(directory, measured, output):
    manifest = directory / "manifest.json"
    trace = read(measured)
    assert trace["manifest_sha256"] == sha(manifest)
    rows = trace["rows"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        total = sum(pool.map(lambda row: read_checks(directory / row["file"], row), rows))
    paired = cadence(read(manifest))
    stat = {"delivered_files": len(rows), "verified_qp_and_picture_clocks": total,
            "source_cadence": dict(Counter("comparable" if row["comparable"] else "different" for row in paired.values()))}
    save(output, stat)
    print(json.dumps(stat, sort_keys=True))


def cameras(result, manifest, measured, journal, output):
    data = read(result)
    headers, rows = bind(manifest, measured, (journal,))
    compare_weights(data["training_headers"] + headers)
    assert all(row["family"] == "ordinary" for row in rows) and len({row["group"] for row in rows}) == 8
    assert not ({row["group"] for row in rows} & {row["group"] for row in data["training"] + data["held"]})
    scores, summary = evaluate(rows, data["calibration"])
    save(output, {"rows": rows, "model_receipt": headers[0], "scores": scores, "summary": summary})
    for cell, record in sorted(summary.items()):
        print(cell, record["reference"],
              {key: value for key, value in record.items() if isinstance(value, dict) and "wider" in value})


def publish(result, roster, audited, camera_result, camera_audit, output):
    primary = read(result)
    side = read(camera_result)
    receipt = primary | {"acquisition": public_roster(read(roster)), "independent_audit": read(audited),
                         "camera_transfer": side | {"independent_audit": read(camera_audit)}}
    verify_transfer(receipt)
    save(output, receipt)


def verify_transfer(receipt):
    side = receipt["camera_transfer"]
    compare_weights([receipt["held_header"], side["model_receipt"]])
    assert evaluate(side["rows"], receipt["calibration"]) == (side["scores"], side["summary"])
    for cell, counts in sorted(side["summary"].items()):
        print(cell, counts["reference"], counts["pooled"]["wider"])


def replay(receipt):
    verify_transfer(read(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="action", required=True)
    for name, function, resources in (
        ("audit", audit, ("directory", "measured", "output")),
        ("cameras", cameras, ("result", "manifest", "measured", "journal", "output")),
        ("publish", publish, ("result", "roster", "audited", "camera_result", "camera_audit", "output")),
        ("replay", replay, ("receipt",)),
    ):
        command = phases.add_parser(name)
        for role in resources:
            command.add_argument(role, type=Path)
        command.set_defaults(function=function, resources=resources)
    args = parser.parse_args()
    args.function(*(getattr(args, role) for role in args.resources))
