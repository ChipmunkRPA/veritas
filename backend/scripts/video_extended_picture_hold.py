"""Reproduce a bounded repeated-picture scan on authored and optional webcam fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from app.detectors.video import picture_hold
from app.detectors.video.decode import probe_frames
from app.evidence import picture_hold_binding as binding


@dataclass(frozen=True)
class Case:
    name: str
    frames: int
    fps: int = 25
    start: int | None = None
    end: int | None = None
    variable_clock: bool = False
    clock_gap_at: int | None = None
    codec: str = "rgb_lossless"
    input_name: str | None = None
    ordinary_cut: bool = False


def _cases(webcams):
    fixtures = [
        Case("late_variable_clock", 250, start=170, end=195, variable_clock=True),
        Case("late_vp9", 250, start=155, end=184, codec="vp9_lossless"),
        Case("holds_cross_clock_break", 250, start=170, end=190, clock_gap_at=180),
        Case("hold_truncated_by_budget", 680, 20, start=580, end=620),
        Case("hold_beyond_budget", 680, 20, start=622, end=652),
        Case("late_lossy_delivery", 250, start=170, end=195, codec="h264_lossy"),
    ]
    for number, file in enumerate(webcams, 1):
        fixtures += [Case(f"webcam_{number}_unchanged", 240, input_name=str(file)),
                     Case(f"webcam_{number}_ordinary_cut", 240, input_name=str(file), ordinary_cut=True),
                     Case(f"webcam_{number}_late", 240, start=158, end=180, input_name=str(file),
                          variable_clock=number % 2 == 0)]
    return fixtures


def _deliver(case, path):
    source = ["-i", "file:" + case.input_name] if case.input_name else [
        "-f", "lavfi", "-i", f"testsrc2=s=480x270:r={case.fps}:d={case.frames / case.fps}"]
    setup = f"[0:v]fps={case.fps},scale=480:270:flags=area,setsar=1,trim=end_frame={case.frames},format=rgb24"
    clock = f"N*{1000 // case.fps}+5000"
    if case.variable_clock:
        clock += "+floor(N/7)*10"
    if case.clock_gap_at is not None:
        clock += f"+200*gte(N,{case.clock_gap_at})".replace(",", "\\,")
    timeline = f"settb=expr=1/1000,setpts={clock}[out]"
    if case.start is not None:
        extra = (f"split=3[a][b][c];[a]trim=end_frame={case.start},setpts=PTS-STARTPTS[pre];"
                 f"[b]trim=start_frame={case.start}:end_frame={case.start + 1},"
                 f"loop=loop={case.end - case.start - 1}:size=1:start=0,setpts=N/{case.fps}/TB[hold];"
                 f"[c]trim=start_frame={case.end},setpts=PTS-STARTPTS[post];"
                 "[pre][hold][post]concat=n=3:v=1:a=0,")
    elif case.ordinary_cut:
        extra = ("split[a][b];[a]trim=end_frame=160,setpts=PTS-STARTPTS[pre];"
                 "[b]trim=start_frame=160,hflip,setpts=PTS-STARTPTS[post];[pre][post]concat=n=2:v=1:a=0,")
    else:
        extra = ""
    codec = {
        "rgb_lossless": ["-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast"],
        "vp9_lossless": ["-c:v", "libvpx-vp9", "-lossless", "1", "-deadline", "realtime", "-cpu-used", "5", "-pix_fmt", "yuv420p"],
        "h264_lossy": ["-c:v", "libx264", "-crf", "23", "-g", "6", "-preset", "fast", "-pix_fmt", "yuv420p"],
    }[case.codec]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *source, "-filter_complex", f"{setup},{extra}{timeline}",
                    "-map", "[out]", "-fps_mode", "passthrough", "-enc_time_base", "1:1000", *codec, str(path)],
                   check=True, timeout=90)


def _frame_hashes(path, table):
    response = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-copyts", "-i",
                               "file:" + str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "format=rgb24", "-fps_mode",
                               "passthrough", "-enc_time_base", table.time_base, "-c:v", "rawvideo", "-f", "framehash",
                               "-hash", "sha256", "pipe:1"], check=True, timeout=90, capture_output=True, text=True)
    lines = response.stdout.splitlines()
    if f"#tb 0: {table.time_base}" not in lines:
        raise ValueError("Independent RGB hash time base differs from the source")
    rows = [line.split(",") for line in lines if line and not line.startswith("#")]
    if ([int(row[2]) for row in rows] != list(table.pts) or table.interpolated
            or any(int(row[4]) != table.width * table.height * 3 for row in rows)):
        raise ValueError("Independent RGB hash clocks or frame size differ from the source")
    return [row[5].strip() for row in rows]


def _oracle(pts, digests, time_base):
    edges = [i for i, (left, right) in enumerate(pairwise(pts), 1)
             if not 0 < (right - left) * time_base <= Fraction(1, 8)]
    eligible = any((pts[right - 1] - pts[left]) * time_base >= Fraction(1, 2)
                   for left, right in pairwise([0, *edges, len(pts)]))
    breaks = set(edges) | {i for i, (left, right) in enumerate(pairwise(digests), 1) if left != right}
    matches = [(pts[right - 1] - pts[left], left, right - 1) for left, right in pairwise([0, *sorted(breaks), len(pts)])
               if (pts[right - 1] - pts[left]) * time_base >= Fraction(1, 2)]
    best = max(matches, default=None, key=lambda row: row[0])
    return (list(best[1:]) if best is not None else None, len(edges), eligible)


def _evaluate(path):
    table = probe_frames(path)
    started = time.perf_counter()
    inspection = picture_hold.inspect(path, table, origin=table.seconds(0))
    elapsed = time.perf_counter() - started
    coverage, match = inspection["coverage"], inspection["match"]
    verified = _frame_hashes(path, table)
    prefix = binding.frame_limit(len(table), table.width, table.height)
    scanned = coverage["frames_scanned"]
    legacy, _, _ = _oracle(table.pts[:prefix], verified[:prefix], table.tb)
    expanded, breaks, eligible = _oracle(table.pts[:scanned], verified[:scanned], table.tb)
    actual = None if match is None else [match["start_frame_index"], match["end_frame_index"]]
    if (actual != expanded or coverage["status"] != ("matched" if expanded else "no_match" if eligible else "unavailable")
            or ("continuity_breaks" in coverage and coverage["continuity_breaks"] != breaks)
            or (match is not None and match["decoded_rgb_sha256"] != verified[actual[0]])):
        raise ValueError("Independent exact RGB frame-hash readback differs from the inspection")
    return {"source_frames": len(table), "old_frames": prefix, "checked_frames": scanned, "scope": coverage.get("scan_scope"),
            "clock_breaks": breaks, "old_interval": legacy, "found_interval": expanded, "seconds": round(elapsed, 3),
            "rgb_work_mib": round(scanned * table.width * table.height * 3 / 1024**2, 1),
            "resolution": [table.width, table.height], "time_base": table.time_base}


def run(output: Path, webcams: list[Path]):
    output.mkdir(mode=0o700)
    cases = _cases(webcams)
    (output / "plan.json").write_text(json.dumps([asdict(case) for case in cases], indent=2) + "\n")
    with (output / "outcomes.jsonl").open("x") as journal:
        for case in cases:
            extension = "webm" if case.codec == "vp9_lossless" else "mkv"
            path = output / f"{case.name}.{extension}"
            _deliver(case, path)
            measured = _evaluate(path)
            if measured["source_frames"] != case.frames:
                raise ValueError("Authored frame count differs from the frozen fixture")
            row = {"case": case.name, "authored": [case.start, case.end - 1] if case.start is not None else None, **measured}
            journal.write(json.dumps(row, sort_keys=True) + "\n")
            journal.flush()
            path.unlink()
        for number, file in enumerate(webcams, 1):
            journal.write(json.dumps({"case": f"webcam_{number}_native_control", "authored": None, **_evaluate(file)}, sort_keys=True) + "\n")
            journal.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--webcam", type=Path, action="append", default=[], help="Optional earlier VCD2 delivery, at least 9.6 s")
    args = parser.parse_args()
    run(args.output, [path.resolve(strict=True) for path in args.webcam])
