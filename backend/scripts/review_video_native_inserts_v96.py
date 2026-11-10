"""Second command-interface review of native inserts; no app decoder or models.

Uses the same ffmpeg codec library, not an independent decoder engine or human.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

import psutil

from scripts.audio_tree_sampler_v64 import sample_tree

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/video-native-inserts-v95"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_framehash(text, expected, step, width, height):
    bases = []
    records = []
    for line in text.splitlines():
        if line.startswith("#tb "):
            stream, value = line.removeprefix("#tb ").split(":", 1)
            require(stream.strip() == "0", "Only the selected video stream may be hashed")
            bases.append(Fraction(value.strip()))
        elif line and not line.startswith("#"):
            values = [p.strip() for p in line.split(",")]
            require(
                len(values) == 6 and values[0] == "0" and re.fullmatch(r"[0-9a-f]{64}", values[-1]),
                "Complete SHA256 raw-frame record required",
            )
            records.append({"pts": int(values[2]), "bytes": int(values[4]), "sha256": values[5]})
    require(
        len(bases) == 1 and bases[0] > 0 and len(records) == len(expected),
        "One complete rational framehash clock required",
    )
    for i, row in enumerate(records):
        require(row["pts"] * bases[0] == i * Fraction(step), "Command-decoded native picture time differs")
        require(
            row["bytes"] == width * height * 3 and row["sha256"] == expected[i],
            "Command-decoded RGB bytes differ",
        )
    return [r["sha256"] for r in records]


class Budget:
    def __init__(self, folder):
        self.folder = folder
        self.started = time.monotonic()
        self.rows = []
        self.errors = []
        self.exits = []
        self.stream = (folder / "resources.jsonl").open("x")

    def observe(self, child_rss=0, initial=False):
        row = {
            "elapsed_s": time.monotonic() - self.started,
            "combined_rss_bytes": psutil.Process().memory_info().rss + child_rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "project_free_disk_bytes": shutil.disk_usage(TRIAL).free,
            "archive_free_disk_bytes": shutil.disk_usage(ARCHIVE).free,
        }
        self.rows.append(row)
        self.stream.write(json.dumps(row) + "\n")
        self.stream.flush()
        require(
            Path("/Volumes/SEAGATE").is_mount()
            and row["elapsed_s"] <= 900
            and row["combined_rss_bytes"] <= 1536 * 1024**2
            and row["available_ram_bytes"] >= (3 * 1024**3 if initial else 1536 * 1024**2)
            and min(row["project_free_disk_bytes"], row["archive_free_disk_bytes"])
            >= (3 * 1024**3 if initial else 1536 * 1024**2),
            "Source-review resource limit exceeded",
        )

    def command(self, argv):
        self.observe()
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stop = threading.Event()
        problem = []

        def watch():
            try:
                try:
                    process = psutil.Process(proc.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    self.exits.append(
                        {
                            "classification": "verified_command_terminal_transition",
                            "exit_code": proc.wait(timeout=0.25),
                        }
                    )
                    return
                while not stop.is_set() and proc.poll() is None:
                    try:
                        rss, events, terminal = sample_tree(process, proc)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        self.exits.append(
                            {
                                "classification": "verified_command_terminal_transition",
                                "exit_code": proc.wait(timeout=0.25),
                            }
                        )
                        return
                    self.exits.extend(events)
                    self.observe(rss)
                    if terminal:
                        return
                    stop.wait(0.25)
            except BaseException as exc:
                problem.append(type(exc).__name__)
                if proc.poll() is None:
                    proc.kill()

        observer = threading.Thread(target=watch, daemon=True)
        observer.start()
        try:
            output, error = proc.communicate(timeout=180)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            stop.set()
            observer.join(timeout=5)
        self.errors.extend(problem)
        require(
            not observer.is_alive() and not problem and proc.returncode == 0 and not error.strip(),
            "Bounded source-review command failed",
        )
        self.observe()
        return output


def review(index):
    require(type(index) is int and 0 <= index < 90, "One explicit source group required")
    folder = TRIAL / f"group-{index:03d}-command-review-v96"
    folder.mkdir(exist_ok=False)
    budget = Budget(folder)
    status = "failed"
    error_type = None
    try:
        budget.observe(initial=True)
        lock = json.loads((TRIAL / "freeze.json").read_text())
        plan = json.loads((TRIAL / "plan.json").read_text())
        require(
            lock["private_plan_sha256"] == sha(TRIAL / "plan.json") and lock["identity"] == plan["identity"],
            "Locked constructor plan required",
        )
        require(
            all(sha(ROOT / p) == v for p, v in lock["identity"]["code_sha256"].items()),
            "Constructor implementation changed",
        )
        require(
            all(sha(Path(shutil.which(p))) == v for p, v in lock["identity"]["binary_sha256"].items()),
            "Recorded codec binaries changed",
        )
        source = plan["groups"][index]
        receipt_path = TRIAL / f"group-{index:03d}-complete.json"
        receipt = json.loads(receipt_path.read_text())
        require(
            receipt["freeze_sha256"] == sha(TRIAL / "freeze.json")
            and receipt["role"] == source["role"]
            and receipt["group"] == source["group"]
            and len(receipt["cases"]) == 6,
            "Complete exact constructed group required",
        )
        require(
            [c["condition"] for c in receipt["cases"]] == lock["identity"]["conditions"],
            "All distinct fixed construction conditions required",
        )
        profile_path = TRIAL / f"group-{index:03d}-profile/profile.json"
        profile = json.loads(profile_path.read_text())
        require(
            profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Successful source construction required",
        )
        media = Path(plan["media_root"]) / f"group-{index:03d}"
        frames = pngs = 0
        actual_by_condition = {}
        with (folder / "command-frames.jsonl").open("x") as journal:
            for case in receipt["cases"]:
                path = media / case["file"]
                require(
                    path.parent == media
                    and path.stat().st_size == case["bytes"]
                    and sha(path) == case["sha256"],
                    "Constructed bytes changed",
                )
                c = case["construction"]
                require(c == source["construction"], "Case construction differs from locked source plan")
                step = Fraction(c["frame_step_seconds"])
                w = case["frame_table"]["width"]
                h = case["frame_table"]["height"]
                saved = case["frame_table"]
                require(
                    (w, h)
                    == (
                        source["sources"][0]["frame_table"]["width"],
                        source["sources"][0]["frame_table"]["height"],
                    )
                    and not saved["truncated"]
                    and not saved["interpolated"]
                    and saved["rotation"] == 0
                    and len(saved["pts"]) == c["frames"]
                    and all(p * Fraction(saved["time_base"]) == i * step for i, p in enumerate(saved["pts"])),
                    "Stored native table differs from constructed source geometry/clock",
                )
                output = budget.command(
                    [
                        shutil.which("ffmpeg"),
                        "-nostdin",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file",
                        "-copyts",
                        "-i",
                        "file:" + str(path.resolve()),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-pix_fmt",
                        "rgb24",
                        "-fps_mode",
                        "passthrough",
                        "-f",
                        "framehash",
                        "-hash",
                        "sha256",
                        "pipe:1",
                    ]
                ).decode()
                actual = parse_framehash(output, case["expected_rgb_sha256"], step, w, h)
                stream = json.loads(
                    budget.command(
                        [
                            shutil.which("ffprobe"),
                            "-v",
                            "error",
                            "-protocol_whitelist",
                            "file",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height,time_base,duration_ts",
                            "-of",
                            "json",
                            "file:" + str(path.resolve()),
                        ]
                    )
                )["streams"]
                require(
                    len(stream) == 1
                    and (stream[0]["width"], stream[0]["height"]) == (w, h)
                    and int(stream[0]["duration_ts"]) * Fraction(stream[0]["time_base"])
                    == len(actual) * step,
                    "Container display duration differs",
                )
                actual_by_condition[case["condition"]] = actual
                frames += len(actual)
                import cv2

                for exhibit in case["representatives"].values():
                    picture = media / exhibit["file"]
                    i = exhibit["frame_index"]
                    require(
                        picture.parent == media and sha(picture) == exhibit["sha256"],
                        "Representative file changed",
                    )
                    bgr = cv2.imread(str(picture))
                    require(bgr is not None, "Representative image unavailable")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    require(
                        rgb.shape == (h, w, 3)
                        and hashlib.sha256(rgb.tobytes()).hexdigest() == actual[i]
                        and exhibit["pts"] * Fraction(exhibit["time_base"]) == i * step,
                        "Representative pixels/timestamp refer to another frame",
                    )
                    pngs += 1
                    budget.observe()
                journal.write(
                    json.dumps({"condition": case["condition"], "command_framehash": output}) + "\n"
                )
                journal.flush()
        original = actual_by_condition["original"]
        n = len(original)
        a, b = source["construction"]["edit_frames"]
        require(
            original == source["sources"][0]["rgb_sha256"][:n],
            "Original output differs from qualified source",
        )
        for condition, slot in [("swap_a", 1), ("swap_b", 2)]:
            expected = original[:a] + source["sources"][slot]["rgb_sha256"][a:b] + original[b:]
            require(actual_by_condition[condition] == expected, "Swap source correspondence differs")
        expected = [original[i + 2] if a <= i < b else original[i] for i in range(n)]
        require(
            actual_by_condition["ordinary_time_step"] == expected,
            "Ordinary time step/source restoration differs",
        )
        for case in receipt["cases"]:
            actual = actual_by_condition[case["condition"]]
            changed = [
                i for i, (first, second) in enumerate(zip(original, actual, strict=True)) if first != second
            ]
            require(
                changed == case["actual_changed_frames"] and all(a <= i < b for i in changed),
                "Changed-frame support differs",
            )
        result = {
            "schema": "chipmunk.native-insert-command-review/96",
            "status": "complete",
            "cases": 6,
            "native_rgb_frames_verified": frames,
            "native_pngs_verified": pngs,
            "constructor_freeze_sha256": sha(TRIAL / "freeze.json"),
            "private_group_receipt_sha256": sha(receipt_path),
            "constructor_profile_sha256": sha(profile_path),
            "command_journal_sha256": sha(folder / "command-frames.jsonl"),
            "reviewer_sha256": sha(Path(__file__)),
            "sample_helper_sha256": sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
            "new_neural_calls": 0,
            "scope": "Second FFmpeg command-interface read of all output pixels/PTS/duration, all native PNG bindings and source/changed-frame correspondence. Same underlying codec library, not an independent decoder engine, outside human review, AI-pixel proof or detector accuracy. Source fingerprints and media remain private.",
        }
        with (folder / "result.json").open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        status = "complete"
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        budget.stream.close()
        with (folder / "profile.json").open("x") as f:
            json.dump(
                {
                    "status": status,
                    "error_type": error_type,
                    "duration_s": time.monotonic() - budget.started,
                    "maximum_observed_combined_rss_bytes": max(
                        (r["combined_rss_bytes"] for r in budget.rows), default=0
                    ),
                    "minimum_observed_available_ram_bytes": min(
                        (r["available_ram_bytes"] for r in budget.rows), default=None
                    ),
                    "sample_errors": len(budget.errors),
                    "terminal_observations": budget.exits,
                    "new_neural_calls": 0,
                    "resource_journal_sha256": sha(folder / "resources.jsonl"),
                    "scope": "Parent plus active command descendants sampled while commands run; before/after checks between commands.900s/1.5GiB RSS,3GiB initial RAM/disk,1.5GiB running reserves. Command timeout180s. Samples are not true peaks. No retry or overwrite.",
                },
                f,
                indent=2,
            )
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    review(parser.parse_args().index)
