"""Native-clock and exact-content qualification of acquired shared video sources.

No detector, generator or network code. Per-record clocks/hashes stay private.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_common_sources_v89 import ROLES, read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "video-native-sources-v92"
ACQUIRED = BASE / "video-common-acquisition-v91"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
MEDIA = ARCHIVE / "video-common-sources-v91"
HISTORY = BASE / "video-specialists-v18/source-frame-audit-v1.jsonl"
PLAN_SHA = "3de3a0bd75fbbebf6da2911a78e341c2ced7c6ef7d03e8fa66c6b8bf088fab99"
BATCH = 28
CODE = (
    "backend/scripts/video_native_sources_v92.py",
    "backend/scripts/video_native_runtime_v92.py",
    "backend/scripts/video_common_sources_v89.py",
    "backend/app/detectors/video/decode.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/tests/test_video_native_sources_v92.py",
    "docs/video-native-source-qualification-v92.md",
)


def identity():
    return {
        "schema": "chipmunk.video-native-source-freeze/92",
        "asset_plan_sha256": sha(ACQUIRED / "asset-plan.json"),
        "historical_rgb_journal_sha256": sha(HISTORY),
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "psutil")},
        "binaries_sha256": {p: sha(Path(shutil.which(p))) for p in ("ffmpeg", "ffprobe")},
        "assets": 336,
        "batch_size": BATCH,
        "batches": 12,
        "new_neural_calls": 0,
        "limits": {
            "seconds": 900,
            "initial_ram_bytes": 3 * 1024**3,
            "tree_rss_bytes": 1536 * 1024**2,
            "running_ram_and_disk_reserve_bytes": 1536 * 1024**2,
            "initial_disk_bytes": 3 * 1024**3,
        },
        "method": "Full native RGB decode with observed PTS, original audio frame/sample clocks, and canonical mono-16k PCM content hashes. Join exact full-video or informative three-frame/three-PCM-block aliases. Exclude components matching historical RGB sequences. No perceptual identity, historical-audio, arbitrary crop/transcode or unknown-source independence guarantee. No inference or serving change.",
    }


def assets():
    plan = read(ACQUIRED / "asset-plan.json")
    require(
        sha(ACQUIRED / "asset-plan.json") == PLAN_SHA
        and plan["planned_assets"] == 336
        and plan["media_root"] == str(MEDIA),
        "Exact acquired source plan required",
    )
    rows = plan["assets"]
    require(
        len(rows) == 336 and [r["index"] for r in rows] == list(range(336)),
        "Complete ordered assets required",
    )
    for r in rows:
        require(
            r["file"]
            == f"asset-{r['index']:03d}"
            + (".flv" if r["source_family"] == "recorded_audiovisual_speech" else ".mp4"),
            "Bound plain source filename required",
        )
    return rows


def checked():
    require(read(TRIAL / "freeze.json") == identity(), "Frozen source audit inputs/runtime changed")


def triplets(values, informative=None):
    """Three distinct informative consecutive values; static/solid frames do not link sources."""
    flags = [True] * len(values) if informative is None else informative
    require(len(flags) == len(values), "Corresponding informativeness flags required")
    return {
        hashlib.sha256(("\0".join(values[i : i + 3])).encode()).hexdigest()
        for i in range(len(values) - 2)
        if all(flags[i : i + 3]) and len(set(values[i : i + 3])) == 3
    }


def audio_clocks(stream, frames):
    rate, tb = int(stream["sample_rate"]), Fraction(stream["time_base"])
    require(rate > 0 and tb > 0, "Positive native audio clock required")
    intervals, missing, gaps, overlap = [], 0, 0, 0
    tolerance = tb + Fraction(1, rate)
    for f in frames:
        raw = f.get("pts")
        if raw in (None, "N/A"):
            raw = f.get("best_effort_timestamp")
        samples = int(f.get("nb_samples", 0))
        if raw in (None, "N/A") or samples <= 0:
            missing += 1
            continue
        start = int(raw) * tb
        end = start + Fraction(samples, rate)
        if intervals:
            delta = start - Fraction(intervals[-1]["end"])
            gaps += delta > tolerance
            overlap += delta < -tolerance
        intervals.append({"pts": int(raw), "samples": samples, "start": str(start), "end": str(end)})
    monotonic = all(Fraction(a["start"]) < Fraction(b["start"]) for a, b in zip(intervals, intervals[1:]))
    return {
        "time_base": str(tb),
        "sample_rate": rate,
        "channels": int(stream["channels"]),
        "frames": intervals,
        "missing_frame_clocks": missing,
        "gap_count": gaps,
        "overlap_count": overlap,
        "clock_qualified": bool(intervals) and not missing and monotonic and not overlap,
        "contiguous": bool(intervals) and not missing and monotonic and not gaps and not overlap,
    }


def probe_json(path, arguments):
    result = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            *arguments,
            "-of",
            "json",
            "file:" + str(path.resolve()),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=True,
    )
    require(not result.stderr.strip(), "Source probe reported decode errors")
    return json.loads(result.stdout)


def pcm_content(path):
    """Canonical content only: native audio frame clocks, not this resampling, define timing."""
    argv = [
        shutil.which("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file",
        "-i",
        "file:" + str(path.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(120, proc.kill)
    timer.start()
    digest, blocks, flags, samples = hashlib.sha256(), [], [], 0
    try:
        while block := proc.stdout.read(32000):
            require(len(block) % 2 == 0, "Complete canonical PCM samples required")
            digest.update(block)
            x = np.frombuffer(block, "<i2").astype(np.int32)
            samples += len(x)
            if len(block) == 32000:
                blocks.append(hashlib.sha256(block).hexdigest())
                flags.append(bool(np.ptp(x) >= 64 and np.std(x) >= 32 and len(np.unique(x)) >= 64))
        stderr = proc.stderr.read()
        require(
            proc.wait(timeout=10) == 0 and not stderr.strip(),
            "Complete error-free canonical audio decode required",
        )
        return {
            "samples": samples,
            "pcm_sha256": digest.hexdigest(),
            "block_samples": 16000,
            "informative_block_count": sum(flags),
            "three_block_aliases": sorted(triplets(blocks, flags)),
            "scope": "Canonical mono-16k signed-16 content fingerprint only; not native sample timing or synthesis evidence.",
        }
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()


def decode_asset(row):
    path = MEDIA / row["file"]
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Source bytes changed")
    table = probe_frames(path, timeout_s=120)
    require(table.width * table.height <= 16_777_216, "Source dimensions exceed decoder budget")
    rgb, flags = [], []
    for frame in decode_frames(path, table, None, timeout_s=120):
        require(
            frame.index == len(rgb) and frame.pts == table.pts[frame.index],
            "Incomplete/reordered native frames",
        )
        rgb.append(hashlib.sha256(frame.rgb.tobytes()).hexdigest())
        flags.append(bool((frame.rgb != frame.rgb[0, 0]).any()))
    require(len(rgb) == len(table), "Incomplete native video decode")
    video_ok = (
        not table.truncated
        and not table.interpolated
        and all(a < b for a, b in zip(table.pts, table.pts[1:]))
    )
    stream_info = probe_json(
        path,
        [
            "-show_entries",
            "stream=index,codec_type,codec_name,sample_rate,channels,time_base,start_pts,duration_ts",
        ],
    )
    streams = [s for s in stream_info.get("streams", []) if s["codec_type"] == "audio"]
    clocks, pcm = None, None
    if streams:
        native = probe_json(
            path,
            [
                "-select_streams",
                "a:0",
                "-show_frames",
                "-show_entries",
                "frame=pts,best_effort_timestamp,nb_samples",
            ],
        )
        clocks = audio_clocks(streams[0], native.get("frames", []))
        pcm = pcm_content(path)
    return {
        "index": row["index"],
        "group": row["group"],
        "provisional_role": row["provisional_role"],
        "source_family": row["source_family"],
        "kind": row["kind"],
        "status": "decoded",
        "input_sha256": row["sha256"],
        "frame_table": table.to_json(),
        "rgb_sha256": rgb,
        "full_rgb_sequence_sha256": hashlib.sha256(("\0".join(rgb)).encode()).hexdigest(),
        "informative_frames": sum(flags),
        "three_frame_aliases": sorted(triplets(rgb, flags)),
        "video_clock_qualified": video_ok,
        "audio_stream_count": len(streams),
        "has_audio": bool(streams),
        "audio_clocks": clocks,
        "canonical_audio": pcm,
    }


def qualify(index):
    require(type(index) is int and 0 <= index < 12, "One fixed source-audit batch required")
    checked()
    selected = assets()[index * BATCH : (index + 1) * BATCH]
    journal = TRIAL / f"batch-{index:02d}.jsonl"
    counts = Counter()
    with journal.open("x") as output:
        for row in selected:
            try:
                record = decode_asset(row)
            except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
                record = {k: row[k] for k in ("index", "group", "provisional_role", "source_family", "kind")}
                record.update(status="failed", error_type=type(exc).__name__, video_clock_qualified=False)
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            counts[record["status"]] += 1
            print(
                json.dumps(
                    {
                        "batch": index,
                        "completed": sum(counts.values()),
                        "decoded": counts["decoded"],
                        "failed": counts["failed"],
                    }
                ),
                flush=True,
            )
    checked()
    write(
        TRIAL / f"batch-{index:02d}-complete.json",
        {
            "batch": index,
            "assets": len(selected),
            "counts": dict(counts),
            "journal_sha256": sha(journal),
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "new_neural_calls": 0,
        },
    )


def refine(records, historical_sequences):
    parent = {r["group"]: r["group"] for r in records}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen, links = {}, Counter()
    historical = set().union(*(triplets(seq) for seq in historical_sequences))
    historical_groups = set()
    for r in records:
        if r["status"] != "decoded":
            continue
        group = r["group"]
        keys = [("video_frames", k) for k in r["three_frame_aliases"]]
        if r["three_frame_aliases"]:
            keys.append(("video_full", r["full_rgb_sequence_sha256"]))
        pcm = r["canonical_audio"]
        if pcm and pcm["informative_block_count"]:
            keys += [("audio_blocks", k) for k in pcm["three_block_aliases"]]
            keys.append(("audio_full", pcm["pcm_sha256"]))
        if set(r["three_frame_aliases"]) & historical:
            historical_groups.add(group)
        for key in keys:
            if key in seen and root(group) != root(seen[key]):
                parent[root(group)] = root(seen[key])
                links[key[0]] += 1
            else:
                seen.setdefault(key, group)
    members, roles = defaultdict(set), defaultdict(set)
    for r in records:
        members[root(r["group"])].add(r["group"])
        roles[root(r["group"])].add(r["provisional_role"])
    crossing = sum(len(v) > 1 for v in roles.values())
    groups_with_failures = {
        r["group"] for r in records if r["status"] != "decoded" or not r["video_clock_qualified"]
    }
    assignments = {}
    ordered = sorted(
        members,
        key=lambda k: hashlib.sha256(
            ("chipmunk-native-video/92\0" + "\0".join(sorted(members[k]))).encode()
        ).digest(),
    )
    for i, key in enumerate(ordered):
        excluded = bool(members[key] & (historical_groups | groups_with_failures))
        role = ROLES[i % len(ROLES)] if crossing else next(iter(roles[key]))
        component = hashlib.sha256(("\0".join(sorted(members[key]))).encode()).hexdigest()
        for group in members[key]:
            assignments[group] = {
                "role": role,
                "component": component,
                "eligible": not excluded,
                "historical_rgb_connection": bool(members[key] & historical_groups),
                "failed_or_unqualified_video_clock": bool(members[key] & groups_with_failures),
            }
    return assignments, {
        "components": len(members),
        "additional_exact_connections": dict(links),
        "components_crossing_provisional_roles": crossing,
        "all_roles_reassigned": bool(crossing),
        "historical_rgb_connected_groups": len(historical_groups),
        "eligible_source_groups": sum(a["eligible"] for a in assignments.values()),
        "excluded_source_groups": sum(not a["eligible"] for a in assignments.values()),
    }


def finalise():
    checked()
    rows = assets()
    records = []
    for i in range(12):
        complete = read(TRIAL / f"batch-{i:02d}-complete.json")
        profile = read(TRIAL / f"batch-{i:02d}-profile/profile.json")
        require(
            profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Successful source-audit runtime required",
        )
        path = TRIAL / f"batch-{i:02d}.jsonl"
        require(
            complete["journal_sha256"] == sha(path) and complete["assets"] == BATCH,
            "Complete unchanged source journal required",
        )
        records += [json.loads(line) for line in path.read_text().splitlines()]
    require([r["index"] for r in records] == list(range(336)), "Every planned source must remain recorded")
    history = [json.loads(line).get("rgb_sha256", []) for line in HISTORY.read_text().splitlines()]
    assignments, graph = refine(records, history)
    output = {
        "schema": "chipmunk.video-native-qualified-source-plan/92",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "assets": [{**r, "qualification": assignments[r["group"]]} for r in records],
        "graph": graph,
        "detector_study_ready": False,
        "new_neural_calls": 0,
    }
    write(TRIAL / "qualified-plan.json", output)
    decoded = [r for r in records if r["status"] == "decoded"]
    summary = {
        "schema": "chipmunk.video-native-source-qualification/92",
        "planned_assets": len(rows),
        "decoded_assets": len(decoded),
        "failed_assets": len(records) - len(decoded),
        "native_video_frames": sum(len(r["rgb_sha256"]) for r in decoded),
        "qualified_video_clocks": sum(r["video_clock_qualified"] for r in decoded),
        "with_audio": sum(r["has_audio"] for r in decoded),
        "without_audio": sum(not r["has_audio"] for r in decoded),
        "qualified_audio_clocks": sum(
            bool(r["audio_clocks"] and r["audio_clocks"]["clock_qualified"]) for r in decoded
        ),
        "contiguous_audio_clocks": sum(
            bool(r["audio_clocks"] and r["audio_clocks"]["contiguous"]) for r in decoded
        ),
        **graph,
        "roles_eligible_groups": dict(Counter(a["role"] for a in assignments.values() if a["eligible"])),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "qualified_private_plan_sha256": sha(TRIAL / "qualified-plan.json"),
        "new_neural_calls": 0,
        "detector_study_ready": False,
        "defaults_changed": False,
        "scope": "Native source clocks and exact decoded-content links only. Silent clips do not test lip sync; audio presence does not prove speech. Near duplicates, historical audio overlap, unknown people/pretraining links and portrait attribution remain unverified. Edit-boundary preparation and detector comparison are separate pending work. No model accuracy or outside human review claim.",
    }
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "qualify", "finalise"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.index is None, "Preparation takes no batch index")
        assets()
        write(TRIAL / "freeze.json", identity())
    elif args.command == "qualify":
        qualify(args.index)
    else:
        require(args.index is None, "Finalization takes no batch index")
        finalise()
