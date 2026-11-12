"""Freeze and check paired-video soundtrack timing on retained licensed recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.video.decode import probe_frames
from app.evidence import audio_reference, video_reference
from app.evidence import video_reference_soundtrack as method
from scripts.video_lipsync_local_delay_transfer import resample_clock, shift

POSITIVE = ("local_late", "local_early")
CONTROLS = ("original", "gain", "noise_mix", "unrelated_insert", "global_shift", "short_offset", "picture_change")
DELIVERIES = {"pcm": ("mkv", ("-c:a", "pcm_s16le")), "aac": ("mp4", ("-c:a", "aac", "-b:a", "48k"))}
GATE = {"min_total": 16, "min_each_sign": 7, "min_groups": 8, "max_added_control_alerts": 1,
        "max_added_global_or_picture_alerts": 0, "min_interval_iou": "3/5", "max_sample_error": 16}


def digest(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(*args):
    return subprocess.run(args, timeout=20, capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout


def freeze(run, prior):
    run.mkdir(mode=0o700, exist_ok=False)
    receipt = read(prior / "intake-receipt.json")
    roster = read(prior / "intake-freeze.json")
    require(receipt["roster_sha256"] == digest(prior / "intake-freeze.json"), "Prior publisher roster hash differs")
    rows = [{**row, "publisher_path": entry["source_path"]} for row, entry in zip(receipt["rows"], roster["sources"], strict=True)]
    require(len(rows) == 15 and len({row["actor"] for row in rows}) == 15, "Separate recorded actors required")
    for row in rows:
        row["donor"] = (row["ordinal"] + 3) % 15 + 1
    save(run / "freeze.json", {"schema": "chipmunk.video-paired-soundtrack-freeze/1", "rows": rows,
         "method_sha256": digest(Path(method.__file__)), "script_sha256": digest(Path(__file__)),
         "source_receipt_sha256": digest(prior / "intake-receipt.json"), "positive": POSITIVE, "controls": CONTROLS,
         "deliveries": DELIVERIES, "gate": GATE, "ffmpeg": command("ffmpeg", "-version").decode().splitlines()[0]})


def pcm_from_source(prior, row):
    path = prior / row["file"]
    require(digest(path) == row["sha256"], "Stored recording differs from the qualified publisher file")
    audio = decode_audio(path, source_sha256=row["sha256"], max_seconds=8)
    _, pcm, _ = resample_clock(audio)
    return pcm


def _mix(pcm, donor, bounds):
    first, end = bounds
    weight = np.ones(end - first)
    fades = np.arange(1, 81) / 80
    weight[:80], weight[-80:] = fades, fades[::-1]
    result = pcm.copy()
    result[first:end] = np.clip(np.rint((1 - weight) * pcm[first:end] + weight * donor[first:end]), -32768, 32767).astype("<i2")
    return result


def variants(row, pcm, donor):
    width = 10240
    middle = len(pcm) // 2 // method.STEP * method.STEP
    bounds = middle - width // 2, middle + width // 2
    offset = (2560, 3200, 3840, 4480)[row["ordinal"] % 4]
    a, b = bounds
    yield "local_late", _mix(pcm, shift(pcm, offset), bounds), bounds, offset
    yield "local_early", _mix(pcm, shift(pcm, -offset), bounds), bounds, -offset
    yield "original", pcm, None, None
    yield "gain", _mix(pcm, pcm * .65, bounds), None, None
    rng = np.random.default_rng(row["ordinal"])
    noise = rng.standard_normal(len(pcm)) * np.sqrt(np.mean(pcm[a:b].astype(np.float64) ** 2)) * .28
    yield "noise_mix", _mix(pcm, pcm + noise, bounds), None, None
    unrelated = np.resize(donor, len(pcm))
    yield "unrelated_insert", _mix(pcm, unrelated, bounds), None, None
    yield "global_shift", shift(pcm, offset), None, None
    yield "short_offset", _mix(pcm, shift(pcm, 640), bounds), None, None
    yield "picture_change", _mix(pcm, shift(pcm, offset), bounds), None, None


def picture_hash(path):
    hashes = command("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(path),
                     "-an", "-map", "0:v:0", "-vf", "format=rgb24", "-fps_mode", "passthrough",
                     "-threads", "1", "-f", "framehash", "-hash", "sha256", "-").decode().splitlines()
    return canonical_hash([line.rsplit(",", 1)[1].strip() for line in hashes if not line.startswith("#")])


def native(path):
    table = probe_frames(path, cap=video_reference.MAX_FRAMES, timeout_s=8, max_pixels=video_reference.MAX_PIXELS)
    track = audio_reference.read_track(path, {"sha256": digest(path)}, max_seconds=12, video=True, origin=table.pts[0] * table.tb)
    require(track[0] == "measured" and not table.interpolated and not table.truncated, "Native picture or audio clock failed")
    return table, track, picture_hash(path)


def encode_base(source, path, *, flip=False):
    command("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(source), "-map", "0:v:0", "-an",
            *(("-vf", "hflip") if flip else ()), "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast", "-crf", "24", "-pix_fmt",
            "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "1:1000", str(path))


def remux(picture, sound, output, audio_options):
    command("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(picture), "-i", str(sound), "-map", "0:v:0",
            "-map", "1:a:0", "-c:v", "copy", *audio_options, "-threads", "1", str(output))


def score(run, prior, row, all_rows):
    number = row["ordinal"]
    folder = run / f"source-{number:02d}"
    folder.mkdir(mode=0o700, exist_ok=False)
    pcm = pcm_from_source(prior, row)
    donor = pcm_from_source(prior, all_rows[row["donor"] - 1])
    source = prior / row["file"]
    original, flipped = folder / "picture.mkv", folder / "flipped.mkv"
    encode_base(source, original)
    encode_base(source, flipped, flip=True)
    wav = folder / "input.wav"
    wav.write_bytes(wav_bytes(pcm.astype(np.float32) / 32768))
    cases = []
    for codec, (suffix, audio_options) in DELIVERIES.items():
        earlier = folder / f"earlier-{codec}.{suffix}"
        remux(original, wav, earlier, audio_options)
        table, track, picture = native(earlier)
        require(len(track[1]) >= method.MIN_SAMPLES, "Recording too short for the fixed local-slip comparison")
        for role, wave, authored, offset in variants(row, pcm, donor):
            wav.write_bytes(wav_bytes(wave.astype(np.float32) / 32768))
            path = folder / f"{role}-{codec}.{suffix}"
            remux(flipped if role == "picture_change" else original, wav, path, audio_options)
            after, later, pixels = native(path)
            require(video_reference._comparison_issue((table, after)) is None and
                    (pixels == picture) == (role != "picture_change"), "Unexpected picture relationship")
            require((track[2]["relative_start"], track[2]["samples"]) == (later[2]["relative_start"], later[2]["samples"]),
                    "Prepared audio clocks differ")
            if codec == "pcm":
                require(np.array_equal(later[1], wave.astype(np.float32) / 32768), "PCM changed in delivery")
            assets = {}
            value = method.assess(path, after, {"changed_frames": int(pixels != picture), "frame_pairs_sha256": picture},
                                  {"status": "measured", "earlier": track[2], "submitted": later[2]},
                                  (track, later), assets)
            evidence = value.get("range")
            found = [evidence["submitted"]["samples"], evidence["shift_samples"]] if evidence else None
            cases.append({"delivery": codec, "role": role, "authored": authored, "offset": offset, "found": found,
                          "state": value["status"], "windows": value.get("matched_windows", 0), "pictures": len(after),
                          "earlier_sha256": digest(earlier), "submitted_sha256": digest(path),
                          "relative_picture_pts_sha256": canonical_hash([str(t) for t in video_reference._elapsed(after)]),
                          "native_picture_rgb_sha256": pixels, "pcm_sha256": later[2]["pcm_sha256"],
                          "assets": {name: hashlib.sha256(content).hexdigest() for name, content in assets.items()}})
        wav.write_bytes(wav_bytes(pcm.astype(np.float32) / 32768))
    wav.unlink()
    save(folder / "results.json", {"ordinal": number, "cases": cases})
    print(f"Verified {number:02d} with {len(cases)} native deliveries", flush=True)


def aggregate(rows):
    summary = {}
    for delivery in DELIVERIES:
        useful, false_flags, assessed = Counter(), Counter(), Counter()
        groups = set()
        for row in rows:
            for case in row["cases"]:
                if case["delivery"] != delivery:
                    continue
                role, observed, expected = case["role"], case["found"], case["authored"]
                assessed[role] += int(case["state"] != "unassessed")
                if observed and expected:
                    (a, b), offset = observed
                    start, end = expected
                    intersection = max(0, min(end, b) - max(start, a))
                    union = max(end, b) - min(start, a)
                    if (Fraction(intersection, union) >= Fraction(GATE["min_interval_iou"]) and
                            abs(offset - case["offset"]) <= GATE["max_sample_error"]):
                        useful[role] += 1
                        groups.add(row["ordinal"])
                elif observed:
                    false_flags[role] += 1
        summary[delivery] = {"useful": dict(useful), "false_flags": dict(false_flags), "assessed": dict(assessed),
                             "groups": len(groups), "admitted": sum(useful.values()) >= GATE["min_total"] and
                             min(useful[role] for role in POSITIVE) >= GATE["min_each_sign"] and len(groups) >= GATE["min_groups"] and
                             sum(false_flags.values()) <= GATE["max_added_control_alerts"] and
                             false_flags["global_shift"] + false_flags["picture_change"] <= GATE["max_added_global_or_picture_alerts"]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    run, prior = args.run, args.prior
    if args.phase == "freeze":
        freeze(run, prior)
        return
    plan = read(run / "freeze.json")
    require(plan["method_sha256"] == digest(Path(method.__file__)) and plan["script_sha256"] == digest(Path(__file__)) and
            plan["source_receipt_sha256"] == digest(prior / "intake-receipt.json"), "Frozen input or method differs")
    held = [row for row in plan["rows"] if row["split"] == "held"]
    if args.phase == "report":
        rows = [read(run / f"source-{row['ordinal']:02d}" / "results.json") for row in held]
        result = {"schema": "chipmunk.video-paired-soundtrack-panel/1", "freeze_sha256": digest(run / "freeze.json"),
                  "gate": GATE, "results": aggregate(rows), "rows": rows}
        save(run / "receipt.json", result)
        print(json.dumps(result["results"], sort_keys=True))
        return
    selected = held if args.phase == "held" else [row for row in plan["rows"] if row["split"] == "development"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda row: score(run, prior, row, plan["rows"]), selected))


if __name__ == "__main__":
    main()
