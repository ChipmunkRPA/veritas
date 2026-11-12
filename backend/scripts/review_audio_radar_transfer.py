# SPDX-License-Identifier: Apache-2.0
"""Independently reopen the RADAR publisher bytes and verify the scored panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
from array import array
from collections import Counter
from decimal import Decimal
from pathlib import Path

RATE = 16_000
SPAN = 48_000
PER_CELL = 16
PREFIX_BYTES = 32 * 1024**2
SOURCES = {
    "archive-prefix.gz": "d50b195fdccb37443c7a5e57169941c1646333afda57f97e72bca18c019c837c",
    "labels.txt": "dc51de1f7a6fcfb4e26ffd7fc458b5e0e290e6541bf6d2f0710250f67d29d3ba",
    "languages.txt": "ff11556268e4b404ac1fba048c7ef7cf9a0baaeb1a3889c5572ff6390d0e6735",
    "LICENSE.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    "traffic.ogg": "9702be26fec457ac8c54df61fbe62f120189119b5734377cda9e5ae442566efd",
    "cafe.ogg": "6360791fdcee1f64638f0fc9abd6d2c7a1aa75fd3d2b80033170a02d025099dc",
}
CELLS = [(length, label) for length in ("short", "long") for label in ("bonafide", "spoof")]
BOUNDS = {"short": (Decimal("2.5"), Decimal(4)), "long": (Decimal(6), Decimal(14))}
SCENES = {"traffic": [2, 6, 10, 14], "cafe": [22, 96, 174, 262]}
DIRECTIONS = ("ai", "original", "inconclusive", "unassessed")
MEMBER = re.compile(rb"RADAR2026-eval/flac/(RADAR2026-EVAL[0-9]{6})\.flac")
FRAME = re.compile(rb"n:\s*(\d+)\s+pts:\s*(-?\d+)\b[^\n]*?nb_samples:\s*(\d+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commitment(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    return json.loads(path.read_bytes())


def table(path):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            identity, *values = line.split()
            require(identity not in rows, "Duplicate publisher table identity")
            rows[identity] = values
    return rows


def tar_prefix(path, *, member=MEMBER):
    """Read complete regular FLAC members from a deliberately truncated gzip stream."""
    with gzip.open(path, "rb") as stream:
        try:
            while header := stream.read(512):
                if len(header) < 512 or header == bytes(512):
                    return
                require(sum(header[:148] + b"        " + header[156:]) == int(header[148:156].strip(b" \0"), 8),
                        "Publisher TAR header checksum differs")
                name = header[:100].split(b"\0", 1)[0]
                length = int(header[124:136].strip(b" \0"), 8)
                require(0 <= length < 2 * 1024**2, "Publisher TAR member exceeds its frozen bounds")
                payload = stream.read((length + 511) // 512 * 512)
                if len(payload) < length:
                    return
                match = member.fullmatch(name)
                if match is not None and header[156:157] in {b"0", b"\0"}:
                    require(payload.startswith(b"fLaC"), "Publisher audio member is not FLAC")
                    yield match[1].decode(), payload[:length]
        except EOFError:
            return


def source_roster(source, panel):
    require((source / "archive-prefix.gz").stat().st_size == PREFIX_BYTES, "Compressed prefix size differs")
    for filename, pin in SOURCES.items():
        require(file_sha(source / filename) == pin, f"Source hash differs for {filename}")
    labels, languages = table(source / "labels.txt"), table(source / "languages.txt")
    require(len(labels) == len(languages) == 102_725 and labels.keys() == languages.keys(), "Publisher label tables differ")
    eligible = {cell: [] for cell in CELLS}
    complete = set()
    for identity, payload in tar_prefix(source / "archive-prefix.gz"):
        require(identity in labels and identity not in complete, "Missing or repeated FLAC identity")
        complete.add(identity)
        duration, label, *segments = labels[identity]
        if languages[identity] != ["EN"] or segments != [f"0.0000-{duration}-{label}"]:
            continue
        for stratum, (lower, upper) in BOUNDS.items():
            if lower <= Decimal(duration) <= upper:
                require((stratum, label) in eligible, "Unrecognized publisher label")
                eligible[stratum, label].append({"id": identity, "duration": duration, "label": label, "stratum": stratum,
                                                "flac_bytes": len(payload), "flac_sha256": digest(payload)})
    roster = []
    for cell in CELLS:
        candidates = sorted(eligible[cell], key=lambda row: digest(("radar-transfer/1/" + row["id"]).encode()))
        require(len(candidates) >= PER_CELL, "Incomplete publisher source cell")
        roster.extend(candidates[:PER_CELL])
    require(roster == read_json(panel / "intake.json"), "The salted native roster differs")
    for row in roster:
        path = panel / "clips" / (row["id"] + ".flac")
        require(path.stat().st_size == row["flac_bytes"] and file_sha(path) == row["flac_sha256"],
                "Extracted FLAC differs from its exact publisher member")
    for scene in SCENES:
        name = f"{scene}.ogg"
        require(file_sha(panel / name) == SOURCES[name], "The delivered background copy differs")
    frozen = read_json(panel / "freeze.json")
    runner = Path(__file__).with_name("audio_radar_transfer.py")
    require(frozen["schema"] == "chipmunk.audio-radar-transfer/1" and frozen["source_files"] == SOURCES
            and frozen["intake_sha256"] == commitment(roster) and frozen["compressed_bytes"] == PREFIX_BYTES
            and frozen["noise"]["offsets_s"] == SCENES and frozen["noise"]["target_db"] == -5
            and frozen["minimum_usefulness"] == {"minimum_long_spoof_flags": 8, "maximum_long_bonafide_flags": 2}
            and frozen["runner_sha256"] == file_sha(runner), "The scored freeze differs from the source protocol")
    return roster, frozen, len(complete), {f"{length}/{label}": len(eligible[length, label]) for length, label in CELLS}


def decode(path, limit):
    filters = f"aformat=sample_fmts=s16:channel_layouts=mono,aresample={RATE},atrim=end_sample={limit},ashowinfo"
    process = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-xerror",
                              "-protocol_whitelist", "file", "-copyts", "-i", str(path), "-map", "0:a:0", "-af", filters,
                              "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"], capture_output=True, timeout=25, check=True)
    pcm = array("h")
    pcm.frombytes(process.stdout)
    if sys.byteorder == "big":
        pcm.byteswap()
    timeline = [(int(ordinal), int(pts), int(count)) for ordinal, pts, count in FRAME.findall(process.stderr)]
    require(timeline and all(ordinal == i and count > 0 for i, (ordinal, _, count) in enumerate(timeline))
            and len(pcm) == sum(count for _, _, count in timeline) <= limit, "Native FFmpeg clock does not cover exact PCM")
    return pcm, timeline


def pcm_sha(samples):
    floats = array("f", (sample / 32768 for sample in samples))
    if sys.byteorder == "big":
        floats.byteswap()
    return digest(floats.tobytes())


def native_ranges(length):
    if length < 96_000:
        return []
    positions = [0, length - SPAN]
    if length >= 144_000:
        positions.insert(1, (length - SPAN) // 2)
    return [[start, start + SPAN] for start in positions]


def level(samples):
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768


def mix_hashes(samples, backgrounds, rank):
    speech = [sample / 32768 for sample in samples]
    signal_level = level(samples)
    require(signal_level > 0, "Silent foreground")
    mixed, snrs = {"gain_control": speech}, []
    for scene, offsets in SCENES.items():
        offset = RATE * offsets[rank % 4]
        noise = backgrounds[scene][offset:offset + len(samples)]
        require(len(noise) == len(samples) and level(noise) > 0, "Background source does not cover the chosen range")
        normalization = signal_level / level(noise)
        scale = 10 ** .25
        component = [(sample / 32768) * normalization * scale for sample in noise]
        mixed[scene] = [voice + background for voice, background in zip(speech, component, strict=True)]
        component_level = math.sqrt(math.fsum(sample**2 for sample in component) / len(component))
        snrs.append(20 * math.log10(signal_level / component_level))
    common_gain = min(1., .98 / max(abs(sample) for track in mixed.values() for sample in track))
    return common_gain, {name: pcm_sha(round(sample * common_gain * 32768) for sample in track)
                         for name, track in mixed.items()}, max(abs(value + 5) for value in snrs)


def direction(response, length):
    name, scope, windows, raw = (response[key] for key in ("decision", "scope", "windows", "raw"))
    require(name in DIRECTIONS and type(scope) is bool and (scope or name not in {"ai", "original"})
            and all(type(a) is type(b) is int and 0 <= a < b <= length for a, b in windows), "Malformed recorded voice response")
    if name == "unassessed":
        require(not scope and not windows and raw is None, "Unassessed response carries a decision or score")
    else:
        require(windows and type(raw) in (int, float) and 0 <= raw <= 1, "Missing recorded voice input")
    return name


def replay(source, panel):
    roster, frozen, members, eligible = source_roster(source, panel)
    result_path = panel / "result.json"
    saved = read_json(result_path) if result_path.exists() else None
    if saved is not None:
        require(saved["intake"] == roster and saved["freeze"] == frozen and saved["rows_sha256"] == commitment(saved["rows"]),
                "Scored source or result commitment differs")
    rows = saved["rows"] if saved is not None else [None] * len(roster)
    backgrounds = {scene: decode(source / f"{scene}.ogg", (max(offsets) + 5) * RATE)[0] for scene, offsets in SCENES.items()}
    totals, noise = {}, {}
    clocks, portions, mixtures, max_snr_error = 0, 0, 0, 0.
    for index, (row, observed) in enumerate(zip(roster, rows, strict=True)):
        path = panel / "clips" / (row["id"] + ".flac")
        samples, blocks = decode(path, 15 * RATE)
        cursor = 0
        for _, point, count in blocks:
            require(point == cursor, "Publisher FLAC did not decode onto a continuous zero-origin clock")
            cursor += count
        require(abs(Decimal(row["duration"]) * RATE - len(samples)) <= 3, "Publisher duration differs from native PCM")
        clocks += 1
        if observed is not None:
            require(all(observed[key] == value for key, value in row.items()) and observed["samples"] == len(samples)
                    and observed["pcm_sha256"] == pcm_sha(samples), "Measured file identity or native PCM differs")
        cell = f"{row['stratum']}/{row['label']}"
        if row["stratum"] == "long":
            ranges = native_ranges(len(samples))
            portions += len(ranges)
            if observed is not None:
                require([item["interval"] for item in observed["portions"]] == ranges and observed["matched_gain"] is None
                        and not observed["mixtures"], "Long range roster or matched-scene scope differs")
                for (a, b), portion in zip(ranges, observed["portions"], strict=True):
                    require(portion["pcm_sha256"] == pcm_sha(samples[a:b]), "Exact long-file portion PCM differs")
        else:
            gain, hashes, error = mix_hashes(samples, backgrounds, index % PER_CELL)
            mixtures += len(hashes)
            max_snr_error = max(max_snr_error, error)
            if observed is not None:
                require(not observed["portions"] and math.isclose(observed["matched_gain"], gain, abs_tol=1e-12, rel_tol=0)
                        and {key: item["pcm_sha256"] for key, item in observed["mixtures"].items()} == hashes,
                        "Peak-safe gain or independently reconstructed scene PCM differs")
        if observed is None:
            continue
        counter = totals.setdefault(cell, Counter(dict.fromkeys(DIRECTIONS, 0)))
        full = direction(observed["full"], len(samples))
        counter.update({"files": 1, full: 1})
        if row["stratum"] == "long":
            require(not observed["full"]["scope"], "A long file received scoped whole-file confidence")
            local = [direction(item["response"], SPAN) for item in observed["portions"]]
            counter.update(two_portions_assessed=sum(value != "unassessed" for value in local) >= 2,
                           any_ai_portion="ai" in local,
                           corroborated_ai_portions=local.count("ai") >= 2 and "original" not in local)
        else:
            for scene, mixed in observed["mixtures"].items():
                scene_cell = f"{scene}/{row['label']}"
                noise.setdefault(scene_cell, Counter(dict.fromkeys(DIRECTIONS, 0)))[direction(mixed["response"], len(samples))] += 1
    summary = None
    if saved is not None:
        advanced = totals["long/spoof"]["corroborated_ai_portions"] >= 8 and totals["long/bonafide"]["corroborated_ai_portions"] <= 2
        summary = {"counts": totals, "matched_short_scenes": noise, "follow_up_criterion_met": advanced}
        require(summary == saved["summary"], "Independent decision aggregate differs")
    return {"complete_flac_members": members, "eligible_by_cell": eligible, "publisher_clips_verified": len(roster),
            "native_zero_origin_clocks": clocks, "long_portions_reconstructed": portions,
            "matched_short_pcm_views_reconstructed": mixtures, "maximum_prequantization_snr_error_db": max_snr_error,
            "scored_pcm_and_decisions_verified": saved is not None, "decision_summary": summary}


def replay_confirmation(source, panel, prior_path):
    pins = {
        "dev-prefix.gz": "6e85fce6e8ee74784769f0bd76f77c379af44a1dab9848d61fc791a9ada05266",
        "dev-labels.txt": "33dbf6ef66df46c50716c419fab0c1c73edc160ca06996dbaf719bebe47204a3",
        "dev-languages.txt": "f28f2ebcad8ce58f0042f3f1fd2fa0c385284a9ca008e776f2d9ac9ae4dccd72",
    }
    require(all(file_sha(source / name) == pin for name, pin in pins.items()), "Confirmation source hashes differ")
    labels, languages = (table(source / name) for name in ("dev-labels.txt", "dev-languages.txt"))
    require(len(labels) == len(languages) == 44_034 and labels.keys() == languages.keys(), "Confirmation metadata differs")
    candidates, seen = {name: [] for name in ("bonafide", "spoof")}, set()
    member = re.compile(rb"RADAR2026-dev/flac/(RADAR2026-DEV[0-9]{6})\.flac")
    for identity, raw in tar_prefix(source / "dev-prefix.gz", member=member):
        require(identity in labels and identity not in seen, "Unknown or repeated confirmation archive member")
        seen.add(identity)
        duration, label, *segments = labels[identity]
        if languages[identity] == ["EN"] and 6 <= Decimal(duration) <= 14 and segments == [f"0.0000-{duration}-{label}"]:
            candidates[label].append({"id": identity, "duration": duration, "label": label, "flac_sha256": digest(raw)})
    ranked = []
    for values in candidates.values():
        ranked.extend(sorted(values, key=lambda row: digest(f"radar-long-confirmation/1/{row['id']}".encode()))[:20])
    prior, result = read_json(prior_path), read_json(panel / "result.json")
    intake, rows, frozen = result["intake"], result["rows"], result["freeze"]
    require(len(ranked) == len(intake) == len(rows) == 40 and intake == read_json(panel / "intake.json")
            and frozen == read_json(panel / "freeze.json") and frozen["source_files"] == pins
            and frozen["prior_freeze"] == commitment(prior["freeze"]) and frozen["roster"] == commitment(intake)
            and result["rows_sha256"] == commitment(rows), "Confirmation source commitments differ")
    prior_pcm = {row["pcm_sha256"] for row in prior["rows"]}
    summary, windows = {}, 0
    for expected, row, saved in zip(ranked, intake, rows, strict=True):
        require(all(row[key] == value for key, value in expected.items()) and all(saved[key] == value for key, value in row.items()),
                "Confirmation rank or row differs")
        path = panel / f"{row['id']}.flac"
        require(file_sha(path) == row["flac_sha256"], "Confirmation FLAC bytes differ")
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
                                         "-show_entries", "stream=codec_name,bits_per_raw_sample,sample_rate,channels",
                                         "-of", "json", str(path)], timeout=10)
        require(json.loads(probe)["streams"] == [{"codec_name": "flac", "sample_rate": "16000", "channels": 1,
                                                  "bits_per_raw_sample": "16"}], "Confirmation source profile differs")
        pcm, clock = decode(path, 15 * RATE)
        fingerprint = pcm_sha(pcm)
        require(len(pcm) == row["samples"] and fingerprint == row["pcm_sha256"] and fingerprint not in prior_pcm
                and abs(Decimal(row["duration"]) * RATE - len(pcm)) <= 3, "Confirmation PCM differs or overlaps")
        prior_pcm.add(fingerprint)
        cursor = 0
        for _, pts, count in clock:
            require(pts == cursor, "Confirmation clock has a gap")
            cursor += count
        intervals = native_ranges(len(pcm))
        portions = saved["portions"]
        require([item["interval"] for item in portions] == intervals, "Confirmation intervals differ")
        windows += len(intervals)
        for (start, end), portion in zip(intervals, portions, strict=True):
            require(pcm_sha(pcm[start:end]) == portion["pcm_sha256"], "Confirmation interval samples differ")
        local = [direction(item["response"], SPAN) for item in portions]
        full = direction(saved["full"], len(pcm))
        require(not saved["full"]["scope"] and full in {"inconclusive", "unassessed"}, "Long whole-file scope differs")
        counts = summary.setdefault(row["label"], Counter(files=0, native_inconclusive=0, native_unassessed=0,
                                                           two_assessed=0, one_ai=0, corroborated_ai=0))
        counts.update({"files": 1, f"native_{full}": 1, "two_assessed": sum(value != "unassessed" for value in local) >= 2,
                       "one_ai": "ai" in local, "corroborated_ai": local.count("ai") >= 2 and "original" not in local})
    passes = summary["spoof"]["corroborated_ai"] >= 10 and summary["bonafide"]["corroborated_ai"] <= 2
    require(result["summary"] == {"counts": summary, "passes": passes}, "Confirmation decision aggregate differs")
    return {"complete_flac_members": len(seen), "native_flac_sources": len(intake), "exact_intervals": windows,
            "decision_summary": result["summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--confirmation-prior", type=Path, help="Review the separate DEV panel using the first result")
    args = parser.parse_args()
    audit = (replay_confirmation(args.source, args.panel, args.confirmation_prior) if args.confirmation_prior
             else replay(args.source, args.panel))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
