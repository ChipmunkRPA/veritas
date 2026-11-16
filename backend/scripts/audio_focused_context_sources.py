# SPDX-License-Identifier: MIT
"""Bind long publisher recordings, their labels and matched AMI mixtures before inference."""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_focused_edit_binding as focused
from scripts import audio_focused_context_transfer as study

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PHONEME = PANELS / "audio-joint-generator-conversion-2026-11-16.json.gz"
RANGE_STARTS = (4 * 1024**3, 8 * 1024**3, 11 * 1024**3)
RANGE_BYTES = 32 * 1024**2
PUBLISHER_BYTES = 12_791_859_200
PER_FAMILY = 3
ORDINARY_COUNT = 18
LEVELS = (-5, -10)
HISTORY = (
    "audio-partial-publisher-transfer-2026-11-12.json",
    "audio-reduced-background-localization-2026-11-12.json",
    "audio-dense-evidence-confirmation-2026-11-12.json",
    "audio-noise-window-transfer-2026-11-12.json",
    "audio-nemotron-noise-transfer-2026-11-12.json",
    "audio-wpe-fresh-room-transfer-2026-11-13.json",
    "audio-spatial-mask-aachen-transfer-2026-11-13.json",
    "audio-focused-48k-transfer-2026-11-14.json",
)


def _sha(file):
    import hashlib

    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rank(identity):
    return study.digest(f"{study.SCHEMA}/{identity}".encode())


def capture(output):
    import struct
    from concurrent.futures import ThreadPoolExecutor

    from scripts.audio_locate_competing_speech import fetch

    plan = study.read(PHONEME)["source_manifest"]["room_plan"]
    output.mkdir(mode=0o700, parents=True, exist_ok=True)

    def extend(row):
        rate = study.RATE
        start, total = row["start_sample"], row["byte_size"]
        head, _ = fetch(row["url"], 0, 44, total)
        *header, extent = struct.unpack("<4sI4s4sIHHIIHH4sI", head)
        assert tuple(header) == (b"RIFF", total - 8, b"WAVE", b"fmt ", 16, 1, 1, rate, 2 * rate, 2, 16, b"data")
        assert study.digest(head) == row["header_sha256"] and 2 * (start + study.MAXIMUM) <= extent
        samples, _ = fetch(row["url"], 44 + 2 * start, 2 * study.MAXIMUM, total)
        assert study.digest(samples[:2 * focused.SAMPLES]) == row["pcm_sha256"]
        (output / f"{row['room']}.pcm").write_bytes(samples)
        return {key: row[key] for key in ("room", "start_sample", "url", "byte_size", "header_sha256")} | {
            "samples": study.MAXIMUM, "pcm_sha256": study.digest(samples)}

    with ThreadPoolExecutor(max_workers=3) as workers:
        captured = list(workers.map(extend, sorted(plan, key=lambda row: row["room"])))
    study.save(output / "rooms.json", captured)


def _partial_sources(part_b, new_ranges):
    from scripts import audio_noise_window_transfer as previous
    from scripts import audio_partial_publisher_transfer as publisher

    available, scanned = previous.archive_partials(part_b)
    labels = {name: parts for name, *parts in map(str.split, (part_b / "part-b.labels.txt").open())}
    added, range_pins = defaultdict(list), {}
    for start in RANGE_STARTS:
        file = new_ranges / f"llama-{start}.tarseg"
        end = start + RANGE_BYTES - 1
        headers = (new_ranges / f"llama-{start}.headers").read_text().casefold().splitlines()
        assert file.stat().st_size == RANGE_BYTES and f"content-range: bytes {start}-{end}/{PUBLISHER_BYTES}" in headers
        assert not any(line.startswith("content-encoding:") and line != "content-encoding: identity" for line in headers)
        range_pins[str(start)] = _sha(file)
        with file.open("rb") as stream:
            for offset in range(0, 2**21, tarfile.BLOCKSIZE):
                block = stream.read(tarfile.BLOCKSIZE)
                if not block.startswith(b"R01TTS.0.b/"):
                    continue
                try:
                    entry = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
                except tarfile.HeaderError:
                    continue
                if entry.isfile() and previous.MEMBER.fullmatch(entry.name):
                    stream.seek(offset)
                    break
            else:
                raise ValueError("Publisher range lacks a complete WAV header")
            try:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        match = previous.MEMBER.fullmatch(member.name)
                        assert match and 44 < member.size < 2**20
                        name, family, recipe, group = match.groups()
                        assert name in labels
                        with archive.extractfile(member) as source:
                            body = source.read()
                        assert len(body) == member.size
                        raw = publisher.native(body)
                        count = len(raw) // 2
                        if not study.MINIMUM <= count <= study.MAXIMUM:
                            continue
                        try:
                            segments = publisher.labels_for(labels[name], count)
                        except ValueError:
                            scanned["invalid_publisher_clocks"] += 1
                            continue
                        edits = [list(span) for span in study.merge((a, b) for a, b, label in segments if label == "spoof")]
                        if any(.35 * study.RATE <= b - a <= 1.4 * study.RATE for a, b in edits):
                            added[family].append(({"id": name, "group": group, "family": family, "recipe": recipe,
                                                  "edits": edits, "source_sha256": study.digest(body),
                                                  "source_pcm_sha256": study.digest(raw), "range_start": start}, raw))
            except (tarfile.ReadError, EOFError) as error:
                assert isinstance(error, EOFError) or "end of data" in str(error)
    return {family: added[family] + available[family] for family in publisher.GENERATORS}, range_pins, scanned


def _taken(options, count, seen, excluded, older_groups):
    chosen = []
    ranked = sorted(options, key=lambda pair: (pair[0]["group"] in older_groups,
                                               "range_start" not in pair[0], _rank(pair[0]["id"])))
    for row, pcm in ranked:
        if len(pcm) // 2 < study.MINIMUM or row["group"] in seen:
            continue
        if any(row[key] in excluded for key in ("id", "source_pcm_sha256", "source_sha256")):
            continue
        chosen.append((row, pcm))
        seen.add(row["group"])
        if len(chosen) == count:
            break
    return chosen


def prepare(args):
    import numpy as np

    from scripts import audio_noise_window_transfer as previous
    from scripts.audio_edit_direction_fresh import mix
    from scripts.audio_reduced_background_localization import source_candidates

    older_groups = previous.prior_groups()[2]
    history = {name: (PANELS / name).read_bytes() for name in HISTORY}
    excluded = set(re.findall(r"[a-f0-9]{64}|(?:dev-clean|test-clean|dev01-[a-z0-9]+-(?:partial-(?:cp|oa|cf)|full))_[\d_]+",
                              "\n".join(body.decode() for body in history.values())))
    pool, range_pins, scanned = _partial_sources(args.part_b, args.new_ranges)
    a_segments = [args.part_a / name for name in ("part-a-192m.prefix.tar", "part-a-192m-352m.tarseg",
                                                 "slice-2g.tarseg", "slice-6g.tarseg", "slice-10g.tarseg")]
    ordinary, scanned_a, _ = source_candidates(*a_segments[:2], args.a_labels, a_segments[2:])
    used_groups, selected = set(), []
    for family in sorted(pool, key=lambda name: (len(pool[name]), name)):
        options = [(row, raw) for row, raw in pool[family]
                   if any(.35 * study.RATE <= b - a <= 1.4 * study.RATE for a, b in row["edits"])]
        selected.extend(_taken(options, PER_FAMILY, used_groups, excluded, older_groups))
    original_options = [({"id": row["publisher_id"], "group": row["group"], "family": None, "recipe": None, "edits": [],
                          "source_sha256": row["source_wav_sha256"], "source_pcm_sha256": row["source_pcm_sha256"]}, raw)
                        for row, raw in ordinary["genuine", None]]
    selected.extend(_taken(original_options, ORDINARY_COUNT, used_groups, excluded, older_groups))
    assert len({row["family"] for row, _ in selected if row["edits"]}) >= 2 and len(selected) >= 12

    background_plan = study.read(args.rooms / "rooms.json")
    backgrounds = {}
    parents = {row["room"]: row for row in study.read(PHONEME)["source_manifest"]["room_plan"]}
    for row in background_plan:
        name = row["room"]
        raw = (args.rooms / f"{name}.pcm").read_bytes()
        assert row["samples"] == study.MAXIMUM and len(raw) == 2 * study.MAXIMUM and study.digest(raw) == row["pcm_sha256"]
        assert row["start_sample"] == parents[name]["start_sample"] and study.digest(raw[:2 * focused.SAMPLES]) == parents[name]["pcm_sha256"]
        backgrounds[name] = np.frombuffer(raw, "<i2").astype(np.float64)
    assert set(backgrounds) == set(parents)

    args.output.mkdir(mode=0o700)
    (args.output / "pcm").mkdir()
    rows, origins = [], []

    def keep(row, values, treatment, kind, room=None, local_db=None):
        raw = values.astype("<i2").tobytes()
        pin = study.digest(raw)
        (args.output / "pcm" / f"{pin}.pcm").write_bytes(raw)
        rows.append({"id": f"{row['id']}/{treatment}/{kind}", "group": row["group"], "treatment": treatment,
                     "kind": kind, "family": row["family"] if kind == "edited" else None, "samples": len(values),
                     "truth": row["edits"] if kind == "edited" else [], "pcm_sha256": pin,
                     "room": room, "local_edit_to_background_db": local_db})

    for row, raw in selected:
        samples = np.frombuffer(raw, "<i2").astype(np.float64)
        assert row["source_pcm_sha256"] == study.digest(raw)
        room = min(backgrounds, key=lambda name: _rank(f"{row['id']}/{name}"))
        noise = backgrounds[room][:len(samples)]
        kind = "edited" if row["edits"] else "ordinary"
        origins.append(row | {"source_group_in_recent_history": row["group"] in older_groups, "room": room})
        keep(row, samples, "clean", kind)
        for level in LEVELS:
            mixture, single = mix(samples[None, :], noise, level)
            local = []
            for a, b in row["edits"]:
                ground = mixture[0, a:b].astype(np.float64) - single[a:b]
                powers = np.linalg.norm(ground), np.linalg.norm(single[a:b])
                local.append(round(float(20 * np.log10(powers[0] / powers[1])), 6) if min(powers) > 0 else None)
            keep(row, mixture[0], f"overlap_{level}dB", kind, room, local or None)
            keep(row, single, f"overlap_{level}dB", "background", room)
    fit = study.read(PHONEME)
    head, = (row for row in fit["heads"] if row["arm"] == "both")
    freeze = {"schema": study.SCHEMA, "method": study.METHOD, "point_units": focused.REVIEW_POINT,
              "focused_head_sha256": head["sha256"], "origins": origins,
              "published_archive_pins": {"new_part_b": range_pins, "prior_part_b": previous.PINS,
                                         "part_a": {file.name: _sha(file) for file in (*a_segments, args.a_labels)}},
              "history_pins": {name: study.digest(body) for name, body in history.items()},
              "backgrounds": background_plan, "scanned": {"part_a": scanned_a, "prior_part_b": scanned}, "cases": rows}
    study.save(args.output / "freeze.json", freeze)
    print(f"Froze {len(rows)} submissions from {len(origins)} sources. " + str(dict(Counter(row["family"] for row in origins))), flush=True)


def audit(media, rooms):
    """Check each mixture from saved bytes without calling the renderer or a model."""
    import numpy as np

    frozen = study.read(media / "freeze.json")
    rows = {row["id"]: row for row in frozen["cases"]}
    assert len(rows) == len(frozen["cases"]) == len(frozen["origins"]) * (1 + 2 * len(LEVELS))
    backgrounds = {}
    for capture in frozen["backgrounds"]:
        raw = (rooms / f"{capture['room']}.pcm").read_bytes()
        assert len(raw) == 2 * capture["samples"] and study.digest(raw) == capture["pcm_sha256"]
        backgrounds[capture["room"]] = np.frombuffer(raw, "<i2").astype(np.float64)

    def pcm(row):
        key = row["pcm_sha256"]
        raw = (media / "pcm" / f"{key}.pcm").read_bytes()
        assert len(raw) == 2 * row["samples"] and study.digest(raw) == key
        return np.frombuffer(raw, "<i2").astype(np.float64)

    worst_delta = 0.0
    count = 0
    for origin in frozen["origins"]:
        ident, room, edits = (origin[key] for key in ("id", "room", "edits"))
        kind = "edited" if edits else "ordinary"
        dry = rows[f"{ident}/clean/{kind}"]
        assert dry["pcm_sha256"] == origin["source_pcm_sha256"] and dry["truth"] == edits
        clean = pcm(dry)
        room_pcm = backgrounds[room][:len(clean)]
        for level in LEVELS:
            treatment = f"overlap_{level}dB"
            delivered, isolated = (rows[f"{ident}/{treatment}/{role}"] for role in (kind, "background"))
            assert delivered["room"] == isolated["room"] == room and delivered["truth"] == edits and not isolated["truth"]
            mixed, sound = pcm(delivered), pcm(isolated)
            signal = mixed - sound
            for source, observed in ((clean, signal), (room_pcm, sound)):
                gain = np.dot(source, observed) / np.dot(source, source)
                assert gain > 0 and np.max(np.abs(source * gain - observed)) <= 1.0
            assert np.max(np.abs(mixed)) <= 32001
            actual_db = 20 * np.log10(np.linalg.norm(signal) / np.linalg.norm(sound))
            worst_delta = max(worst_delta, abs(actual_db - level))
            observed = []
            for a, b in edits:
                rms = np.linalg.norm(signal[a:b]), np.linalg.norm(sound[a:b])
                observed.append(round(float(20 * np.log10(rms[0] / rms[1])), 6) if min(rms) > 0 else None)
            assert delivered["local_edit_to_background_db"] == (observed or None)
            count += 1
    assert worst_delta < .05
    print(f"Verified {len(rows)} delivery hashes and {count} exact background controls.")
    print(f"Largest foreground-to-background ratio deviation {worst_delta:.6f} dB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("capture").add_argument("output", type=Path)
    audit_sources = actions.add_parser("audit")
    audit_sources.add_argument("media", type=Path)
    audit_sources.add_argument("rooms", type=Path)
    intake = actions.add_parser("prepare")
    for name in ("part_b", "new_ranges", "part_a", "a_labels", "rooms", "output"):
        intake.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.output)
    elif args.action == "audit":
        audit(args.media, args.rooms)
    else:
        prepare(args)
