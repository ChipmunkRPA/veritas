# SPDX-License-Identifier: MIT
"""Verify the longer-file two-second speech scan on fresh labeled inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from app.evidence import audio_long_review_binding as current
from app.evidence import audio_short_review_binding as short
from app.evidence import audio_two_second_binding as binding
from scripts import audio_focused_context_transfer as prior

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PREVIOUS = PANELS / "audio-long-focused-context-paired-2026-11-16.json.gz"
BACKGROUND_HISTORY = "04c91bd0f03f31c447c7bafa45154974ad4866d8"
SCHEMA = "chipmunk.two-second-speech-transfer/1"
RANGES = tuple(n * 512 * 1024**2 for n in (2, 3, 6, 10, 12, 15, 20, 23))
RANGE_BYTES, ARCHIVE_BYTES = 32 * 1024**2, 12_791_859_200
PER_GENERATOR, ORIGINALS, ROOMS = 4, 24, 6
LLAMA_ORIGINALS = 10
LIBRISPEECH_MD5 = "fb5a50374b501bb3bac4815ee91d3135"
LEVELS = (-5, -10)
ARMS = ("current", "two_second", "adjacent_support")
METHOD = {
    "sources": "Four native 9–14 second clips from each of six generator families, ten untouched publisher recordings "
               "and fourteen LibriSpeech test-other recordings. "
               "Use distinct foreground readers within each label and exclude previously recorded source IDs and hashes. "
               "Prefer readers absent from the preceding comparison and recorded historical ledgers where eligible.",
    "background": "Use six different publisher meetings with distinct participants from each other and from the "
                  "previous paired study. Exclude meetings named in the inspected project history. "
                  "A 14-second range must include at least ten seconds of annotated speech and four seconds "
                  "of overlap. Deliver each voice without added sound or with competing speech at −5 or −10 dB. "
                  "Retain the exact scaled background by itself for each mixture.",
    "comparison": "Retain the existing evidence schedule and responses. Compare adding the strongest two-second "
                  "response in each six-second excerpt at the earlier fixed native point. A separate arm requires a "
                  "different two-second window at the same point within the measured readings whose overlap is at least one second.",
    "outcomes": "Report one-to-one correct range matches at temporal IoU ≥ 0.30 and recall ≥ 50%, unassessed cases, "
                "ordinary versus isolated flags, and all genuine samples highlighted including inside edited recordings.",
}


def rank(*values):
    return prior.digest("/".join((SCHEMA, *map(str, values))).encode())


def capture(annotations, output, *, additional_meetings=()):
    from concurrent.futures import ThreadPoolExecutor
    from xml.etree import ElementTree

    from scripts.audio_harmonic_overlap_transfer import room_activity
    from scripts.audio_locate_competing_speech import capture_url, fetch

    previous = prior.read(PANELS / "audio-locate-competing-speech-2026-11-13.json")
    assert prior.digest(annotations.read_bytes()) == previous["freeze"]["annotations_sha256"]
    revision = BACKGROUND_HISTORY
    grep = subprocess.check_output(["git", "grep", "-h", "-o", "-E", r"(ES|IS|TS)[0-9]{4}[a-d]", revision, "--", "docs"],
                                   cwd=ROOT, text=True)
    prior_rooms = {row["room"] for row in prior.read(PREVIOUS)["freeze"]["backgrounds"]}
    prior_rooms.update(additional_meetings)
    excluded = set(grep.splitlines()) | prior_rooms
    output.mkdir(mode=0o700)
    selected = []
    with ZipFile(annotations) as archive:
        files = set(archive.namelist())
        root = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {node.get("observation"): {row.get("global_name") for row in node if row.get("global_name")}
                        for node in root.iter() if node.get("observation")}
        seen = set().union(*(participants[room] for room in prior_rooms))
        for room in sorted(participants, key=rank):
            if (room in excluded or not re.fullmatch(r"(ES|IS|TS)[0-9]{4}[a-d]", room)
                    or len(participants[room]) != 4 or seen & participants[room]):
                continue
            if any(f"words/{room}.{speaker}.words.xml" not in files for speaker in "ABCD"):
                continue
            activity = room_activity(archive, room)
            choices = []
            for cell in range(1800, len(activity) - 1400, 350):
                part = activity[cell:cell + 1400]
                speech, overlap = int((part > 0).sum()), int((part >= 2).sum())
                if speech >= 1000 and overlap >= 400:
                    choices.append((rank(room, cell), cell, speech, overlap))
            if choices:
                _, cell, speech, overlap = min(choices)
                selected.append({"room": room, "start_sample": cell * 160, "speech_samples": speech * 160,
                                 "overlap_samples": overlap * 160, "url": capture_url(room)})
                seen.update(participants[room])
            if len(selected) == ROOMS:
                break
    assert len(selected) == ROOMS

    def retrieve(url, offset, size, total=None):
        for attempt in range(3):
            try:
                return fetch(url, offset, size, total)
            except (TimeoutError, ConnectionError):
                if attempt == 2:
                    raise

    def acquire(row):
        header, total = retrieve(row["url"], 0, 44)
        *fields, size = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
        rate, start = binding.RATE, row["start_sample"]
        assert tuple(fields) == (b"RIFF", total - 8, b"WAVE", b"fmt ", 16, 1, 1, rate, 2 * rate, 2, 16, b"data")
        assert 2 * (start + binding.MAXIMUM) <= size
        raw, _ = retrieve(row["url"], 44 + 2 * start, 2 * binding.MAXIMUM, total)
        (output / f"{row['room']}.pcm").write_bytes(raw)
        return row | {"byte_size": total, "header_sha256": prior.digest(header), "pcm_sha256": prior.digest(raw)}
    with ThreadPoolExecutor(max_workers=3) as workers:
        rooms = list(workers.map(acquire, selected))
    prior.save(output / "rooms.json", {"history_revision": revision, "excluded_meetings": sorted(excluded),
                                       "participant_excluded_meetings": sorted(prior_rooms), "rooms": rooms})
    print("Verified six publisher room clocks and competing-speech ranges.", flush=True)


def publisher_edits(ranges, labels_path, *, edit_extent=(9600, 28800), range_starts=RANGES):
    from scripts import audio_noise_window_transfer as source
    from scripts import audio_partial_publisher_transfer as publisher

    assert prior.digest(labels_path.read_bytes()) == source.PINS["part-b.labels.txt"]
    labels = {name: fields for name, *fields in map(str.split, labels_path.read_text().splitlines())}
    options, pins = defaultdict(list), {}
    for start in range_starts:
        archive = ranges / f"llama-{start}.tarseg"
        headers = (ranges / f"llama-{start}.headers").read_text().casefold()
        assert archive.stat().st_size == RANGE_BYTES
        assert f"content-range: bytes {start}-{start + RANGE_BYTES - 1}/{ARCHIVE_BYTES}" in headers
        assert all(line == "content-encoding: identity" for line in headers.splitlines() if line.startswith("content-encoding:"))
        pins[str(start)] = prior.digest(archive.read_bytes())
        with archive.open("rb") as file:
            for offset in range(0, 2**21, tarfile.BLOCKSIZE):
                block = file.read(tarfile.BLOCKSIZE)
                if block.startswith(b"R01TTS.0.b/"):
                    try:
                        member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
                    except tarfile.HeaderError:
                        continue
                    if member.isfile() and source.MEMBER.fullmatch(member.name):
                        file.seek(offset)
                        break
            else:
                raise ValueError("Bounded Part B range contains no complete WAV header")
            try:
                with tarfile.open(fileobj=file, mode="r|") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        match = source.MEMBER.fullmatch(member.name)
                        assert match and 44 < member.size < 2**21
                        identity, family, recipe, group = match.groups()
                        with tar.extractfile(member) as stream:
                            wav = stream.read()
                        pcm = publisher.native(wav)
                        samples = len(pcm) // 2
                        if not binding.excerpt_starts(samples):
                            continue
                        try:
                            edits = prior.merge((a, b) for a, b, label in publisher.labels_for(labels[identity], samples) if label == "spoof")
                        except ValueError:
                            continue
                        if any(edit_extent[0] <= b - a <= edit_extent[1] for a, b in edits):
                            options[family].append(({"id": identity, "group": group, "family": family, "recipe": recipe, "corpus": "llama",
                                                     "edits": [list(span) for span in edits], "source_sha256": prior.digest(wav),
                                                     "native_pcm_sha256": prior.digest(pcm), "archive_start": start}, pcm))
            except (EOFError, tarfile.ReadError) as error:
                assert isinstance(error, EOFError) or "end of data" in str(error)
    if range_starts == RANGES:
        assert set(options) == set(publisher.GENERATORS)
    return options, pins


def librispeech_controls(archive, excluded, earlier, exposed, occupied, *, count_limit=ORIGINALS - LLAMA_ORIGINALS,
                        admit=None, subset="test-other"):
    pins = {"test-other": LIBRISPEECH_MD5, "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931"}
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "md5").hexdigest() == pins[subset]
    options = defaultdict(list)
    with tarfile.open(archive, mode="r|gz") as collection:
        for member in collection:
            match = re.fullmatch(rf"LibriSpeech/{subset}/(\d+)/(\d+)/(\d+-\d+-\d+)\.flac", member.name)
            if not member.isfile() or match is None or not 42 <= member.size < 2**21:
                continue
            reader, chapter, identity = match.groups()
            assert identity.startswith(f"{reader}-{chapter}-")
            if reader in occupied or identity in excluded or (admit is not None and not admit(identity)):
                continue
            with collection.extractfile(member) as stream:
                header = stream.read(42)
                assert header[:4] == b"fLaC" and header[4] & 0x7f == 0 and header[5:8] == b"\x00\x00\x22"
                packed = int.from_bytes(header[18:26], "big")
                samples = packed & ((1 << 36) - 1)
                if (packed >> 44, (packed >> 41) & 7, (packed >> 36) & 31) != (binding.RATE, 0, 15):
                    continue
                if not binding.excerpt_starts(samples):
                    continue
                options[reader].append((rank(identity), identity, samples, header + stream.read()))
    count = 0
    for reader in sorted(options, key=lambda group: (group in earlier, group in exposed, rank(group))):
        for _, identity, samples, body in sorted(options[reader]):
            if (source_pin := prior.digest(body)) in excluded:
                continue
            completed = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "flac", "-i", "pipe:0",
                                        "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
                                       input=body, capture_output=True, check=True)
            pcm = completed.stdout
            assert len(pcm) == 2 * samples
            if (pcm_pin := prior.digest(pcm)) in excluded:
                continue
            yield {"id": identity, "group": reader, "family": None, "edits": [], "corpus": f"librispeech_{subset.replace('-', '_')}",
                   "source_sha256": source_pin, "native_pcm_sha256": pcm_pin}, pcm
            count += 1
            break
        if count == count_limit:
            return
    raise ValueError("Insufficient complete native public audio controls")


def prepare(args):
    import numpy as np
    from scripts import audio_noise_window_transfer as source
    from scripts.audio_edit_direction_fresh import mix
    from scripts.audio_focused_context_sources import HISTORY, _sha
    from scripts.audio_reduced_background_localization import source_candidates

    paired = prior.read(PREVIOUS)["freeze"]
    earlier_readers = {row["group"] for row in paired["origins"]}
    exposed = source.prior_groups()[2]
    old_pins = {row[key] for row in paired["origins"] for key in ("source_sha256", "source_pcm_sha256", "id")}
    history_pins = {}
    for name in HISTORY:
        data = (PANELS / name).read_bytes()
        history_pins[name] = prior.digest(data)
        old_pins.update(re.findall(r"[a-f0-9]{64}|(?:dev-clean|test-clean|dev01-[a-z0-9]+-(?:partial-(?:cp|oa|cf)|full))_[\d_]+", data.decode()))
    options, range_pins = publisher_edits(args.ranges, args.labels_b)
    part_a = [args.part_a / name for name in ("part-a-192m.prefix.tar", "part-a-192m-352m.tarseg",
                                              "slice-2g.tarseg", "slice-6g.tarseg", "slice-10g.tarseg")]
    pins_a = paired["published_archive_pins"]["part_a"]
    assert all(_sha(path) == pins_a[path.name] for path in (*part_a, args.labels_a))
    bank, scanned_a, _ = source_candidates(*part_a[:2], args.labels_a, part_a[2:])
    options[None] = [({"id": row["publisher_id"], "group": row["group"], "family": None, "edits": [], "corpus": "llama",
                       "source_sha256": row["source_wav_sha256"], "native_pcm_sha256": row["source_pcm_sha256"]}, pcm)
                     for row, pcm in bank["genuine", None] if binding.excerpt_starts(len(pcm) // 2)]
    used, selected = {True: set(), False: set()}, []
    for family in sorted(options, key=lambda key: len({row["group"] for row, _ in options[key]})):
        count = LLAMA_ORIGINALS if family is None else PER_GENERATOR
        chosen = 0
        seen = used[family is not None]
        order = sorted(options[family], key=lambda pair: (pair[0]["group"] in earlier_readers, pair[0]["group"] in exposed,
                                                         rank(pair[0]["id"])))
        for row, pcm in order:
            if row["group"] in seen or any(row[key] in old_pins for key in ("source_sha256", "native_pcm_sha256", "id")):
                continue
            selected.append((row, pcm))
            seen.add(row["group"])
            chosen += 1
            if chosen == count:
                break
        assert chosen == count, f"Only {chosen} eligible new foreground groups for {family}"
    for row, pcm in librispeech_controls(args.librispeech, old_pins, earlier_readers, exposed, used[False]):
        selected.append((row, pcm))
        used[False].add(row["group"])
    room_plan = prior.read(args.rooms / "rooms.json")
    rooms = room_plan["rooms"]
    background = {}
    for row in rooms:
        raw = (args.rooms / f"{row['room']}.pcm").read_bytes()
        assert prior.digest(raw) == row["pcm_sha256"] and len(raw) == 2 * binding.MAXIMUM
        background[row["room"]] = np.frombuffer(raw, "<i2").astype(np.float64)
    cases, origins = [], []
    args.output.mkdir(mode=0o700)
    (args.output / "pcm").mkdir()

    def keep(row, values, condition, kind, room=None):
        raw = values.astype("<i2").tobytes()
        pin = prior.digest(raw)
        (args.output / "pcm" / f"{pin}.pcm").write_bytes(raw)
        cases.append({"id": f"{row['id']}/{condition}/{kind}", "group": row["group"], "condition": condition,
                      "kind": kind, "family": row["family"] if kind == "edited" else None, "corpus": row["corpus"],
                      "samples": len(values),
                      "truth": row["edits"] if kind == "edited" else [], "pcm_sha256": pin, "room": room})

    for index, (row, pcm) in enumerate(sorted(selected, key=lambda pair: rank(pair[0]["id"]))):
        signal = np.frombuffer(pcm, "<i2").astype(np.float64)
        room = rooms[index % ROOMS]["room"]
        origins.append(row | {"room": room, "earlier_reader": row["group"] in exposed,
                              "paired_panel_reader": row["group"] in earlier_readers})
        kind = "edited" if row["edits"] else "ordinary"
        keep(row, signal, "clean", kind)
        for level in LEVELS:
            mixture, isolated = mix(signal[None, :], background[room][:len(signal)], level)
            keep(row, mixture[0], f"overlap_{level}dB", kind, room)
            keep(row, isolated, f"overlap_{level}dB", "background", room)
    freeze = {"schema": SCHEMA, "method": METHOD, "history_pins": history_pins,
              "review_point": short.REVIEW_POINT, "cases": cases, "origins": origins, "rooms": room_plan,
              "prior_panel_sha256": prior.digest(PREVIOUS.read_bytes()), "archive_ranges": range_pins, "part_a_sha256": pins_a,
              "scanned_a": scanned_a, "librispeech_archive_sha256": _sha(args.librispeech)}
    prior.save(args.output / "freeze.json", freeze)
    print(f"Froze {len(cases)} deliveries from {len(origins)} recordings and {len(used[True] | used[False])} distinct readers.", flush=True)


def locations(row):
    excerpts = binding.summarize_readings(row["excerpts"], row["samples"], schema=binding.CONFIRMED_SCHEMA)
    selected = [(region["start_sample"], region["end_sample"]) for region in binding.regions(excerpts)]
    corroborated = [(region["start_sample"], region["end_sample"])
                    for region in binding.regions(excerpts, schema=binding.CONFIRMED_SCHEMA)]
    windows = row["current_windows"]
    dense = current.focused_intervals(row["samples"])
    assert [(w["start_sample"], w["end_sample"]) for w in windows] == (dense or current.intervals(row["samples"]))
    assert all(w["direction"] in current.DIRECTIONS for w in windows)
    if dense:
        primary = current.focused_regions(row["samples"], windows)
    else:
        primary = [window for window in windows if window["direction"] == "ai"] if current.status(windows) == "corroborated" else []
    baseline = [(window["start_sample"], window["end_sample"]) for window in primary]
    return dict(zip(ARMS, (baseline, sorted(set(baseline + selected)), sorted(set(baseline + corroborated))), strict=True))


def verify_serving(rows):
    from app.core.canonical import canonical_json

    for row in rows:
        rounded = json.loads(canonical_json(row["excerpts"]))
        for schema in (binding.SCHEMA, binding.CONFIRMED_SCHEMA):
            original = binding.summarize_readings(row["excerpts"], row["samples"], schema=schema)
            stored = binding.summarize_readings(rounded, row["samples"], schema=schema)
            assert stored == original, (row["id"], schema)


def geometry_audit(rows):
    """Check individual placement limits and compare intersections at the frozen points."""
    counts = defaultdict(Counter)
    for row in rows:
        tally = counts[row["condition"]]
        truth = row["truth"]
        arms = locations(row)
        excerpts = binding.summarize_readings(row["excerpts"], row["samples"], schema=binding.CONFIRMED_SCHEMA)
        paired = binding.regions(excerpts, schema=binding.CONFIRMED_SCHEMA)
        intersections = {(max(region["start_sample"], region["confirming_window"]["start_sample"]),
                          min(region["end_sample"], region["confirming_window"]["end_sample"])) for region in paired}
        full = arms["adjacent_support"]
        narrowed = sorted(set(arms["current"]) | intersections)
        before, after = prior.matches(full, truth), prior.matches(narrowed, truth)
        tally["truth_ranges"] += len(truth)
        tally["confirmed_full_matches"] += len(before)
        tally["intersection_matches"] += len(after)
        tally["intersection_added_matches"] += len(after - before)
        tally["intersection_lost_matches"] += len(before - after)
        if row["kind"] == "edited":
            scheduled = [(offset + a, offset + b) for offset in binding.excerpt_starts(row["samples"]) for a, b in short.GRID]
            admitted = [(excerpt["start_sample"] + reading["start_sample"], excerpt["start_sample"] + reading["end_sample"])
                        for excerpt in row["excerpts"] for reading in excerpt["responses"]]
            tally["shorter_than_600_ms"] += sum(b - a < 9600 for a, b in truth)
            tally["covered_at_scheduled_geometry"] += sum(bool(prior.matches(scheduled, [interval])) for interval in truth)
            tally["covered_at_admitted_geometry"] += sum(bool(prior.matches(admitted, [interval])) for interval in truth)
        else:
            tally[f"{row['kind']}_full_flags"] += bool(full)
            tally[f"{row['kind']}_intersection_flags"] += bool(narrowed)
        if row["kind"] != "background":
            for label, proposed in (("full", full), ("intersection", narrowed)):
                tally[f"genuine_samples_marked_{label}"] += sum(b - a - sum(prior.overlap((a, b), exact) for exact in truth)
                                                                 for a, b in prior.merge(proposed))
    return dict(sorted(counts.items()))


def summarize(freeze, rows):
    assert freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["review_point"] == short.REVIEW_POINT
    assert len(freeze["cases"]) == len(rows) == (6 * PER_GENERATOR + ORIGINALS) * 5
    assert len({row["group"] for row in freeze["origins"] if row["edits"]}) == 6 * PER_GENERATOR
    assert len({row["group"] for row in freeze["origins"] if not row["edits"]}) == ORIGINALS
    assert len({case["id"] for case in freeze["cases"]}) == len(rows)
    tallies, gained = defaultdict(Counter), defaultdict(set)
    for case, row in zip(freeze["cases"], rows, strict=True):
        assert all(row[key] == value for key, value in case.items())
        truth, kind = row["truth"], row["kind"]
        all_ranges = locations(row)
        old = prior.matches(all_ranges["current"], truth)
        scanned = any(reading["responses"] for reading in row["excerpts"])
        current_assessed = any(window["direction"] != "unassessed" for window in row["current_windows"])
        for arm, ranges in all_ranges.items():
            key = row["condition"], arm
            tally = tallies[key]
            tally[kind] += 1
            tally[f"{kind}_flagged"] += bool(ranges)
            if kind == "ordinary":
                tally[f"{row['corpus']}_ordinary"] += 1
                tally[f"{row['corpus']}_ordinary_flagged"] += bool(ranges)
            tally[f"{kind}_unassessed"] += not (current_assessed or (arm != "current" and scanned))
            found = prior.matches(ranges, truth)
            tally["truth_ranges"] += len(truth)
            tally["located"] += len(found)
            tally["lost_locations"] += len(old - found)
            if found - old:
                gained[key].add(row["group"])
            tally["flagged_genuine_samples"] += sum(b - a - sum(prior.overlap((a, b), span) for span in truth)
                                                  for a, b in prior.merge(ranges)) if kind != "background" else 0
    return {f"{condition}/{arm}": dict(tally, added_groups=len(gained[condition, arm]))
            for (condition, arm), tally in sorted(tallies.items())}


def score_case(case, pcm, voice, activity):
    from app.detectors.audio import extended_short_voice_review
    from app.engine.boundary import Deadline
    from scripts.audio_radar_transfer import adapter_result

    schedule = current.focused_intervals(len(pcm))
    windows = [{"start_sample": a, "end_sample": b, "direction": adapter_result(pcm[a:b], voice, activity)["decision"]}
               for a, b in schedule or current.intervals(len(pcm))]
    excerpts = extended_short_voice_review.scan(pcm, activity, voice, Deadline(None))
    return case | {"current_windows": windows, "excerpts": excerpts}


def score(media, weights, output):
    import numpy as np

    from scripts.audio_joint_generator_transfer import panels
    from scripts.audio_joint_generator_transfer import prior as previous
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = prior.read(media / "freeze.json")
    assert frozen["method"] == METHOD and frozen["schema"] == SCHEMA
    assets = previous.read(panels.PRIMARY)["incumbent"]["assets"]
    voice = load_voice({"weights": weights, "assets": assets})
    activity = previous.runtime(weights / "silero_vad.jit")
    rows = []
    for index, case in enumerate(frozen["cases"], 1):
        raw = (media / "pcm" / f"{case['pcm_sha256']}.pcm").read_bytes()
        assert len(raw) == 2 * case["samples"] and prior.digest(raw) == case["pcm_sha256"]
        pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.
        rows.append(score_case(case, pcm, voice, activity))
        if index % 12 == 0:
            print(f"Compared {index}/{len(frozen['cases'])} deliveries.", flush=True)
    prior.save(output, {"freeze": frozen, "rows": rows, "model_sha256": assets, "summary": summarize(frozen, rows)})


def export(receipt, output):
    from app.bench import independent
    from app.bench.independent_locations import TRUTH_SCHEMA, count_events, count_locations

    freeze, rows = receipt["freeze"], receipt["rows"]
    summary = summarize(freeze, rows)
    assert summary == receipt["summary"]
    known = {row["id"]: row for row in freeze["origins"]}
    backgrounds = {row["room"]: row for row in freeze["rooms"]["rooms"]}
    inputs, truth, selected = [], [], {arm: [] for arm in ARMS}
    for row in rows:
        origin = known[row["id"].split("/")[0]]
        room = backgrounds.get(row["room"])
        ordinary_background = row["kind"] == "background"
        reader = f"reader/{row['group']}"
        case = {"id": row["id"], "input_sha256": row["pcm_sha256"]}
        condition = row["condition"] + ("/background" if ordinary_background else "")
        source_sha = room["pcm_sha256"] if ordinary_background else origin["source_sha256"]
        inputs.append(case | {"modality": "audio", "split": "confirmation", "source_sha256": source_sha, "condition": condition,
                              "group": f"ami/{room['room']}" if ordinary_background else reader,
                              "label": int(row["kind"] == "edited"), "family": row["family"] or row["kind"],
                              "lineage": [reader, room["pcm_sha256"]] if room else [reader],
                              "license": {"id": "CC BY 4.0 (LlamaPartialSpoof, LibriSpeech, AMI)", "evaluation_permitted": True,
                                          "redistribution_permitted": True},
                              "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                                     "target_units": sum(b-a for a, b in row["truth"])}})
        truth.append(case | {"extent": [row["samples"]], "regions": row["truth"]})
        base_assessed = any(window["direction"] != "unassessed" for window in row["current_windows"])
        shorter_assessed = any(excerpt["responses"] for excerpt in row["excerpts"])
        for arm, spans in locations(row).items():
            assessed = base_assessed or (arm != "current" and shorter_assessed)
            selected[arm].append(case | ({"status": "ok", "response": int(bool(spans)),
                                         "localization": {"status": "ok", "regions": [list(span) for span in spans]}}
                                        if assessed else {"status": "not_applicable", "response": None,
                                                          "reason": "No eligible speech window was assessed.",
                                                          "localization": {"status": "not_applicable",
                                                                           "reason": "No eligible speech window was assessed."}}))
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": binding.RATE, "cases": inputs}
    identity = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    targets = identity | {"schema": TRUTH_SCHEMA, "cases": truth}
    output.mkdir(mode=0o700)
    prior.save(output / "manifest.json", manifest)
    prior.save(output / "truth.json", targets)
    for arm, predictions in selected.items():
        response = identity | {"system": arm, "implementation_sha256": independent.digest([SCHEMA, METHOD, arm]),
                               "method_kind": "weighted", "score_kind": "raw", "response_units": "A source range was flagged (0 or 1)",
                               "weight_sha256": sorted(receipt["model_sha256"].values()), "predictions": predictions}
        events = count_events(manifest, targets, response, minimum_iou=.30, minimum_recall=.50)
        points = count_locations(manifest, targets, response)
        for condition in ("clean", "overlap_-5dB", "overlap_-10dB"):
            expected = summary[f"{condition}/{arm}"]
            strata = [record for record in events["strata"] if record["condition"] == condition and record["label"] == 1]
            assert sum(record["matched_events"] for record in strata) == expected["located"]
            mask = (record["localization"] for case, record in zip(inputs, points["predictions"], strict=True)
                    if case["condition"] == condition and record["status"] == "ok")
            assert sum(item["reported_units"] - item["overlap_units"] for item in mask) == expected["flagged_genuine_samples"]
        prior.save(output / f"{arm}.json", response)
        prior.save(output / f"{arm}-events.json", events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, fields in (("capture", ("annotations", "output")),
                           ("prepare", ("ranges", "labels_b", "part_a", "labels_a", "librispeech", "rooms", "output")),
                           ("score", ("media", "weights", "output")), ("review", ("receipt",)),
                           ("geometry", ("receipt",)), ("export", ("receipt", "output"))):
        command = actions.add_parser(action)
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.annotations, args.output)
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "score":
        score(args.media, args.weights, args.output)
    elif args.action == "export":
        export(prior.read(args.receipt), args.output)
    else:
        result = prior.read(args.receipt)
        counts = summarize(result["freeze"], result["rows"])
        assert counts == result["summary"]
        verify_serving(result["rows"])
        if args.action == "geometry":
            counts = geometry_audit(result["rows"])
        print(json.dumps(counts, indent=2))
