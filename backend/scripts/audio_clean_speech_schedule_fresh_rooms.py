# SPDX-License-Identifier: Apache-2.0
"""Transfer the frozen stereo window schedules to new recorded room interference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from scripts import audio_clean_speech_schedule_transfer as schedule

room, old = schedule.room, schedule.old
RATE, SAMPLES = room.RATE, room.SAMPLES
ROOMS = {
    "EN2005a": {"file_bytes": 173287554, "pcm_bytes": 173287510},
    "IB4001": {"file_bytes": 56986418, "pcm_bytes": 56986284},
    "IN1016": {"file_bytes": 115340162, "pcm_bytes": 115340118},
}
MICS = ("Array1-03", "Array1-07")
PUBLISHER = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
SCHEMA = "chipmunk.audio-clean-speech-schedule-fresh-rooms/1"
PLAN = {
    "background": "Three non-scenario AMI meetings with microphones 03 and 07, using 12 s windows with "
                  "at least 2.4 s of words in each half and words during half of each staged synthetic edit.",
    "selection": "Retain all 24 earlier ICSI foreground groups. Assign early and late groups independently "
                 "to the three meetings in hash-determined order, balance the quieter target side, and choose "
                 "disjoint annotated windows by hash before downloading or scoring them.",
    "decision": "Keep the current stereo result if present. Compare both denser schedules separately "
                "using the frozen measurement settings, together with ordinary and wrong-reference controls.",
}


def source_groups():
    groups = {}
    for name, panel in schedule.spatial.input_panels().items():
        measured = defaultdict(dict)
        for row in panel["rows"]:
            if row["bank"] == "icsi":
                measured[row["group"]][row["kind"]] = row["truth"]
        for row in panel["plan"]["rows"]:
            if row["bank"] == "icsi":
                key = row["group"]
                groups[key] = {"group": key, "earlier_panel": name, "truth": measured[key]}
                groups[key].update({field: row[field] for field in ("bank", "companion", "half", "earlier_sha256")})
    old.require(len(groups) == 24 and Counter(row["half"] for row in groups.values()) == {"early": 12, "late": 12},
                "The retained foreground groups differ")
    return groups


def placements(groups):
    meeting_names = tuple(ROOMS)
    for half in ("early", "late"):
        order = sorted((group for group, row in groups.items() if row["half"] == half),
                       key=lambda group: old.sha(f"new-room-order/1/{group}".encode()))
        for index, group in enumerate(order):
            yield group, meeting_names[index % len(meeting_names)], ("left", "right")[index // len(meeting_names) % 2]


def freeze(annotations):
    import numpy as np

    from scripts.prepare_natural_overlap_transfer import track_words

    with annotations.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    old.require(digest == room.ANNOTATIONS, "The AMI manual annotations differ")
    groups = source_groups()
    available = {}
    with ZipFile(annotations) as archive:
        for meeting, extent in ROOMS.items():
            words = np.zeros(extent["pcm_bytes"] // 320, dtype=bool)
            for speaker in "ABCD":
                for start, end in track_words(archive.read(f"words/{meeting}.{speaker}.words.xml")):
                    words[start:end] = True
            available[meeting] = words
    rows = []
    used = defaultdict(set)
    for group, meeting, quiet_side in placements(groups):
        words = available[meeting]
        truth = groups[group]["truth"]
        edit_spans = [span for kind, span in truth.items() if kind.startswith("edited/")]
        edits = [(a // 160, (b + 159) // 160) for a, b in edit_spans]
        candidates = [
            start for start in range(6000, min(60000, len(words) - 1200 + 1), 1200)
            if start not in used[meeting]
            and all(words[start + shift:start + shift + 600].sum() >= 240 for shift in (0, 600))
            and all(2 * words[start + a:start + b].sum() >= b - a for a, b in edits)
        ]
        old.require(bool(candidates), f"No eligible annotation window for {group}")
        start = min(candidates, key=lambda offset: old.sha(f"new-room-window/1/{group}/{offset}".encode()))
        used[meeting].add(start)
        rows.append(groups[group] | {
            "meeting": meeting, "start": start * 160, "quiet_side": quiet_side,
            "word_centiseconds_in_edits": [int(words[start + a:start + b].sum()) for a, b in edits],
        })
    return {"schema": SCHEMA, "method": PLAN, "annotations_sha256": digest,
            "prior": {name: old.sha(path.read_bytes()) for name, path in schedule.spatial.PANELS.items()},
            "rooms": ROOMS, "microphones": list(MICS), "rows": rows}


def fetch_range(meeting, microphone, first, count):
    last = first + count - 1
    size = ROOMS[meeting]["file_bytes"]
    url = f"{PUBLISHER}/{meeting}/audio/{meeting}.{microphone}.wav?chipmunk-range={first}"
    headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=28) as reply:
                old.require(reply.status == 206 and reply.headers.get("Content-Range") ==
                            f"bytes {first}-{last}/{size}"
                            and reply.headers.get("Content-Encoding", "identity") == "identity",
                            "The publisher returned a different byte range")
                raw = reply.read(count + 1)
            old.require(len(raw) == count, "The publisher range was truncated")
            return raw
        except HTTPError as error:
            if attempt == 2 or error.code not in {429, 500, 502, 503, 504}:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise


def fetch_headers():
    signatures = {}
    for meeting, extent in ROOMS.items():
        signatures[meeting] = {}
        for microphone in MICS:
            raw = fetch_range(meeting, microphone, 0, 44)
            old.require(struct.unpack("<4sI4s4sIHHIIHH4sI", raw) ==
                        (b"RIFF", extent["file_bytes"] - 8, b"WAVE", b"fmt ", 16, 1, 1, RATE, 2 * RATE,
                         2, 16, b"data", extent["pcm_bytes"]), "The original microphone PCM clock differs")
            signatures[meeting][microphone] = old.sha(raw)
    return signatures


def capture(plan, icsi):
    import numpy as np

    dry = room.dry_icsi_inputs(icsi)
    headers = fetch_headers()
    groups = {}
    samples = []
    with TemporaryDirectory(prefix="chipmunk-fresh-room-audio-") as work:
        path = Path(work) / "submission.m4a"
        for source in plan["rows"]:
            meeting = source["meeting"]
            raw = [fetch_range(meeting, mic, 44 + 2 * source["start"], 2 * SAMPLES) for mic in MICS]
            groups[str(source["group"])] = dict(zip(MICS, map(old.sha, raw), strict=True))
            tracks = [np.frombuffer(channel, dtype="<i2") for channel in raw]
            cases = room.materialize(source | {"start": 0}, {meeting: tracks}, dry)
            carriers = {}
            for kind, (reference, submitted) in cases.items():
                pin = old.sha(submitted)
                if pin not in carriers:
                    carriers[pin] = room.stereo.stereo_pcm(path, submitted)
                views, clock = carriers[pin]
                measurement = room.stereo.check(reference, views)
                values = schedule.measure(np.frombuffer(reference, dtype="<i2"), views, measurement)
                samples.append({"group": source["group"], "kind": kind, "measurement": measurement, "views": values,
                                "clock": clock, "submitted": pin, "reference": old.sha(reference)})
            print(f"{len(groups)}/{len(plan['rows'])} source groups measured ({meeting}).", flush=True)
    return {"plan": plan, "headers": headers, "rooms_pcm_sha256": groups, "samples": samples}


def review(receipt):
    plan = receipt["plan"]
    groups = source_groups()
    roster = {row["group"]: row for row in plan["rows"]}
    old.require(plan["schema"] == SCHEMA and plan["method"] == PLAN and plan["rooms"] == ROOMS
                and plan["microphones"] == list(MICS) and plan["annotations_sha256"] == room.ANNOTATIONS
                and plan["prior"] == {name: old.sha(path.read_bytes()) for name, path in schedule.spatial.PANELS.items()}
                and len(plan["rows"]) == len(groups) and set(roster) == set(groups)
                and len({(r["meeting"], r["start"]) for r in roster.values()}) == len(groups),
                "The frozen source roster differs")
    for key, meeting, quiet_side in placements(groups):
        row = roster[key]
        old.require(all(row[field] == value for field, value in groups[key].items())
                    and row["meeting"] == meeting and row["quiet_side"] == quiet_side and row["start"] % SAMPLES == 0
                    and 60 * RATE <= row["start"] < 600 * RATE
                    and 2 * (row["start"] + SAMPLES) <= ROOMS[meeting]["pcm_bytes"],
                    "A foreground source or room clock differs")
    for collection, keys in ((receipt["headers"], ROOMS), (receipt["rooms_pcm_sha256"], map(str, groups))):
        old.require(set(collection) == set(keys) and all(set(clips) == set(MICS) and
                    all(room.DIGEST.fullmatch(pin) for pin in clips.values())
                    for clips in collection.values()), "A source range lacks a microphone commitment")
    rows = {(row["group"], row["kind"]): row for row in receipt["samples"]}
    expected = {(key, kind) for key, group in groups.items() for kind in group["truth"]}
    old.require(len(rows) == len(receipt["samples"]) and set(rows) == expected, "A required case is missing or duplicated")
    totals = defaultdict(Counter)
    for (key, kind), row in rows.items():
        source = roster[key]
        measurement = row["measurement"]
        views = measurement["readings"]
        clock = row["clock"]
        old.require(set(views) == {"downmix", "left", "right"} and set(measurement["support"]) == {"left", "right"}
                    and measurement["selected"] == room.channel_check.select(views, measurement["support"])
                    and clock["native_clock"] == room.stereo.CLOCK and 0 <= clock["decoded_samples"] - SAMPLES <= 342
                    and all(room.DIGEST.fullmatch(pin) for pin in (row["submitted"], row["reference"], clock["encoded_sha256"])),
                    "Corroboration or source clock differs")
        for reading in views.values():
            span = reading["range"]
            old.require(reading["reason"] in room.stereo.REASONS and room.DIGEST.fullmatch(reading["pcm_sha256"])
                        and (span is not None) == (reading["reason"] == "local_change")
                        and (span is None or (0 <= span[0] < span[1] <= SAMPLES and span[1] - span[0] == schedule.WIDTH)),
                        "A current measurement does not match the source clock")
        if kind in {"background_reference", "unrelated", "half_matched"}:
            unchanged = rows[key, "untouched"]
            old.require(row["clock"] == unchanged["clock"] and row["submitted"] == unchanged["submitted"],
                        "A reference-only substitution changed the submitted recording")
        if kind == "unmixed":
            old.require(row["reference"] == rows[key, "background_reference"]["reference"]
                        and row["submitted"] == rows[key, "background_only"]["submitted"]
                        and row["clock"] == rows[key, "background_only"]["clock"],
                        "The background sound controls no longer share the intended source")
        prior = views["downmix"]["range"] if room.downmix_with_channel_support(measurement) else None
        picks = {"current": prior}
        for arm in schedule.SEPARATIONS:
            candidate = schedule.arm_result(row, row, arm)
            picks[arm] = prior or candidate
        truth = source["truth"][kind]
        category = room.category(kind)
        for arm, picked in picks.items():
            for group in ("all", source["meeting"], source["earlier_panel"]):
                useful = old.location(picked, truth)
                totals[f"{group}/{arm}/{category}"].update(
                    files=1, flags=bool(picked), useful=useful,
                    gained=useful and not old.location(prior, truth),
                    extra_flag=bool(picked and not prior),
                    wrong_on_edit=bool(truth and picked and not useful),
                )
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan_mode = modes.add_parser("freeze")
    capture_mode = modes.add_parser("capture")
    review_mode = modes.add_parser("review")
    for mode, arguments in ((plan_mode, ("annotations", "output")), (capture_mode, ("plan", "annotations", "icsi", "output")),
                            (review_mode, ("receipt",))):
        for argument in arguments:
            mode.add_argument(argument, type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        with args.output.open("x") as output:
            json.dump(freeze(args.annotations), output, indent=2, sort_keys=True)
    elif args.mode == "capture":
        plan = old.read(args.plan)
        old.require(plan == freeze(args.annotations), "Source selection changed after freezing")
        receipt = capture(plan, args.icsi)
        receipt["results"] = review(receipt)
        body = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with args.output.open("xb") as output:
            output.write(gzip.compress(body, mtime=0))
    else:
        receipt = schedule.spatial.load(args.receipt)
        results = review(receipt)
        old.require(results == receipt["results"], "The recorded counts differ")
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
