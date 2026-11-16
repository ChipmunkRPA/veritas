"""Select a prospective noisy-speech panel with separate reader and room commitments."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np

from scripts import audio_focused_context_sources as history
from scripts import audio_long_context_inputs as prep
from scripts import audio_two_second_transfer as serving
from scripts.audio_edit_direction_fresh import mix, ordinary_edit
from scripts.audio_focused_context_transfer import read, save

SCHEMA = "chipmunk.audio-native-context-holdout/1"
LEVELS = (-5, -10)
MATCHABLE = (4000, 14400)
MARGIN = prep.RATE // 10
FAMILY_TARGET = 4
PART_A_ORIGINALS = 3
ORDINARY_COUNT = 24
EXTRA_STARTS = (4 * 1024**3, 8 * 1024**3, 11 * 1024**3)


def centered_views(samples):
    if not 9 * prep.RATE <= samples <= 14 * prep.RATE:
        raise ValueError("Held input is outside the measured native duration range")
    longest = min(samples, prep.SAMPLES["long"])
    return {"short": ((samples - prep.SAMPLES["short"]) // 2, prep.SAMPLES["short"]),
            "long": ((samples - longest) // 2, longest)}


def in_scope(samples, truth):
    start, length = centered_views(samples)["short"]
    return [interval for interval in truth if start <= interval[0] < interval[1] <= start + length]


def eligible_edit(row, values):
    start, length = centered_views(len(values) // 2)["short"]
    return any(MATCHABLE[0] <= b - a <= MATCHABLE[1] and start + MARGIN <= a < b <= start + length - MARGIN
               for a, b in row["edits"])


def native_utterance(identity):
    match = re.search(r"(\d+)[_-](\d+)[_-](\d+)(?:[_-]\d+)?$", identity)
    return tuple(map(int, match.groups())) if match else None


def exposures(development, retained, older, directory):
    development_readers = {str(row["reader"]) for row in development["sources"]}
    previous_readers = set()
    pins, older_receipts = set(), {}
    for package in (retained, older):
        for row in package["origins"]:
            previous_readers.add(str(row["group"]))
            pins.update(row[key] for key in ("id", "source_sha256"))
            pins.update(row[key] for key in ("native_pcm_sha256", "source_pcm_sha256") if key in row)
    for name in history.HISTORY:
        receipt = (directory / name).read_bytes()
        older_receipts[name] = hashlib.sha256(receipt).hexdigest()
        pins.update(re.findall(r"[a-f0-9]{64}|(?:dev-clean|test-clean|dev01-[a-z0-9]+-(?:partial-(?:cp|oa|cf)|full))_[\d_]+",
                               receipt.decode()))
    utterances = {identity for key in pins if (identity := native_utterance(key)) is not None}
    return development_readers, previous_readers, pins, utterances, older_receipts


def _available(pairs, used, pins, seen_utterances):
    return [(row, data) for row, data in pairs if row["group"] not in used
            and native_utterance(row["id"]) not in seen_utterances
            and all(row[key] not in pins for key in ("id", "source_sha256", "native_pcm_sha256"))]


def distinct_assignment(groups, sizes, earlier):
    slots = [family for family, count in sizes.items() for _ in range(count)]
    qualified = {}
    for family, pairs in groups.items():
        assigned = {}
        for row, values in sorted(pairs, key=lambda item: (item[0]["group"] in earlier, prep.rank("prospective", item[0]["id"]))):
            assigned.setdefault(row["group"], (row, values))
        qualified[family] = assigned
    selected, occupied = {}, {}

    def choose(slot, visited):
        for reader, item in qualified[slots[slot]].items():
            if reader in visited:
                continue
            visited.add(reader)
            if reader not in occupied or choose(occupied[reader], visited):
                occupied[reader] = slot
                selected[slot] = item
                return True
        return False

    if not all(choose(slot, set()) for slot in sorted(range(len(slots)), key=lambda i: len(qualified[slots[i]]))):
        raise ValueError("The publisher inputs do not supply enough distinct reader groups")
    return [selected[slot] for slot in range(len(slots))]


def part_a_originals(part_a, labels):
    from scripts.audio_focused_context_sources import _sha
    from scripts.audio_reduced_background_localization import source_candidates

    known = read(serving.PREVIOUS)["freeze"]["published_archive_pins"]["part_a"]
    files = [part_a / name for name in ("part-a-192m.prefix.tar", "part-a-192m-352m.tarseg", "slice-2g.tarseg",
                                        "slice-6g.tarseg", "slice-10g.tarseg")]
    assert all(_sha(source) == known[source.name] for source in (*files, labels))
    candidates, _, _ = source_candidates(*files[:2], labels, files[2:])
    return [({"id": row["publisher_id"], "group": row["group"], "family": None, "edits": [], "corpus": "llama",
              "source_sha256": row["source_wav_sha256"], "native_pcm_sha256": row["source_pcm_sha256"]}, raw)
            for row, raw in candidates["genuine", None] if 9 * prep.RATE <= len(raw) // 2 <= 14 * prep.RATE]


def roster(args, development, retained, older):
    from scripts import audio_noise_window_transfer as history_checks
    from scripts import audio_partial_publisher_transfer as publisher

    used, previous_readers, pins, seen_utterances, historical = exposures(development, retained, older, serving.PANELS)
    previous_groups = history_checks.prior_groups()[2] | previous_readers
    options, archive = serving.publisher_edits(args.ranges, args.labels_b, edit_extent=MATCHABLE)
    if args.extra_ranges:
        extra, pins_extra = serving.publisher_edits(args.extra_ranges, args.labels_b, edit_extent=MATCHABLE,
                                                    range_starts=EXTRA_STARTS)
        for label, values in extra.items():
            options[label].extend(values)
        archive.update(pins_extra)
    labels = tuple(sorted(options))
    if labels != tuple(sorted(publisher.GENERATORS)):
        raise ValueError("The prospective panel must cover every pinned generator")
    groups = {label: _available([pair for pair in options[label] if eligible_edit(*pair)], used, pins, seen_utterances)
              for label in labels}
    groups[None] = _available(part_a_originals(args.part_a, args.labels_a), used, pins, seen_utterances)
    print("Reader counts with unused source utterances", {str(key): len({row["group"] for row, _ in items}) for key, items in groups.items()},
          flush=True)
    selected = distinct_assignment(groups, dict.fromkeys(labels, FAMILY_TARGET), previous_groups)
    edited_readers = {row["group"] for row, _ in selected}
    edited_utterances = {native_utterance(row["id"]) for row, _ in selected}
    genuine = sorted(groups[None], key=lambda item: (item[0]["group"] in edited_readers, item[0]["group"] in previous_groups,
                                                     prep.rank("prospective-ordinary", item[0]["id"])))
    ordinary_readers = set()
    for row, values in genuine:
        if row["group"] in ordinary_readers or native_utterance(row["id"]) in edited_utterances:
            continue
        selected.append((row, values))
        ordinary_readers.add(row["group"])
        if len(ordinary_readers) == PART_A_ORIGINALS:
            break
    already_selected = used | {row["group"] for row, _ in selected}
    selected.extend(serving.librispeech_controls(args.librispeech, pins, previous_readers, previous_groups, already_selected,
                                                 count_limit=ORDINARY_COUNT - len(ordinary_readers),
                                                 admit=lambda key: native_utterance(key) not in seen_utterances, subset="dev-other"))
    assert len({native_utterance(row["id"]) for row, _ in selected}) == len(selected)
    return [(row | {"prior_held_reader": row["group"] in previous_readers}, pcm) for row, pcm in selected], archive, historical


def prepare(args):
    development, retained, older = [read(path) for path in (args.development, args.retained, args.older)]
    selected, archives, histories = roster(args, development, retained, older)
    room_plan = read(args.rooms / "rooms.json")
    room_names = [row["room"] for row in room_plan["rooms"]]
    previous_rooms = {row["room"] for row in retained["rooms"]["rooms"]} | {
        room for group in development["room_folds"].values() for room in group
    }
    if set(room_names) & previous_rooms or not previous_rooms <= set(room_plan["participant_excluded_meetings"]):
        raise ValueError("The background rooms overlap prior fit or transfer rooms")
    backgrounds = {}
    for room in room_plan["rooms"]:
        raw = (args.rooms / f"{room['room']}.pcm").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == room["pcm_sha256"] and len(raw) == 2 * 14 * prep.RATE
        backgrounds[room["room"]] = np.frombuffer(raw, "<i2")
    args.output.mkdir(mode=0o700)
    (args.output / "pcm").mkdir()
    cases, origins = [], []

    def keep(source, values, condition, kind, room):
        raw = values.astype("<i2").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        (args.output / "pcm" / f"{digest}.pcm").write_bytes(raw)
        cases.append({"id": f"{source['id']}/{kind}/{condition}", "group": source["group"], "condition": condition,
                      "kind": kind, "family": source["family"] if kind == "edited" else None,
                      "truth": source["edits"] if kind == "edited" else [], "samples": len(values), "pcm_sha256": digest,
                      "room": room, "corpus": source["corpus"]})

    for index, (source, raw) in enumerate(sorted(selected, key=lambda item: prep.rank("room-assignment", item[0]["id"]))):
        signal = np.frombuffer(raw, "<i2")
        edited = bool(source["edits"])
        room = room_names[index % len(room_names)]
        origins.append(source | {"room": room})
        kind = "edited" if edited else "ordinary"
        if edited:
            variants = signal[None, :]
        else:
            start, extent = centered_views(len(signal))["short"]
            a = start + extent // 2
            variants = np.asarray([signal, ordinary_edit(signal, (a, a + 8000))])
        keep(source, signal, "clean", kind, None)
        if not edited:
            keep(source, variants[1], "clean", "ordinary_processed", None)
        for db in LEVELS:
            mixture, isolated = mix(variants, backgrounds[room][:len(signal)], db)
            condition = f"overlap_{db}dB"
            keep(source, mixture[0], condition, kind, room)
            if not edited:
                keep(source, mixture[1], condition, "ordinary_processed", room)
            keep(source, isolated, condition, "background", room)
    plan = {"schema": SCHEMA, "input_sha256": {path.name + f"/{i}": history._sha(path)
                                               for i, path in enumerate((args.development, args.retained, args.older))},
            "rooms": room_plan, "archive_ranges": archives, "earlier_commitments": histories,
            "origins": origins, "cases": cases}
    save(args.output / "freeze.json", plan)
    print(f"Froze {len(cases)} deliveries from {len(origins)} source recordings, "
          f"{len({row['group'] for row in origins})} readers and {len(room_names)} rooms", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "retained", "older", "ranges", "labels_b", "part_a", "labels_a", "librispeech", "rooms", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--extra-ranges", type=Path)
    prepare(parser.parse_args())
