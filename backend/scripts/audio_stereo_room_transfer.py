# SPDX-License-Identifier: Apache-2.0
"""Evaluate earlier-copy comparison with real two-microphone room interference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_clean_speech_stereo_long as stereo

old = stereo.old
SAMPLES, RATE = stereo.prior.SAMPLES, 16_000
MICS = ("Array1-01", "Array1-05")
SOURCES = {
    "ES2011a": ("eef59d96abe292a2b421b39d8ea8fea250210a9028bfe92758d58e51d18f9f7a",
                "409bd94990397de8706f8f209d01935b532c57323010f178928f2c262b1be455"),
    "IS1004a": ("94cbb0f0cadfe3d8d4d4bfd699d1e40f5a66cece979de24718721ebcc0f08f6a",
                "3f6d6e61e896c91c5f1ad90726ccc1f5283fa764156394f699750c3a18fc914a"),
}
ANNOTATIONS = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
SCHEMA = "chipmunk.audio-stereo-room-transfer/1"
LEVEL = -5
WEAK_CORROBORATION = .15
ARMS = ("downmix", "agreement", "reference_selected", "corroborated")
CONTROLS = ("background_reference", "unrelated", "half_matched", "unmixed", "background_only")
DIGEST = re.compile(r"[0-9a-f]{64}")
DIAGNOSTIC = "downmix_with_channel_support"
METHOD = {
    "clock": "Native AMI tracks at 16 kHz from two microphone positions. Add a dry foreground with a fixed "
             "0.55 amplitude on the quieter target side and matched overall −5 dB foreground-to-room ratio.",
    "holdout": "Retain two of every four foreground groups per bank with the earlier clip pairings. Use two "
               "new meeting backgrounds with nonoverlapping 12 s windows, covering the staged edit with AMI words.",
    "comparison": "Keep the frozen waveform check. Compare the downmix, channel agreement, reference-based "
                  "selection, and the same selection with at least .15 other-channel correlation on both halves.",
    "controls": "Use ordinary and resampled speech, wrong and half-matching target references, exact room "
                "references in the mixed inputs, room alone with its matching track, and room alone with target reference.",
}


def foreground_rows():
    previous = stereo.baseline()
    selected = []
    for bank in ("icsi", "ami"):
        groups = sorted({group for source, group, _kind in previous if source == bank})
        for index, group in enumerate(groups):
            if index % 4 < 2:
                original = previous[bank, group, "untouched"]
                selected.append({key: original[key] for key in ("bank", "group", "companion", "half", "earlier_sha256")})
    return previous, selected


def read_rooms(folder):
    from contextlib import ExitStack

    import numpy as np

    tracks = {}
    for meeting, pins in SOURCES.items():
        channels = []
        with ExitStack() as stack:
            for mic, pin in zip(MICS, pins, strict=True):
                path = folder / f"{meeting}.{mic}.wav"
                with path.open("rb") as f:
                    old.require(hashlib.file_digest(f, "sha256").hexdigest() == pin, "AMI source bytes changed")
                source = stack.enter_context(wave.open(str(path), "rb"))
                old.require(source.getparams()[:3] == (1, 2, RATE) and source.getcomptype() == "NONE",
                            "AMI microphone PCM changed")
                channels.append(np.frombuffer(source.readframes(source.getnframes()), dtype="<i2"))
        old.require(len(channels[0]) == len(channels[1]), "AMI microphone clocks differ")
        tracks[meeting] = channels
    return tracks


def plan(annotations, rooms):
    from zipfile import ZipFile

    import numpy as np

    from scripts.prepare_natural_overlap_transfer import track_words

    previous, rows = foreground_rows()
    with annotations.open("rb") as f:
        old.require(hashlib.file_digest(f, "sha256").hexdigest() == ANNOTATIONS, "AMI annotation archive differs")
    speech = {}
    with ZipFile(annotations) as archive:
        for name, channels in rooms.items():
            mask = np.zeros(len(channels[0]) // 160, dtype=bool)
            for speaker in "ABCD":
                for start, end in track_words(archive.read(f"words/{name}.{speaker}.words.xml")):
                    mask[start:end] = True
            speech[name] = mask
    used = defaultdict(set)
    for index, row in enumerate(rows):
        meeting = tuple(SOURCES)[index % len(SOURCES)]
        mask = speech[meeting]
        labels = [case["truth"] for (bank, group, kind), case in previous.items()
                  if (bank, group) == (row["bank"], row["group"]) and stereo.prior.category(kind) == "synthetic_edit"]
        offsets = tuple((a // 160, (b + 159) // 160) for a, b in labels)
        choices = [
            start for start in range(6000, 60000, 1200)
            if start not in used[meeting]
            and all(mask[start + shift:start + shift + 600].sum() >= 240 for shift in (0, 600))
            and all(2 * mask[start + a:start + b].sum() >= b - a for a, b in offsets)
        ]
        start = min(choices, key=lambda offset: old.sha(f"room-aac-1:{index}:{offset}".encode()))
        used[meeting].add(start)
        background = np.column_stack([voice[start * 160:start * 160 + SAMPLES] for voice in rooms[meeting]])
        row.update(meeting=meeting, start=start * 160, quiet_side="left" if index // 2 % 2 else "right",
                   observed_word_centiseconds=[int(mask[start + a:start + b].sum()) for a, b in offsets],
                   room_pcm_sha256=old.sha(background.astype("<i2").tobytes()))
    return {"sources": SOURCES, "annotations": ANNOTATIONS, "parameters": METHOD,
            "prior": old.sha(stereo.BASE.read_bytes()), "rows": rows}


def dry_inputs(icsi, ami):
    import numpy as np

    from scripts.audio_short_edit_detector_training import load_plan

    result = defaultdict(dict)
    frozen, digest = load_plan(icsi)
    old.require(digest == old.source_hash("icsi"), "ICSI sources changed")
    for row in frozen["rows"]:
        if row["role"] == "held" and row["condition"] == "dry":
            pin = row["pcm_sha256"]
            raw = (icsi / "pcm" / f"{pin}.pcm").read_bytes()
            old.require(len(raw) == SAMPLES and old.sha(raw) == pin, "ICSI dry excerpt changed")
            result["icsi", row["group"]][row["kind"]] = raw
    freeze_path = ami / "placements-freeze.json"
    old.require(old.sha(freeze_path.read_bytes()) == old.source_hash("ami"), "AMI foreground sources changed")
    with np.load(ami / "placements-native.npz", allow_pickle=False) as storage:
        for row, dry in zip(old.read(freeze_path)["native_manifest_rows"], storage["dry"], strict=True):
            if row["snr"] == -5:
                raw = dry.astype("<i2", copy=False).tobytes()
                old.require(len(raw) == SAMPLES and old.sha(raw) == row["dry_pcm_sha256"], "AMI dry excerpt changed")
                result["ami", row["group"]][row["kind"]] = raw
    return result


def pcm16(x):
    import numpy as np

    return np.clip(np.rint(x), -32768, 32767).astype("<i2").tobytes()


def materialize(row, rooms, dry):
    import numpy as np

    start, bank, group = row["start"], row["bank"], row["group"]
    bed = np.column_stack([c[start:start + SAMPLES] for c in rooms[row["meeting"]]]).astype(np.float64)
    foregrounds = dry[bank, group]
    companion = dry[bank, row["companion"]]["untouched"]

    def joined(target):
        return target + companion if row["half"] == "early" else companion + target

    earlier = joined(foregrounds["untouched"])
    old.require(old.sha(earlier) == row["earlier_sha256"], "The earlier foreground changed")
    gain = np.array([.55, 1.]) if row["quiet_side"] == "left" else np.array([1., .55])
    clean = {kind: np.frombuffer(joined(pcm), dtype="<i2")[:, None] * gain for kind, pcm in foregrounds.items()}
    bed *= np.linalg.norm(clean["untouched"]) * 10 ** (-LEVEL / 20) / max(1., np.linalg.norm(bed))
    scale = 32500 / max(float(np.max(np.abs(pcm + bed))) for pcm in clean.values())
    scale = min(1., scale)
    inputs = {kind: (earlier, pcm16((pcm + bed) * scale)) for kind, pcm in clean.items()}
    background = pcm16(bed * scale)
    bed_reference = pcm16(bed[:, 0] * scale)
    alternate = next(key for key, source in dry.items() if key[0] == bank and key[1] not in {group, row["companion"]}
                     and source["untouched"] not in {foregrounds["untouched"], companion})
    other_target = dry[alternate]["untouched"]
    inputs.update(background_reference=(bed_reference, inputs["untouched"][1]),
                  unrelated=(other_target + other_target, inputs["untouched"][1]),
                  half_matched=(joined(other_target), inputs["untouched"][1]),
                  unmixed=(bed_reference, background), background_only=(earlier, background))
    return inputs


def choose(entry):
    pick = entry["selected"]["reference_selected"]
    if pick in ("left", "right"):
        other = "right" if pick == "left" else "left"
        if any((part["upper_correlation"] or 0.) < WEAK_CORROBORATION for part in entry["support"][other]):
            return entry["selected"]["agreement"]
    return pick


def category(kind):
    return kind if kind in CONTROLS else stereo.prior.category(kind)


def downmix_with_channel_support(entry):
    views = entry["readings"]
    downmix = views["downmix"]["range"]
    selected = views.get(choose(entry))
    channel = selected["range"] if selected else None
    return "downmix" if downmix and channel and stereo.overlap(downmix, channel) else None


def summary(rows, arms=ARMS):
    totals = defaultdict(Counter)
    for row in rows:
        measurement = row["measurement"]
        views = measurement["readings"]
        useful_before = old.location(views["downmix"]["range"], row["truth"])
        choices = measurement["selected"] | {"downmix": "downmix", "corroborated": choose(measurement)}
        if DIAGNOSTIC in arms:
            choices[DIAGNOSTIC] = downmix_with_channel_support(measurement)
        for arm in arms:
            result = views.get(choices[arm])
            span = result["range"] if result else None
            useful = old.location(span, row["truth"])
            for location in ("all", row["meeting"]):
                totals[f"{location}/{arm}/{category(row['kind'])}"].update(
                    files=1, listed=bool(result), admitted=bool(result and result["reason"] != "reference_match_not_supported"),
                    flagged=bool(span), useful=useful, gained=useful and not useful_before, lost=useful_before and not useful,
                    late=useful and row["half"] == "late",
                )
    return dict(totals)


def native_check(plan_data, rooms, icsi, ami):
    sources = dry_inputs(icsi, ami)
    previous = stereo.baseline()
    rows = []
    with TemporaryDirectory(prefix="chipmunk-room-aac-") as folder:
        path = Path(folder) / "capture.m4a"
        for source in plan_data["rows"]:
            bank, group = source["bank"], source["group"]
            pairs = materialize(source, rooms, sources)
            untouched = pairs["untouched"][1]
            for kind, (reference, mixture) in pairs.items():
                truth = None if kind in CONTROLS else previous[bank, group, kind]["truth"]
                if truth is not None:
                    a, b = (4 * tick for tick in truth)
                    old.require(mixture[:a] == untouched[:a] and mixture[b:] == untouched[b:]
                                and mixture[a:b] != untouched[a:b], "The staged PCM edit differs from its source interval")
                views, clock = stereo.stereo_pcm(path, mixture)
                rows.append({**source, "kind": kind, "truth": truth, "mixture_sha256": old.sha(mixture),
                             "reference_sha256": old.sha(reference), "clock": clock, "measurement": stereo.check(reference, views)})
            print(f"Completed {source['meeting']} placement {source['start'] // RATE}s, {len(rows)} total.", flush=True)
    return {"schema": SCHEMA, "plan": plan_data, "rows": rows, "summary": summary(rows)}


def review(receipt):
    previous, admitted = foreground_rows()
    frozen = receipt["plan"]
    old.require(receipt["schema"] == SCHEMA and frozen["parameters"] == METHOD
                and frozen["annotations"] == ANNOTATIONS and frozen["prior"] == old.sha(stereo.BASE.read_bytes())
                and frozen["sources"] == {key: list(pins) for key, pins in SOURCES.items()},
                "The frozen test identity differs")
    roster = {(r["bank"], r["group"]): r for r in frozen["rows"]}
    for i, row in enumerate(admitted):
        held = roster[row["bank"], row["group"]]
        old.require(all(held[key] == value for key, value in row.items()) and held["meeting"] == tuple(SOURCES)[i % 2]
                    and held["quiet_side"] == ("left" if i // 2 % 2 else "right")
                    and held["start"] in range(60 * RATE, 600 * RATE, SAMPLES), "The source placement changed")
    offsets = {(r["meeting"], r["start"]) for r in roster.values()}
    old.require(len(offsets) == len(roster) == len(admitted) == len(frozen["rows"]) == 16
                and all(DIGEST.fullmatch(row["room_pcm_sha256"]) for row in roster.values()),
                "A meeting window is missing or reused")
    actual = {}
    for row in receipt["rows"]:
        bank, group, kind = row["bank"], row["group"], row["kind"]
        actual[bank, group, kind] = row
        held = roster[bank, group]
        truth = None if kind in CONTROLS else previous[bank, group, kind]["truth"]
        entry = row["measurement"]
        clock = row["clock"]
        old.require(all(row[key] == value for key, value in held.items()) and row["truth"] == truth
                    and clock["native_clock"] == stereo.CLOCK and 0 <= clock["decoded_samples"] - SAMPLES <= 342
                    and all(DIGEST.fullmatch(pinned) for pinned in
                            (row["mixture_sha256"], row["reference_sha256"], clock["encoded_sha256"]))
                    and entry["readings"].keys() == {"downmix", "left", "right"} and entry["support"].keys() == {"left", "right"}
                    and entry["selected"] == stereo.select(entry["readings"], entry["support"]),
                    "A comparison lost its source selection or playback trace")
        for reading in entry["readings"].values():
            span = reading["range"]
            old.require(reading["reason"] in stereo.REASONS and DIGEST.fullmatch(reading["pcm_sha256"])
                        and (span is not None) == (reading["reason"] == "local_change")
                        and (span is None or (0 <= span[0] < span[1] <= SAMPLES and span[1] - span[0] == stereo.RANGE_SAMPLES)),
                        "A measured range is outside the evidence clock")
    expected = {(bank, group, kind) for bank, group in roster for kind in CONTROLS}
    expected.update((bank, group, kind) for bank, group, kind in previous if (bank, group) in roster
                    and kind not in CONTROLS and kind != "background")
    old.require(actual.keys() == expected and len(receipt["rows"]) == len(expected), "A comparison is missing or duplicated")
    for bank, group in roster:
        for base, controls in (("untouched", ("background_reference", "unrelated", "half_matched")),
                               ("unmixed", ("background_only",))):
            anchor = actual[bank, group, base]
            for control in controls:
                matched = actual[bank, group, control]
                old.require(all(anchor[key] == matched[key] for key in ("mixture_sha256", "clock")),
                            "Changing the reference also changed the submitted recording")
        exact_room = actual[bank, group, "background_reference"]["reference_sha256"]
        old.require(exact_room == actual[bank, group, "unmixed"]["reference_sha256"], "The room reference changed")
    observed = summary(receipt["rows"])
    old.require(observed == receipt["summary"], "The measured counts differ")
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("plan_or_receipt", type=Path)
    parser.add_argument("--rooms", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--icsi", type=Path)
    parser.add_argument("--ami", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--diagnostic", action="store_true", help="Review the later downmix and channel range agreement probe")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = json.loads(gzip.decompress(args.plan_or_receipt.read_bytes()))
        output = review(receipt)
        if args.diagnostic:
            output = summary(receipt["rows"], (DIAGNOSTIC,))
        print(json.dumps(output, indent=2, sort_keys=True))
        return
    if not args.rooms or not args.annotations:
        parser.error("Native comparison requires --rooms and --annotations")
    if args.phase == "run" and not (args.icsi and args.ami and args.receipt):
        parser.error("Native scoring requires --icsi, --ami and --receipt")
    rooms = read_rooms(args.rooms)
    frozen = plan(args.annotations, rooms)
    encoded = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    if args.phase == "freeze":
        with args.plan_or_receipt.open("xb") as f:
            f.write(encoded)
        print(old.sha(encoded))
        return
    old.require(args.plan_or_receipt.read_bytes() == encoded, "A prospective room plan is required")
    receipt = native_check(frozen, rooms, args.icsi, args.ami)
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    review(json.loads(raw))
    with args.receipt.open("xb") as f:
        f.write(gzip.compress(raw, mtime=0))


if __name__ == "__main__":
    main()
