# SPDX-License-Identifier: MIT
"""Build matched training and held speech edits with published word clocks."""

from __future__ import annotations

import argparse
import wave
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts import audio_nested_voice_window_transfer as parent

SOURCE = parent.SOURCE
RATE, SAMPLES = parent.RATE, parent.SAMPLES
FAMILIES = parent.ROLES[1:]
ROLES = {"fit": 6, "calibration": 2, "held": 3}
PROMPTS = 8
CONDITIONS = ("dry", "competing_speech")
KINDS = ("untouched", "genuine_splice", *(f"{kind}/{family}" for family in FAMILIES
                                        for kind in ("resampled", "edited")), "background")
SCHEMA = "chipmunk.audio-short-edit-training-sources/1"
METHOD = {
    "sources": "Select eight distinct aligned utterances per declared reader, with four short and four longer edits. "
               "Exclude foreground utterances used in the nested AMI/ICSI training follow-up and exact historical v9 keys. "
               "Partition readers by hash before measuring any classifier. Retain the same word sequence in both TTS renditions.",
    "background": "Select source-clock AMI excerpts for fitting and calibration from two different sessions. "
                  "Prefer disjoint excerpts within AMI sessions. Use the three frozen ICSI rooms for held trials, "
                  "balance excerpt reuse and align the held foreground crop to annotated overlap. "
                  "Require annotated overlapping background speech over at least half the edit.",
    "variants": "Use matched untouched, same-reader genuine word substitution, ordinary same-word resampling for each "
                "generator, generated same-word substitution for each generator, and isolated-background controls. "
                "Apply the same 20 ms fades and local RMS matching for all edits. Whole-host mixture ratio is -5 dB. "
                "Dry and wet inputs use the same PCM target with exact shared integer background addition.",
}


def order(value):
    return SOURCE.digest(f"{SCHEMA}/{value}".encode())


def foregrounds(arrays, labels, history):
    from scripts import review_audio_melo_short_sources as census

    previous = SOURCE.read(parent.AMI_PARENT)["freeze"]["plan"]["sources"]
    consumed = {row["bindings"]["real"]["wav_sha256"] for row in previous}
    older = SOURCE.read(history / census.HISTORY["v9"][0])["sources"]
    old_keys = {value.removeprefix("real:").removeprefix("librispeech:")
                for row in older for name in ("source_key", "source_group", "id") if (value := row.get(name))}
    candidates = defaultdict(lambda: ([], []))
    for key, tiers in labels.items():
        recording = arrays["real"][key][0]
        if recording in consumed or key in old_keys:
            continue
        reader = key.split("-", 1)[0]
        options = [[], []]
        for option in parent._ranges(tiers, arrays, key):
            a, b = option["truth"]
            duration = b - a
            # A genuine substitution is taken from the same utterance, never from the held vocabulary labels.
            native_start, native_stop = option["native"]["real"]
            dry = arrays["real"][key][2]
            alternate = [start for start in (native_start - duration - 800, native_stop + 800)
                         if 0 <= start <= len(dry) - duration]
            if alternate:
                donor = max(alternate, key=lambda start: int(abs(dry[start:start + duration].astype("int32")).sum()))
                options[int(duration >= 9600)].append((order((key, option)), option | {"donor": donor}))
        for band, choices in enumerate(options):
            if choices:
                rank, chosen = min(choices)
                candidates[reader][band].append((rank, key, chosen))
    eligible = {}
    for reader, by_size in candidates.items():
        selected = []
        used = set()
        for options in by_size:
            available = [entry for entry in sorted(options) if entry[1] not in used]
            block = available[:PROMPTS // 2]
            used.update(item[1] for item in block)
            selected.extend(block)
        if len(selected) == PROMPTS:
            eligible[reader] = selected
    readers = sorted(eligible, key=order)
    SOURCE.check(len(readers) >= sum(ROLES.values()), f"Only {len(readers)} readers have a balanced roster")
    selected = []
    for role, reader_count in ROLES.items():
        members, readers = readers[:reader_count], readers[reader_count:]
        for reader in members:
            for _, key, option in eligible[reader]:
                selected.append(option | {"role": role, "group": len(selected), "key": key,
                                          "reader_sha256": order(reader)})
    return selected


def ami_audio(groups):
    import numpy as np

    from scripts.audio_harmonic_overlap_transfer import room_activity
    from scripts.audio_locate_competing_speech import ANNOTATIONS, PANEL

    SOURCE.check(SOURCE.path_sha(ANNOTATIONS) == SOURCE.read(PANEL)["freeze"]["annotations_sha256"], "AMI annotations differ")
    previous = SOURCE.read(parent.AMI_PARENT)["freeze"]["plan"]["sources"]
    with ZipFile(ANNOTATIONS) as annotations:
        for role, (room, capture_sha) in zip(("fit", "calibration"), parent.ROOMS.items(), strict=True):
            prior = {r["background"]["start_sample"] for r in previous if r["background"]["room"] == room}
            uses = Counter()
            activity = room_activity(annotations, room)
            path = ANNOTATIONS.parent / f"{room}.Array1-01.wav"
            SOURCE.check(SOURCE.path_sha(path) == capture_sha, "AMI source differs")
            with wave.open(str(path), "rb") as wav:
                SOURCE.check(wav.getparams()[:3] == (1, 2, RATE), "AMI audio profile differs")
                possibilities = [(tick * 160, activity[tick:tick+600])
                                 for tick in range(900, min(len(activity), wav.getnframes() // 160) - 600, 100)
                                 if np.count_nonzero(activity[tick:tick+600] >= 2) >= 120]
                for src in (src for src in groups if src["role"] == role):
                    a, b = src["truth"]
                    viable = []
                    for start, active in possibilities:
                        covered = 160 * int(np.count_nonzero(active[(a+159)//160:b//160] >= 2))
                        if 2 * covered >= b-a:
                            overlaps = sum(abs(start - taken) < SAMPLES for taken in prior | uses.keys())
                            viable.append((overlaps, uses[start], order((src["key"], start)), start, covered))
                    SOURCE.check(bool(viable), f"No annotated AMI range covers {role} group {src['group']}")
                    _, _, _, start, covered = min(viable)
                    uses[start] += 1
                    wav.setpos(start)
                    raw = wav.readframes(SAMPLES)
                    SOURCE.check(len(raw) == 2*SAMPLES, "AMI background samples are truncated")
                    yield src["group"], raw, {"room": room, "start": start, "support": covered, "pcm_sha256": SOURCE.digest(raw)}


def icsi_audio(root, groups, arrays):
    from scripts import audio_icsi_overlap_intake as intake

    report = SOURCE.read(root / "receipt.json")
    plan = report["plan"]
    SOURCE.check(plan["revision"] == intake.REVISION and plan["schema"] == intake.SCHEMA
                 and plan["sample_rate"] == RATE and plan["sample_count"] == SAMPLES, "ICSI source differs")
    uses = Counter()
    for src in (src for src in groups if src["role"] == "held"):
        left, right = src["native"]["real"]
        length = len(arrays["real"][src["key"]][2])
        lower = max(0, right + RATE - SAMPLES)
        upper = min(length - SAMPLES, left - RATE)
        crops = range(((lower + 1599)//1600) * 1600, upper + 1, 1600)
        available = []
        for case in report["cases"]:
            if case["condition"] != "strong":
                continue
            key = case["room"], case["start_sample"]
            for crop in crops:
                a, b = left - crop, right - crop
                support = sum(max(0, min(b, hi) - max(a, lo)) for lo, hi in case["overlap_ranges"])
                if 2 * support >= b-a:
                    available.append((uses[key], abs(crop-src["crop"]), order((src["key"], key, crop)), crop, support, case))
        SOURCE.check(bool(available), f"No strong ICSI range overlaps group {src['group']}")
        _, _, _, crop, support, case = min(available)
        src.update(crop=crop, truth=[left-crop, right-crop])
        room, start = key = case["room"], case["start_sample"]
        uses[key] += 1
        raw = (root / f"{room}-{start}.pcm").read_bytes()
        SOURCE.check(len(raw) == 2*SAMPLES and SOURCE.digest(raw) == case["pcm_sha256"], "ICSI excerpt differs")
        yield src["group"], raw, {"room": room, "start": start, "support": support, "pcm_sha256": SOURCE.digest(raw)}


def materialize(args):
    import numpy as np

    from scripts import audio_phoneme_short_edit_intake as intake
    from scripts.audio_odss_matched_room import rms

    arrays, labels, pins = parent._inputs(args)
    groups = foregrounds(arrays, labels, args.history)
    backgrounds = {group: (raw, pin) for group, raw, pin in
                   (*ami_audio(groups), *icsi_audio(args.icsi, groups, arrays))}
    args.output.mkdir(mode=0o700)
    pcm = args.output / "pcm"
    pcm.mkdir(mode=0o700)
    rows, sources = [], []
    for group in groups:
        identifier, start, (a, b) = group["key"], group["crop"], group["truth"]
        foreground = arrays["real"][identifier][2].astype(np.float64)
        host = foreground[start:start+SAMPLES]
        word = host[a:b]
        donor = group["donor"]
        genuine = foreground[donor:donor+len(word)]
        materials = {"genuine_splice": genuine}
        for family in FAMILIES:
            i, j = group["native"][family]
            original = arrays[family][identifier][2][i:j].astype(np.float64)
            materials[f"resampled/{family}"] = intake.resample(intake.resample(word, len(original)), len(word))
            materials[f"edited/{family}"] = intake.resample(original, len(word))
        variants = {"untouched": host}
        for kind, part in materials.items():
            SOURCE.check(rms(part) > 0 and len(part) == len(word), "Invalid edit donor")
            variants[kind] = intake.replace(host, part * rms(word) / rms(part), a, b)
        raw, background = backgrounds[group["group"]]
        masker = np.frombuffer(raw, "<i2").astype(np.float64)
        masker *= rms(host) / rms(masker) * 10 ** .25
        gain = 26000 / max(float(np.max(abs(value) + abs(masker))) for value in variants.values())
        noise = np.rint(masker * gain).astype(np.int32)
        dry = {kind: np.rint(value * gain).astype(np.int32) for kind, value in variants.items()}
        for condition in CONDITIONS:
            types = KINDS[:-1] if condition == "dry" else KINDS
            for kind in types:
                truth = group["truth"] if kind.startswith("edited/") else None
                if kind == "background":
                    samples = noise
                else:
                    control = dry["untouched"]
                    samples = dry[kind]
                    SOURCE.check(np.array_equal(samples[:a], control[:a]) and np.array_equal(samples[b:], control[b:]),
                                 "An edit changed outside its clock range")
                    if condition == "competing_speech":
                        samples = samples + noise
                SOURCE.check(np.max(abs(samples)) <= 26001, "A delivered recording clips")
                data = samples.astype("<i2").tobytes()
                digest = SOURCE.digest(data)
                path = pcm / f"{digest}.pcm"
                if not path.exists():
                    path.write_bytes(data)
                rows.append({"group": group["group"], "role": group["role"], "condition": condition, "kind": kind,
                             "truth": truth, "pcm_sha256": digest})
        sources.append({key: value for key, value in group.items() if key != "key"} |
                       {"input_sha256": {family: arrays[family][identifier][0] for family in parent.ROLES},
                        "alignment_sha256": {family: labels[identifier][family][1] for family in parent.ROLES},
                        "background": background})
    plan = {"schema": SCHEMA, "method": METHOD, "archives": pins, "rows": rows, "sources": sources,
            "icsi_intake_sha256": SOURCE.path_sha(args.icsi / "receipt.json")}
    SOURCE.write(args.output / "sources.json", plan)
    print(f"Froze {len(sources)} matched prompts and {len(rows)} deliveries across {len(ROLES)} roles.", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    for name in ("melo", "aligned", "earlier", "icsi"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    cli.add_argument("--history", type=Path, default=parent.ROOT / "backend/data/independent-evaluation")
    materialize(cli.parse_args())


if __name__ == "__main__":
    main()
