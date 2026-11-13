"""Test overlapping six-second clean-reference checks on longer native recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.evidence import audio_clean_speech
from scripts import audio_clean_reference_transfer as short
from scripts import audio_phoneme_short_edit_intake as intake
from scripts.audio_overlap_pitch_transfer import canonical, read, require, useful

RATE = audio_clean_speech.RATE
SIZE = 12 * RATE
OFFSET = 19200
WINDOWS = tuple(range(0, SIZE - audio_clean_speech.SAMPLES + 1, OFFSET))
SCHEMA = "chipmunk.audio-clean-reference-duration/1"
LEVELS = (0, -5, -10)
PHASES = ("early", "middle", "late")
TARGETS = {"early": (28800, 54400), "middle": (78400, 113600), "late": (137600, 163200)}
POSITIVE = {"synthetic_tts", "synthetic_conversion", "ordinary_replacement"}
WRONG = {"wrong_reader", "clock_shifted", "partial_match"}
METHOD = {
    "input": "Exactly twelve seconds of continuous native aligned mono signed16 at 16 kHz, with a separately supplied earlier clean copy. Use six overlapping fixed six-second windows starting every 1.20 seconds. Keep the signed six-second calculation, match criteria and numeric point frozen.",
    "rule": "Require at least four of six accepted matches, including the first and last window. A reportable pair of six-second window results must come from adjacent windows and have at least half of the 720 ms interval in common. Select the stronger interval from the pair ranked by the smaller response and then earlier range. Compare also the best single flag on the same grid and the first six seconds of the original lane.",
    "sources": "Use the short-reference study's frozen source role identities, with twelve-second continuous source segments chosen by sample clock, exact publisher word alignment, and hash ranking. One early, middle and late edit per eligible reader where available. Competing speech comes from the role-disjoint recorded AMI sessions used in the short-reference study with independently selected continuous twelve-second excerpts. Exclude exact six-second prior cuts. Keep wrong-reference readers in the same role.",
    "controls": "Compare unchanged, local gain, ordinary same-speech resampling, whole-file EQ and one different natural passage from the same reader. Refuse wrong reader, same-source 320 ms clock shift, and an earlier recording sharing only one six-second half. Real and synthetic replacements both count as physical-change positives. Mix at 0, -5 and -10 dB with identical background and gain per prompt.",
    "gate": "At -5 dB report at least four useful synthetic locations from at least two assessment readers and two zones, with no more than two flagged unchanged/gain/resampling/EQ controls in that condition, no wrong-reference admissions at any level, and no misplaced held positives at -5 dB. Use 0.30 IoU and 0.50 edit recall. Maintain distinct missing-match coverage. No confidence impact.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ranked(value):
    return digest(f"{SCHEMA}:{value}".encode())


def donors(args):
    old, ancestry = short.pair.ancestry(args)
    assets, roles, range_pins = short.select(args, old, ancestry)
    people = {row["source_key"].split("-")[0]: (row["case"], row["role"]) for row in roles}
    keys = set.intersection(*(set(values) for values in assets.values()))
    keys = {key for key in keys if key.split("-")[0] in people and len(assets["real"][key][2]) >= SIZE}
    alignments = intake.grid_inputs(args, keys, assets)
    history = read(short.audio.ROOT / "backend/data/independent-evaluation/audio-v9/source-plan.json")
    denied = {value.removeprefix("librispeech:") for row in history["sources"]
              for field in ("source_key", "source_group") if (value := row.get(field) or "").startswith("librispeech:")}
    opportunities = defaultdict(list)
    for key in sorted(keys - denied):
        if set(alignments[key]) != set(intake.ORIGINS):
            continue
        group, role = people[key.split("-")[0]]
        for left, a, b, bounds, _, _ in intake.possibilities(alignments[key], assets, key):
            start, stop = left + a, left + b
            for zone, (minimum, maximum) in TARGETS.items():
                middle = (start + stop - minimum - maximum) // 2
                offset = max(0, min(len(assets["real"][key][2]) - SIZE, middle))
                a, b = start - offset, stop - offset
                if a < minimum or b > maximum:
                    continue
                opportunities[(group, role, zone)].append((key, offset, a, b, bounds))
    selected, used = [], set()
    for (group, role, zone), possible in sorted(opportunities.items()):
        choices = sorted(possible, key=lambda row: ranked(f"{group}:{zone}:{row[:4]}"))
        chosen = next((row for row in choices if row[0] not in used), None)
        if chosen is None:
            continue
        key, offset, a, b, bounds = chosen
        used.add(key)
        selected.append({"group": group, "case": f"{group}-{zone}", "zone": zone, "role": role, "key": key, "offset": offset,
                         "truth": [a, b], "donors": bounds})
    require(len({row["group"] for row in selected if row["role"] == "assessment"}) == 3, "Need all three assessment readers")
    require({row["zone"] for row in selected if row["role"] == "assessment"} == set(PHASES), "Missing held edit zone")
    require(any(row["role"] == "development" for row in selected), "No development source")
    return selected, assets, roles, range_pins


def rooms(args, roster, previous):
    _, reserved = short.backgrounds(args, previous)
    archive_path = args.ami / "ami_public_manual_1.6.2.zip"
    settings = read(short.audio.ROOT / "docs/evaluation-panels/audio-gain-public-replay-2026-11-13.json")
    captures, pins = {}, {}
    with ZipFile(archive_path) as archive:
        for role, names in short.RIG.items():
            rows = [item for item in roster if item["role"] == role]
            for index, row in enumerate(rows):
                meeting = names[index % len(names)]
                name = f"{meeting}.Array1-01.wav"
                path = args.gain_ami / name
                if name not in pins:
                    expected = settings["sources"][name]["sha256"]
                    require(short.audio.file_digest(path) == expected, "AMI recording changed")
                    blocked = [item["start"] // 160 for item in reserved.get(meeting, {}).get("excerpts", [])]
                    pins[name] = {"sha256": expected, "reserved": [(start, start + 600) for start in blocked], "used": []}
                a, b = row["truth"]
                tracks = [short.audio.ami.track_words(archive.read(f"words/{meeting}.{letter}.words.xml")) for letter in "ABCD"]
                n = max(end for track in tracks for _, end in track)
                active = np.zeros(n, dtype=np.int16)
                for track in tracks:
                    occupied = np.zeros(n, dtype=np.int16)
                    for start, end in track:
                        occupied[start:end] = 1
                    active += occupied
                with wave.open(str(path), "rb") as source:
                    require(source.getparams()[:3] == (1, 2, RATE), "AMI PCM clock changed")
                    candidates = []
                    for start in range(500, min(n, source.getnframes() // 160) - 1200, 25):
                        if any(left < start + 1200 and right > start for left, right in pins[name]["reserved"]):
                            continue
                        view = active[start:start + 1200]
                        overlap = int(np.count_nonzero(view[(a + 159) // 160:b // 160] >= 2))
                        if overlap >= 25 and np.count_nonzero(view) >= 870:
                            candidates.append((start, overlap))
                    require(candidates, f"No continuous overlapping room excerpt for {row['case']}")
                    tick, overlap = min(candidates, key=lambda candidate: ranked(f"{row['case']}:{name}:{candidate[0]}"))
                    source.setpos(tick * 160)
                    raw = source.readframes(SIZE)
                require(len(raw) == 2 * SIZE, "Room excerpt truncated")
                pins[name]["reserved"].append((tick, tick + 1200))
                pins[name]["used"].append({"case": row["case"], "offset_samples": tick * 160, "overlap_cs": overlap, "sha256": digest(raw)})
                captures[row["case"]] = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return captures, pins


def render(args):
    roster, assets, previous, publisher = donors(args)
    captures, pins = rooms(args, roster, previous)
    dry = {row["case"]: assets["real"][row["key"]][2][row["offset"]:row["offset"] + SIZE].astype(np.float64) for row in roster}
    rows, audio = [], {}
    for row in roster:
        case, key = row["case"], row["key"]
        reference = dry[case]
        other = next((dry[other["case"]] for other in roster if other["group"] != row["group"] and other["role"] == row["role"]), None)
        a, b = row["truth"]
        words = reference[a:b]
        span = b - a
        donor_start = 0 if a >= span + 8000 else SIZE - span
        natural = short.pair.normed(reference[donor_start:donor_start + span], words)
        transformed = {"untouched": reference, "ordinary_replacement": intake.replace(reference, natural, a, b),
                       "local_gain": intake.replace(reference, words * 1.4, a, b)}
        equalized = sosfiltfilt(butter(2, 2450, fs=RATE, btype="lowpass", output="sos"), reference)
        transformed["global_eq"] = short.pair.normed(equalized, reference)
        for family, origin in zip(intake.FAMILIES, intake.ORIGINS[1:], strict=True):
            first, end = row["donors"][origin]
            donor = assets[origin][key][2][first:end].astype(np.float64)
            resampled = short.pair.normed(intake.resample(donor, span), words)
            ordinary = short.pair.normed(intake.resample(intake.resample(words, len(donor)), span), words)
            transformed[f"synthetic_{family}"] = intake.replace(reference, resampled, a, b)
            transformed[f"ordinary_{family}"] = intake.replace(reference, ordinary, a, b)
        half = SIZE // 2
        refs = {"clock_shifted": np.r_[reference[5120:], np.zeros(5120)]}
        if other is not None:
            other = short.pair.normed(other, reference)
            refs.update(wrong_reader=other, partial_match=np.r_[reference[:half], other[half:]])
        noise = short.pair.normed(captures[case], reference, -10)
        maximum = max(max(float(np.max(abs(v) + abs(noise))) for v in transformed.values()),
                      max(float(np.max(abs(v))) for v in refs.values()))
        gain = 24000 / maximum
        for level in LEVELS:
            noise_at_level = noise * 10 ** (-(level + 10) / 20)
            for kind in sorted(set(transformed) | set(refs)):
                first = short.pair.quantized(refs.get(kind, reference) * gain)
                last = short.pair.quantized((transformed.get(kind, reference) + noise_at_level) * gain)
                label = f"{case}_{level}_{kind}"
                rows.append({"case": case, "group": row["group"], "zone": row["zone"], "role": row["role"], "level": level,
                             "kind": kind, "truth": row["truth"] if kind in POSITIVE else None,
                             "earlier_sha256": digest(first.tobytes()), "submitted_sha256": digest(last.tobytes()), "id": label})
                audio[label] = first, last
    return rows, audio, {"publisher_prefixes": publisher, "room_pins": pins, "source_roster_sha256": digest(canonical(roster))}


def compare(first, second):
    values = []
    for start in WINDOWS:
        end = start + audio_clean_speech.SAMPLES
        reason, result = audio_clean_speech.locate(first[start:end], second[start:end])
        if result:
            result = {**result, "samples": [x + start for x in result["samples"]]}
        values.append({"start": start, "reason": reason, "result": result})
    matched = [r["reason"] != "reference_match_not_supported" for r in values]
    admitted = sum(matched) >= 4 and matched[0] and matched[-1]
    available = [r["result"] for r in values if r["result"]]
    first_only = values[0]["result"]
    any_window = max(available, key=lambda r: r["relative_drop"], default=None) if admitted else None
    witnessed = []
    if admitted:
        for left, right in zip(values[:-1], values[1:], strict=True):
            a, b = left["result"], right["result"]
            if a and b and abs(a["samples"][0] - b["samples"][0]) <= audio_clean_speech.RANGE_SAMPLES // 2:
                strongest = a if a["relative_drop"] >= b["relative_drop"] else b
                witnessed.append((min(a["relative_drop"], b["relative_drop"]), strongest, [left["start"], right["start"]]))
    best = max(witnessed, key=lambda entry: (entry[0], -entry[1]["samples"][0]), default=None)
    return {"windows": values, "admitted": admitted, "first_only": first_only, "single_window": any_window,
            "corroborated": None if best is None else best[1] | {"supporting_window_starts": best[2]}}


def report(rows):
    output = {}
    for role in ("development", "assessment"):
        output[role] = {}
        for level in LEVELS:
            output[role][str(level)] = {}
            selected = [row for row in rows if row["role"] == role and row["level"] == level]
            for arm in ("first_only", "single_window", "corroborated"):
                counts, groups, zones = Counter(), set(), set()
                for row in selected:
                    kind, data = row["kind"], row["comparison"]
                    found = data[arm]
                    counts[f"planned_{kind}"] += 1
                    eligible = data["windows"][0]["reason"] != "reference_match_not_supported" if arm == "first_only" else data["admitted"]
                    counts[f"admitted_{kind}"] += int(eligible)
                    if found is None:
                        continue
                    hit = row["truth"] is not None and useful(found["samples"], row["truth"])
                    counts[f"flag_{kind}"] += 1
                    if hit:
                        counts[f"localized_{kind}"] += 1
                        if kind.startswith("synthetic"):
                            groups.add(row["group"])
                            zones.add(row["zone"])
                    elif row["truth"] is not None:
                        counts["misplaced_positive"] += 1
                output[role][str(level)][arm] = {"counts": dict(counts), "synthetic_reader_groups": len(groups), "synthetic_edit_zones": len(zones)}
    held = output["assessment"]["-5"]["corroborated"]
    c = held["counts"]
    output["gate_pass"] = (sum(c.get(f"localized_synthetic_{k}", 0) for k in intake.FAMILIES) >= 4
                           and held["synthetic_reader_groups"] >= 2 and held["synthetic_edit_zones"] >= 2
                           and sum(c.get(f"flag_{kind}", 0) for kind in ("untouched", "local_gain", "ordinary_tts",
                                                                         "ordinary_conversion", "global_eq")) <= 2
                           and c.get("misplaced_positive", 0) == 0
                           and all(all(output["assessment"][str(level)]["corroborated"]["counts"].get(f"admitted_{kind}", 0) == 0
                                       for kind in WRONG) for level in LEVELS))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "gain-ami", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    rows, signals, source = render(args)
    freeze = {"schema": SCHEMA, "method": METHOD, "size": SIZE, "clock_hz": RATE, "window_starts": WINDOWS,
              "point": audio_clean_speech.POINT, "runner_sha256": digest(Path(__file__).read_bytes()),
              "serving_lane_sha256": digest(Path(audio_clean_speech.__file__).read_bytes()), "sources": source,
              "inputs_sha256": digest(canonical(rows))}
    pin = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        pin.write_bytes(canonical(freeze))
        (args.run / "roster.json").write_bytes(canonical(rows))
        print(f"Locked {len(rows)} recordings across {len({row['case'] for row in rows})} prompts before scoring")
        return
    require(pin.read_bytes() == canonical(freeze) and (args.run / "roster.json").read_bytes() == canonical(rows), "Frozen input differs")
    data = [row | {"comparison": compare(*signals[row["id"]])} for row in rows]
    result = {"freeze": freeze, "rows": data, "summary": report(data)}
    output = args.run / "result.json"
    if args.phase == "score":
        require(not output.exists(), "Held data already scored")
        output.write_bytes(canonical(result))
    else:
        require(output.read_bytes() == canonical(result), "Native replay differs")
    print(json.dumps(result["summary"]["gate_pass"]))


if __name__ == "__main__":
    main()
