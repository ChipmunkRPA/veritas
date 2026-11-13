# SPDX-License-Identifier: Apache-2.0
"""Recompute the sealed PartialEdit schedule comparison without models or media libraries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

SCHEMA = "chipmunk.audio-partialedit-dense-transfer/1"
RATE, WINDOW, STEP = 16_000, 48_000, 24_000
SHORT_LENGTH = 6 * RATE
SCENES = ("clean", "construction", "leicester")
FAMILIES = ("genuine", "E1", "E2")
ARMS = ("current", "dense", "combined")
STATES = {"ai", "original", "inconclusive", "unassessed"}
PIN = re.compile(r"[0-9a-f]{64}")
UTTERANCE = re.compile(r"(p\d{3})_\d{3}")
CALIBRATION = "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
INPUT_PINS = {
    "labels.csv": "adeecb0a7dd39a982223c07970e02e41ca5395ea3d484fcc6ce03174ebdf6cbc",
    "speakers.zip": "8be8f5b907163bcd886c4e487f173347a7972080d4e77bfed14ec66298edf086",
    "vctk-directory": "ff07294e20e89207c2620a5fb0ce8a404478268b80788ce29c6c9837ce2fafac",
    "E1.prefix": "39e84aa4fddb39585368f787788a8ea5f6ce8719b97213fdd24bc6df0c9a8466",
    "E1.more": "8177c5089fc771d5731616d9e6e5b35ad4d1ea94a5dba000441ea136da552889",
    "E2.prefix": "6ecfb5790a6fc3b326a957fed4820e63726b5b0aea26458085ff0f9cd03fde47",
    "E2.more": "8a5c365419981391420ab32558ca7d09ec27ac0d90f7e6389dce5ce868191465",
}
BACKGROUNDS = {
    "construction": ["7b69cda9671d635bd20ed17f1f7db728fbca3ba6a67cc7ce2512777a8e02534a", [5, 30, 55, 80], -5],
    "leicester": ["42d2f7a9989748fe3f190b39665e7519bc205b336b3237c0788e2f36c6bab707", [4, 21, 38, 55], 5],
}
VCTK_DIRECTORY_START = 11_734_391_133
VOICE_ASSETS = {
    "pellav2_detector.pt": "d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac",
    "w2v-bert-config.json": "f5572bd5998b68182e9c328a43127ed21fed687f6910497136b91a4e3b0e3675",
    "silero_vad.jit": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
    "panns-Cnn10.pth": "5240bdc47444e331bbeb54fd741d2b3f933a4526e84b0c17bf3593df94b13962",
    "class_labels_indices.csv": "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429",
}
METHOD = {
    "source": "Every common eligible utterance from the two frozen publisher prefixes and official eval speaker list.",
    "eligibility": "Mono 16 kHz WAV at 6–14 s, at least one edit 0.6–2.4 s, label duration tolerance 20 ms. Canonical PCM16.",
    "control": "The same utterance's official VCTK microphone 1 recording, decoded to 16 kHz at its own full length.",
    "comparison": "Preserve incumbent locations. Add only the shared 1.5 s of two adjacent AI-leaning 3 s responses.",
    "useful": "Cover half a known edit with at least 0.30 temporal intersection over union.",
    "min_speakers_for_confirmation": 8,
    "deployment": "Transfer pilot only. No automatic serving change or fit to this panel.",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pins(values):
    return all(isinstance(value, str) and PIN.fullmatch(value) for value in values)


def span(value):
    require(isinstance(value, list) and len(value) == 2 and all(type(n) is int for n in value),
            "An interval needs exact integer sample positions")
    return tuple(value)


def schedules(length):
    require(type(length) is int and 6 * RATE <= length <= 14 * RATE, "Speech clock outside six to fourteen seconds")
    last = length - WINDOW
    sparse = (0, last) if length < 9 * RATE else (0, last // 2, last)
    return tuple((a, a + WINDOW) for a in sparse), tuple((a, a + WINDOW) for a in range(0, last + 1, STEP))


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def useful(ranges, truth):
    for start, end in ranges:
        for a, b in truth:
            common = overlap((start, end), (a, b))
            if 2 * common >= b - a and 10 * common >= 3 * (end - start + b - a - common):
                return True
    return False


def covered_genuine(ranges, truth):
    # Sweep distinct endpoints to count each flagged original sample exactly once.
    edges = sorted({point for pair in ranges + truth for point in pair})
    return sum(b - a for a, b in zip(edges, edges[1:]) if any(lo <= a < hi for lo, hi in ranges)
               and not any(lo <= a < hi for lo, hi in truth))


def validate_roster(roster, lock):
    require(roster and len(roster) % 9 == 0, "Each source needs three recordings in three delivery conditions")
    exclusions = {}
    for row in lock["excluded_original_clocks"]:
        name, count = row["utterance"], row["samples"]
        require(set(row) == {"utterance", "samples"} and UTTERANCE.fullmatch(name) and name not in exclusions
                and type(count) is int and count > 0 and not 6 * RATE <= count <= 14 * RATE,
                "An excluded original does not have a distinct unsupported clock")
        exclusions[name] = count
    names = [row["utterance"] for row in roster[::9]]
    require(all(UTTERANCE.fullmatch(name) for name in names) and names == sorted(set(names)) and set(names).isdisjoint(exclusions),
            "Retained and excluded sources overlap or their ordering differs")
    rank = {name: index for index, name in enumerate(sorted([*names, *exclusions]))}
    source_keys, delivery_pins, ranges = set(), set(), set()
    for group, name in enumerate(names):
        for family_index, family in enumerate(FAMILIES):
            start = group * 9 + family_index * 3
            clean = roster[start]
            size = clean["samples"]
            schedules(size)
            truth = [span(item) for item in clean["truth"]]
            require(all(0 <= a < b <= size for a, b in truth) and all(a[1] < b[0] for a, b in zip(truth, truth[1:])),
                    "Publisher edit truth is outside its own sample clock or overlaps")
            require(not truth if family == "genuine" else 1 <= len(truth) <= 2
                    and any(9_599 <= b - a <= 38_401 for a, b in truth),
                    "Speech edit truth does not match the publisher family and duration scope")
            original = clean["vctk_range"]
            if family == "genuine":
                a, b = span(original)
                require(0 <= a < b < VCTK_DIRECTORY_START and 4_095 < b - a < 2**20 + 4_095,
                        "An original source is outside its frozen publisher ZIP range")
                require((a, b) not in ranges, "An original ZIP range was reused by another utterance")
                ranges.add((a, b))
            else:
                require(original is None, "An edited member claims an original ZIP range")
            require(clean["source_sha256"] not in source_keys, "A source file is shared across unrelated clips or edit families")
            source_keys.add(clean["source_sha256"])
            for scene_index, scene in enumerate(SCENES):
                index = start + scene_index
                item = roster[index]
                expected_offset = None if scene == "clean" else BACKGROUNDS[scene][1][rank[name] % 4] * RATE
                require(set(item) == {"id", "utterance", "family", "scene", "offset", "truth", "samples", "pcm_sha256",
                         "source_sha256", "vctk_range"} and type(item["id"]) is int and item["id"] == index
                        and (item["utterance"], item["family"], item["scene"]) == (name, family, scene)
                        and item["offset"] == expected_offset
                        and (expected_offset is None or type(item["offset"]) is int)
                        and all(item[k] == clean[k] for k in ("samples", "truth", "source_sha256", "vctk_range"))
                        and type(item["samples"]) is int and all(span(raw) == expected for raw, expected in zip(item["truth"], truth, strict=True))
                        and pins([item["pcm_sha256"], item["source_sha256"]]) and item["pcm_sha256"] not in delivery_pins,
                        "Paired scene source, sample count, truth, offset or delivery commitment differs")
                delivery_pins.add(item["pcm_sha256"])
    return names


def verify_publisher_truth(roster, exclusions, directory):
    labels = (directory / "labels.csv").read_bytes()
    speaker_archive = (directory / "speakers.zip").read_bytes()
    require(sha(labels) == INPUT_PINS["labels.csv"] and sha(speaker_archive) == INPUT_PINS["speakers.zip"],
            "Publisher labels or evaluation speakers differ from the original pins")
    with zipfile.ZipFile(io.BytesIO(speaker_archive)) as archive:
        speakers = set(archive.read("eval/eval.spk").decode().split())
    require(len(speakers) == 70, "Publisher evaluation roster differs")
    edited = {f"{row['family']}/{row['utterance'][:4]}/{row['utterance']}_edited_partial_16k.wav": row
              for row in roster if row["family"] != "genuine" and row["scene"] == "clean"}
    checked = set()
    count = 0
    for name, *values in csv.reader(io.StringIO(labels.decode())):
        count += 1
        if name not in edited:
            continue
        row = edited[name]
        require(name not in checked and row["utterance"][:4] in speakers, "A source is duplicated or outside the evaluation roster")
        boundaries = list(map(Decimal, values))
        intervals = list(zip(boundaries[:-1:2], boundaries[1:-1:2], strict=True))
        require(len(boundaries) in (3, 5) and 6 <= boundaries[-1] <= 14
                and abs(row["samples"] - round(boundaries[-1] * RATE)) <= 320
                and any(Decimal(".6") <= b - a <= Decimal("2.4") for a, b in intervals)
                and row["truth"] == [[round(a * RATE), round(b * RATE)] for a, b in intervals],
                "A saved edit interval differs from the publisher label on its own clock")
        checked.add(name)
    require(count == 85_007 and checked == set(edited)
            and all(row["utterance"][:4] in speakers for row in exclusions), "Incomplete publisher truth readback")


def row_outputs(row, check, pcm_dir):
    sparse, dense = schedules(row["samples"])
    scheduled = sorted(set(sparse + dense))
    require(type(check["id"]) is int and check["id"] == row["id"] and set(check) == {"id", "windows", "short", "short_primary"}
            and [span(item["span"]) for item in check["windows"]] == scheduled,
            "Window scores do not cover the exact sparse and dense union")
    native = None
    if pcm_dir is not None:
        with (pcm_dir / f"{row['id']:03}.pcm").open("rb") as source:
            native = source.read(2 * row["samples"] + 1)
        require(len(native) == 2 * row["samples"] and sha(native) == row["pcm_sha256"], "A frozen source PCM differs")
    directions = {}
    for item, (a, b) in zip(check["windows"], scheduled, strict=True):
        require(set(item) == {"span", "direction", "pcm_sha256"} and item["direction"] in STATES and pins([item["pcm_sha256"]]),
                "Invalid speech-window direction or hash")
        if native is not None:
            require(sha(native[2 * a:2 * b]) == item["pcm_sha256"], "Speech-window hash does not match the exact source samples")
        directions[(a, b)] = item["direction"]
    selected = [window for window in sparse if directions[window] == "ai"]
    current = selected if len(selected) >= 2 and all(directions[w] != "original" for w in sparse) else []
    short = check["short"]
    status = check["short_primary"]
    require(status in {"ok", "not_applicable"} if row["samples"] == SHORT_LENGTH else status is None,
            "Primary six-second eligibility is outside the exact short clock")
    require(short is None or status == "ok", "Short evidence requires a completed primary response")
    if short is not None:
        require(set(short) == {"start_sample", "end_sample"}, "Invalid short-review fields")
        a, b = short["start_sample"], short["end_sample"]
        require(type(a) is int and type(b) is int and row["samples"] == SHORT_LENGTH
                and a in range(0, 5 * RATE, RATE) and b == a + 2 * RATE, "Short review is outside the fixed two-second grid")
        current.append((a, b))
    proposed = [(second[0], first[1]) for first, second in zip(dense, dense[1:])
                if directions[first] == directions[second] == "ai"]
    return (current, proposed, current + proposed), any(value != "unassessed" for value in directions.values())


def evaluate(roster, checks, names, pcm_dir):
    cells = defaultdict(Counter)
    gains, alarms = defaultdict(set), defaultdict(set)
    for row, check in zip(roster, checks, strict=True):
        outputs, assessed = row_outputs(row, check, pcm_dir)
        truth = list(map(span, row["truth"]))
        scene, family = row["scene"], row["family"]
        cell = cells[f"{scene}/{family}"]
        cell.update(files=1, assessed=assessed, genuine_samples=row["samples"] - sum(b - a for a, b in truth))
        located = {}
        for arm, ranges in zip(ARMS, outputs, strict=True):
            located[arm] = useful(ranges, truth)
            cell.update({f"{arm}_flagged": bool(ranges), f"{arm}_useful": located[arm],
                         f"{arm}_genuine_samples_flagged": covered_genuine(ranges, truth)})
        if located["combined"] and not located["current"]:
            gains[scene].add(row["utterance"])
        if family == "genuine" and outputs[1] and not outputs[0]:
            alarms[scene].add(row["utterance"])
    return {"cells": dict(cells), "added_useful_utterances": {scene: len(gains[scene]) for scene in SCENES},
            "added_genuine_utterances": {scene: len(alarms[scene]) for scene in SCENES},
            "publisher_speakers": len({UTTERANCE.fullmatch(name)[1] for name in names}), "admitted_to_serving": False}


def review(receipt, *, run=None, publisher=None):
    lock, roster, checks = (receipt[key] for key in ("lock", "roster", "checks"))
    require(lock["schema"] == SCHEMA and lock["calibration_sha256"] == CALIBRATION
            and lock["sources"] == INPUT_PINS and lock["backgrounds"] == BACKGROUNDS
            and type(lock["known_model_responses_at_freeze"]) is int and lock["known_model_responses_at_freeze"] == 0
            and lock["method"] == METHOD and lock["voice_assets"] == VOICE_ASSETS
            and set(lock["complete_archive_members"]) == {"E1", "E2"}
            and all(type(n) is int and n > 0 for n in lock["complete_archive_members"].values())
            and pins([lock["runner_sha256"]]), "Frozen pilot method or source pins differ")
    require(sha(canonical(roster)) == lock["roster_sha256"], "Frozen delivery roster hash differs")
    require(sha(canonical(checks)) == receipt["checks_sha256"], "Saved window decisions differ from their receipt commitment")
    names = validate_roster(roster, lock)
    if publisher is not None:
        verify_publisher_truth(roster, lock["excluded_original_clocks"], publisher)
    if run is not None:
        for name, content in (("freeze", lock), ("roster", roster)):
            require((run / f"{name}.json").read_bytes() == canonical(content), f"Retained {name} differs from the receipt")
        original = json.loads((run / "scores.json").read_bytes())
        require(original["freeze_sha256"] == sha(canonical(lock)) and original["checks"] == checks,
                "Retained score ledger differs from its frozen input or receipt")
        for row in roster[::9]:
            with (run / "controls" / f"{row['utterance']}.flac").open("rb") as source:
                body = source.read(2**20)
            require(body.startswith(b"fLaC") and len(body) < 2**20 and sha(body) == row["source_sha256"],
                    "A retained original VCTK member differs from its source commitment")
    result = evaluate(roster, checks, names, run / "pcm" if run is not None else None)
    require(result == receipt["result"], "Reported flags, localizations or genuine sample totals differ")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path, help="Also verify frozen files and every exact PCM window in this private run")
    parser.add_argument("--publisher", type=Path, help="Also check the pinned publisher labels and evaluation speakers")
    args = parser.parse_args()
    result = review(json.loads(args.receipt.read_bytes()), run=args.run, publisher=args.publisher)
    print(json.dumps(result, sort_keys=True, indent=2))
