# SPDX-License-Identifier: Apache-2.0
"""Replay environmental audio decisions without model weights or a native runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts import environmental_embedding_transfer as study

SHA = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def review_rows(rows, native):
    expected = []
    for group, role in enumerate(study.ROLES):
        lanes = ("audiogen", "audiolcm", "real") if role == "held" else ("audioldm1", "audioldm2", "real")
        expected.extend((group, role, lane, scene) for lane in lanes for scene in study.SCENES[role])
    require(len(rows) == len(expected) == 348, "Incomplete comparison roster")
    source_pins = {}
    foreground_pins = defaultdict(set)
    voices_by_group = defaultdict(set)
    for index, (row, placement) in enumerate(zip(rows, expected, strict=True)):
        group, role, lane, scene = placement
        require(row["id"] == index and tuple(row[key] for key in ("group", "role", "lane", "scene")) == placement,
                "Source group or view order differs")
        pin = row["native_pcm_sha256"]
        original = source_pins.setdefault((group, lane), pin)
        require(SHA.fullmatch(pin) and original == pin and SHA.fullmatch(row["pcm_sha256"])
                and (scene != "native" or row["pcm_sha256"] == pin), "Native waveform commitment differs")
        voice = row["voice_pcm_sha256"]
        if scene in ("speech_mix", "balanced_mix"):
            require(SHA.fullmatch(voice), "Invalid foreground commitment")
            foreground_pins[role].add(voice)
            voices_by_group[group].add(voice)
        else:
            require(voice is None, "Isolated input has a speech component")
        if native is not None:
            data = (native / "pcm" / f"{index:03}.pcm").read_bytes()
            require(len(data) == 96000 and digest(data) == row["pcm_sha256"], "Saved native PCM differs")
    require(len(set(source_pins.values())) == 120 and all(len(item) == 1 for item in voices_by_group.values())
            and [len(foreground_pins[role]) for role in ("fit", "selection", "held")] == [4, 4, 7]
            and len(set.union(*foreground_pins.values())) == 15, "Speaker or environment crosses roles")


def bound_responses(rows, observations):
    require(len(rows) == len(observations), "Missing responses")
    for row, record in zip(rows, observations, strict=True):
        require(row["id"] == record["id"] and set(record["responses"]) == set(study.ARMS)
                and all(type(value) in (float, int) and math.isfinite(value) for value in record["responses"].values()),
                "Invalid measurement binding")


def audit(document, native=None):
    freeze, rows = document["freeze"], document["rows"]
    require(freeze["schema"] == study.SCHEMA and freeze["method"] == study.METHOD and freeze["pins"] == study.PINS
            and freeze["runner"] == digest(Path(study.__file__).read_bytes()) and freeze["responses_at_freeze"] == 0
            and freeze["prior_public_sha256"] == digest(study.BASELINE.read_bytes())
            and freeze["roster_sha256"] == digest(canonical(rows)), "Frozen inputs changed")
    review_rows(rows, native)
    selected = [row for row in rows if row["role"] == "selection"]
    held = [row for row in rows if row["role"] == "held"]
    bound_responses(selected, document["selection"])
    bound_responses(held, document["held"])
    recalculated = {arm: {} for arm in study.ARMS}
    for arm in study.ARMS:
        for scene in ("native", "speech_mix"):
            scores = [record["responses"][arm] for row, record in zip(selected, document["selection"], strict=True)
                      if row["lane"] == "real" and row["scene"] == scene]
            require(len(scores) == 10, "Wrong genuine calibration denominator")
            recalculated[arm][scene] = sorted(scores, reverse=True)[1]
    require(recalculated == document["points"], "Operating point differs")
    result = {}
    for arm in study.ARMS:
        counts = defaultdict(Counter)
        for row, record in zip(held, document["held"], strict=True):
            scene = row["scene"]
            cutoff = recalculated[arm]["speech_mix" if scene.endswith("mix") else "native"]
            counts[scene][row["lane"]] += record["responses"][arm] > cutoff
        passes = {scene: count["real"] <= (2 if scene == "speech_mix" else 3)
                  and min(count["audiogen"], count["audiolcm"]) >= (11 if scene == "speech_mix" else 9)
                  for scene, count in counts.items()}
        result[arm] = {"flags": dict(counts), "passes_view": passes, "passes": all(passes.values())}
    require(result == document["result"], "Published counts differ")
    for key in ("heads_sha256", "held_features_sha256"):
        require(SHA.fullmatch(document[key]), "Missing model fit or response commitment")
    if native is not None:
        require(read(native / "freeze.json") == freeze and read(native / "roster.json") == rows
                and digest((native / "heads.json").read_bytes()) == document["heads_sha256"]
                and digest((native / "held.json").read_bytes()) == document["held_features_sha256"],
                "Local stage commitments differ")
    rankings = {}
    for scene in study.POINTS:
        scene_rows = [(row["lane"], item["responses"]) for row, item in zip(held, document["held"], strict=True)
                      if row["scene"] == scene]
        rankings[scene] = {}
        for arm in study.ARMS:
            above = set()
            scores = sorted({values[arm] for _, values in scene_rows}, reverse=True)
            counts = [{"real": 0, "audiogen": 0, "audiolcm": 0}]
            for score in scores:
                above.update(index for index, (_, values) in enumerate(scene_rows) if values[arm] == score)
                counts.append({lane: sum(index in above and kind == lane for index, (kind, _) in enumerate(scene_rows))
                               for lane in ("real", "audiogen", "audiolcm")})
            rankings[scene][arm] = {
                str(budget): max((row for row in counts if row["real"] <= budget),
                                 key=lambda row: (row["audiogen"] + row["audiolcm"], -row["real"]))
                for budget in (2, 4)
            }
    return {"fixed": result, "matched_false_alarm_ranking": rankings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", type=Path, help="Verify local PCM and stage commitments")
    args = parser.parse_args()
    print(canonical(audit(read(args.receipt), args.native)).decode(), end="")
