"""Reserve untouched source utterances for a prospective lip-sync transfer test."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts import video_lip_articulation_transfer as prior
from scripts import video_lip_correspondence as receipt
from scripts import video_lip_sequence_panel as sequence
from scripts import video_visual_speech_train as training

CLIPS = 24


def input_rows(path):
    payload = receipt.read(path) if path.suffix == ".gz" else json.loads(path.read_bytes())
    if isinstance(payload, list):
        # The acoustic fit inventory has no picture or donor fields.
        return [{"target": {"id": row["uid"], "audio_sha256": row["audio_sha256"]}} for row in payload]
    return payload.get("rows", payload.get("cases", []))


def digit_bounds(phones):
    words = []
    start = None
    for info in phones:
        name = info["phone"]
        if name.startswith("SIL"):
            if start is not None:
                return None
            continue
        tail = name.rsplit("_", 1)[-1]
        a = round(Fraction(str(info["offset"])) * training.RATE)
        b = round((Fraction(str(info["offset"])) + Fraction(str(info["duration"]))) * training.RATE)
        if tail in {"S", "B"}:
            if start is not None:
                return None
            start = a
        if start is None or b <= a or tail not in {"S", "B", "I", "E"}:
            return None
        if tail in {"S", "E"}:
            words.append([start, b])
            start = None
    if len(words) == 6 and start is None and all(x[1] <= y[0] for x, y in zip(words, words[1:])):
        a, b = words[4]
        if 3840 <= b - a <= 8320:
            return a, b
    return None


def matched_digit(pool, audio, front, waves, pictures, used_audio, used_pictures):
    order = sorted(pool, key=lambda uid: training.rank(f"digit/{uid}"))
    available = {}

    def member(uid):
        if uid not in available:
            available[uid] = prior.source_record(uid, pool[uid], audio, front, waves, pictures, used_audio, used_pictures)
        return available[uid]

    def sound_only(row):
        return {key: value for key, value in row.items() if key not in {"front", "front_sha256"}}

    for target in order:
        a, b = pool[target][0]
        length = b - a
        candidates = [uid for uid in order if uid != target and 85 * length <= 100 * (pool[uid][0][1] - pool[uid][0][0]) <= 115 * length]
        keep = [uid for uid in candidates if uid[-2] == target[-2]]
        changed = [uid for uid in candidates if uid[-2] != target[-2]]
        if not keep or not changed or (origin := member(target)) is None:
            continue
        for same in keep:
            control = member(same)
            if not control:
                continue
            c, d = pool[same][0]
            for swapped in changed:
                p, q = pool[swapped][0]
                if abs((d - c) - (q - p)) <= 960 and (donor := member(swapped)):
                    return {"target": origin, "same_words": sound_only(control), "different_words": sound_only(donor)}
    return None


def prepare(source, fit_plan, history, output, *, clips=CLIPS, separate_speakers=(), recordings_per_speaker=1):
    from scripts.lombard_mouth_source_intake import index_zip

    plan = receipt.read(fit_plan)
    assert plan["schema"] == training.SCHEMA
    denied = {r["speaker"] for r in plan["rows"] if r["split"] in {"fit", "calibration"}}
    denied.update(row["speaker"] for path in separate_speakers for row in input_rows(path))
    seen_actors = {r["speaker"] for r in plan["rows"]}
    known = [entry for path in (fit_plan, *history) for row in input_rows(path)
             for key in ("target", "same_words", "different_words") if isinstance(entry := row.get(key), dict)]
    excluded = {item["id"] for item in known if "id" in item}
    used_audio = {item["audio_sha256"] for item in known if "audio_sha256" in item}
    used_pictures = {item["front_sha256"] for item in known if "front_sha256" in item}
    available, rows = defaultdict(dict), []
    with (ZipFile(source / "lombardgrid_front.zip") as front, ZipFile(source / "lombardgrid_audio.zip") as audio,
          ZipFile(source / "alignment.zip") as aligned, ZipFile(source / "publisher-json.zip") as metadata):
        pictures, waves, labels = (index_zip(archive, ending) for archive, ending in
                                   ((front, ".mov"), (audio, ".wav"), (aligned, ".json")))
        for name in index_zip(metadata, ".json").values():
            for row in json.loads(metadata.read(name)):
                actor, condition = prior.keyed(row["SPKR"]), row["COND"]
                uid = f"{row['SPKR']}_{condition}_{row['UTTERANCE']}"
                if (actor in denied or condition not in {"p", "l"} or row["STATUS"] != "CORRECT" or uid in excluded
                        or not all(uid in archive for archive in (pictures, waves, labels))):
                    continue
                data = aligned.read(labels[uid])
                if span := digit_bounds(json.loads(data)[uid]):
                    available[actor, condition][uid] = span, data
        used_actors = Counter()
        order = sorted(available, key=lambda key: (key[0] in seen_actors, training.rank(f"prospective/{key[0]}/{key[1]}")))
        for actor, condition in order:
            if used_actors[actor] >= recordings_per_speaker:
                continue
            pair = matched_digit(available[actor, condition], audio, front, waves, pictures, used_audio, used_pictures)
            if pair:
                rows.append({"speaker": actor, "condition": condition, "split": "fresh", **pair})
                used_actors.update((actor,))
                used_pictures.add(pair["target"]["front_sha256"])
                used_audio.update(row["audio_sha256"] for row in pair.values())
                if len(rows) == clips:
                    break
    assert len(rows) >= min(16, clips) and max(used_actors.values()) <= recordings_per_speaker, len(rows)
    output.mkdir(mode=0o700)
    print(f"Selected {len(rows)} recordings from {len(used_actors)} readers, "
          f"including {len(used_actors.keys() - seen_actors)} outside the earlier held groups", flush=True)
    cases = {"schema": training.SCHEMA, "fit_plan": receipt.sha(fit_plan), "rows": rows, "publisher": plan["publisher"],
             "history": {path.name: receipt.sha(path) for path in history}}
    if separate_speakers:
        cases["speaker_exclusions"] = [receipt.sha(path) for path in separate_speakers]
    receipt.save(output / "plan.json.gz", cases)
    receipt.save(output / "incumbent-plan.json.gz", cases | {
        "schema": sequence.SCHEMA, "method": sequence.STUDY_METHOD, "counts": {"held": {"recordings": len(rows)}},
        "rows": [row | {"split": "held"} for row in rows],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fit_plan", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("history", nargs="+", type=Path)
    parser.add_argument("--clips", type=int, choices=range(1, 55), default=CLIPS)
    parser.add_argument("--separate-speakers", action="append", type=Path, default=[])
    parser.add_argument("--recordings-per-speaker", type=int, choices=(1, 2), default=1)
    prepare(**vars(parser.parse_args()))
