"""Validate tone phase evidence on different voices and field recordings."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_tonal_phase_binding as tonal
from scripts import audio_tonal_phase_transfer as first
from scripts.audio_focused_window_transfer import SCENES as NOISE_SOURCES

files = first.io
SCHEMA = "chipmunk.audio-persistent-tone-transfer/1"
ARMS = ("single", "harmonic", "persistent")
SHARDS = {
    "libri-0000.parquet": "3098c6e44d1d49f8c62bd123775f1c492bda2cabd80f3ee70fe7800572371401",
    "libri-0001.parquet": "011cb76e296b2650bab3ea54044c57d10293dadb05e9e96605b37c88e4bc89bd",
}
TONES = {
    "513490-hq.mp3": "07b11152decfdb11caea721687e9337354766cbabdfae75bc473d1a57a7b5215",
    "785990-hq.mp3": "988c55f6190fa054d0c9c048272a1fbe5a2ef200f067b4b0bec1f3c894f40883",
}
CONTROLS = (*first.CONTROLS, "ordinary_clean")
METHOD = {
    "inputs": "First 24 distinct LibriSpeech speakers with a usable native 16 kHz take in the two pinned shards. "
              "Use different recorded tone sources and noise excerpts. Score all frozen clips.",
    "point": "Use the original broader point for each rule. The persistent variant applies the same single-tone point. "
             "Require at least eight complete 40 ms frame centers on each outer side and support in at least 60 percent.",
    "controls": "Keep clean speech, mixtures with no added tone, room-only excerpts, level and timing changes, "
                "ordinary local speech-only shifts, low-pass filtering and smoothly varying tone timing.",
    "outcome": "Count useful positions, missed events, wrong locations and ordinary flags. A localized tone change "
               "has zero authorship weight.",
}


def foregrounds(root):
    import pyarrow.parquet as pq
    import soundfile as sf

    seen = set()
    for name, digest in SHARDS.items():
        path = root / name
        if files.sha(path) != digest:
            raise ValueError(f"Publisher shard bytes differ for {name}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["id", "speaker_id", "audio"]):
            for row in batch.to_pylist():
                speaker = row["speaker_id"]
                if speaker in seen:
                    continue
                content = row["audio"]["bytes"]
                samples, rate = sf.read(io.BytesIO(content), dtype="int16")
                if rate != first.RATE or samples.ndim != 1 or len(samples) < first.SAMPLES:
                    continue
                samples = samples[:first.SAMPLES]
                if first.rms(samples) < 160:
                    continue
                seen.add(speaker)
                source = {"id": row["id"], "reader": speaker, "member_sha256": files.digest(content),
                          "pcm_sha256": files.digest(samples.astype("<i2", copy=False).tobytes()), "shard": name}
                yield source, samples
                if len(seen) == 24:
                    return
    raise ValueError("Fewer than 24 distinct source speakers qualified")


def cases(sources):
    for index, source in enumerate(sources):
        for slot, scene in enumerate(first.SCENES):
            start = 17500 + ((index * 13 + slot * 11) % 21) * 480
            end = start + (8960, 11520, 14080, 16640)[(index + 3 * slot) % 4]
            offset = (1421, 1738, 2127, 2256)[(index * 5 + slot) % 4]
            for role in (*first.POSITIVES, *CONTROLS):
                yield {"scene": scene, "slot": index, "id": source["id"], "reader": source["reader"],
                       "tone": list(TONES)[(index + slot) % 2], "tone_db": -13 if (index // 2 + slot) % 2 else -7,
                       "offset": -offset if (index // 3 + slot) % 2 else offset,
                       "authored_interval": [start, end], "role": role,
                       "truth": [start, end] if role in first.POSITIVES else None}
    for scene in first.SCENES:
        for index in range(12):
            yield {"id": None, "reader": None, "scene": scene, "slot": index, "role": "isolated", "truth": None}


def freeze(parent, root):
    previous = files.read(parent)
    first.review(previous)
    points = {arm: previous["points"][arm]["broader"] for arm in first.ARMS}
    sources = [row for row, _ in foregrounds(root)]
    return {"schema": SCHEMA, "method": METHOD, "parent_sha256": files.sha(parent),
            "shards": SHARDS, "tones": TONES, "sources": sources,
            "points": points | {"persistent": points["single"]}}


def beds(folder):
    import numpy as np

    noises = {}
    for scene, spec in NOISE_SOURCES.items():
        file = folder / f"{scene}.ogg"
        if files.sha(file) != spec["sha256"]:
            raise ValueError("Background capture differs from its pinned bytes")
        source = np.frombuffer(files.decode(file), dtype="<i2")
        offsets = range(0, len(source) - first.SAMPLES + 1, first.RATE * 5)
        starts = [start for start in offsets if all(start + first.SAMPLES <= old * first.RATE
                  or start >= (old + 6) * first.RATE for old in spec["offsets"])]
        if not starts:
            raise ValueError("No disjoint background excerpts remain")
        noises[scene] = [source[start:start + first.SAMPLES] for start in starts]
    return noises


def recordings(folder):
    import numpy as np

    output = {}
    for name, pin in TONES.items():
        path = folder / name
        if files.sha(path) != pin:
            raise ValueError("Recorded tone differs from its frozen bytes")
        pcm = np.frombuffer(files.decode(path), dtype="<i2").astype(np.float64)
        clips = [pcm[i:i + first.SAMPLES] for i in range(0, len(pcm) - first.SAMPLES + 1, 5 * first.RATE)]
        output[name] = [clip - clip.mean() for clip in clips]
        if not output[name]:
            raise ValueError("Short field recording")
    return output


def samples(plan, root, tones, sounds):
    import numpy as np

    admitted = list(foregrounds(root))
    if [row for row, _ in admitted] != plan["sources"]:
        raise ValueError("Foreground samples differ from the frozen roster")
    voice = [pcm for _, pcm in admitted]
    noise, capture = beds(sounds), recordings(tones)
    previous, group = None, {}
    for case in cases(plan["sources"]):
        scene, index = case["scene"], case["slot"]
        if scene == "competing_speech":
            room = voice[(index + 7) % len(voice)]
        else:
            room = noise[scene][index % len(noise[scene])]
        if case["role"] == "isolated":
            yield case, room
            continue
        key = case["id"], scene
        if key != previous:
            tone = capture[case["tone"]]
            edited = first.splice(voice[index].astype(np.float64), *case["authored_interval"], case["offset"])
            group = first.variants(case, voice[index], np.rint(edited).astype("<i2"), room, tone[index % len(tone)])
            group["ordinary_clean"] = voice[index]
            previous = key
        yield case, group[case["role"]]


def observation(pcm):
    from app.evidence import audio_continuity, audio_tonal_phase

    decoded = pcm.astype("float64") / 32768
    counts, ranges = audio_tonal_phase.measure(decoded, persistent=True)
    spectral = audio_continuity.select(audio_continuity.candidates(decoded))
    return {"sha256": files.digest(pcm.astype("<i2", copy=False).tobytes()), "comparisons": counts, "ranges": ranges,
            "spectral_boundaries": sorted(row["boundary_sample"] for row in spectral)}


def summarize(rows, points):
    cells, flagged_groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        case = row["case"]
        truth = case["truth"]
        bounds = row["spectral_boundaries"]
        results = [("spectral", [[a, b] for a in bounds for b in bounds if 8320 <= b - a <= 18560], True)]
        for arm in ARMS:
            best = row["ranges"][arm]
            shown = [best["tones"][0]["samples"]] if best and best["score"] > points[arm] else []
            results.append((arm, shown, row["comparisons"][arm] > 0))
        for arm, spans, assessed in results:
            for grouping in ("all", case["scene"], case.get("tone", "isolated")):
                cell = cells[f"{arm}/{grouping}/{case['role']}"]
                cell.update(files=1, assessed=int(assessed), flagged=int(bool(spans)))
                if truth is not None:
                    cell["useful"] += int(any(first.useful(span, truth) for span in spans))
                    cell["near_boundaries"] += int(any(
                        max(abs(a - b) for a, b in zip(span, truth, strict=True)) <= 1920 for span in spans
                    ))
            if truth is None and spans and case["reader"] is not None:
                flagged_groups[arm].add(case["reader"])
    return {"counts": {key: dict(value) for key, value in cells.items()},
            "ordinary_reader_groups": {key: len(group) for key, group in flagged_groups.items()}}


def review(receipt, parent):
    plan = receipt["freeze"]
    reference = files.read(parent)
    first.review(reference)
    points = {arm: reference["points"][arm]["broader"] for arm in first.ARMS}
    earlier_readers = {str(row["case"]["reader"]) for phase in ("development", "held") for row in reference[phase]}
    readers = {str(source["reader"]) for source in plan["sources"]}
    if (plan["schema"] != SCHEMA or plan["method"] != METHOD or plan["parent_sha256"] != files.sha(parent)
            or plan["shards"] != SHARDS or plan["tones"] != TONES or plan["points"] != points | {"persistent": points["single"]}
            or len(readers) != 24 or len(plan["sources"]) != 24 or readers & earlier_readers):
        raise ValueError("The transfer roster differs from its fixed sources")
    for case, row in zip(cases(plan["sources"]), receipt["rows"], strict=True):
        if case != row["case"] or not first.prior.clock.SHA.fullmatch(row["sha256"]) or set(row["ranges"]) != set(ARMS):
            raise ValueError("A comparison source differs")
        if (match := row["ranges"]["harmonic"]) and match["score"] > points["harmonic"]:
            tonal.validate({"comparisons": row["comparisons"]["harmonic"], "status": "measured_change",
                            "measurement": match}, first.SAMPLES)
        if best := row["ranges"]["persistent"]:
            tone, = best["tones"]
            if any(total < 8 or not 0 <= present <= total or present * 5 < total * 3
                   for present, total in tone["outside_support_frames"]):
                raise ValueError("A selected tone lacks its supporting outer measurements")
    result = summarize(receipt["rows"], plan["points"])
    if result != receipt["results"]:
        raise ValueError("The held tally changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--hums", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        first.write(args.path, freeze(args.parent, args.shards))
    elif args.phase == "run":
        plan = files.read(args.path)
        rows = []
        for case, pcm in samples(plan, args.shards, args.hums, args.sounds):
            rows.append({"case": case, **observation(pcm)})
            if len(rows) % 120 == 0:
                print("held", len(rows), flush=True)
        result = {"freeze": plan, "rows": rows, "results": summarize(rows, plan["points"])}
        review(result, args.parent)
        first.write(args.output, result)
    else:
        print(json.dumps(review(files.read(args.path), args.parent), sort_keys=True))


if __name__ == "__main__":
    main()
