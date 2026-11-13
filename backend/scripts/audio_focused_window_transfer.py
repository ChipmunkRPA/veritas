# SPDX-License-Identifier: Apache-2.0
"""Compare narrow corroborated speech ranges under independently recorded noise."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_dense_evidence_confirmation as previous
from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

SCHEMA = "chipmunk.audio-focused-window-transfer/1"
RATE, SPAN = clock.RATE, clock.WINDOW
FAMILIES = ("cosyvoice", "elevenlab", "gptsovits", "ljjets", "xttsv2", "yourtts")
SCENES = {
    "rain": {"sha256": "e0233652d8173650149adcf3d6e5ddc01b73e8a93ce61b8d7809abfe582251a8", "offsets": [3, 15, 29, 43]},
    "train": {"sha256": "06746a387a2995595b99da5c3f43b2008681fc98fadf6ad513d9a4226973f179", "offsets": [32, 58, 84, 110]},
}
METHOD = {
    "source": "Three Part B edits per generator and 12 genuine Part A groups, ranked before any new inference.",
    "duration_samples": [6 * RATE, 11 * RATE], "genuine_groups": 12, "edits_per_family": 3,
    "scene_snr_db": -5, "scenes": SCENES, "isolated_samples": 8 * RATE,
    "calibration_sha256": clock.METHOD["primary_calibration_sha256"],
    "sparse": "Retain the original corroborated two or three 3-second windows.",
    "dense": "Use the 1.5-second intersection of two adjacent 3-second AI directions.",
    "focused": "Keep every dense intersection. Keep a sparse window only if it contains no entire dense intersection.",
    "advance": {"min_added_noisy_groups": 3, "min_added_noisy_families": 2, "max_lost_per_scene": 1,
                "max_added_genuine_groups": 1, "max_added_genuine_time_fraction": .02,
                "max_new_clean_or_isolated_groups": 0},
}
ARMS = ("sparse", "dense", "union", "focused")
SOURCE_LEDGER = "audio-dense-evidence-confirmation-2026-11-12.json"
NOISE_LEDGER = "audio-nemotron-noise-transfer-2026-11-12.json"
REQUIREMENTS = (
    "audio_dense_evidence_confirmation.py", "audio_noise_window_transfer.py", "audio_reduced_background_localization.py",
    "audio_radar_transfer.py", "audio_noise_views_transfer.py", "review_audio_reduced_background_localization.py",
)
require = previous.require


def histories():
    groups, ids, pins = previous.history()
    paths = {name: io.PUBLIC / f"{name}-2026-11-12.json" for name in previous.BASES}
    later = io.PUBLIC / SOURCE_LEDGER
    noise = io.PUBLIC / NOISE_LEDGER
    for source in io.read(later)["roster"]:
        groups.add(source["group"])
        ids.add(source["publisher_id"])
        pins.add(source["source_pcm_sha256"])
    for item in io.read(noise)["held"]:
        if item["label"] == "isolated":
            continue
        origin = item["origin"]
        groups.add(origin["group"])
        ids.add(origin.get("publisher_id", origin.get("id")))
        pins.add(origin.get("source_pcm_sha256", origin.get("pcm_sha256")))
    return groups, ids, pins, {path.name: io.sha(path) for path in (*paths.values(), later, noise)}


def dependencies():
    folder = Path(__file__).parent
    return {name: io.sha(folder / name) for name in REQUIREMENTS}


def choose(part_a, part_b, labels):
    from scripts.audio_noise_window_transfer import archive_partials
    from scripts.audio_reduced_background_localization import source_candidates

    groups, ids, pins, _ = histories()
    paths = [part_a / name for name in previous.PREFIXES]
    pool, scans = archive_partials(part_b)
    original, examined, _ = source_candidates(*paths[:2], labels, paths[2:])
    options = {}
    for family, choices in pool.items():
        options[family] = [(dict(row, label="partial", samples=len(raw) // 2), raw) for row, raw in choices]
    options[None] = [(dict(row, id=row["publisher_id"], source_sha256=row["source_wav_sha256"], recipe=None), raw)
                     for row, raw in original["genuine", None]]
    for family, choices in options.items():
        unique = {}
        for row, raw in sorted(choices, key=lambda entry: (entry[0]["group"] in groups,
                              io.digest(f"{SCHEMA}/{entry[0]['id']}".encode()))):
            if (row["id"] not in ids and row["source_pcm_sha256"] not in pins
                    and METHOD["duration_samples"][0] < row["samples"] <= METHOD["duration_samples"][1]):
                unique.setdefault(row["group"], (row, raw))
        options[family] = unique
    slots = [family for family in FAMILIES for _ in range(METHOD["edits_per_family"])]
    slots.extend([None] * METHOD["genuine_groups"])
    selected, assigned = {}, {}

    def assign(slot, visited):
        for group, pair in options[slots[slot]].items():
            if group in visited:
                continue
            visited.add(group)
            if group not in assigned or assign(assigned[group], visited):
                assigned[group], selected[slot] = slot, pair
                return True
        return False

    for slot in sorted(range(len(slots)), key=lambda n: (len(options[slots[n]]), n)):
        require(assign(slot, set()), f"Insufficient distinct groups for {slots[slot]}")
    return [selected[slot] for slot in range(len(slots))], {"part_a": examined, "part_b": scans}


def prepare(run, part_a, part_b, labels, background):
    from scripts.audio_gtcrn_traffic import float_pcm, pcm
    from scripts.audio_noise_views_transfer import match_views
    from scripts.audio_noise_window_transfer import PINS

    selection, scans = choose(part_a, part_b, labels)
    seen, _, _, history = histories()
    sounds = {}
    for scene, spec in SCENES.items():
        path = background / f"{scene}.ogg"
        require(io.sha(path) == spec["sha256"], "Background recording differs")
        sounds[scene] = float_pcm(io.decode(path))
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    (run / "responses").mkdir(mode=0o700)
    rows = []

    def add(wave, source, scene, offset):
        raw = pcm(wave)
        case = len(rows)
        rows.append({"case": case, "scene": scene, "offset": offset, "samples": len(wave),
                     "source": source, "pcm_sha256": io.digest(raw)})
        (run / "pcm" / f"{case:03}.pcm").write_bytes(raw)

    for rank, (entry, raw) in enumerate(selection):
        source = {key: entry[key] for key in ("id", "group", "family", "label", "recipe", "edits", "source_sha256", "source_pcm_sha256")}
        source["speaker_in_history"] = source["group"] in seen
        wave = float_pcm(raw)
        offsets = {scene: spec["offsets"][rank % 4] * RATE for scene, spec in SCENES.items()}
        clips = {scene: sounds[scene][start:start + len(wave)] for scene, start in offsets.items()}
        views = match_views(wave, clips, METHOD["scene_snr_db"])
        for scene, values in views.items():
            add(values, source, scene, offsets.get(scene))
    for scene, spec in SCENES.items():
        for offset in spec["offsets"]:
            start = offset * RATE
            span = sounds[scene][start:start + METHOD["isolated_samples"]]
            require(len(span) == METHOD["isolated_samples"], "Incomplete isolated audio")
            add(span, None, scene, start)
    require(len(rows) == 98 and len({row["pcm_sha256"] for row in rows}) == 98, "Unexpected delivery roster")
    io.save(run / "roster.json", rows)
    pins = {name: io.sha(part_a / name) for name in previous.PREFIXES} | {"part-a.labels.txt": io.sha(labels)} | PINS
    lock = {"schema": SCHEMA, "method": METHOD, "history": history, "dependencies": dependencies(),
            "runner_sha256": io.sha(Path(__file__)), "roster_sha256": io.sha(run / "roster.json"),
            "source_files": pins, "members_examined": scans, "voice_assets": io.assets(), "responses_at_freeze": 0}
    io.save(run / "freeze.json", lock)
    print("Frozen", len(rows), "deliveries", io.sha(run / "freeze.json"), flush=True)


def native_wave(run, row):
    raw = (run / "pcm" / f"{row['case']:03}.pcm").read_bytes()
    require(len(raw) == 2 * row["samples"] and io.digest(raw) == row["pcm_sha256"], "Delivered PCM differs")
    return raw


def schedule(samples):
    return sorted(set(map(tuple, sum(clock.schedules(samples), []))))


def verified_lock(lock, rows):
    seen, ids, pins, history = histories()
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["history"] == history
            and lock["runner_sha256"] == io.sha(Path(__file__)) and lock["dependencies"] == dependencies()
            and lock["roster_sha256"] == io.digest(io.encoded(rows)) and lock["responses_at_freeze"] == 0,
            "Frozen scope or inputs changed")
    require(len(rows) == len({row["pcm_sha256"] for row in rows}) == 98, "Delivery count or uniqueness changed")
    roles = Counter()
    sources = set()
    for index in range(0, 90, 3):
        paired = rows[index:index + 3]
        source = paired[0]["source"]
        group = source["group"]
        family, label = source["family"], source["label"]
        length = paired[0]["samples"]
        truth = source["edits"]
        require(all(row["source"] == source and row["samples"] == length for row in paired)
                and [row["scene"] for row in paired] == ["gain_control", *SCENES]
                and [row["offset"] for row in paired] == [None, *(value["offsets"][(index // 3) % 4] * RATE for value in SCENES.values())]
                and group not in sources and source["speaker_in_history"] == (group in seen)
                and source["id"] not in ids and source["source_pcm_sha256"] not in pins
                and all(clock.SHA.fullmatch(source[k]) for k in ("source_sha256", "source_pcm_sha256"))
                and METHOD["duration_samples"][0] < length <= METHOD["duration_samples"][1]
                and all(0 <= a < b <= length for a, b in truth) and truth == clock.intervals_union(truth)
                and ((label == "partial" and family in FAMILIES and bool(truth) and source["recipe"] in {"cp", "oa"})
                     or (label == "genuine" and family is None and not truth and source["recipe"] is None)),
                "Invalid source group, role, native label clock or matched views")
        roles[family] += 1
        sources.add(group)
    expected = [(scene, start * RATE) for scene, value in SCENES.items() for start in value["offsets"]]
    require(roles == (dict.fromkeys(FAMILIES, METHOD["edits_per_family"]) | {None: METHOD["genuine_groups"]})
            and [(row["scene"], row["offset"]) for row in rows[90:]] == expected
            and all(row["source"] is None and row["samples"] == METHOD["isolated_samples"] for row in rows[90:])
            and all(row["case"] == i and clock.SHA.fullmatch(row["pcm_sha256"]) for i, row in enumerate(rows)),
            "Role counts or isolated clocks differ")


def verify_windows(row, response, raw=None):
    windows = response["windows"]
    require(response["case"] == row["case"] and [tuple(item["span"]) for item in windows] == schedule(row["samples"]),
            "Scored window schedule differs")
    for item in windows:
        start, end = item["span"]
        pin = item["pcm_sha256"]
        require(item["direction"] in {"ai", "original", "inconclusive", "unassessed"} and clock.SHA.fullmatch(pin)
                and (raw is None or io.digest(raw[2*start:2*end]) == pin), "Window direction or source commitment differs")


def score(run):
    from app.detectors.audio import calibration
    from scripts.audio_gtcrn_traffic import float_pcm
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_reduced_background_localization import assets_for_scores

    lock, rows = (io.read(run / name) for name in ("freeze.json", "roster.json"))
    verified_lock(lock, rows)
    require(METHOD["calibration_sha256"] == calibration.SHA256, "Installed voice calibration differs")
    model, activity = assets_for_scores(lock)
    output = []
    for row in rows:
        raw = native_wave(run, row)
        path = run / "responses" / f"{row['case']:03}.json"
        if path.is_file():
            response = io.read(path)
        else:
            wave = float_pcm(raw)
            windows = []
            for start, end in schedule(len(wave)):
                entry = adapter_result(wave[start:end], model, activity)
                require((entry["scope"] and entry["windows"] == [[0, SPAN]]) or entry["decision"] == "unassessed",
                        "Voice result outside the original three-second scope")
                windows.append({"span": [start, end], "direction": entry["decision"], "pcm_sha256": io.digest(raw[2*start:2*end])})
            response = {"case": row["case"], "windows": windows}
            temp = path.with_suffix(".tmp")
            io.save(temp, response)
            temp.replace(path)
        verify_windows(row, response, raw)
        output.append(response)
        if len(output) % 10 == 0:
            print("Completed", len(output), "deliveries", flush=True)
    io.save(run / "receipt.json", {"freeze": lock, "roster": rows, "responses": output, "result": compare(rows, output)})


def ranges(samples, directions):
    sparse, dense = clock.schedules(samples)
    positive = {span for span, value in directions.items() if value == "ai"}
    incumbent = [span for span in sparse if tuple(span) in positive]
    if len(incumbent) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
        incumbent = []
    precise = clock.intersections(dense, positive)
    fallback = [span for span in incumbent if not any(span[0] <= a < b <= span[1] for a, b in precise)]
    return dict(zip(ARMS, (incumbent, precise, incumbent + precise, precise + fallback), strict=True))


def compare(rows, responses):
    cells = defaultdict(Counter)
    added, families, false, clean_false = set(), set(), set(), set()
    for row, response in zip(rows, responses, strict=True):
        source = row["source"]
        label, truth = (source["label"], source["edits"]) if source else ("isolated", [])
        scene = row["scene"]
        directions = {tuple(item["span"]): item["direction"] for item in response["windows"]}
        proposals = ranges(row["samples"], directions)
        before = clock.useful(proposals["sparse"], truth)
        cell = cells[scene, label]
        cell["files"] += 1
        cell["assessed"] += any(value != "unassessed" for value in directions.values())
        cell["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
        for name, spans in proposals.items():
            found = clock.useful(spans, truth)
            cell[f"{name}_useful"] += found
            cell[f"{name}_added"] += found and not before
            cell[f"{name}_lost"] += before and not found
            cell[f"{name}_flags"] += bool(spans)
            union = clock.intervals_union(spans)
            cell[f"{name}_genuine_samples"] += sum(b - a for a, b in union) - sum(
                clock.overlap(span, edit) for span in union for edit in truth)
        if label == "partial" and scene != "gain_control" and clock.useful(proposals["focused"], truth) and not before:
            added.add(source["group"])
            families.add(source["family"])
        if label != "partial" and proposals["focused"] and not proposals["sparse"]:
            if label == "isolated" or scene == "gain_control":
                clean_false.add((label, source["group"] if source else row["case"]))
            else:
                false.add(source["group"])
    genuine_time = sum(cell["genuine_samples"] for (_, label), cell in cells.items() if label != "isolated")
    extra = sum(cell["focused_genuine_samples"] - cell["sparse_genuine_samples"] for (_, label), cell in cells.items()
                if label != "isolated") / genuine_time
    target = METHOD["advance"]
    gate = {"additions": len(added) >= target["min_added_noisy_groups"], "families": len(families) >= target["min_added_noisy_families"],
            "losses": all(cell["focused_lost"] <= target["max_lost_per_scene"] for cell in cells.values()),
            "false_groups": len(false) <= target["max_added_genuine_groups"],
            "clean_or_isolated": len(clean_false) <= target["max_new_clean_or_isolated_groups"],
            "genuine_time": extra <= target["max_added_genuine_time_fraction"]}
    return {"cells": {f"{scene}/{label}": dict(cell) for (scene, label), cell in cells.items()}, "added_noisy_groups": len(added),
            "added_noisy_families": sorted(families), "added_noisy_genuine_groups": len(false),
            "added_clean_or_isolated_groups": len(clean_false), "extra_genuine_time_fraction": extra,
            "gate": gate, "passed": all(gate.values())}


def review(path, native=None):
    document = io.read(path)
    lock, rows, responses = (document[key] for key in ("freeze", "roster", "responses"))
    verified_lock(lock, rows)
    for row, response in zip(rows, responses, strict=True):
        verify_windows(row, response, native_wave(native, row) if native else None)
    result = compare(rows, responses)
    require(result == document["result"], "Published results differ")
    print(io.encoded(result).decode(), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "review"))
    parser.add_argument("path", type=Path)
    for name in ("part-a", "part-b", "labels", "background", "native"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        review(args.path, args.native)
    elif args.phase == "score":
        score(args.path)
    else:
        if not all((args.part_a, args.part_b, args.labels, args.background)):
            parser.error("prepare requires --part-a, --part-b, --labels and --background")
        prepare(args.path, args.part_a, args.part_b, args.labels, args.background)


if __name__ == "__main__":
    main()
