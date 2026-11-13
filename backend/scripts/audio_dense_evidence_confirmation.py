# SPDX-License-Identifier: Apache-2.0
"""Confirm fixed dense source windows on distinct published partial speech edits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_vocal_event_mixtures as io
from scripts import review_audio_reduced_background_localization as clock

SCHEMA = "chipmunk.audio-dense-evidence-confirmation/1"
PREFIXES = ("part-a-192m.prefix.tar", "part-a-192m-352m.tarseg", "slice-2g.tarseg", "slice-6g.tarseg", "slice-10g.tarseg")
BASES = ("audio-partial-publisher-transfer", "audio-reduced-background-localization", "audio-noise-window-transfer")
RATE, SPAN = clock.RATE, clock.WINDOW
METHOD = {
    "genuine_groups": 12, "edits_per_family": 3, "duration_samples": [6*RATE, 11*RATE],
    "background": "Port Harcourt traffic, foreground 8 dB above added sound, common peak-safe gain.",
    "background_sha256": "9702be26fec457ac8c54df61fbe62f120189119b5734377cda9e5ae442566efd",
    "background_offsets": [3, 6, 9, 13],
    "calibration_sha256": clock.METHOD["primary_calibration_sha256"],
    "geometry": "Original sparse sampler versus existing fixed 1.5-second hop. Use two adjacent AI directions and report their shared 1.5 seconds.",
    "complement": "Keep every original sparse range and add the dense intersections.",
    "advance": {"minimum_gained_edit_groups": 5, "minimum_gained_families": 3,
                "maximum_added_genuine_groups": 1, "maximum_added_genuine_time_fraction": .04},
}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def commits():
    return {key: io.sha(io.PUBLIC / f"{key}-2026-11-12.json") for key in BASES}


def history():
    seen, ids, pins = set(), set(), set()
    for key in BASES:
        panel = io.read(io.PUBLIC / f"{key}-2026-11-12.json")
        for row in panel["roster"]:
            if source := row["source"]:
                seen.add(source.get("group", source.get("speaker")))
                ids.add(source.get("publisher_id", source.get("id")))
                pins.add(source.get("source_pcm_sha256", source.get("pcm_sha256")))
    return seen, ids, pins


def choose(options):
    from scripts.audio_partial_publisher_transfer import GENERATORS

    selected, used = [], set()
    for label, family in [("genuine", None), *(("partial", name) for name in GENERATORS)]:
        count = METHOD["genuine_groups"] if family is None else METHOD["edits_per_family"]
        admitted = 0
        for row, raw in options[label, family]:
            if row["group"] in used:
                continue
            used.add(row["group"])
            selected.append((row, raw))
            admitted += 1
            if admitted == count:
                break
        require(admitted == count, f"Too few qualifying source groups for {label}/{family}")
    return selected


def prepare(run, archive, labels, background):
    from scripts.audio_gtcrn_traffic import float_pcm, pcm
    from scripts.audio_noise_views_transfer import match_views
    from scripts.audio_reduced_background_localization import source_candidates

    paths = [archive / name for name in PREFIXES]
    seen, ids, pins = history()
    candidates, count, _ = source_candidates(*paths[:2], labels, paths[2:])
    eligible = {}
    for key, sources in candidates.items():
        matches = [(row, raw) for row, raw in sources if row["publisher_id"] not in ids
                   and row["source_pcm_sha256"] not in pins
                   and METHOD["duration_samples"][0] <= row["samples"] <= METHOD["duration_samples"][1]]
        eligible[key] = sorted(matches, key=lambda pair: (pair[0]["group"] in seen,
                          io.digest(f"{SCHEMA}/{pair[0]['publisher_id']}".encode())))
    sources = choose(eligible)
    require(io.sha(background) == METHOD["background_sha256"], "Background recording differs")
    bed = float_pcm(io.decode(background))
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    rows = []
    for index, (source, raw) in enumerate(sources):
        start = METHOD["background_offsets"][index % 4] * RATE
        wave = float_pcm(raw)
        views = match_views(wave, {"traffic": bed[start:start + len(wave)]}, 8)
        row = {k: source[k] for k in ("group", "publisher_id", "source_wav_sha256", "source_pcm_sha256",
                                       "label", "family", "edits", "samples")}
        row["speaker_in_history"] = row["group"] in seen
        for scene, values in views.items():
            delivery = pcm(values)
            identifier = len(rows)
            (run / "pcm" / f"{identifier:03}.pcm").write_bytes(delivery)
            rows.append({**row, "id": identifier, "scene": scene, "pcm_sha256": io.digest(delivery),
                         "background_offset": start if scene == "traffic" else None})
    io.save(run / "roster.json", rows)
    lock = {"schema": SCHEMA, "method": METHOD, "history": commits(), "runner": io.sha(Path(__file__)),
            "roster_sha256": io.sha(run / "roster.json"), "source_labels_sha256": io.sha(labels),
            "source_ranges": {path.name: io.sha(path) for path in paths}, "source_members_examined": count,
            "voice_assets": io.assets(), "responses_at_freeze": 0}
    io.save(run / "freeze.json", lock)
    print("Frozen", len(rows), "previously unused source deliveries.", flush=True)


def read_run(run):
    lock, rows = [io.read(run / name) for name in ("freeze.json", "roster.json")]
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["history"] == commits()
            and lock["runner"] == io.sha(Path(__file__)) and lock["roster_sha256"] == io.sha(run / "roster.json"),
            "Frozen procedures or source inputs differ")
    return lock, rows


def native_wave(run, row):
    raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    require(len(raw) == 2 * row["samples"] and io.digest(raw) == row["pcm_sha256"], "Original PCM changed")
    return raw


def scores(run):
    from app.detectors.audio import calibration
    from scripts.audio_gtcrn_traffic import float_pcm
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_reduced_background_localization import assets_for_scores

    lock, rows = read_run(run)
    require(METHOD["calibration_sha256"] == calibration.SHA256, "Voice calibration differs")
    model, speech = assets_for_scores(lock)
    measurements = []
    for row in rows:
        raw = native_wave(run, row)
        wave = float_pcm(raw)
        ranges = sorted(set(tuple(span) for grid in clock.schedules(len(wave)) for span in grid))
        windows = []
        for start, end in ranges:
            entry = adapter_result(wave[start:end], model, speech)
            direction = entry["decision"]
            require((entry["scope"] and entry["windows"] == [[0, SPAN]]) or direction == "unassessed",
                    "Voice response outside three-second scope")
            windows.append({"span": [start, end], "direction": direction, "pcm_sha256": io.digest(raw[2*start:2*end])})
        measurements.append({"id": row["id"], "windows": windows})
        if len(measurements) % 10 == 0:
            print("Completed", len(measurements), "deliveries", flush=True)
    io.save(run / "responses.json", measurements)


def report(rows, responses):
    result = defaultdict(Counter)
    added, families, ordinary = set(), set(), set()
    for row, observed in zip(rows, responses, strict=True):
        sparse, dense = clock.schedules(row["samples"])
        directions = {tuple(item["span"]): item["direction"] for item in observed["windows"]}
        require(set(directions) == set(map(tuple, sparse + dense)) and observed["id"] == row["id"],
                "Scored source schedule differs")
        positive = {span for span, direction in directions.items() if direction == "ai"}
        baseline = [span for span in sparse if tuple(span) in positive]
        if len(baseline) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
            baseline = []
        candidate = clock.intersections(dense, positive)
        cell = result[f"{row['scene']}/{row['label']}"]
        truth = row["edits"]
        cell["files"] += 1
        cell["assessed"] += any(d != "unassessed" for d in directions.values())
        cell["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
        for name, spans in (("sparse", baseline), ("dense", candidate), ("combined", baseline + candidate)):
            cell[f"{name}_useful"] += clock.useful(spans, truth)
            cell[f"{name}_flagged"] += bool(spans)
            union = clock.intervals_union(spans)
            cell[f"{name}_genuine_samples"] += sum(b - a for a, b in union) - sum(
                clock.overlap(span, edit) for span in union for edit in truth)
        if clock.useful(candidate, truth) and not clock.useful(baseline, truth):
            added.add(row["group"])
            families.add(row["family"])
        if not truth and candidate and not baseline:
            ordinary.add(row["group"])
    delta = sum(cell["combined_genuine_samples"] - cell["sparse_genuine_samples"] for cell in result.values())
    fraction = delta / sum(cell["genuine_samples"] for cell in result.values())
    target = METHOD["advance"]
    decision = (len(added) >= target["minimum_gained_edit_groups"] and len(families) >= target["minimum_gained_families"]
                and len(ordinary) <= target["maximum_added_genuine_groups"]
                and fraction <= target["maximum_added_genuine_time_fraction"])
    return {"cells": dict(result), "added_groups": len(added), "added_families": sorted(families),
            "added_ordinary_groups": len(ordinary), "extra_genuine_time_fraction": fraction,
            "passes_local_evidence_gate": decision}


def review(path, run=None):
    document = io.read(path)
    lock, rows, responses = (document[key] for key in ("freeze", "roster", "responses"))
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["history"] == commits()
            and lock["runner"] == io.sha(Path(__file__)) and lock["responses_at_freeze"] == 0
            and lock["roster_sha256"] == io.digest(io.encoded(rows)), "Public freeze differs")
    seen, ids, pins = history()
    require(len(rows) == len(responses) == 60 and len({row["group"] for row in rows}) == 30, "Source groups differ")
    for index, (row, response) in enumerate(zip(rows, responses, strict=True)):
        require(row["id"] == response["id"] == index and row["publisher_id"] not in ids
                and row["source_pcm_sha256"] not in pins and row["speaker_in_history"] == (row["group"] in seen)
                and row["scene"] == ("traffic" if index % 2 else "gain_control")
                and row["source_pcm_sha256"] == rows[index ^ 1]["source_pcm_sha256"]
                and row["group"] == rows[index ^ 1]["group"], "Incorrect pairing or prior exposure")
        native = native_wave(run, row) if run else None
        for item in response["windows"]:
            start, end = item["span"]
            require(item["direction"] in {"ai", "original", "inconclusive", "unassessed"}
                    and clock.SHA.fullmatch(item["pcm_sha256"])
                    and (native is None or io.digest(native[2*start:2*end]) == item["pcm_sha256"]),
                    "Invalid original window commitment")
    outcome = report(rows, responses)
    require(outcome == document["result"], "Reported counts differ")
    print(io.encoded(outcome).decode(), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    for name in ("archive", "labels", "background", "native"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        review(args.path, args.native)
    else:
        if not all((args.archive, args.labels, args.background)):
            parser.error("run requires --archive, --labels and --background")
        prepare(args.path, args.archive, args.labels, args.background)
        scores(args.path)
        lock, rows = read_run(args.path)
        responses = io.read(args.path / "responses.json")
        output = args.path / "receipt.json"
        io.save(output, {"freeze": lock, "roster": rows, "responses": responses, "result": report(rows, responses)})
        review(output, args.path)


if __name__ == "__main__":
    main()
