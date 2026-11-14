"""Test unpadded overlap intervals on four previously unused AMI meetings."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import audio_locate_competing_speech as prior
from scripts import audio_paired_loud_background as common

ROOMS = ("ES2009a", "ES2014d", "IS1000d", "IS1005a")
SCHEMA = "chipmunk.audio-overlap-padding-transfer/1"
ARMS = ("right_padded", "unpadded", "agreement")
METHOD = {
    "source": "Freeze four new meeting cohorts, at least ten seconds between starts, eight high and eight minimal windows apiece using fixed annotation and hash order.",
    "comparison": "No added context beyond the three-second source for the unpadded arm. Compare the right-padded and unpadded input, and an intersection on the shared sample grid.",
    "threshold": "Reuse the original 250 ms minimum, 0.30 temporal IoU and 0.50 annotation recall. Measure all source-selected cases and retain failures.",
    "evidence": "For a new user-visible overlap region require four correct localizations per meeting with at most one minimal-overlap mark each, and four more high-overlap locations than the right-padded route across at least two meetings.",
    "purpose": "Acoustic context only. No speaker identity, authorship result or score change.",
}


def source_commitment(first):
    """Bind the earlier source and interval readout independently of runner packaging."""
    return prior.sha(json.dumps(first["fresh"], sort_keys=True, separators=(",", ":")).encode())


def freeze(folder, annotations, first):
    import subprocess
    from xml.etree import ElementTree
    from zipfile import ZipFile

    from scripts.audio_harmonic_overlap_transfer import room_activity

    initial = common.read(first)
    common.check(initial["schema"] == prior.SCHEMA and len(initial["fresh"]) == 64, "The first comparison is required")
    expected = initial["freeze"]["annotations_sha256"]
    common.check(common.digest(annotations) == expected, "Manual timing archive differs")
    used_rooms = set(initial["freeze"]["earlier_meetings"]) | {row["room"] for row in initial["fresh"]}
    verify = subprocess.run(["git", "rev-parse", initial["freeze"]["earlier_revision"]], capture_output=True, text=True,
                            cwd=common.ROOT, check=True).stdout.strip()
    cases = []
    with ZipFile(annotations) as archive:
        tree = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {item.get("observation"): {p.get("global_name") for p in item if p.get("global_name")}
                  for item in tree.iter() if item.get("observation")}
        occupied = set().union(*(people[room] for room in used_rooms))
        for room in ROOMS:
            members = people[room]
            common.check(len(members) == 4 and not occupied.intersection(members)
                         and room[:-1] not in {name[:-1] for name in used_rooms}, "Meeting cohorts overlap")
            occupied.update(members)
            activity = room_activity(archive, room)
            starts = []
            for label in ("high", "minimal"):
                choices = []
                for pos in range(400, len(activity) - 300, 400):
                    values = activity[pos:pos+300]
                    if (values > 0).sum() < prior.SPEECH_MIN:
                        continue
                    intervals = prior.true_ranges(values >= 2, 160)
                    duration = sum(end-start for start, end in intervals)
                    qualifies = duration >= 19_200 if label == "high" else duration <= 1_600
                    if qualifies:
                        choices.append((prior.sha(f"chipmunk/overlap-padding/1/{room}/{pos}".encode()), pos, intervals))
                selected = 0
                for _, pos, intervals in sorted(choices):
                    if any(abs(pos-earlier) < 1_000 for earlier in starts):
                        continue
                    starts.append(pos)
                    cases.append({"id": len(cases), "room": room, "start_sample": pos*160,
                                  "annotation": label, "truth": intervals})
                    selected += 1
                    if selected == 8:
                        break
                common.check(selected == 8, f"Need eight spaced {label} windows for {room}")
    folder.mkdir(mode=0o700)
    common.write(folder / "freeze.json", {"schema": SCHEMA, "recipe": METHOD,
        "runner_sha256": common.digest(Path(prior.__file__)), "comparison_sha256": common.digest(Path(__file__)),
        "first_source_intervals_sha256": source_commitment(initial), "annotations_sha256": expected,
        "earlier_revision": verify, "cases": cases})


def measure(session, data):
    import numpy as np

    raw = np.frombuffer(data, "<i2").astype(np.float32)[None, None, :] / 32768
    common.check(raw.shape == (1, 1, prior.SAMPLES), "Unexpected sample clock")
    masks = []
    for size in (160_000, prior.SAMPLES):
        source = np.pad(raw, ((0, 0), (0, 0), (0, size-prior.SAMPLES)))
        [logp] = session.run(["y"], {"x": source})
        expected = (size - 991)//prior.STRIDE+1
        common.check(logp.shape == (1, expected, 7) and np.isfinite(logp).all()
                     and np.allclose(np.exp(logp).sum(axis=-1), 1., rtol=0, atol=2e-5), "Unexpected model response")
        masks.append(logp[0, :175].argmax(axis=-1) >= 4)
    a, b = masks
    return {arm: prior.true_ranges(mask, prior.STRIDE, prior.CELL_FIRST, prior.MIN_RUN)
            for arm, mask in zip(ARMS, (a, b, a & b), strict=True)}


def score(folder, runtime):
    frozen, bindings = (common.read(folder / name) for name in ("freeze.json", "input.json"))
    common.check(frozen["comparison_sha256"] == common.digest(Path(__file__))
                 and bindings["freeze_sha256"] == common.digest(folder / "freeze.json"), "The frozen inputs differ")
    session, checkpoint = prior.model(runtime)
    rows = []
    for case, binding in zip(frozen["cases"], bindings["rows"], strict=True):
        pin = binding["pcm_sha256"]
        common.check(case["id"] == binding["id"], "Missing ordered source commitment")
        if pin is None:
            measured = None
        else:
            data = (folder / "pcm" / f"{case['id']:03}.pcm").read_bytes()
            common.check(prior.sha(data) == pin, "The original excerpt changed")
            measured = measure(session, data)
        rows.append(case | {"pcm_sha256": pin, "observations": measured})
        if len(rows) % 16 == 0:
            print(f"Replayed {len(rows)} native excerpts", flush=True)
    result = {"schema": SCHEMA, "freeze": frozen, "captures": bindings["captures"], "checkpoint_sha256": checkpoint,
              "rows": rows, "summary": summarize(rows)}
    common.write(folder / "receipt.json", result)
    print(json.dumps(result["summary"], indent=2))


def summarize(rows):
    report = {name: {room: Counter() for room in ROOMS} for name in ARMS}
    common.check(len(rows) == 64 and [row["id"] for row in rows] == list(range(64)), "Incorrect assessment coverage")
    for row in rows:
        label, room, measurements = row["annotation"], row["room"], row["observations"]
        common.check(label in ("high", "minimal") and room in ROOMS, "Unexpected evaluation group")
        for arm in ARMS:
            count = report[arm][room]
            count[f"{label}_planned"] += 1
            if measurements is None:
                count[f"{label}_unavailable"] += 1
                continue
            intervals = measurements[arm]
            prior.validate_intervals(intervals)
            hit, support, union = prior.overlap(intervals, row["truth"])
            count[f"{label}_marked"] += bool(intervals)
            count["high_localized"] += label == "high" and hit*10 >= 3*union and hit*2 >= support
    choices = {}
    for arm, counts in report.items():
        gain = {room: counts[room]["high_localized"]-report["right_padded"][room]["high_localized"] for room in ROOMS}
        choices[arm] = {"rooms": counts, "gain": gain, "evidence_gate": sum(gain.values()) >= 4
                        and sum(number > 0 for number in gain.values()) >= 2
                        and all(c["high_localized"] >= 4 and c["minimal_marked"] <= 1
                                and c["high_unavailable"] == c["minimal_unavailable"] == 0 for c in counts.values())}
    return choices


def validate_receipt(result):
    common.check(result["schema"] == SCHEMA and result["freeze"]["recipe"] == METHOD
                 and result["freeze"]["comparison_sha256"] == common.digest(Path(__file__))
                 and result["freeze"]["runner_sha256"] == common.digest(Path(prior.__file__))
                 and result["summary"] == summarize(result["rows"]), "The sealed output does not recount")
    prior.validate_cases(result["freeze"]["cases"], result["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "acquire", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--annotations", type=Path, default=prior.ANNOTATIONS)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        common.check(args.first, "Supply the first evaluation receipt")
        freeze(args.path, args.annotations, args.first)
    elif args.phase == "acquire":
        prior.acquire(args.path)
    elif args.phase == "score":
        common.check(args.runtime, "Specify the pinned CPU runtime")
        score(args.path, args.runtime)
    else:
        result = common.read(args.path)
        validate_receipt(result)
        if args.first:
            common.check(result["freeze"]["first_source_intervals_sha256"] == source_commitment(common.read(args.first)),
                         "The first trial's source and interval records differ")
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
