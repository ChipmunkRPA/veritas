"""Compare source-aligned overlap intervals with annotation-selected AMI controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from urllib.request import Request, build_opener
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts import audio_paired_loud_background as common

PANEL = common.PANELS / "audio-harmonic-overlap-2026-11-11.json"
OLD_BANK = common.ROOT / "backend/data/independent-evaluation/audio-harmonic-overlap-2026-11-11/run"
ANNOTATIONS = common.ROOT / "backend/data/independent-evaluation/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
SCHEMA = "chipmunk.audio-locate-competing-speech/1"
ROOMS = ("ES2006a", "ES2011c", "IS1002c", "IS1008c")
RATE, SAMPLES, STRIDE, CELL_FIRST = 16_000, 48_000, 270, 360
SPEECH_MIN, OVERLAP_MIN = 245, 120
SPANS = ("right_padded", "centered", "agreement")
MIN_RUN = 4_000
RECIPE = {
    "sources": "Eight high-overlap and eight minimal-overlap nonoverlapping 3-second native audio excerpts per fresh AMI meeting, selected on manual words only. Cohorts and annotated participants are disjoint from meetings named in repository evaluation documents at freeze.",
    "reference": "Gold is at least two distinct speakers with concurrent manually timed word ranges, quantized to 10 ms. High has at least 1.2 s of overlap, minimal has at most 0.1 s. All excerpts contain at least 2.45 s of annotated speech.",
    "candidate": "Compare a right-padded input and a separately centered, silence-padded input. Their native receptive fields use the same 270-sample grid. Agreement retains only frames whose two-speaker-class flags reproduce in both placements. Require 250 ms of contiguous frames in each arm.",
    "evidence_gate": "At least half of high excerpts correctly localized in every fresh meeting with no more than one marker on minimal overlap per meeting. Correct localization requires at least 0.30 temporal IoU and 0.50 recall of annotated overlap. Counts are not an AI decision.",
    "decision_gate": "Replay a separate 93-clip bank with known genuine and generated speech. A marker may withhold a primary AI flag only. Require at least two genuine false flags withheld, at most one synthetic hit withheld and zero clean synthetic losses.",
}
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def capture_url(room):
    return f"{MIRROR}/{room}/audio/{room}.Array1-01.wav"


def true_ranges(active, size, first=0, minimum=1):
    intervals = []
    start = None
    for index, enabled in enumerate([*active, False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            if (index - start) * size >= minimum:
                intervals.append([first + start * size, first + index * size])
            start = None
    return intervals


def plan(annotations):
    from scripts.audio_harmonic_overlap_transfer import room_activity

    expected = common.read(PANEL)["freeze"]["annotations_sha256"]
    common.check(common.digest(annotations) == expected, "Pinned AMI manual word timing release required")
    # Bind the scan to the publication history immediately before this study.
    import subprocess

    revision = "f5080f4"
    pattern = r"\b(ES|IS|TS)[0-9]{4}[a-d]\b"
    grep = subprocess.run(["git", "grep", "-h", "-o", "-E", r"(ES|IS|TS)[0-9]{4}[a-d]", revision, "--", "docs"],
                          cwd=common.ROOT, text=True, check=True, capture_output=True)
    earlier_rooms = set(grep.stdout.splitlines())
    common.check(all(re.fullmatch(pattern, room) for room in earlier_rooms), "Unexpected earlier source list")
    with ZipFile(annotations) as archive:
        tree = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {row.get("observation"): {p.get("global_name") for p in row if p.get("global_name")}
                        for row in tree.iter() if row.get("observation")}
        used = set().union(*(participants.get(room, set()) for room in earlier_rooms))
        cases = []
        for room in ROOMS:
            members = participants[room]
            common.check(len(members) == 4 and not used.intersection(members)
                         and room[:-1] not in {name[:-1] for name in earlier_rooms}, "Known meeting cohort exposure")
            used.update(members)
            activity = room_activity(archive, room)
            for label in ("high", "minimal"):
                choices = []
                for pos in range(400, len(activity) - 300, 400):
                    values = activity[pos:pos + 300]
                    if (values > 0).sum() < SPEECH_MIN:
                        continue
                    intervals = true_ranges(values >= 2, 160)
                    duration = sum(end - start for start, end in intervals)
                    if (duration >= OVERLAP_MIN * 160 if label == "high" else duration <= 1_600):
                        choices.append((sha(f"chipmunk/overlap-localization/1/{room}/{pos}".encode()), pos, intervals))
                common.check(len(choices) >= 8, "Insufficient annotation-only intervals")
                for _, pos, intervals in sorted(choices)[:8]:
                    cases.append({"id": len(cases), "room": room, "start_sample": pos * 160,
                                  "annotation": label, "truth": intervals})
    return {"schema": SCHEMA, "recipe": RECIPE, "runner_sha256": common.digest(Path(__file__)),
            "annotations_sha256": expected, "earlier_revision": revision,
            "earlier_meetings": sorted(earlier_rooms), "cases": cases}


def fetch(url, start, size, expected_total=None):
    from scripts.replay_audio_bandwidth_public import NoRedirect

    end = start + size - 1
    request = Request(f"{url}?chipmunk-byte-offset={start}", headers={"Range": f"bytes={start}-{end}",
                      "Accept-Encoding": "identity", "User-Agent": "Chipmunk-Verify-overlap-replay/1"})
    with build_opener(NoRedirect()).open(request, timeout=30) as reply:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", reply.headers.get("Content-Range", ""))
        common.check(reply.status == 206 and match is not None and tuple(map(int, match.groups()[:2])) == (start, end)
                     and reply.headers.get("Content-Encoding", "identity") == "identity", "Incorrect publisher byte range")
        total = int(match[3])
        common.check(total > end and expected_total in (None, total), "Source object length changed")
        data = reply.read(size + 1)
    common.check(len(data) == size, "Incomplete publisher range")
    return data, total


def acquire(folder):
    from concurrent.futures import ThreadPoolExecutor

    frozen = common.read(folder / "freeze.json")
    common.check(frozen["runner_sha256"] == common.digest(Path(__file__)), "Prospective method differs")
    voice = folder / "pcm"
    voice.mkdir(exist_ok=True)
    captures = {}
    rooms = tuple(dict.fromkeys(case["room"] for case in frozen["cases"]))
    for room in rooms:
        payload, total = fetch(capture_url(room), 0, 44)
        *head, count = struct.unpack("<4sI4s4sIHHIIHH4sI", payload)
        common.check(tuple(head) == (b"RIFF", total-8, b"WAVE", b"fmt ", 16, 1, 1, RATE, RATE*2, 2, 16, b"data")
                     and count % 2 == 0 and 0 < count <= total - 44, "Unsupported native AMI WAV header")
        captures[room] = {"url": capture_url(room), "bytes": total, "header_sha256": sha(payload), "samples": count // 2}

    def download(case):
        try:
            record = captures[case["room"]]
            beg = case["start_sample"]
            common.check(beg + SAMPLES <= record["samples"], "Excerpt exceeds AMI sample extent")
            payload, _ = fetch(record["url"], 44 + beg*2, SAMPLES*2, record["bytes"])
            path = voice / f"{case['id']:03}.pcm"
            if path.exists():
                common.check(path.read_bytes() == payload, "Existing native PCM differs")
            else:
                with path.open("xb") as out:
                    out.write(payload)
            return {"id": case["id"], "pcm_sha256": sha(payload)}
        except (ValueError, OSError) as error:
            print(f"Input {case['id']} unavailable: {error}", flush=True)
            return {"id": case["id"], "pcm_sha256": None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        materialized = list(pool.map(download, frozen["cases"]))
    for room in rooms:
        print(f"{room}: {sum(row['pcm_sha256'] is not None for row, spec in zip(materialized, frozen['cases'], strict=True) if spec['room'] == room)}/16 excerpts backed by native samples", flush=True)
    common.write(folder / "input.json", {"freeze_sha256": common.digest(folder / "freeze.json"),
                 "captures": captures, "rows": materialized})


def model(runtime, model_archive=None):
    import sys

    from scripts import audio_overlap_context_v180 as overlap
    from scripts.audio_strong_traffic_view import network_free

    site = overlap.verify_runtime(runtime)
    sys.path.insert(0, str(site))
    import onnxruntime as ort

    if model_archive is None:
        payload = overlap.weights()
    else:
        from scripts.video_appearance_corrected_v112 import scoped

        common.check(model_archive.name == f"{overlap.ARCHIVE_NAME}.tar.bz2", "Unexpected publisher asset name")
        payload = scoped(overlap.weights, {"SOURCE": model_archive.parent})()
    network_free()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(payload, sess_options=options, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    common.check(Path(ort.__file__).resolve().is_relative_to(site) and ort.__version__ == "1.23.2"
                 and session.get_providers() == ["CPUExecutionProvider"]
                 and tuple(int(meta[key]) for key in ("sample_rate", "window_size", "receptive_field_size",
                     "receptive_field_shift", "num_classes")) == (RATE, 160_000, 991, STRIDE, 7),
                 "Unexpected segmentation model or CPU runtime")
    return session, overlap.ASSETS["model.onnx"]


def observations(session, data):
    import numpy as np

    samples = np.frombuffer(data, "<i2").astype(np.float32) / 32768
    common.check(samples.shape == (SAMPLES,), "Expected exactly three seconds")
    masks = []
    for offset in (0, 55_890):
        model_input = np.zeros((1, 1, 160_000), dtype=np.float32)
        model_input[0, 0, offset:offset+SAMPLES] = samples
        [logp] = session.run(["y"], {"x": model_input})
        start = offset // STRIDE
        values = logp[0, start:start+(SAMPLES-991)//STRIDE+1]
        common.check(values.shape == (175, 7) and np.isfinite(values).all()
                     and np.allclose(np.exp(values).sum(axis=-1), 1., rtol=0, atol=2e-5), "Unexpected frame response")
        masks.append(values.argmax(axis=-1) >= 4)
    left, right = masks
    return {arm: true_ranges(array, STRIDE, CELL_FIRST, MIN_RUN)
            for arm, array in zip(SPANS, (left, right, left & right), strict=True)}


def held_rows(bank):
    published = common.read(PANEL)
    for filename, expected in (("freeze", published["freeze_sha256"]), ("baseline", published["baseline_sha256"]),
                               ("roster", published["freeze"]["roster_sha256"])):
        common.check(common.digest(bank / f"{filename}.json") == expected, "Earlier matched decision bank differs")
    records = {r["id"]: r for r in common.read(bank / "baseline.json")["rows"]}
    groups = {}
    for row in common.read(bank / "roster.json"):
        if row["role"] != "confirmation":
            continue
        outcome = records.pop(row["id"])
        common.check(outcome["pcm_sha256"] == row["pcm_sha256"], "Archived decision belongs to different samples")
        group = groups.setdefault(row["group"], len(groups))
        pcm = (bank / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        common.check(len(pcm) == 2*SAMPLES and sha(pcm) == row["pcm_sha256"], "Incorrect archived source samples")
        yield row, outcome, group, pcm
    common.check(not records, "Archived primary response ledger has unbound inputs")


def score(folder, runtime, bank, model_archive=None):
    frozen, sources = (common.read(folder / name) for name in ("freeze.json", "input.json"))
    common.check(frozen["runner_sha256"] == common.digest(Path(__file__))
                 and sources["freeze_sha256"] == common.digest(folder / "freeze.json"), "Input freeze changed")
    session, checkpoint = model(runtime, model_archive)
    fresh = []
    for spec, source in zip(frozen["cases"], sources["rows"], strict=True):
        common.check(spec["id"] == source["id"], "Unmatched native input")
        pin = source["pcm_sha256"]
        if pin is not None:
            data = (folder / "pcm" / f"{spec['id']:03}.pcm").read_bytes()
            common.check(sha(data) == pin, "Native excerpt changed")
        measured = observations(session, data) if pin is not None else None
        fresh.append(spec | {"pcm_sha256": pin, "observations": measured})
    for room in ROOMS:
        print(f"Measured {room}", flush=True)
    matched = []
    for row, outcome, group, data in held_rows(bank):
        matched.append({"id": len(matched), "condition": row["condition"], "family": row["family"], "group": group,
                        "primary": outcome["decision"], "observations": observations(session, data)})
    common.check(len(matched) == 93, "Unexpected number of matched voice decisions")
    result = {"schema": SCHEMA, "freeze": frozen, "captures": sources["captures"],
              "decision_bank_sha256": common.digest(PANEL), "checkpoint_sha256": checkpoint,
              "fresh": fresh, "matched": matched, "summary": summarize(fresh, matched)}
    common.write(folder / "receipt.json", result)
    print(json.dumps(result["summary"], indent=2))


def overlap(predicted, target):
    area = sum(max(0, min(end, right) - max(start, left)) for start, end in predicted for left, right in target)
    reference = sum(end - start for start, end in target)
    union = reference + sum(end - start for start, end in predicted) - area
    return area, reference, union


def validate_intervals(intervals):
    cursor = 0
    for start, end in intervals:
        common.check(type(start) is type(end) is int and cursor <= start < end <= SAMPLES
                     and end-start >= MIN_RUN and all((edge-CELL_FIRST) % STRIDE == 0 for edge in (start, end)),
                     "Invalid measured source interval")
        cursor = end


def summarize(fresh, matched):
    for row in [*fresh, *matched]:
        values = row["observations"]
        if values is None:
            continue
        common.check(set(values) == set(SPANS), "Invalid view inventory")
        for intervals in values.values():
            validate_intervals(intervals)
    report = {}
    for arm in SPANS:
        rooms = {room: Counter() for room in ROOMS}
        for row in fresh:
            results = rooms[row["room"]]
            label = row["annotation"]
            results[f"{label}_planned"] += 1
            if row["observations"] is None:
                results[f"{label}_unavailable"] += 1
                continue
            spans = row["observations"][arm]
            area, annotated, union = overlap(spans, row["truth"])
            results[f"{label}_marked"] += bool(spans)
            results["high_localized"] += label == "high" and 10*area >= 3*union and 2*area >= annotated
        by_condition = {}
        for row in matched:
            label = row["family"] if row["family"] in {"genuine", "background"} else "synthetic"
            tally = by_condition.setdefault(f"{row['condition']}/{label}", Counter())
            tally["files"] += 1
            tally[f"primary_{row['primary']}"] += 1
            tally["withheld_ai"] += row["primary"] == "ai" and bool(row["observations"][arm])
        removed_false = sum(count["withheld_ai"] for key, count in by_condition.items() if key.endswith("/genuine"))
        lost_synth = sum(count["withheld_ai"] for key, count in by_condition.items() if key.endswith("/synthetic"))
        evidence = all(room["high_localized"] >= 4 and room["minimal_marked"] <= 1
                       and room["minimal_unavailable"] == 0 for room in rooms.values())
        decisions = removed_false >= 2 and lost_synth <= 1 and not by_condition["clean/synthetic"]["withheld_ai"]
        report[arm] = {"rooms": rooms, "decision_bank": by_condition, "withheld_false_ai": removed_false,
                       "withheld_synthetic_ai": lost_synth, "evidence_gate": evidence, "decision_gate": decisions}
    return report


def validate_receipt(record):
    frozen, cases = record["freeze"], record["fresh"]
    common.check(record["schema"] == SCHEMA and frozen["recipe"] == RECIPE
                 and frozen["runner_sha256"] == common.digest(Path(__file__))
                 and record["decision_bank_sha256"] == common.digest(PANEL)
                 and record["summary"] == summarize(cases, record["matched"]), "Saved interval counts differ")
    validate_cases(frozen["cases"], cases)


def validate_cases(frozen, measured):
    keys = ("id", "room", "start_sample", "annotation", "truth")
    common.check([{key: row[key] for key in keys} for row in measured] == frozen, "Native source decisions differ from the freeze")
    for row in measured:
        pin = row["pcm_sha256"]
        common.check((pin is None) == (row["observations"] is None) and
                     (pin is None or bool(re.fullmatch("[0-9a-f]{64}", pin))), "Native source commitment differs")


def replay_publisher(receipt, folder, phase, *, runtime=None, model_archive=None):
    """Rebuild the publicly available excerpts and optionally verify the native model output for each."""
    cases = receipt.get("fresh", receipt.get("rows"))
    validate_cases(receipt["freeze"]["cases"], cases)
    if phase == "fetch":
        folder.mkdir(mode=0o700)
        common.write(folder / "freeze.json", receipt["freeze"])
        acquire(folder)
    inputs = common.read(folder / "input.json")
    common.check(inputs["captures"] == receipt["captures"] and inputs["freeze_sha256"] == common.digest(folder / "freeze.json")
                 and common.read(folder / "freeze.json") == receipt["freeze"]
                 and inputs["rows"] == [{key: row[key] for key in ("id", "pcm_sha256")} for row in cases],
                 "Reopened native AMI references differ")
    if phase == "model":
        session, checkpoint = model(runtime, model_archive)
        common.check(checkpoint == receipt["checkpoint_sha256"], "Different checkpoint digest")
    for row in cases:
        if row["pcm_sha256"] is None:
            continue
        data = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        common.check(len(data) == 2*SAMPLES and sha(data) == row["pcm_sha256"], "Different native source bytes")
        if phase == "model":
            if receipt["schema"] == SCHEMA:
                measured = observations(session, data)
            else:
                from scripts.audio_overlap_padding_transfer import measure

                measured = measure(session, data)
            common.check(measured == row["observations"], f"Source interval replay differs on {row['id']}")
    print(f"Verified {sum(row['pcm_sha256'] is not None for row in cases)} native excerpts ({phase})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "acquire", "score", "review", "replay-fetch", "replay-model"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--model-archive", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--bank", type=Path, default=OLD_BANK)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        common.write(args.path / "freeze.json", plan(args.annotations))
    elif args.phase == "acquire":
        acquire(args.path)
    elif args.phase == "score":
        common.check(args.runtime is not None, "Specify the pinned local CPU runtime")
        score(args.path, args.runtime, args.bank, args.model_archive)
    else:
        record = common.read(args.path)
        if record["schema"] == SCHEMA:
            validate_receipt(record)
        else:
            from scripts.audio_overlap_padding_transfer import validate_receipt as validate_transfer

            validate_transfer(record)
        if args.phase == "review":
            print(json.dumps(record["summary"], indent=2))
        else:
            phase = args.phase.removeprefix("replay-")
            common.check(args.cache and (phase != "model" or args.runtime), "Specify a cache and the model runtime as needed")
            replay_publisher(record, args.cache, phase, runtime=args.runtime, model_archive=args.model_archive)


if __name__ == "__main__":
    main()
