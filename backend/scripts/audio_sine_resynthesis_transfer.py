"""Replay a bounded SINE comparison with resynthesis and competing speech controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import math
import re
import struct
import tarfile
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

from scripts import audio_interval_learning as learned
from scripts import audio_locate_competing_speech as ami
from scripts import audio_source_isolation_diagnostic as prior
from scripts import review_audio_sine_preview as sine
from scripts import review_audio_stock_voice_transfer as stock

SHARDS = (2, 3, 13, 17, 25, 29)
PREFIX = 192 * 1024**2
LIST_SHA = "c3429d3d092b24ef18256a588cfbe9ef90818cbb0ee0a06c717fe41dfe985e63"
FIT_SHA = "8b40a4ba5fc3315c6b27fd235bcffff2a7273575dafa9f8aa41fe3443ea93926"
POINTS_PARENT = prior.PANELS / "audio-interval-learning-2026-11-13.json"
FIT = prior.PANELS / "audio-interval-learning-fit-2026-11-13.json"
RUNTIME = ("numpy", "scipy", "torch", "transformers", "pyloudnorm")
MEMBER = re.compile(r"dev_(real|resyn|edit|cut_paste)_medium-(\d+-[a-zA-Z0-9_-]+)\.wav")
ROOMS = ("ES2015a", "ES2016a", "TS3006a", "TS3008a", "TS3009a", "TS3010a", "TS3011a", "TS3012a")
ROLES = ("resyn", "cut_paste", "edit")
TREATMENTS = ("clean", "competing_speech")
SCHEMA = "chipmunk.audio-sine-resynthesis-transfer/1"
RATE, SAMPLES = learned.RATE, learned.SAMPLES
METHOD = {
    "sources": "Take first 192 MiB from each of six pinned SINE shards. Find exact filename-stem matches between originals and each of the three treatments in its own shard. Exclude reader 100. Use hash order to select up to 12 pairs per treatment, at most one per reader within each treatment and no reused utterance across treatments. A reader may recur across treatments.",
    "clock": "Use one six-second view of each member. For cut-and-paste use the publisher's single interior region, require 0.20–1.00 s edit length and use an edit-containing crop chosen by hash on a 100 ms grid within both matched clocks. Target is supplied only to fixture preparation and evaluation. For resynthesis and infill use a common start at 0 s. Report infill clip-level flags only: the bounded prefix does not supply medium_edit.txt, and the cut-and-paste label is not assigned to infill.",
    "background": "Use distinct six-second AMI ranges from eight rooms with no shared declared participants among the named earlier room bank at commit 2fdcae31. Select by hash on a three-second start grid, with at least five seconds of annotated speech and two seconds of simultaneous words by different speakers. Add one identical PCM background to each matched pair with its amplitude fixed from the original at nominal -5 dB. Apply one common peak-safe scale and quantize the two stems separately.",
    "comparison": "Compare the stored five two-second window point and both frozen interval heads. Use no new training or point selection. Report flag counts separately from useful source locations and report paired original/variant transitions by condition. Reuse the earlier overlap criterion. Include each AMI background alone as a control.",
}


def source_listing(cache):
    manifest = cache / "list.json"
    prior.check(prior.path_sha(manifest) == LIST_SHA, "SINE publisher listing differs")
    return {entry["path"]: entry for entry in prior.read(manifest)}


def shard_members(cache, shard, expected):
    filename = f"split-{shard}-192"
    path = cache / (filename + ".range")
    headers = (cache / (filename + ".headers")).read_text().lower().splitlines()
    prior.check(f"content-range: bytes 0-{PREFIX-1}/{expected['size']}" in headers
                and f'etag: "{expected["xetHash"]}"' in headers and path.stat().st_size == PREFIX,
                "Unexpected publisher range response")
    members, labels = defaultdict(dict), {}
    with tarfile.open(path, "r|gz") as container:
        while True:
            try:
                entry = container.next()
                if entry is None:
                    break
                name = Path(entry.name).name
                if entry.isfile() and name in ("medium_real.txt", "medium_resyn.txt", "medium_cut_paste.txt"):
                    raw = container.extractfile(entry).read()
                    labels[name] = dict((key, (clock, bit)) for key, clock, bit in
                                        (line.split() for line in raw.decode().splitlines()))
                match = MEMBER.fullmatch(name)
                if entry.isfile() and match:
                    prior.check(0 < entry.size <= 480000, "Unexpected SINE member extent")
                    raw = container.extractfile(entry).read()
                    role, key = match.groups()
                    prior.check(len(raw) == entry.size and key not in members[role], "Duplicate or incomplete member")
                    members[role][key] = raw
            except (EOFError, tarfile.ReadError):
                break  # The final member of a bounded compressed prefix may be incomplete.
    prior.check(len(labels) == 3 and labels["medium_real.txt"] == labels["medium_resyn.txt"], "Unexpected SINE label layout")
    return members, labels, {"shard": shard, "publisher_bytes": expected["size"], "publisher_sha256": expected["lfs"]["oid"],
                              "publisher_etag": expected["xetHash"], "prefix_sha256": prior.path_sha(path),
                              "complete_wavs": sum(map(len, members.values()))}


def wav(raw):
    with wave.open(io.BytesIO(raw)) as src:
        prior.check(src.getparams()[:3] == (1, 2, sine.RATE) and src.getcomptype() == "NONE", "Unexpected SINE WAV format")
        count, data = src.getnframes(), src.readframes(src.getnframes())
    prior.check(len(data) == count * 2, "Incomplete native SINE waveform")
    return count, prior.digest(data)


def extract(cache, output):
    root = output / "sine"
    root.mkdir()
    lookup, candidates, pins = source_listing(cache), defaultdict(list), []
    for shard in SHARDS:
        members, labels, commitment = shard_members(cache, shard, lookup[f"data/split-{shard}.tar.gz"])
        pins.append(commitment)
        for key, real in members["real"].items():
            reader = key.split("-", 1)[0]
            if reader == "100" or not any(key in members[role] for role in ROLES):
                continue
            host_extent, host_pin = wav(real)
            clock, label = labels["medium_real.txt"][f"dev_real_medium-{key}"]
            sine.label_clock(clock, host_extent, "real")
            prior.check(label == "1", "Original label disagrees with its member role")
            for role in ROLES:
                alternate = members[role].get(key)
                if alternate is None:
                    continue
                alt_extent, alt_pin = wav(alternate)
                max_start = min(host_extent, alt_extent) - 6 * sine.RATE
                if max_start < 0:
                    continue
                truth, start, label_clock = None, 0, None
                if role == "resyn":
                    sine.label_clock(clock, alt_extent, "real")
                elif role == "cut_paste":
                    label_clock, label = labels["medium_cut_paste.txt"][f"dev_cut_paste_medium-{key}"]
                    prior.check(label == "0", "Edit label disagrees with its member role")
                    intervals, _ = sine.label_clock(label_clock, alt_extent, role)
                    a, b, _ = intervals[1]
                    if not .20 * sine.RATE <= b - a <= sine.RATE:
                        continue
                    starts = [i for i in range(0, max_start + 1, sine.RATE // 10) if i <= a < b <= i + 6 * sine.RATE]
                    if not starts:
                        continue
                    start = min(starts, key=lambda i: prior.digest(f"sine-clock-v1/{key}/{i}".encode()))
                    truth = [(a - start) * 2 // 3, (b - start) * 2 // 3]
                payloads = {"original": (real, host_extent, host_pin), "variant": (alternate, alt_extent, alt_pin)}
                record = {"role": role, "reader": reader, "stem": key, "shard": shard,
                          "native_start": start, "truth": truth, "publisher_cut_paste_clock": label_clock,
                          "files": {name: {"wav_sha256": prior.digest(raw), "pcm_sha256": pin, "samples": extent}
                                    for name, (raw, extent, pin) in payloads.items()}}
                order = prior.digest(f"sine-compare-v1/{role}/{key}".encode())
                candidates[role].append((order, record, payloads))
    utterances, selected = set(), []
    for role in ROLES:
        readers = set()
        for _, row, payloads in sorted(candidates[role], key=lambda item: item[0]):
            reader, stem = row["reader"], row["stem"]
            if reader in readers or stem in utterances:
                continue
            readers.add(reader)
            utterances.add(stem)
            group = len(selected)
            for name, (data, _, _) in payloads.items():
                (root / f"{group:03}-{name}.wav").write_bytes(data)
            selected.append(row | {"id": group})
            if len(readers) == 12:
                break
    prior.check(all(sum(row["role"] == role for row in selected) >= 8 for role in ROLES), "Insufficient independently read pairs")
    return selected, pins


def background_plan(annotations):
    import subprocess
    from xml.etree import ElementTree

    from scripts.audio_harmonic_overlap_transfer import room_activity

    pin = prior.read(prior.PANELS / "audio-locate-competing-speech-2026-11-13.json")["freeze"]["annotations_sha256"]
    prior.check(prior.path_sha(annotations) == pin, "Unexpected AMI release")
    previous = subprocess.run(["git", "grep", "-h", "-o", "-E", r"(ES|IS|TS)[0-9]{4}[a-d]", "2fdcae31", "--", "docs"],
                              cwd=prior.ROOT, text=True, capture_output=True, check=True)
    with ZipFile(annotations) as archive:
        tree = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        members = {r.get("observation"): {i.get("global_name") for i in r if i.get("global_name")}
                   for r in tree.iter() if r.get("observation")}
        exposed = set().union(*(members.get(room, set()) for room in set(previous.stdout.splitlines())))
        targets = []
        for room in ROOMS:
            prior.check(len(members[room]) == 4 and not (members[room] & exposed), "AMI participants repeat across roles")
            exposed.update(members[room])
            activity = room_activity(archive, room)
            choices = []
            for pos in range(1800, len(activity) - 600, 300):
                one, two = int((activity[pos:pos+600] > 0).sum()), int((activity[pos:pos+600] >= 2).sum())
                if one >= 500 and two >= 200:
                    choices.append((prior.digest(f"sine-background-v1/{room}/{pos}".encode()), pos, one, two))
            _, pos, one, two = min(choices)
            targets.append({"room": room, "start_sample": pos * 160, "speech_samples": one * 160,
                            "overlap_samples": two * 160, "url": ami.capture_url(room)})
    return pin, targets


def download_background(row, root):
    location = row["url"]
    header, total = ami.fetch(location, 0, 44)
    *fields, extent = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
    prior.check(tuple(fields) == (b"RIFF", total - 8, b"WAVE", b"fmt ", 16, 1, 1, RATE, RATE * 2, 2, 16, b"data")
                and row["start_sample"] + SAMPLES <= extent // 2, "Unexpected AMI audio header")
    raw, _ = ami.fetch(location, 44 + 2 * row["start_sample"], 2 * SAMPLES, total)
    (root / f"{row['room']}.pcm").write_bytes(raw)
    return row | {"byte_size": total, "header_sha256": prior.digest(header), "pcm_sha256": prior.digest(raw)}


def freeze(folder, cache):
    folder.mkdir(mode=0o700)
    rows, pins = extract(cache, folder)
    print(f"Selected {len(rows)} matched SINE pairs at source clocks", flush=True)
    annotation_hash, targets = background_plan(ami.ANNOTATIONS)
    background_path = folder / "background"
    background_path.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        background = list(pool.map(lambda row: download_background(row, background_path), targets))
    for row in rows:
        row["room"] = background[row["id"] % len(background)]["room"]
    prior.write(folder / "freeze.json", {"schema": SCHEMA, "method": METHOD, "revision": sine.REVISION,
                "annotations_sha256": annotation_hash, "points_parent_sha256": prior.path_sha(POINTS_PARENT),
                "fit_sha256": FIT_SHA, "source_prefix_bytes": PREFIX, "shards": pins, "pairs": rows, "backgrounds": background})
    print(f"Verified {len(background)} new AMI source ranges", flush=True)


def mix(real, other, background):
    import numpy as np

    real, other = (x.astype(np.float64) for x in (real, other))
    room_norm = np.linalg.norm(background)
    prior.check(room_norm > 0 and np.any(real) and np.any(other), "Silent input in a matched speech pair")
    added = background.astype(np.float64) * (np.linalg.norm(real) / room_norm) * 10**.25
    peak = max(np.max(np.abs(x + added)) for x in (real, other))
    scale = min(1., 32000 / peak)
    # Independent rounding preserves the contribution of the common background exactly.
    return [(np.rint(x * scale) + np.rint(added * scale)).astype("<i2") for x in (real, other)]


def native_rows(folder, locked):
    import numpy as np
    from scipy.signal import resample_poly

    decoded = {}
    for row in locked["backgrounds"]:
        raw = (folder / "background" / f"{row['room']}.pcm").read_bytes()
        prior.check(len(raw) == 2 * SAMPLES and prior.digest(raw) == row["pcm_sha256"], "Background input differs")
        decoded[row["room"]] = np.frombuffer(raw, dtype="<i2")
    for row in locked["pairs"]:
        clips = []
        for side in ("original", "variant"):
            facts = row["files"][side]
            raw = (folder / "sine" / f"{row['id']:03}-{side}.wav").read_bytes()
            prior.check(prior.digest(raw) == facts["wav_sha256"] and wav(raw) == (facts["samples"], facts["pcm_sha256"]), "SINE input differs")
            with wave.open(io.BytesIO(raw)) as stream:
                stream.setpos(row["native_start"])
                pcm = np.frombuffer(stream.readframes(6 * sine.RATE), dtype="<i2")
            resampled = np.clip(np.rint(resample_poly(pcm.astype(np.float64), 2, 3)), -32768, 32767).astype("<i2")
            prior.check(len(resampled) == SAMPLES, "Decoded crop readout changed")
            clips.append(resampled)
        noisy = mix(*clips, decoded[row["room"]])
        for condition, signals in zip(TREATMENTS, (clips, noisy), strict=True):
            for side, values in zip(("original", "variant"), signals, strict=True):
                yield {"id": row["id"], "side": side, "condition": condition, "role": row["role"],
                       "truth": row["truth"] if side == "variant" else None}, values
    for index, (room, pcm) in enumerate(decoded.items()):
        yield {"id": index, "side": room, "condition": "competing_speech", "role": "background_only", "truth": None}, pcm


def audit_native(folder, result):
    for (binding, values), recorded in zip(native_rows(folder, result["freeze"]), result["records"], strict=True):
        prior.check(all(recorded[key] == value for key, value in binding.items())
                    and prior.digest(values.tobytes()) == recorded["pcm_sha256"], "Native delivery changed")


def audit_sources(cache, locked):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="chipmunk-sine-source-audit-") as tmp:
        scratch = Path(tmp)
        rows, shards = extract(cache, scratch)
    expected = [{key: value for key, value in row.items() if key != "room"} for row in locked["pairs"]]
    pin, targets = background_plan(ami.ANNOTATIONS)
    measured = [{key: row[key] for key in targets[0]} for row in locked["backgrounds"]]
    prior.check(rows == expected and shards == locked["shards"] and pin == locked["annotations_sha256"]
                and targets == measured, "Publisher inputs, source labels, or native room ranges differ")
    return {"matched_pairs": len(rows), "background_ranges": len(targets), "native_source_audit": True}


def score(folder, locked, fit, weights):
    import numpy as np

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    prior.check(prior.path_sha(fit) == FIT_SHA, "Frozen fitted heads required")
    coefficients = prior.read(fit)
    network_free()
    model = load_voice({"weights": weights, "assets": ASSETS})
    capture, activity = prior.FrameCapture(model), SpeechActivity(weights / "silero_vad.jit")
    records = []
    for row, pcm in native_rows(folder, locked):
        signal = pcm.astype(np.float32) / 32768
        features, spans = learned.design(signal, activity, capture)
        existing = [logit(model.predict([signal[a:b]])[0]) if window_in_speech(spans, a / RATE, b / RATE) else None for a, b in stock.WINDOWS]
        result = learned.summarize({"existing_scores": existing, "features": features, "group": row["id"],
                                    "kind": row["role"], "family": None, "truth": row["truth"],
                                    "pcm_sha256": prior.digest(pcm.tobytes())}, coefficients)
        records.append(row | {key: result[key] for key in ("pcm_sha256", "scores")})
        if len(records) % 12 == 0:
            print(f"{len(records)} original, edited or control deliveries scored", flush=True)
    return records


def score_receipt(locked, records):
    from scripts.prepare_audio_specialists_v15 import ASSETS

    return {"freeze": locked, "points": {"existing": stock.SHORT_POINT, **prior.read(POINTS_PARENT)["points"]},
            "records": records, "assets": ASSETS, "runtime": {key: importlib.metadata.version(key) for key in RUNTIME}}


def review(result):
    locked, rows = result["freeze"], result["records"]
    points = {"existing": stock.SHORT_POINT, **prior.read(POINTS_PARENT)["points"]}
    prior.check(locked["schema"] == SCHEMA and locked["method"] == METHOD and result["points"] == points
                and locked["points_parent_sha256"] == prior.path_sha(POINTS_PARENT) and locked["fit_sha256"] == FIT_SHA,
                "Fixed method or points changed")
    prior.check([row["id"] for row in locked["pairs"]] == list(range(len(locked["pairs"])))
                and len({row["stem"] for row in locked["pairs"]}) == len(locked["pairs"])
                and len({(row["role"], row["reader"]) for row in locked["pairs"]}) == len(locked["pairs"])
                and [row["room"] for row in locked["backgrounds"]] == list(ROOMS), "Incomplete or repeated source groups")
    for row in locked["pairs"]:
        start = row["native_start"]
        prior.check(row["room"] == ROOMS[row["id"] % len(ROOMS)] and row["shard"] in SHARDS and row["reader"] != "100"
                    and all(0 <= start <= facts["samples"] - 6 * sine.RATE for facts in row["files"].values()), "Source range changed")
        if row["role"] == "cut_paste":
            intervals, _ = sine.label_clock(row["publisher_cut_paste_clock"], row["files"]["variant"]["samples"], row["role"])
            a, b, _ = intervals[1]
            prior.check(row["truth"] == [(a - start) * 2 // 3, (b - start) * 2 // 3]
                        and 0 <= row["truth"][0] < row["truth"][1] <= SAMPLES, "Publisher timing differs")
        else:
            prior.check(row["truth"] is None and start == 0, "Unexpected localized label")
    prior.check(locked["revision"] == sine.REVISION and locked["source_prefix_bytes"] == PREFIX
                and [row["shard"] for row in locked["shards"]] == list(SHARDS)
                and Counter(row["role"] for row in locked["pairs"]) == dict.fromkeys(ROLES, 12), "Unexpected pinned source set")
    expected = {(row["id"], side, condition, row["role"]): row["truth"] if side == "variant" else None
                for row in locked["pairs"] for condition in TREATMENTS for side in ("original", "variant")}
    expected.update({(index, row["room"], "competing_speech", "background_only"): None
                     for index, row in enumerate(locked["backgrounds"])})
    known = {}
    allowed = {"existing": {tuple(span) for span in stock.WINDOWS},
               **{arm: {tuple(learned.interval(i)) for i in range(len(learned.STARTS))} for arm in learned.ARMS}}
    for row in rows:
        key = tuple(row[field] for field in ("id", "side", "condition", "role"))
        prior.check(key in expected and key not in known and expected[key] == row["truth"] and stock.sha256(row["pcm_sha256"])
                    and set(row["scores"]) == set(points), "Unexpected assessment input or response")
        for arm, response in row["scores"].items():
            if response is not None:
                prior.check(math.isfinite(response[0]) and tuple(response[1]) in allowed[arm], "Response is outside the native time grid")
        known[key] = row
    prior.check(len(known) == len(expected), "An assessment delivery is missing")
    counts = {arm: defaultdict(Counter) for arm in points}
    for (group, side, condition, role), row in known.items():
        for arm, point in points.items():
            reading = row["scores"][arm]
            summary = counts[arm][f"{role}/{condition}"]
            status = "unassessed" if reading is None else "flag" if reading[0] > point else "no_flag"
            summary[f"{side if role != 'background_only' else 'background'}_{status}"] += 1
            if side != "variant":
                continue
            reference = known[group, "original", condition, role]["scores"][arm]
            if reference is not None and reading is not None:
                summary["paired_assessed"] += 1
                summary["variant_score_higher"] += int(reading[0] > reference[0])
                summary["new_flag_after_treatment"] += int(status == "flag" and reference[0] <= point)
            if row["truth"] is not None and status == "flag":
                summary["useful" if stock.useful(reading[1], row["truth"]) else "misplaced"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review", "audit"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--fit", type=Path, default=FIT)
    parser.add_argument("--weights", type=Path, default=prior.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.cache is None:
            parser.error("--cache is required for freeze")
        freeze(args.path, args.cache)
        return
    if args.phase == "score":
        locked = prior.read(args.path / "freeze.json")
        prior.check(locked["schema"] == SCHEMA and locked["method"] == METHOD, "Frozen analysis changed")
        records = score(args.path, locked, args.fit, args.weights)
        result = score_receipt(locked, records)
        result["summary"] = review(result)
        prior.write(args.path / "result.json", result)
    elif args.phase == "audit":
        result = prior.read(args.path / "result.json")
        prior.check(prior.read(args.path / "freeze.json") == result["freeze"], "The source freeze differs from the scored result")
        audit_native(args.path, result)
        if args.cache is not None:
            print(json.dumps(audit_sources(args.cache, result["freeze"]), sort_keys=True))
    else:
        result = prior.read(args.path)
    summary = review(result)
    prior.check(summary == result["summary"], "Published results differ from replay")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
