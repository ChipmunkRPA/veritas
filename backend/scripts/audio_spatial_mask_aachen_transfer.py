"""Replicate frozen stereo voice localization in two previously unused rooms."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import tarfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly

from scripts import audio_blind_spatial_transfer as auxiva
from scripts import audio_partial_publisher_transfer as publisher
from scripts import audio_spatial_mask_voice_transfer as directional
from scripts import prepare_natural_overlap_transfer as ami
from scripts import review_audio_spatial_mask_voice_transfer as review

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/evaluation-panels"
BASE = ROOT / "backend/data/independent-evaluation"
SCHEMA = "chipmunk.aachen-spatial-voice/1"
RATE = directional.RATE
SAMPLES = 6 * RATE
ROOMS = {"stairway": (2, "", "ES2005a"), "aula_carolina": (3, "_3", "ES2008a")}
FORMER_AMI = ("ES2004a", "IS1009a", "ES2010d", "IS1007a")
ANGLES = {"balanced": 90, "lateral": 0}
PREFIX_BYTES = 288 * 1024 ** 2
PARTIAL = re.compile(r"R01TTS\.0\.a/((?:dev-clean|test-clean|dev01-[a-z0-9]+-partial-cf)_(\d+)_\d+_\d+_\d+)\.wav")
TAR_MEMBERS_MAX = 4096
MINIMUM_EDITS, MINIMUM_GENUINE = 6, 8
MAX_EDITS_PER_FAMILY = 3
METHOD = {
    "source": "Bounded first 288 MiB of the publisher's uncompressed tar delivery. Strictly exclude every prior partial publisher speaker, and foreground and donor speaker IDs and source bytes from the three earlier directional rooms. Rank exact source IDs and six-second crop starts by SHA-256. Each selected positive has a complete .64–1.28 second labeled edit wholly inside 1–3 or 3–5 seconds; any other complete visible edit is retained in truth. Do not select with model scores.",
    "controls": "Up to one publisher genuine delivery per unique unexposed reader and one same-reader same-exact-utterance genuine counterpart when present. For a matched utterance add a same-reader unchanged crop and a same-reader reinsert from a distinct earlier or later speech segment where the complete range is available. Distinct unpaired genuine controls remain separate groups. Add three AMI-only controls per room.",
    "mix": "Aachen publisher angle label 0 for lateral foreground and label 90 for balanced competing feed in each fixed rig. Retain complete measured coefficients, resampled from native 48 kHz to 16 kHz with the same polyphase convention as earlier room tests. Select disjoint six-second excerpts on a quarter-second grid from AMI participants excluded from the earlier directional test. Require 80% coverage by any speech, at least 400 ms of simultaneous annotated background words within the principal edit and 1.2 seconds of overlap overall for edits and their paired controls. Distinct ordinary and background-only controls require at least 600 ms overall overlap. Mix at −5 dB mono foreground-to-masker ratio, using exact integer summation and the same competing feed and gain for matched versions.",
    "inference": "Run unchanged directional separator, prior AuxIVA both-output maximum, and rounded stereo mono. Gate each waveform with the pinned speech admission. Preserve the two frozen sample ranges and the three frozen points. Add directional responses to mono only as a parallel option. Neither prior labels nor dry stems enter inference.",
    "rule": "Count a location when the highest admitted above-point window has at least .30 temporal IoU and covers at least half of any visible publisher-labeled edit. Require at least two new source groups spanning at least two generators, zero added background-only group flags and at most one added ordinary group flag per room. Count each source speaker once in the cross-room replication.",
}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(raw):
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    return directional.hash_file(path)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def rank(value):
    return sha(f"{SCHEMA}:{value}".encode())


def prior_sources(prior, used_publisher):
    public = read(DOCS / "audio-spatial-mask-voice-transfer-2026-11-13.json")
    lock, points = (read(prior / name) for name in ("freeze.json", "points.json"))
    require(lock == public["freeze"] and points["points"] == review.points(public["rows"]), "Prior method or points differ from publication")
    old_rooms = [BASE / "audio-measured-spatial-transfer-final-2026-11-12/roster.json",
                 BASE / "audio-remote-neighbor-replication-2026-11-12/roster.json"]
    for path, receipt in zip(old_rooms, ("audio-measured-spatial-transfer", "audio-remote-neighbor-replication"), strict=True):
        check = read(DOCS / f"{receipt}-2026-11-12.json")
        require(hash_file(path) == check["commitments"]["roster"], "Former exact room roster differs")
    used_ids = {r["source"]["speaker"] for r in read(used_publisher) if r["source"]}
    used_bytes = {r["source"][key] for r in read(used_publisher) if r["source"] for key in ("wav_sha256", "pcm_sha256")}
    for path in old_rooms:
        for row in read(path):
            for field in ("host", "donor"):
                if origin := row.get(field):
                    identity = origin["canonical_speaker"]
                    if identity.startswith("librispeech:"):
                        used_ids.add(identity.removeprefix("librispeech:"))
                    used_bytes.update(origin[k] for k in ("source_sha256", "decoder_input_sha256", "pcm_sha256") if origin.get(k))
    return lock, points["points"], used_ids, used_bytes, {path.parent.name: hash_file(path) for path in old_rooms}


def eligible_crops(clock, samples):
    truth = [(a, b) for a, b, label in clock if label == "spoof"]
    for start in range(0, samples - SAMPLES + 1, RATE // 4):
        stop = start + SAMPLES
        visible = [(a - start, b - start) for a, b in truth if start < b and a < stop]
        if any(a < 0 or b > SAMPLES for a, b in visible):
            continue
        targets = [(a, b) for a, b in visible if 10240 <= b - a <= 20480 and any(x <= a < b <= y for x, y in review.WINDOWS)]
        if targets and sum(b - a for a, b in visible) <= .5 * SAMPLES:
            yield start, visible, max(targets, key=lambda pair: (pair[1] - pair[0], -pair[0]))


def read_publisher(prefix, labels_path, used_ids, used_bytes, *, prefix_bytes=PREFIX_BYTES):
    require(prefix.stat().st_size == prefix_bytes and hash_file(labels_path) == publisher.PINS["part-a.labels.txt"],
            "The bounded publisher prefix or label commitment is missing")
    labels = {name: parts for name, *parts in map(str.split, labels_path.read_text().splitlines())}
    require(len(labels) == 76_228, "Unexpected publisher label count")
    cases, examined, complete = [], set(), 0
    try:
        with tarfile.open(prefix, "r|") as tar:
            for entry in tar:
                if not entry.isfile():
                    continue
                match = PARTIAL.fullmatch(entry.name)
                if match is None:
                    continue
                name, speaker = match.groups()
                require(name not in examined and name in labels and 44 < entry.size < 2 * 1024 ** 2,
                        "Unexpected publisher WAV member")
                with tar.extractfile(entry) as stream:
                    body = stream.read()
                if len(body) != entry.size:
                    break
                examined.add(name)
                complete += 1
                if complete > TAR_MEMBERS_MAX:
                    raise ValueError("Publisher member cap exceeded")
                if speaker in used_ids or not 6 <= float(labels[name][0]) <= 24:
                    continue
                pcm = publisher.native(body)
                if sha(pcm) in used_bytes or sha(body) in used_bytes:
                    continue
                samples = len(pcm) // 2
                clock = publisher.labels_for(labels[name], samples)
                if samples < SAMPLES:
                    continue
                genuine = labels[name][1] == "bonafide"
                crops = [(0, [], None)] if genuine else list(eligible_crops(clock, samples))
                if not crops:
                    continue
                start, truth, principal = min(crops, key=lambda crop: rank((name, crop[0])))
                family = None if genuine else name.split("-")[1]
                require(genuine or family in publisher.GENERATORS, "Unexpected publisher generator label")
                cases.append({"source": name, "speaker": speaker, "utterance": name.split("_", 1)[1],
                              "family": family, "start": start,
                              "truth": truth, "principal": principal, "native_samples": samples,
                              "source_sha256": sha(body), "source_pcm_sha256": sha(pcm), "raw": pcm})
    except (EOFError, tarfile.ReadError) as error:
        require(isinstance(error, EOFError) or "end of data" in str(error), "Unexpected publisher archive error")
    return cases, complete


def select(cases):
    edits, ordinary, voices = [], [], set()
    by_family = defaultdict(list)
    clean = {row["utterance"]: row for row in cases if row["family"] is None}
    for row in cases:
        if row["family"]:
            by_family[row["family"]].append(row)
    families = sorted(by_family, key=lambda name: (len({row["speaker"] for row in by_family[name]}), name))
    for _ in range(MAX_EDITS_PER_FAMILY):
        for family in families:
            available = [row for row in by_family[family] if row["speaker"] not in voices]
            if not available:
                continue
            selected = min(available, key=lambda row: (row["utterance"] not in clean, rank(row["source"])))
            voices.add(selected["speaker"])
            edits.append(selected)
    for row in sorted(clean.values(), key=lambda item: rank(item["source"])):
        if row["speaker"] not in voices:
            voices.add(row["speaker"])
            ordinary.append(row)
        if len(ordinary) == 12:
            break
    require(len(edits) >= MINIMUM_EDITS and len(ordinary) >= MINIMUM_GENUINE and len({r["family"] for r in edits}) >= 3,
            f"Bounded publisher roster has only {len(edits)} edits and {len(ordinary)} genuine groups")
    return edits, ordinary, clean


def rir_inputs(path):
    records, values = {}, {}
    with ZipFile(path) as archive:
        notice = archive.read("AIR_1_4/license.txt")
        require(b"MIT License" in notice and b"without restriction" in notice, "Aachen license differs")
        for room, (position, suffix, _) in ROOMS.items():
            values[room], records[room] = {}, {}
            for role, angle in ANGLES.items():
                coefficients, pins = [], []
                for channel in (1, 0):
                    name = f"AIR_1_4/air_binaural_{room}_{channel}_1_{position}_{angle}{suffix}.mat"
                    entry = archive.getinfo(name)
                    require(0 < entry.file_size <= 5 * 1024 ** 2, "Measured room member exceeds size bound")
                    raw = archive.read(entry)
                    data = loadmat(io.BytesIO(raw), simplify_cells=True)
                    impulse, meta = data["h_air"], data["air_info"]
                    require(meta["fs"] == 48000 and meta["room"] == room and meta["head"] == 1 and meta["channel"] == channel
                            and impulse.ndim == 1 and 0 < len(impulse) <= 480000 and np.isfinite(impulse).all(),
                            "Measured room geometry or coefficient type differs")
                    pins.append({"member": name, "sha256": sha(raw), "native_samples": len(impulse)})
                    coefficients.append(impulse)
                response = np.ascontiguousarray(resample_poly(np.column_stack(coefficients), 1, 3, axis=0), dtype="<f8")
                peaks = np.argmax(abs(response), axis=0).tolist()
                early = response[max(0, min(peaks) - 10):max(peaks) + 40]
                balance = float(20 * np.log10(np.linalg.norm(early[:, 0]) / np.linalg.norm(early[:, 1])))
                require((abs(peaks[0] - peaks[1]) <= 2 if role == "balanced" else 4 <= abs(peaks[0] - peaks[1]) <= 22),
                        "Aachen early response does not support the fixed angle role")
                values[room][role] = response
                records[room][role] = {"members": pins, "rendered_sha256": sha(response.tobytes()), "peaks": peaks,
                                       "early_left_minus_right_db": round(balance, 5), "angle_label": angle}
    require(len({item["rendered_sha256"] for room in records.values() for item in room.values()}) == 4,
            "Measured room responses are repeated")
    return records, values, sha(notice)


def ami_inputs(folder, annotations):
    require(hash_file(annotations) == ami.ANNOTATION_SHA256, "AMI annotation archive differs")
    public = read(DOCS / "audio-gain-public-replay-2026-11-13.json")
    specs = {}
    with ZipFile(annotations) as archive:
        document = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))

        def participants(room):
            meeting = next(item for item in document.iter() if item.get("observation") == room)
            return {item.get("global_name") for item in meeting if item.get("global_name")}

        earlier = set().union(*(participants(room) for room in FORMER_AMI))
        for _, _, meeting in ROOMS.values():
            people = participants(meeting)
            require(len(people) == 4 and not earlier & people, "AMI participant identities overlap the former directional trial")
            earlier.update(people)
            file = folder / f"{meeting}.Array1-01.wav"
            pin = public["sources"][file.name]["sha256"]
            require(hash_file(file) == pin, "Original AMI microphone differs")
            tracks = [ami.track_words(archive.read(f"words/{meeting}.{channel}.words.xml")) for channel in "ABCD"]
            active = np.zeros((4, max(end for track in tracks for _, end in track)), dtype=bool)
            for i, track in enumerate(tracks):
                for start, end in track:
                    active[i, start:end] = True
            specs[meeting] = file, pin, active.sum(axis=0), public["plan"]["window_starts_s"]
    return specs


def masker(spec, used, meeting, identity, principal=None):
    path, pin, counts, prior = spec
    first, last = (a // 160 for a in principal) if principal else (0, 600)
    with wave.open(str(path), "rb") as source:
        require(source.getparams()[:3] == (1, 2, RATE), "AMI source is not native 16 kHz mono PCM")
        stops = min(len(counts), source.getnframes() // 160)
        candidates = []
        for tick in range(500, stops - 600 + 1, 25):
            start = tick * 160
            if any(abs(start - seen) < SAMPLES for seen in used) or any(start < (second + 3) * RATE and second * RATE < start + SAMPLES for second in prior):
                continue
            active = counts[tick:tick + 600]
            overlap, local = int(np.count_nonzero(active >= 2)), int(np.count_nonzero(active[first:last] >= 2))
            if np.count_nonzero(active) >= 480 and overlap >= (120 if principal else 60) and local >= 40:
                candidates.append((rank((meeting, identity, start)), start, overlap, local))
        require(candidates, f"No unused annotation-supported competing window for {identity} in {meeting}")
        _, start, overlap, local = min(candidates)
        source.setpos(start)
        raw = source.readframes(SAMPLES)
    require(len(raw) == SAMPLES * 2, "Native AMI excerpt ended early")
    used.add(start)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64), {"meeting": meeting, "start": start,
           "capture_sha256": pin, "pcm_sha256": sha(raw), "overlap_ms": overlap * 10, "principal_overlap_ms": local * 10}


def render(dry, masker_pcm, room, ref=None):
    def power(pcm):
        return np.square(pcm.mean(axis=1)).sum()

    reference = dry[0] if ref is None else ref
    foregrounds = [fftconvolve(clip[:, None], room["lateral"], axes=0)[:SAMPLES] for clip in dry]
    reference = fftconvolve(reference[:, None], room["lateral"], axes=0)[:SAMPLES]
    background = fftconvolve(masker_pcm[:, None], room["balanced"], axes=0)[:SAMPLES]
    require(power(reference) > 0 and power(background) > 0, "Room mix has an empty path")
    background *= math.sqrt(power(reference) * 10 ** .5 / power(background))
    gain = 22500 / max(np.max(abs(waveform) + abs(background)) for waveform in [*foregrounds, reference])
    integer_noise = np.rint(background * gain).astype("<i2")
    ref_i = np.rint(reference * gain).astype("<i2")
    actual_snr = 10 * math.log10(power(ref_i) / power(integer_noise))
    require(abs(actual_snr + 5) <= 0.02, "Rendered competition differs from the fixed −5 dB ratio")
    return [(np.rint(clip * gain).astype("<i2"), integer_noise, round(actual_snr, 6)) for clip in foregrounds]


def pcm(row, start=None):
    samples = np.frombuffer(row["raw"], dtype="<i2")
    a = row["start"] if start is None else start
    require(0 <= a <= len(samples) - SAMPLES, "The exact source crop cannot be read")
    return samples[a:a + SAMPLES].astype(np.float64)


def source_receipt(row):
    return {key: row[key] for key in ("source", "source_sha256", "source_pcm_sha256", "native_samples", "start")}


def pairs(selected, clean):
    truth = selected["truth"]
    if not truth:
        return [("genuine", selected, pcm(selected))]
    output = [("publisher_edit", selected, pcm(selected))]
    counterpart = clean.get(selected["utterance"])
    if counterpart is not None and counterpart["native_samples"] >= selected["start"] + SAMPLES:
        ordinary = pcm(counterpart, selected["start"])
        a, b = selected["principal"]
        width = b - a
        src = SAMPLES - width if b <= SAMPLES - width else 0
        require(src + width <= a or src >= b, "Ordinary splice overlaps its destination")
        variant = ordinary.copy()
        fade = np.r_[np.linspace(0, 1, 320), np.ones(width - 640), np.linspace(1, 0, 320)]
        variant[a:b] = np.rint(ordinary[a:b] * (1 - fade) + ordinary[src:src + width] * fade)
        match = {**counterpart, "start": selected["start"]}
        output += [("matched_genuine", match, ordinary), ("ordinary_same_reader", match, variant)]
    return output


def assign(edits, ordinary):
    layouts = list(ROOMS)
    taken = Counter()
    selected = []
    for index, row in enumerate(edits):
        same = sorted(layouts, key=lambda room: (taken[row["family"], room], taken[room], layouts.index(room)))
        room = same[0]
        taken[row["family"], room] += 1
        taken[room] += 1
        selected.append((room, f"edit-{index:02d}", row))
    selected.extend((layouts[index % len(layouts)], f"genuine-{index:02d}", row) for index, row in enumerate(ordinary))
    return selected


def stage(args):
    old, points, used_ids, used_bytes, former = prior_sources(args.prior, args.publisher / "locked/roster.json")
    prefix, rir = (args.input / name for name in ("llama-first-288m.tar", "air_database_release_1_4.zip"))
    require(hash_file(args.publisher / "LICENSE.txt") == publisher.PINS["LICENSE.txt"], "Publisher voice rights receipt differs")
    cases, scanned = read_publisher(prefix, args.publisher / "part-a.labels.txt", used_ids, used_bytes)
    edits, ordinary, clean = select(cases)
    room_pins, rooms, notice = rir_inputs(rir)
    meeting_specs = ami_inputs(args.ami, args.annotations)
    exposures, results = defaultdict(set), []
    for room, group, selection in assign(edits, ordinary):
        meeting = ROOMS[room][-1]
        variants = pairs(selection, clean)
        background, support = masker(meeting_specs[meeting], exposures[meeting], meeting, group, selection["principal"])
        mixes = render([row[2] for row in variants], background, rooms[room], ref=variants[1][2] if len(variants) > 1 else None)
        for (kind, row, _), (target, noise, level) in zip(variants, mixes, strict=True):
            results.append(({"case": len(results), "group": group, "room": room, "kind": kind, "family": row["family"],
                             "truth": row["truth"], "principal": row["principal"], "source": source_receipt(row),
                             "ami": support, "reference_input_snr_db": level}, target, noise))
    for room, (_, _, meeting) in ROOMS.items():
        for index in range(3):
            group = f"{room}-background-{index}"
            background, support = masker(meeting_specs[meeting], exposures[meeting], meeting, group)
            mask = fftconvolve(background[:, None], rooms[room]["balanced"], axes=0)[:SAMPLES]
            noise = np.rint(22500 * mask / np.max(abs(mask))).astype("<i2")
            results.append(({"case": len(results), "group": group, "room": room, "kind": "background_only",
                             "family": None, "truth": [], "principal": None, "source": None,
                             "ami": support, "reference_input_snr_db": None}, np.zeros_like(noise), noise))
    lock = {"schema": SCHEMA, "method": METHOD, "points": points, "assets": old["assets"],
            "model_code": hash_file(Path(directional.__file__)), "prior_auxiva_code": hash_file(Path(auxiva.__file__)),
            "runner_code": hash_file(Path(__file__)), "prior_comparison": hash_file(DOCS / "audio-spatial-mask-voice-transfer-2026-11-13.json"),
            "prior_exact_rosters": former, "prior_publisher_roster": hash_file(args.publisher / "locked/roster.json"),
            "prior_speaker_exclusions": len(used_ids), "prior_source_hash_exclusions": len(used_bytes),
            "prefix_sha256": hash_file(prefix), "prefix_bytes": prefix.stat().st_size, "complete_native_members": scanned,
            "rooms": room_pins, "aachen_archive_sha256": hash_file(rir), "aachen_license_sha256": notice,
            "ami_annotations": hash_file(args.annotations), "windows": review.WINDOWS,
            "foreground_and_genuine_speakers": len(edits) + len(ordinary), "background_participants": 8}
    rows, payloads = [], []
    for row, target, noise in results:
        integers = target.astype(np.int32) + noise.astype(np.int32)
        require(np.max(abs(integers)) <= 22501, "Integer stereo mixture exceeded the fixed peak allowance")
        stereo = integers.astype("<i2")
        rows.append({**row, "stereo_pcm_sha256": sha(stereo.tobytes()), "dry_room_pcm_sha256": sha(target.tobytes()),
                     "background_room_pcm_sha256": sha(noise.tobytes())})
        payloads.append(stereo)
    return lock, rows, payloads


def views(stereo, *, alternate=False):
    signal = stereo.astype(np.float64)
    options = {"fft": np.fft.rfft, "inverse": np.fft.irfft} if alternate else {}
    masked, modes = directional.separate(signal, **options)
    old = auxiva.separate(signal, **options)
    earlier = []
    if old is not None:
        for waveform in old[0].T:
            values = np.rint(waveform)
            earlier.append(values.astype("<i2") if np.min(values) >= -32768 and np.max(values) <= 32767 else None)
    return {"mono": [np.rint(signal.mean(axis=1)).astype("<i2")], "earlier_either": earlier, "directional": masked}, modes


def score(args, lock, rows, payloads, *, audit=False):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    require(hash_file(args.weights / "silero_vad.jit") == lock["assets"]["silero_vad.jit"], "VAD weights changed")
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    model = None if audit else load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    previous = read(args.run / "responses.json")["rows"] if audit else None
    records, cache = [], {}
    started = time.monotonic()
    for index, (row, stereo) in enumerate(zip(rows, payloads, strict=True)):
        tracks, modes = views(stereo, alternate=audit)
        results = {}
        for arm, values in tracks.items():
            readings = []
            for track, audio in enumerate(values):
                if audio is None:
                    readings.append(None)
                    continue
                pcm = audio.astype(np.float32) / 32768
                speech = activity.spans(pcm)
                windows = []
                for slot, (start, end) in enumerate(review.WINDOWS):
                    digest = sha(audio[start:end].tobytes())
                    accepted = window_in_speech(speech, start / RATE, end / RATE)
                    value = None
                    if accepted and audit:
                        value = previous[index]["reads"][arm][track][slot]["score"]
                        require(value is not None and math.isfinite(value), "Speech admission changed")
                    elif accepted:
                        if digest not in cache:
                            probability, = model.predict([pcm[start:end]])
                            require(math.isfinite(probability) and 0 <= probability <= 1, "Voice model returned an invalid response")
                            cache[digest] = float(logit(probability))
                        value = cache[digest]
                    windows.append({"pcm_sha256": digest, "score": value})
                readings.append(windows)
            results[arm] = readings
        records.append({"case": row["case"], "group": row["group"], "room": row["room"], "kind": row["kind"],
                        "family": row["family"], "truth": row["truth"], "modes": modes, "reads": results})
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print("audit" if audit else "score", index + 1, len(rows), "forwards", len(cache), "elapsed",
                  round(time.monotonic() - started, 2), flush=True)
    require(not audit or json.dumps(records, sort_keys=True) == json.dumps(previous, sort_keys=True),
            "Independent FFT changes the waveform or admitted-window proof")
    return {"rows": records, "forwards": len(cache)}


def pick(readings):
    choices = [(reading["score"], -slot) for view in readings if view is not None
               for slot, reading in enumerate(view) if reading["score"] is not None]
    return max(choices, default=None)


def summarize(rows, points):
    layouts, all_added = {}, {}
    for name in (*ROOMS, "pooled"):
        group = [r for r in rows if r["room"] == name or name == "pooled"]
        tallies, outcomes = {}, {}
        for arm in review.ARMS:
            good, ordinary, background, assessed, wrong = set(), set(), set(), 0, 0
            for row in group:
                choice = pick(row["reads"][arm])
                if choice is None:
                    continue
                assessed += 1
                value, index = choice
                if value <= points[arm]:
                    continue
                if row["truth"]:
                    window = review.WINDOWS[-index]
                    if any(review.location_matches(span, window) for span in row["truth"]):
                        good.add(row["group"])
                        if name == "pooled" and arm == "directional":
                            all_added[row["group"]] = {"room": row["room"], "family": row["family"],
                                                       "truth": [list(span) for span in row["truth"]], "window": list(window)}
                    else:
                        wrong += 1
                else:
                    ordinary.add(row["group"])
                    if row["kind"] == "background_only":
                        background.add(row["group"])
            tallies[arm] = {"assessed": assessed, "unassessed": len(group) - assessed, "located": len(good),
                            "wrong_edit_windows": wrong, "flagged_ordinary_groups": len(ordinary),
                            "flagged_background_only_groups": len(background)}
            outcomes[arm] = good, ordinary, background
        mono, mask = outcomes["mono"], outcomes["directional"]
        additional = mask[0] - mono[0]
        if name == "pooled":
            all_added = {key: all_added[key] for key in sorted(additional)}
        tallies["additive"] = {"located": len(mono[0] | mask[0]), "added_locations": len(additional),
                               "added_ordinary_groups": len(mask[1] - mono[1]), "added_background_groups": len(mask[2] - mono[2]),
                               "flagged_ordinary_groups": len(mask[1] | mono[1])}
        tallies["totals"] = {"cases": len(group), "edits": sum(r["truth"] != [] for r in group),
                             "ordinary_groups": len({r["group"] for r in group if not r["truth"]})}
        layouts[name] = tallies
    gates = {"two_edits": len(all_added) >= 2, "two_families": len({r["family"] for r in all_added.values()}) >= 2,
             "ordinary_per_room": all(layouts[room]["additive"]["added_ordinary_groups"] <= 1 for room in ROOMS),
             "background_per_room": all(layouts[room]["additive"]["added_background_groups"] == 0 for room in ROOMS)}
    return {"layouts": layouts, "added_locations": all_added, "gates": gates, "passed": all(gates.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "audit", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--publisher", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ami", type=Path)
    parser.add_argument("--annotations", type=Path, default=BASE / "spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.run)
        require(receipt["schema"] == SCHEMA and summarize(receipt["responses"], receipt["freeze"]["points"]) == receipt["summary"],
                "Public decision ledger differs")
        print(json.dumps(receipt["summary"], indent=2))
        return
    lock, roster, payloads = stage(args)
    freeze, inputs = args.run / "freeze.json", args.run / "roster.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(freeze, lock)
        write(inputs, roster)
        print("Frozen", len(roster), Counter(row["kind"] for row in roster), hash_file(freeze), flush=True)
        return
    require(json.dumps(read(freeze), sort_keys=True) == json.dumps(lock, sort_keys=True)
            and json.dumps(read(inputs), sort_keys=True) == json.dumps(roster, sort_keys=True), "Frozen source or implementation differs")
    result = score(args, lock, roster, payloads, audit=args.phase == "audit")
    if args.phase == "score":
        write(args.run / "responses.json", result)
    else:
        rows = [{**row, "reads": {arm: [None if view is None else [{"score": window["score"]} for window in view] for view in tracks]
                                  for arm, tracks in row["reads"].items()}} for row in result["rows"]]
        write(args.run / "public.json", {"schema": SCHEMA, "freeze": lock, "freeze_sha256": hash_file(freeze),
              "private_roster_sha256": hash_file(inputs), "response_ledger_sha256": hash_file(args.run / "responses.json"),
              "exact_alternate_fft_replay": len(rows), "forwards": read(args.run / "responses.json")["forwards"],
              "responses": rows, "summary": summarize(rows, lock["points"])})
    print(json.dumps(summarize(result["rows"], lock["points"]), indent=2))


if __name__ == "__main__":
    main()
