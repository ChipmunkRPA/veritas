# SPDX-License-Identifier: Apache-2.0
"""Qualify blind bilabial speech timestamps before using them for lip-sync evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cache
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.video_short_soundtrack_transfer import sha

SCHEMA = "chipmunk.video-neural-phone-clock/1"
MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
WEIGHT_PIN = "3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e"
MODEL_BLOBS = {
    "README.md": "d3d413893434f36cef6042639ab0be56769b95fe",
    "config.json": "c8fc04b8bd95c7a3361e606579a36f335a2b0c57",
    "preprocessor_config.json": "36ebe8b7c1cc967b3059f0494ae8a1069dd67655",
    "vocab.json": "14929da774c31eccf9e8c25fa3a966d8fc17d3fa",
}
SPHINX_PIN = "833d0ba73affdec98883e0828dbacb69ab4932f4e739062076ed679e54878f21"
RATE, STRIDE, CENTER = 16000, 320, 200
LABEL_PADDING, STABLE_PADDING = 640, 640
SHIFT = 3200
HALF_EDIT = 5600
LOCAL_WINDOW = 19200
LOCAL_STEP = 7680
LOCAL_EDGE = 2560
EXCLUDED = frozenset({"s2", "s7", "s9", "s17", "s23", "s27", "s37", "s38", "s46", "s47", "s48", "s49", "s55"})
TREATMENTS = ("original", "filtered", "early", "late")
DESIGN = {
    "sources": "CC BY 4.0 Lombard frontal and audio archive members with published phone alignment labels. "
               "Choose four development and eight held speaker codes by fixed hash order. "
               "Each speaker supplies two plain and two Lombard utterances. Verify the full picture archive "
               "and the size, CRC and SHA-256 of selected audio members read from publisher byte ranges.",
    "neural": "Unprompted CTC acoustic token emissions at a verified 20 ms hop. "
              "Keep the largest emission in each run of the same token if posterior is at least 0.35.",
    "stable": "Keep the original timestamp only if the same phone is observed within 40 ms "
              "after an independent 3.4 kHz low-pass delivery.",
    "reference": "Match independent publisher B or P intervals one-to-one on the native 16 kHz audio clock, "
                 "with 40 ms interval padding. Maximize distinct matches, then minimize total absolute midpoint error.",
    "edits": "Stage copies of the target speech 200 ms earlier and later in a local 700 ms window "
             "with 8 ms endpoint blends and unchanged PCM outside. Match only the independently labeled target in edited views.",
}
CONTEXT_DESIGN = DESIGN | {
    "additional_sources": "Exclude every speaker used in the first neural phone study. Reuse the unchanged source ordering rule.",
    "local_context": "Read 1.2 second windows every 480 ms. Omit 160 ms at each crop edge. "
                     "Keep a phone event supported by at least two different inputs within 40 ms of one another. "
                     "The input set includes the complete waveform. All short windows start at an exact CTC hop.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, record):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def stable_order(value):
    return digest(f"{SCHEMA}/{value}".encode())


def model_assets(path):
    for name, expected in MODEL_BLOBS.items():
        contents = (path / name).read_bytes()
        prefix = f"blob {len(contents)}\0".encode()
        require(hashlib.sha1(prefix + contents, usedforsecurity=False).hexdigest() == expected,
                f"Configuration file differs from the pinned publisher release: {name}")
    require(sha(path / "pytorch_model.bin") == WEIGHT_PIN, "Published checkpoint hash changed")
    readme = (path / "README.md").read_text()
    require("license: apache-2.0" in readme, "Commercial checkpoint terms were not verified")


def source_archives(folder):
    from scripts.lombard_mouth_source_intake import md5, publisher

    metadata = publisher(folder)
    locations = {name: folder / f"lombardgrid_{name}.zip" for name in ("front", "audio")}
    require(all(p.stat().st_size == metadata[role]["bytes"]
                and metadata[role]["published_checksum"] == "md5:" + md5(p)
                for role, p in locations.items() if role == "front" or p.exists()),
            "A media archive differs from the publisher commitment")
    return locations, metadata


def read_audio_member(archive, metadata, member, directory_cache):
    if directory_cache is None:
        return archive.read(member)
    from scripts.lombard_mouth_source_intake import MemberReader

    reader = MemberReader(metadata)
    reader.cache = list(directory_cache)
    reader.seek(member.header_offset)
    reader.read(min(metadata["bytes"] - member.header_offset, member.compress_size + 4096))
    with ZipFile(reader) as remote:
        restored = remote.getinfo(member.filename)
        require((restored.file_size, restored.compress_size, restored.CRC, restored.header_offset)
                == (member.file_size, member.compress_size, member.CRC, member.header_offset),
                "The publisher directory commitment changed")
        return remote.read(restored)


def exact_labels(payload, uid):
    labels = []
    for phone in json.loads(payload)[uid]:
        name = phone["phone"].split("_", 1)[0]
        if name not in ("b", "p"):
            continue
        start = Fraction(str(phone["offset"]))
        end = start + Fraction(str(phone["duration"]))
        labels.append([name, round(start * RATE), round(end * RATE)])
    require(all(0 <= a < b for _, a, b in labels)
            and all(left[2] <= right[1] for left, right in zip(labels, labels[1:])), "Invalid publisher phone intervals")
    return labels


def pcm(payload):
    import numpy as np
    from scipy.io import wavfile

    rate, values = wavfile.read(io.BytesIO(payload))
    require(rate == RATE and values.ndim == 1 and 20000 <= len(values) <= 64000,
            "Expected complete bounded 16 kHz mono publisher audio")
    if values.dtype == np.int16:
        return values.astype("<i2", copy=False)
    require(values.dtype == np.float32 and np.isfinite(values).all() and np.max(np.abs(values)) <= 1,
            "Expected 16-bit PCM or finite publisher float audio")
    return np.rint(values.astype(np.float64) * 32768).clip(-32768, 32767).astype("<i2")


def source_rows(folder, previous=None):
    from scripts.lombard_mouth_source_intake import MemberReader, index_zip, target

    locations, archive_metadata = source_archives(folder)
    groups = defaultdict(list)
    earlier = None
    excluded = set(EXCLUDED)
    if previous is not None:
        first = json.loads(previous.read_bytes())
        require(first.get("method") == DESIGN and review(first) == first.get("outcomes"), "The earlier benchmark cannot replay")
        earlier = {"sha256": sha(previous), "speakers": sorted({row["speaker"] for row in first["rows"]})}
        excluded.update(earlier["speakers"])
    member_folder = folder / "audio-members"
    member_folder.mkdir(mode=0o700, exist_ok=True)
    remote_audio = archive_metadata["audio"] | {"url": "https://zenodo.org/api/records/3736465/files/lombardgrid_audio.zip/content"}
    reader = MemberReader(remote_audio) if not locations["audio"].is_file() else None
    with (ZipFile(folder / "publisher-json.zip") as info, ZipFile(folder / "alignment.zip") as alignments,
          ZipFile(locations["front"]) as front, ZipFile(locations["audio"] if reader is None else reader) as audio):
        aligned = index_zip(alignments, ".json")
        eyes, ears = index_zip(front, ".mov"), index_zip(audio, ".wav")
        directory_cache = reader.cache if reader is not None else None
        for entry in index_zip(info, ".json").values():
            for row in json.loads(info.read(entry)):
                speaker, condition = row["SPKR"], row["COND"]
                uid = f"{speaker}_{condition}_{row['UTTERANCE']}"
                if (speaker in excluded or condition not in ("p", "l") or row["STATUS"] != "CORRECT"
                        or uid not in aligned or uid not in eyes or uid not in ears):
                    continue
                label_bytes = alignments.read(aligned[uid])
                event = target(json.loads(label_bytes)[uid])
                if not event:
                    continue
                a = round(Fraction(event["aligned_start_s"]) * RATE)
                b = round(Fraction(event["aligned_following_phone_s"]) * RATE)
                groups[speaker].append((uid, condition, event["phoneme"], a, b))
        rows, speakers = [], []
        for speaker in sorted(groups, key=stable_order):
            by_condition = defaultdict(list)
            for uid, condition, phone, a, b in sorted(groups[speaker], key=lambda row: stable_order(row[0])):
                if len(by_condition[condition]) >= 2:
                    continue
                center = (a + b) // 2
                start, stop = center - HALF_EDIT, center + HALF_EDIT
                if start < SHIFT or not start + 128 <= a - SHIFT < b + SHIFT <= stop - 128:
                    continue
                sound = read_audio_member(audio, remote_audio, ears[uid], directory_cache)
                wave_data = pcm(sound)
                label_bytes = alignments.read(aligned[uid])
                labels = exact_labels(label_bytes, uid)
                if stop > len(wave_data) - SHIFT or [phone, a, b] not in labels or labels[-1][2] > len(wave_data):
                    continue
                picture = front.read(eyes[uid])
                pin = digest(sound)
                path = member_folder / f"{pin}.wav"
                if not path.exists():
                    path.write_bytes(sound)
                by_condition[condition].append({
                    "speaker": speaker, "condition": condition, "source": uid, "samples": len(wave_data),
                    "labels": labels, "target": [phone, a, b], "edit": [start, stop],
                    "members": {"audio": {"name": ears[uid].filename, "sha256": pin, "crc32": ears[uid].CRC,
                                          "bytes": ears[uid].file_size},
                                "front": {"name": eyes[uid].filename, "sha256": digest(picture)},
                                "alignment": {"name": aligned[uid].filename, "sha256": digest(label_bytes)}},
                })
            if any(len(by_condition[c]) != 2 for c in ("p", "l")):
                continue
            split = "development" if len(speakers) < 4 else "held"
            speakers.append(speaker)
            rows.extend(row | {"split": split} for condition in ("p", "l") for row in by_condition[condition])
            if len(speakers) == 12:
                break
    require(len(speakers) == 12 and len(rows) == 48, "Incomplete disjoint source roster")
    document = {"schema": SCHEMA, "method": DESIGN if earlier is None else CONTEXT_DESIGN,
            "revision": MODEL_REVISION, "model_sha256": WEIGHT_PIN,
            "baseline_sha256": SPHINX_PIN, "archives": archive_metadata,
            "metadata_sha256": {name: sha(folder / name) for name in ("record.json", "alignment.zip", "publisher-json.zip")},
            "rows": rows}
    if earlier is not None:
        document["earlier"] = earlier
    return document


def neural_decoder(path, *, names=("b", "bʲ", "bʰ", "bː", "p", "pʲ", "pʰ", "pː")):
    import numpy as np
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

    model_assets(path)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCTC.from_pretrained(path, local_files_only=True, trust_remote_code=False, dtype=dtype).to(device).eval()
    feature = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
    vocabulary = json.loads((path / "vocab.json").read_text())
    phones = {vocabulary[name]: name[0] for name in names}
    hop, width = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride, strict=True):
        width += hop * (kernel - 1)
        hop *= step
    require((hop, width) == (STRIDE, 2 * CENTER) and feature.sampling_rate == RATE, "Acoustic clock changed")

    def infer(samples):
        material = samples.astype(np.float32) / 32768
        inputs = feature(material, sampling_rate=RATE, return_tensors="pt").input_values.to(device, dtype)
        with torch.inference_mode():
            outputs = model(inputs).logits.float().cpu()[0].softmax(-1)
        require(len(outputs) == (len(samples) - width) // STRIDE + 1, "CTC frame count does not match the input")
        maxima = outputs.argmax(1).tolist()
        events = []
        for token, positions in groupby(range(len(maxima)), key=maxima.__getitem__):
            if token not in phones:
                continue
            index = max(positions, key=lambda p: float(outputs[p, token]))
            value = float(outputs[index, token])
            if value >= .35:
                events.append([phones[token], index * STRIDE + CENTER, round(value, 6)])
        return events

    return infer, device


def old_decoder(wheel_path, directory):
    require(sha(wheel_path) == SPHINX_PIN, "Baseline wheel differs from the reviewed release")
    directory.mkdir(exist_ok=True)
    with ZipFile(wheel_path) as wheel:
        for name in wheel.namelist():
            if name.startswith("pocketsphinx/") and not name.endswith("/"):
                require(".." not in Path(name).parts, "Unexpected baseline wheel path")
                asset = directory / name
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(wheel.read(name))
    sys.path.insert(0, str(directory))
    from pocketsphinx import Decoder

    assets = directory / "pocketsphinx/model/en-us"
    decoder = Decoder(hmm=str(assets / "en-us"), allphone=str(assets / "en-us-phone.lm.bin"),
                      allphone_ci=True, samprate=RATE, frate=100, logfn="/dev/null")

    def infer(samples):
        decoder.start_utt()
        decoder.process_raw(samples.tobytes(), no_search=False, full_utt=True)
        decoder.end_utt()
        return [[seg.word.lower(), (seg.start_frame + seg.end_frame + 1) * 80, None] for seg in decoder.seg()
                if seg.word in ("B", "P") and 0 <= (seg.start_frame + seg.end_frame + 1) * 80 < len(samples)]

    return infer


def variants(original, edit):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 3400, fs=RATE, output="sos")
    filtered = np.rint(sosfiltfilt(sos, original)).clip(-32768, 32767).astype("<i2")
    start, stop = edit
    fades = np.ones(stop - start)
    fades[:128] = np.linspace(0, 1, 128, endpoint=False)
    fades[-128:] = fades[:128][::-1]
    yield "original", original
    yield "filtered", filtered
    for role, displacement in (("early", -SHIFT), ("late", SHIFT)):
        delivery = original.copy()
        delivery[start:stop] = np.rint(original[start:stop] * (1 - fades)
                                       + original[start - displacement:stop - displacement] * fades).astype("<i2")
        require(np.count_nonzero(delivery[start:stop] != original[start:stop]) >= STRIDE
                and np.array_equal(delivery[:start], original[:start])
                and np.array_equal(delivery[stop:], original[stop:]), "Authored waveform timing changes lost their source bounds")
        yield role, delivery


def match(events, labels, tolerance):
    """One-to-one matches over the same phone class, maximizing count then clock accuracy."""
    def align(expected, observed):
        @cache
        def best(i, j):
            if i >= len(expected) or j >= len(observed):
                return 0, 0, ()
            choices = [best(i + 1, j), best(i, j + 1)]
            _, a, b = expected[i]
            _, sample, _ = observed[j]
            if a - tolerance <= sample <= b + tolerance:
                count, error, pairs = best(i + 1, j + 1)
                choices.append((count + 1, error - abs(sample - (a + b) // 2), ((i, j), *pairs)))
            return max(choices, key=lambda item: item[:2])

        return [(expected[i], observed[j]) for i, j in best(0, 0)[2]]

    return [pair for name in ("b", "p")
            for pair in align([row for row in labels if row[0] == name], [row for row in events if row[0] == name])]


def stable(events, filtered):
    labels = [[name, sample, sample] for name, sample, _ in filtered]
    return sorted((reading for _, reading in match(events, labels, STABLE_PADDING)), key=lambda event: event[1])


def context_starts(samples):
    end_start = (samples - LOCAL_WINDOW) // STRIDE * STRIDE
    if end_start <= 0:
        return []
    return sorted({*range(0, end_start + 1, LOCAL_STEP), end_start})


def context_readings(decoder, samples):
    readings = []
    for start in context_starts(len(samples)):
        anchors = []
        for name, point, score in decoder(samples[start:start + LOCAL_WINDOW]):
            if LOCAL_EDGE <= point < LOCAL_WINDOW - LOCAL_EDGE:
                anchors.append([name, start + point, score])
        readings.append({"start": start, "events": anchors})
    return readings


def consensus(full, contexts):
    sources = [(name, p, confidence, -1) for name, p, confidence in full]
    sources.extend((name, p, score, row["start"]) for row in contexts for name, p, score in row["events"])
    detected = []
    for phone in ("b", "p"):
        pending = sorted([item for item in sources if item[0] == phone], key=lambda row: row[1])
        while pending:
            cutoff = pending[0][1] + STABLE_PADDING
            end = next((i for i, row in enumerate(pending) if row[1] > cutoff), len(pending))
            cluster, pending = pending[:end], pending[end:]
            if len({row[3] for row in cluster}) >= 2:
                detected.append(list(max(cluster, key=lambda row: (row[3] == -1, row[2], -row[1]))[:3]))
    chronological = sorted(detected, key=lambda row: (row[1], -row[2], row[0]))
    return [next(events) for _, events in groupby(chronological, key=lambda row: row[1])]


def measure(folder, model_path, wheel, frozen, output):
    source = json.loads(frozen.read_bytes())
    require(source["method"] in (DESIGN, CONTEXT_DESIGN) and source["schema"] == SCHEMA, "Unknown source roster")
    output.mkdir(mode=0o700)
    neural, device = neural_decoder(model_path)
    baseline = old_decoder(wheel, output / "baseline")
    rows = []
    for index, row in enumerate(source["rows"], 1):
        item = row["members"]["audio"]
        content = (folder / "audio-members" / f"{item['sha256']}.wav").read_bytes()
        require(digest(content) == item["sha256"], "Source audio bytes differ")
        original = pcm(content)
        require(len(original) == row["samples"], "Source sample count differs")
        readings = {}
        for role, delivery in variants(original, row["edit"]):
            full = neural(delivery)
            readings[role] = {"sha256": digest(delivery.tobytes()), "old": baseline(delivery), "neural": full}
            if source.get("earlier"):
                windows = context_readings(neural, delivery)
                readings[role].update(windows=windows, contexts=consensus(full, windows))
        rows.append(row | {"measurements": readings})
        if index % 4 == 0:
            print("Measured", index, "of", len(source["rows"]), flush=True)
    result = source | {"rows": rows, "roster_sha256": sha(frozen), "device": device}
    result["outcomes"] = review(result)
    write(output / "receipt.json", result)


def review(document):
    extended = document["method"] == CONTEXT_DESIGN
    earlier = document.get("earlier")
    require(document["schema"] == SCHEMA and document["method"] in (DESIGN, CONTEXT_DESIGN)
            and bool(earlier) == extended
            and document["revision"] == MODEL_REVISION and document["model_sha256"] == WEIGHT_PIN
            and document["baseline_sha256"] == SPHINX_PIN, "The comparison commitments differ")
    speakers = defaultdict(Counter)
    totals, errors = defaultdict(Counter), defaultdict(list)
    for row in document["rows"]:
        source = row["speaker"], row["split"]
        speakers[source][row["condition"]] += 1
        readings = row["measurements"]
        require(set(readings) == set(TREATMENTS), "Missing planned delivery")
        for delivery in readings.values():
            if extended:
                windows = delivery["windows"]
                require([window["start"] for window in windows] == context_starts(row["samples"])
                        and all(window["start"] + LOCAL_EDGE <= sample < window["start"] + LOCAL_WINDOW - LOCAL_EDGE
                                and (sample - CENTER) % STRIDE == 0 and .35 <= score <= 1
                                and name in ("b", "p") for window in windows for name, sample, score in window["events"])
                        and delivery["contexts"] == consensus(delivery["neural"], windows),
                        "Short-window support differs from its native timing or consensus")
            for method in (("old", "neural", "contexts") if extended else ("old", "neural")):
                events = delivery[method]
                require(all(name in ("b", "p") and 0 <= p < row["samples"] for name, p, _ in events)
                        and all(a[1] < b[1] for a, b in zip(events, events[1:])),
                        "Measured phone timestamps fall outside the verified audio clock")
                if method != "old":
                    require(all((point - CENTER) % STRIDE == 0 and .35 <= score <= 1.0
                                for _, point, score in events), "Neural readings do not bind to the acoustic frame clock")
        methods = ["old", "neural", "old_stable", "neural_stable"]
        if extended:
            methods.append("contexts")
        for method in methods:
            for role in TREATMENTS:
                if method.endswith("_stable") and role != "original":
                    continue
                if method.endswith("_stable"):
                    key = method.removesuffix("_stable")
                    events = stable(readings["original"][key], readings["filtered"][key])
                else:
                    events = readings[role][method]
                bucket = f"{row['split']}/{row['condition']}/{method}/{role}"
                stats = totals[bucket]
                shift = {"early": -SHIFT, "late": SHIFT}.get(role, 0)
                labels = row["labels"] if not shift else [[row["target"][0], row["target"][1] + shift, row["target"][2] + shift]]
                hits = match(events, labels, LABEL_PADDING)
                stats["files"] += 1
                stats["target_intervals"] += len(labels)
                stats["matched"] += len(hits)
                stats["files_with_hit"] += int(bool(hits))
                if not shift:
                    stats["unmatched_events"] += len(events) - len(hits)
                else:
                    original_hits = match(readings["original"][method], [row["target"]], LABEL_PADDING)
                    stats["paired_shifts_within_60ms"] += int(bool(hits and original_hits and
                        abs(hits[0][1][1] - original_hits[0][1][1] - shift) <= 3 * STRIDE))
                errors[bucket].extend(abs(obs[1] - (label[1] + label[2]) // 2) for label, obs in hits)
    require(Counter(split for _, split in speakers) == {"development": 4, "held": 8}
            and len({speaker for speaker, _ in speakers}) == 12
            and all(value == {"p": 2, "l": 2} and key[0] not in EXCLUDED
                    and (earlier is None or key[0] not in earlier["speakers"]) for key, value in speakers.items()),
            "Source roles or paired speech conditions differ")
    return {key: dict(value) | {"median_midpoint_error_samples": statistics.median(errors[key]) if errors[key] else None}
            for key, value in sorted(totals.items())}


def verify_previous(document, path):
    earlier = json.loads(path.read_bytes())
    require(earlier.get("method") == DESIGN and review(earlier) == earlier.get("outcomes")
            and document.get("earlier") == {
                "sha256": sha(path), "speakers": sorted({row["speaker"] for row in earlier["rows"]}),
            }, "The earlier speaker exclusions differ from the released benchmark")


def replay(path, folder, output):
    """Verify independent labels, authored PCM and exact surrounding native pictures."""
    from app.detectors.video.decode import decode_frames, probe_frames

    document = json.loads(path.read_bytes())
    require(review(document) == document["outcomes"], "The numeric benchmark did not replay")
    locations, metadata = source_archives(folder)
    require(metadata == document["archives"] and all(sha(folder / name) == pin
            for name, pin in document["metadata_sha256"].items()), "Publisher commitments differ")
    frames_total, waveforms = 0, 0
    bindings = []
    with (ZipFile(folder / "alignment.zip") as labels, ZipFile(locations["audio"]) as audio,
          ZipFile(locations["front"]) as front, TemporaryDirectory(prefix="phone-clock-replay-") as staging):
        clip = Path(staging) / "source.mov"
        for row in document["rows"]:
            files = {name: archive.read(row["members"][name]["name"])
                     for name, archive in (("alignment", labels), ("audio", audio), ("front", front))}
            require(all(digest(data) == row["members"][role]["sha256"] for role, data in files.items()),
                    "An original archive member differs from the study")
            require(exact_labels(files["alignment"], row["source"]) == row["labels"],
                    "A publisher phone interval differs from the study")
            for role, samples in variants(pcm(files["audio"]), row["edit"]):
                require(digest(samples.tobytes()) == row["measurements"][role]["sha256"], "An authored waveform changed")
                waveforms += 1
            clip.write_bytes(files["front"])
            table = probe_frames(clip, cap=160, timeout_s=18, max_pixels=1920 * 1080)
            require(len(table) >= 24 and not table.truncated and not table.interpolated
                    and all(a < b for a, b in zip(table.pts, table.pts[1:])), "No exact increasing source picture clock")
            frames_total += len(table)
            if row["split"] != "held":
                continue
            hits = match(row["measurements"]["original"]["neural"], [row["target"]], LABEL_PADDING)
            if not hits:
                continue
            (_, sample, _) = hits[0][1]
            relative_times = [(pts - table.pts[0]) * table.tb for pts in table.pts]
            closest = min(range(1, len(table) - 1), key=lambda i: abs(relative_times[i] - Fraction(sample, RATE)))
            if abs(relative_times[closest] - Fraction(sample, RATE)) > Fraction(1, 20):
                continue
            indices = [closest - 1, closest, closest + 1]
            records = []
            for picture in decode_frames(clip, table, indices, timeout_s=18):
                require(picture.pts_verified and picture.pts == table.pts[picture.index], "Picture timestamp differs")
                records.append({"frame": picture.index, "pts": picture.pts, "rgb_sha256": digest(picture.rgb.tobytes())})
            require([item["frame"] for item in records] == indices, "The picture context is incomplete")
            bindings.append({"source": row["source"], "original_video_sha256": row["members"]["front"]["sha256"],
                             "audio_sample": sample, "time_base": table.time_base,
                             "first_pts": table.pts[0], "frames": records})
    write(output, {"schema": SCHEMA + "/native-replay", "receipt_sha256": sha(path), "source_files": len(document["rows"]),
                   "verified_waveforms": waveforms, "source_picture_timestamps": frames_total, "examples": bindings})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review", "replay"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        write(args.path, source_rows(args.sources, args.previous))
    elif args.phase == "measure":
        measure(args.sources, args.model, args.wheel, args.path, args.output)
    elif args.phase == "replay":
        replay(args.path, args.sources, args.output)
    else:
        document = json.loads(args.path.read_bytes())
        if args.previous is not None:
            verify_previous(document, args.previous)
        outcomes = review(document)
        require(outcomes == document["outcomes"], "Replayed results differ")
        print(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    main()
