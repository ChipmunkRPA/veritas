"""Independently read back the environmental-sound source and scoring receipts."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal.windows import hann

from scripts.review_audio_landmark_repeat_transfer import digest, quantized, read, require

RATE, SAMPLES = 16_000, 48_000
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
SHARD = "test-00013-of-00014"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
SALT = "chipmunk-environmental-20261111-v1"
ROLES = ("fit",) * 12 + ("selection",) * 10 + ("held",) * 18
DEVELOPMENT = ("audioldm1", "audioldm2")
TRANSFER = ("audiogen", "audiolcm")
LANES = ("real", *DEVELOPMENT, *TRANSFER)
VIEWS = ("native", "opus", "speech_mix")
HEADS = ("magnitude", "texture")
METADATA = (f"{SHARD}-rows.json", "envsdd-test-metadata.csv", "clotho-development.csv",
            "clotho-validation.csv", "clotho-eval.csv", "env-card.md", "upstream-commit.json")
BANDS = (93, 218, 437, 843, 1500, 2500, 4218, 6875)


def csv_rows(path, encoding):
    with path.open(encoding=encoding, newline="") as stream:
        return list(csv.DictReader(stream))


def verify_roster(folder, freeze):
    base = folder / "metadata"
    require(freeze["inputs"] == {name: digest(base / name) for name in METADATA}
            and (freeze["dataset_revision"], freeze["shard"]) == (REVISION, SHARD)
            and freeze["upstream_protocol_revision"] == read(base / "upstream-commit.json")["sha"],
            "Publisher metadata commitments differ")
    require("license: cc-by-4.0" in (base / "env-card.md").read_text(), "Missing publisher attribution terms")
    env = csv_rows(base / METADATA[1], "utf-8")
    by_file = {row["wavename"]: row for row in env}
    require(len(by_file) == len(env), "Repeated publisher clip filename")
    rights = defaultdict(set)
    for name in METADATA[2:5]:
        for row in csv_rows(base / name, "latin1"):
            rights[row["file_name"]].add(tuple(sorted(row.items())))
    sources = defaultdict(dict)
    publisher = read(base / METADATA[0])
    require(len(publisher) >= 1100, "Missing pinned publisher rows")
    for index, clip in enumerate(publisher[:1100]):
        meta = by_file[clip["filename"]]
        require(clip["dataset"] == meta["source dataset"] == "Clotho" and meta["faketype"] == clip["attack_type"],
                "Publisher filename or attack type differs")
        if meta["faketype"] not in {"real", "tta"}:
            continue
        lane = "real" if meta["faketype"] == "real" else clip["generative_model"]
        require((lane == "real" and meta["generator"] == "none" and clip["label"] == "real")
                or (lane == meta["generator"] and clip["label"] == "fake"), "Publisher generator or label differs")
        parts = sources[meta["source wavename"]]
        require(lane not in parts, "Duplicate lane for one source")
        parts[lane] = [index, clip["filename"]]
    qualified = []
    for source, parts in sources.items():
        exact = rights[source]
        if len(exact) != 1 or not set(LANES) <= parts.keys():
            continue
        licence = dict(next(iter(exact)))
        if licence["license"].replace("http:", "https:") == CC0:
            qualified.append((digest((SALT + licence["sound_id"]).encode()), source, parts, licence))
    require(len(qualified) == freeze["qualified_groups"], "Number of unambiguous CC0 source groups differs")
    authors, ids, expected = set(), set(), []
    for _, source, parts, licence in sorted(qualified):
        author = licence["manufacturer"]
        owner, source_id, url = author.casefold().strip(), licence["sound_id"], licence["sound_link"]
        if owner in authors or source_id in ids:
            continue
        parsed = urlsplit(url)
        require(bool(owner) and source_id.isdecimal() and parsed.hostname == "freesound.org"
                and parsed.path.rstrip("/").endswith(f"/sounds/{source_id}"), "Source attribution or identifier differs")
        group = len(expected)
        role = ROLES[group]
        lanes = TRANSFER if role == "held" else DEVELOPMENT
        expected.append({"role": role, "group": group, "source": source, "license": CC0,
                         "creator": author, "source_id": source_id, "source_url": url,
                         "parts": {lane: parts[lane] for lane in ("real", *lanes)}})
        authors.add(owner)
        ids.add(source_id)
        if len(expected) == len(ROLES):
            break
    require(expected == freeze["groups"] and len(authors) == len(ids) == len(ROLES), "Selected sources, authors or roles differ")
    return expected


def verify_clips(folder, acquisition, groups):
    receipt = acquisition["clips"]
    names = {f"{row['group']:02}-{lane}" for row in groups for lane in row["parts"]}
    require(set(receipt) == names and len(names) == 120, "Incomplete acquisition")
    result = {}
    for name, row in receipt.items():
        base = folder / "clips"
        wav = (base / f"{name}.wav").read_bytes()
        pcm = (base / f"{name}.pcm").read_bytes()
        require(len(wav) == row["size"] and digest(wav) == row["wav_sha256"]
                and len(pcm) == SAMPLES * 2 and digest(pcm) == row["pcm_sha256"], "Source WAV or frozen PCM differs")
        with wave.open(io.BytesIO(wav), "rb") as source:
            require((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()) ==
                    (1, 2, RATE, 64_000), "Publisher WAV is not four seconds of 16 kHz signed mono")
            start = (source.getnframes() - SAMPLES) // 2
            source.setpos(start)
            require(start == row["start_sample"] and source.readframes(SAMPLES) == pcm,
                    "Committed interval is not exact native center PCM")
        result[name] = pcm
    for key in ("wav_sha256", "pcm_sha256"):
        require(len({clip[key] for clip in receipt.values()}) == len(names), "Repeated source or analysis bytes")
    return result


def verify_voices(parent, freeze):
    roster = parent / "roster.json"
    require(digest(roster) == freeze["audio_parent_sha256"], "Speech parent roster changed")
    voices = [row for row in read(roster) if (row["role"], row["placement"], row["family"]) == ("transfer", "clean", "genuine")]
    require(voices == freeze["speech_rows"] and len({row["group"] for row in voices}) == len(voices) == 7,
            "Speech mixtures do not use the seven committed genuine groups")
    pcm = [(parent / "mono" / f"{row['id']:03}.pcm").read_bytes() for row in voices]
    require(all(len(data) == SAMPLES * 2 and digest(data) == row["mono_sha256"]
                for row, data in zip(voices, pcm, strict=True)), "Speech mixture input differs")
    return pcm


def fixed_features(data):
    signal = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
    offsets = np.arange(0, SAMPLES - 511, 128)[:, None] + np.arange(512)
    power = np.abs(np.fft.rfft(signal[offsets] * hann(512, sym=False))) ** 2
    frequency = np.fft.rfftfreq(512, 1 / RATE)
    total = np.maximum(power.sum(axis=1), 1e-12)
    mag, texture = [], []
    for lower, upper in zip(BANDS[:-1], BANDS[1:], strict=True):
        bins = np.maximum(power[:, (frequency >= lower) & (frequency < upper)], 1e-12)
        envelope = bins.sum(axis=1)
        log_ratio = np.log(envelope / total)
        q1, q3 = np.percentile(log_ratio, [25, 75])
        flatness = np.exp(np.log(bins).mean(axis=1)) / bins.mean(axis=1)
        mag.extend((np.median(log_ratio), q3 - q1, np.median(flatness)))
        log_amplitude = .5 * np.log(envelope)
        residual = log_amplitude - uniform_filter1d(log_amplitude, size=11, mode="nearest")
        amplitude = np.sqrt(envelope)
        middle = amplitude[1:-1] ** 2
        energy = np.clip((middle - amplitude[:-2] * amplitude[2:]) / np.maximum(middle, 1e-12), -2, 2)
        texture.extend((np.median(np.abs(residual)), np.median(energy), np.percentile(np.abs(energy), 90)))
        for lag in (1, 3):
            a, b = residual[lag:], residual[:-lag]
            texture.append(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
    return {"magnitude": np.asarray(mag), "texture": np.asarray(texture)}


def responses(heads, data):
    vectors = fixed_features(data)
    return {key: float(np.dot((vectors[key] - model["head"]["mean"]) / model["head"]["scale"], model["head"]["coef"]))
            for key, model in heads.items()}


def verify_points(heads, groups, clips):
    selected = [(lane != "real", responses(heads, clips[f"{row['group']:02}-{lane}"]))
                for row in groups if row["role"] == "selection" for lane in row["parts"]]
    require(len(selected) == 30 and sum(not label for label, _ in selected) == 10, "Selection split differs")
    points = {}
    for key, model in heads.items():
        maximums = sorted(scores[key] for label, scores in selected if not label)
        point = float(np.nextafter(maximums[-2], np.inf))
        true = sum(label and score[key] >= point for label, score in selected)
        false = sum(not label and score[key] >= point for label, score in selected)
        require((point, true, false) == tuple(model[name] for name in ("cutoff", "selection_true_flags", "selection_false_flags")),
                f"{key} cutoff or selection counts differ")
        points[key] = {"cutoff": point, "second_highest_control": maximums[-2], "selection_true_flags": true, "selection_false_flags": false}
    return points


def opus(pcm, scratch):
    wav, encoded = scratch / "test.wav", scratch / "test.opus"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    subprocess.run([*command, "-y", "-i", str(wav), "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", "-vbr", "off", str(encoded)],
                   check=True, timeout=30)
    filters = "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,atrim=end_sample=48000,asetpts=N/SR/TB"
    decoded = subprocess.run([*command, "-copyts", "-i", str(encoded), "-map", "0:a:0", "-af", filters, "-vn", "-t", "3",
                              "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"],
                             capture_output=True, check=True, timeout=30).stdout
    require(len(decoded) == SAMPLES * 2, "Opus sample count differs")
    return decoded


def speech_mix(pcm, voice):
    environment, speech = (np.frombuffer(data, dtype="<i2").astype(np.float64) for data in (pcm, voice))
    ratio = np.sqrt(np.dot(speech, speech) / np.dot(environment, environment)) * 10 ** (-6 / 20)
    signal = environment * ratio + speech
    return quantized(signal * min(1., 30_000 / np.max(np.abs(signal))))


def verify_scores(scores, groups, clips, voices, heads, points):
    rows = {(row["group"], row["lane"], row["view"]): row for row in scores}
    require(len(rows) == len(scores) == 162, "Missing or repeated held views")
    cells = {head: defaultdict(Counter) for head in HEADS}
    with tempfile.TemporaryDirectory(prefix="chipmunk-env-review-") as directory:
        for ordinal, source in enumerate(group for group in groups if group["role"] == "held"):
            for lane in ("real", *TRANSFER):
                pcm = clips[f"{source['group']:02}-{lane}"]
                versions = (pcm, opus(pcm, Path(directory)), speech_mix(pcm, voices[ordinal % len(voices)]))
                for view, data in zip(VIEWS, versions, strict=True):
                    row = rows.pop((source["group"], lane, view))
                    require(digest(data) == row["sha256"], f"{view} reconstructed PCM differs")
                    values = responses(heads, data)
                    require(values == row["responses"], f"{view} fixed-head responses differ")
                    for key, value in values.items():
                        cells[key][f"{view}/{lane}"].update(planned=1, flagged=value >= points[key]["cutoff"])
    require(not rows, "Unexpected held rows")
    outcome = {}
    for key, counts in cells.items():
        passed = {view: counts[f"{view}/real"]["flagged"] <= 2
                  and min(counts[f"{view}/{lane}"]["flagged"] for lane in TRANSFER) >= 11 for view in VIEWS}
        outcome[key] = {"cells": counts, "passes_view": passed, "advance_isolated": passed["native"] and passed["opus"],
                        "advance_speech_mix": all(passed.values()),
                        **{name: points[key][name] for name in ("selection_true_flags", "selection_false_flags")}}
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "speech_parent", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    paths = {name: args.run / f"{name}.json" for name in ("freeze", "acquisition", "heads", "scores", "aggregate")}
    freeze, acquisition, heads, scores, aggregate = (read(path) for path in paths.values())
    require(freeze["code_sha256"] == digest(Path(__file__).with_name("environmental_sound_transfer.py"))
            and aggregate["schema"] == freeze["schema"] == "chipmunk.environmental-sound-transfer/1"
            and aggregate["method"] == freeze["method"]
            and all(aggregate[key] == freeze[key] for key in ("dataset_revision", "upstream_protocol_revision")),
            "Frozen method or revision differs")
    for item, key, objects in (("freeze", "freeze_sha256", (acquisition, aggregate)), ("acquisition", "intake_sha256", (heads, aggregate)),
                               ("heads", "head_sha256", (scores, aggregate)), ("scores", "scores_sha256", (aggregate,))):
        require(all(obj[key] == digest(paths[item]) for obj in objects), f"{item} commitment differs")
    groups = verify_roster(args.run, freeze)
    clips = verify_clips(args.run, acquisition, groups)
    voices = verify_voices(args.speech_parent, freeze)
    require(set(heads["models"]) == set(HEADS), "Unexpected frozen heads")
    points = verify_points(heads["models"], groups, clips)
    print("Verified 40 CC0 groups, 120 source WAVs and PCM clips, and both fixed cutoffs", flush=True)
    arms = verify_scores(scores["rows"], groups, clips, voices, heads["models"], points)
    require(aggregate["arms"] == arms and aggregate["exact_held_views"] == 162
            and aggregate["unique_source_groups"] == 40, "Aggregate or held decision differs")
    receipt = {"status": "verified", "aggregate_sha256": digest(paths["aggregate"]), "qualified_cc0_groups": freeze["qualified_groups"],
               "unique_source_ids": 40, "unique_authors": 40, "source_groups_by_role": Counter(ROLES),
               "source_wavs_and_exact_pcm_clips": len(clips), "native_center_range_samples": [8000, 56_000],
               "genuine_mix_source_groups": len(voices), "exact_held_views": dict.fromkeys(VIEWS, 54), "points": points, "arms": arms}
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print("Verified all 162 held PCM hashes, fixed responses, and decisions", flush=True)


if __name__ == "__main__":
    main()
