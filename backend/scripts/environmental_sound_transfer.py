"""Test local non-speech origin cues with source- and generator-disjoint EnvSDD inputs."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal.windows import hann

from app.detectors.audio.decode import decode_audio, wav_bytes
from scripts.audio_mono_dereverb_transfer import read, sha, write
from scripts.audio_phase_corroboration import sample_bytes

SCHEMA = "chipmunk.environmental-sound-transfer/1"
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
SHARD = "test-00013-of-00014"
URL = f"https://huggingface.co/datasets/EnvSDD/EnvSDD/resolve/{REVISION}/data/{SHARD}.parquet"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
ROLE_SIZES = {"fit": 12, "selection": 10, "held": 18}
DEVELOPMENT_GENERATORS = ("audioldm1", "audioldm2")
HELD_GENERATORS = ("audiogen", "audiolcm")
BANDS = (93, 218, 437, 843, 1500, 2500, 4218, 6875)
RATE, FFT, HOP, N = 16_000, 512, 128, 48_000
WINDOW = hann(FFT, sym=False)
FREQUENCIES = np.fft.rfftfreq(FFT, 1/RATE)
SALT = "chipmunk-environmental-20261111-v1"
METHOD = {
    "eligible": "First 1100 publisher rows. Original recording must have one unambiguous exact Clotho source match and CC0 grant. "
                "No duplicate Freesound creator or sound identifier. Use only TTA synthetic outputs.",
    "role_sizes": ROLE_SIZES,
    "order": SALT,
    "development_generators": DEVELOPMENT_GENERATORS,
    "held_generators": HELD_GENERATORS,
    "input": "Center three seconds of exact decoded mono at 16 kHz. Test native, 32 kb/s Opus, and -6 dB "
             "environment-to-genuine-speech mixtures on held source pairs.",
    "feature_frame": [FFT, HOP],
    "bands": BANDS,
    "magnitude": "Per-band relative energy median, interquartile range and spectral flatness.",
    "texture": "Per-band log-envelope residue, normalized discrete Teager energy, and 8 and 24 ms lag consistency.",
    "fit": "Equal weight for the two origins and each source within an origin. Standardized ridge head with penalty 0.50.",
    "selection": "Lowest point above the second-highest genuine response. At most one false result in ten source controls.",
    "advance": "Each unseen generator requires at least 11/18 isolated hits in native and Opus, and each delivery "
               "at most 2/18 genuine false alarms. For a mixed claim require that gate separately in the speech mixture.",
}


def read_csv(path, encoding="utf-8"):
    with path.open(encoding=encoding, newline="") as stream:
        return list(csv.DictReader(stream))


def metadata_paths(folder):
    base = folder / "metadata"
    return [base / name for name in (f"{SHARD}-rows.json", "envsdd-test-metadata.csv", "clotho-development.csv",
                                     "clotho-validation.csv", "clotho-eval.csv", "env-card.md", "upstream-commit.json")]


def prepare(folder, speech_parent):
    manifests = metadata_paths(folder)
    publisher, match = read(manifests[0]), {row["wavename"]: row for row in read_csv(manifests[1])}
    licensed = {}
    ambiguous = set()
    for manifest in manifests[2:5]:
        for record in read_csv(manifest, "latin1"):
            filename = record["file_name"]
            if filename in licensed and record != licensed[filename]:
                ambiguous.add(filename)
            licensed[filename] = record
    groups = defaultdict(dict)
    for index, sample in enumerate(publisher[:1100]):
        item = match[sample["filename"]]
        if (sample["dataset"] != "Clotho" or item["source dataset"] != "Clotho"
                or item["faketype"] != sample["attack_type"]):
            raise ValueError("Publisher shard differs from the source protocol")
        if sample["attack_type"] in {"real", "tta"}:
            kind = "real" if sample["attack_type"] == "real" else sample["generative_model"]
            if kind in groups[item["source wavename"]]:
                raise ValueError("Duplicate generator in source group")
            groups[item["source wavename"]][kind] = [index, sample["filename"]]
    eligible, qualified = [], 0
    required = {"real", *DEVELOPMENT_GENERATORS, *HELD_GENERATORS}
    for source, parts in groups.items():
        rights = licensed.get(source)
        if (not rights or source in ambiguous or rights["license"].replace("http:", "https:") != CC0
                or not required <= parts.keys()):
            continue
        if any(not rights[field] for field in ("manufacturer", "sound_id", "sound_link")):
            raise ValueError("A qualified recording is missing source provenance")
        eligible.append((sha((SALT + rights["sound_id"]).encode()), source, parts, rights))
        qualified += 1
    result, creators, ids = [], set(), set()
    roles = [role for role, number in ROLE_SIZES.items() for _ in range(number)]
    for _, source, parts, rights in sorted(eligible):
        creator = rights["manufacturer"].casefold().strip()
        if creator in creators or rights["sound_id"] in ids:
            continue
        creators.add(creator)
        ids.add(rights["sound_id"])
        role = roles[len(result)]
        selected = HELD_GENERATORS if role == "held" else DEVELOPMENT_GENERATORS
        result.append({"role": role, "group": len(result), "source": source, "license": CC0,
                       "creator": rights["manufacturer"], "source_id": rights["sound_id"], "source_url": rights["sound_link"],
                       "parts": {name: parts[name] for name in ("real", *selected)}})
        if len(result) == len(roles):
            break
    if len(result) != len(roles):
        raise ValueError("Insufficient commercially usable and creator-disjoint source groups")
    speech_rows = [r for r in read(speech_parent/"roster.json")
                   if r["role"] == "transfer" and r["placement"] == "clean" and r["family"] == "genuine"]
    if len(speech_rows) != 7 or len({r["group"] for r in speech_rows}) != 7:
        raise ValueError("Expected seven exact held genuine foreground voices")
    for row in speech_rows:
        sample_bytes(speech_parent, row)
    locked = {"schema": SCHEMA, "method": METHOD, "dataset_revision": REVISION, "shard": SHARD,
              "upstream_protocol_revision": read(manifests[-1])["sha"], "inputs": {p.name: sha(p) for p in manifests},
              "audio_parent_sha256": sha(speech_parent/"roster.json"), "speech_rows": speech_rows,
              "qualified_groups": qualified, "groups": result, "code_sha256": sha(Path(__file__))}
    write(folder/"freeze.json", locked)
    print(f"Frozen {len(result)} source groups and {len(result)*3} isolated clips", flush=True)


def bound(folder):
    freeze = read(folder/"freeze.json")
    if (freeze["code_sha256"] != sha(Path(__file__)) or freeze["method"] != json.loads(json.dumps(METHOD))
            or freeze["inputs"] != {p.name: sha(p) for p in metadata_paths(folder)}):
        raise ValueError("A source manifest or the procedure changed after freeze")
    return freeze


def acquire(folder):
    import fsspec
    import pyarrow.parquet as pq

    lock = bound(folder)
    wanted = {spec[0]: (group["group"], key, spec[1]) for group in lock["groups"] for key, spec in group["parts"].items()}
    (folder/"clips").mkdir()
    receipt, offset = {}, 0
    with fsspec.open(URL, block_size=32768, cache_type="readahead") as remote:
        archive = pq.ParquetFile(remote)
        for number in range(archive.num_row_groups):
            count = archive.metadata.row_group(number).num_rows
            needed = [index for index in wanted if offset <= index < offset+count]
            if needed:
                table = archive.read_row_group(number, columns=["audio", "filename"], use_threads=False)
                for index in needed:
                    group, lane, filename = wanted[index]
                    row = table.slice(index-offset, 1).to_pylist()[0]
                    data = row["audio"]["bytes"]
                    if row["filename"] != filename or not data or len(data) > 2_000_000:
                        raise ValueError("Unexpected audio body or source order in the pinned shard")
                    name = f"{group:02}-{lane}"
                    file = folder/"clips"/f"{name}.wav"
                    file.write_bytes(data)
                    decoded = decode_audio(file, source_sha256=sha(data), max_seconds=4)
                    if len(decoded.pcm) < N:
                        raise ValueError("Publisher clip is shorter than the fixed analysis input")
                    start = (len(decoded.pcm)-N)//2
                    raw = np.rint(decoded.pcm[start:start+N]*32768).astype("<i2").tobytes()
                    (folder/"clips"/f"{name}.pcm").write_bytes(raw)
                    receipt[name] = {"wav_sha256": sha(data), "pcm_sha256": sha(raw), "start_sample": start, "size": len(data)}
                print(f"Row group {number}: {len(receipt)} of {len(wanted)} clips", flush=True)
            offset += count
    if len(receipt) != len(wanted):
        raise ValueError("The frozen audio intake is incomplete")
    write(folder/"acquisition.json", {"freeze_sha256": sha(folder/"freeze.json"), "clips": receipt})


def features(data):
    x = np.frombuffer(data, "<i2").astype(float) / 32768
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT)[::HOP] * WINDOW
    power = np.abs(np.fft.rfft(frames))**2
    total = np.maximum(power.sum(axis=1), 1e-12)
    magnitude, texture = [], []
    for a, b in zip(BANDS[:-1], BANDS[1:], strict=True):
        selected = np.maximum(power[:, (a <= FREQUENCIES) & (b > FREQUENCIES)], 1e-12)
        envelope = selected.sum(axis=1)
        relative = np.log(envelope/total)
        flatness = np.exp(np.log(selected).mean(axis=1)) / selected.mean(axis=1)
        magnitude.extend((np.median(relative), np.subtract(*np.percentile(relative, [75, 25])), np.median(flatness)))
        logamp = .5*np.log(envelope)
        residue = logamp - uniform_filter1d(logamp, size=11, mode="nearest")
        amplitude = np.sqrt(envelope)
        energy = (amplitude[1:-1]**2 - amplitude[:-2]*amplitude[2:]) / np.maximum(amplitude[1:-1]**2, 1e-12)
        texture.extend((np.median(abs(residue)), np.median(np.clip(energy, -2, 2)),
                        np.percentile(abs(np.clip(energy, -2, 2)), 90)))
        for lag in (1, 3):
            u, v = residue[lag:], residue[:-lag]
            texture.append(float(np.dot(u, v)/max(1e-12, np.linalg.norm(u)*np.linalg.norm(v))))
    return {"magnitude": list(map(float, magnitude)), "texture": list(map(float, texture))}


def clip(folder, receipt, name):
    data = (folder/"clips"/f"{name}.pcm").read_bytes()
    if len(data) != N*2 or sha(data) != receipt[name]["pcm_sha256"]:
        raise ValueError("Decoded environmental input differs from the frozen receipt")
    return data


def fit_head(inputs, key):
    matrix = np.array([v[key] for _, v in inputs])
    positive = np.array([label for label, _ in inputs], dtype=bool)
    weight = np.where(positive, .5/positive.sum(), .5/(~positive).sum())
    mean = weight @ matrix
    scale = np.maximum(np.sqrt(weight @ (matrix-mean)**2), 1e-5)
    z = (matrix-mean)/scale
    coef = np.linalg.solve((z.T*weight)@z + .50*np.eye(z.shape[1]), z.T@(weight*(positive-.5)))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist()}


def raw(head, vector):
    return float(((np.asarray(vector)-head["mean"])/head["scale"])@head["coef"])


def intake(folder):
    lock, receipt = bound(folder), read(folder/"acquisition.json")
    if receipt["freeze_sha256"] != sha(folder/"freeze.json"):
        raise ValueError("Intake does not match the frozen source roles")
    return lock, receipt["clips"]


def fit(folder):
    lock, receipt = intake(folder)
    data = defaultdict(list)
    for row in lock["groups"]:
        if row["role"] == "held":
            continue
        for lane in row["parts"]:
            name = f"{row['group']:02}-{lane}"
            data[row["role"]].append((lane != "real", features(clip(folder, receipt, name))))
    result = {}
    for key in ("magnitude", "texture"):
        head = fit_head(data["fit"], key)
        control = sorted([raw(head, vec[key]) for synthetic, vec in data["selection"] if not synthetic])
        cutoff = float(np.nextafter(control[-2], np.inf))
        result[key] = {"head": head, "cutoff": cutoff,
                       "selection_true_flags": sum(s and raw(head, v[key]) >= cutoff for s, v in data["selection"]),
                       "selection_false_flags": sum(not s and raw(head, v[key]) >= cutoff for s, v in data["selection"])}
    write(folder/"heads.json", {"intake_sha256": sha(folder/"acquisition.json"), "models": result})
    print("Two small heads and separate selection points frozen", flush=True)


def convert(folder, data):
    source, target = folder/"delivery.wav", folder/"delivery.opus"
    source.write_bytes(wav_bytes(np.frombuffer(data, "<i2").astype(np.float32)/32768))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-ar", "48000", "-c:a", "libopus",
                    "-b:a", "32k", "-vbr", "off", str(target)], check=True, timeout=30)
    out = decode_audio(target, source_sha256=sha(target), max_seconds=3).pcm
    if len(out) != N:
        raise ValueError("Opus did not preserve the committed three-second grid")
    return np.rint(out*32768).astype("<i2").tobytes()


def mix(environment, speech):
    a, b = (np.frombuffer(data, "<i2").astype(float) for data in (environment, speech))
    level = np.linalg.norm(b)/np.linalg.norm(a)*10**(-.3)
    mixture = a*level + b
    mixture *= min(1., 30_000/np.max(abs(mixture)))
    return np.rint(mixture).astype("<i2").tobytes()


def score(folder, speech_parent):
    lock, receipt = intake(folder)
    if lock["audio_parent_sha256"] != sha(speech_parent/"roster.json"):
        raise ValueError("Genuine overlap sources changed")
    heads = read(folder/"heads.json")
    if heads["intake_sha256"] != sha(folder/"acquisition.json"):
        raise ValueError("Points do not match this intake")
    voices = [sample_bytes(speech_parent, row) for row in lock["speech_rows"]]
    rows, counts = [], {key: defaultdict(Counter) for key in heads["models"]}
    for index, row in enumerate(r for r in lock["groups"] if r["role"] == "held"):
        for lane in row["parts"]:
            pcm = clip(folder, receipt, f"{row['group']:02}-{lane}")
            for view, data in (("native", pcm), ("opus", convert(folder, pcm)), ("speech_mix", mix(pcm, voices[index%len(voices)]))):
                vec = features(data)
                values = {key: raw(model["head"], vec[key]) for key, model in heads["models"].items()}
                rows.append({"group": row["group"], "lane": lane, "view": view, "sha256": sha(data), "responses": values})
                for key, value in values.items():
                    counts[key][f"{view}/{lane}"].update(planned=1, flagged=value >= heads["models"][key]["cutoff"])
        if index % 6 == 5:
            print(f"{len(rows)} held responses", flush=True)
    for name in ("delivery.wav", "delivery.opus"):
        (folder/name).unlink()
    write(folder/"scores.json", {"head_sha256": sha(folder/"heads.json"), "rows": rows})
    arms = {}
    for key, cells in counts.items():
        meets = {view: cells[f"{view}/real"]["flagged"] <= 2 and all(cells[f"{view}/{lane}"]["flagged"] >= 11
                 for lane in HELD_GENERATORS) for view in ("native", "opus", "speech_mix")}
        arms[key] = {"cells": cells, "passes_view": meets, "advance_isolated": meets["native"] and meets["opus"],
                     "advance_speech_mix": all(meets.values()),
                     **{name: heads["models"][key][name] for name in ("selection_true_flags", "selection_false_flags")}}
    write(folder/"aggregate.json", {"schema": SCHEMA, "method": METHOD, "dataset_revision": REVISION,
          "upstream_protocol_revision": lock["upstream_protocol_revision"], "freeze_sha256": sha(folder/"freeze.json"),
          "intake_sha256": sha(folder/"acquisition.json"), "head_sha256": sha(folder/"heads.json"),
          "scores_sha256": sha(folder/"scores.json"), "unique_source_groups": len(lock["groups"]),
          "exact_held_views": len(rows), "arms": arms})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "acquire", "fit", "score"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--speech-parent", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.folder, args.speech_parent)
    elif args.stage == "score":
        score(args.folder, args.speech_parent)
    else:
        {"acquire": acquire, "fit": fit}[args.stage](args.folder)
