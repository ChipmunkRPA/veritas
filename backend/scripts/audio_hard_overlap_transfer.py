"""Test a frozen voice-overlap veto on similar, disjoint Parliament speakers."""
from __future__ import annotations

import argparse
import hashlib
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_independent_overlap_count as count_model
from scripts import audio_measured_speaker_routing as diarization
from scripts import audio_overlap_count_level_replay as level_replay

REVISION = "db0a600a0776f910769e9dc1fdd7f218e8f78a45"
LANGUAGE_QUOTAS = {"et": 2, "lt": 2, "sl": 3, "fi": 8, "nl": 9}
DB_CONDITIONS = {"two": None, "three_balanced": 0, "three_quiet": -9, "three_faint": -15}
ECHOES = ((0, 1.), (211, .29), (487, -.18), (913, .11), (1760, -.06))
METHOD = {
    "corpus": "VoxPopuli CC0 test splits from one pinned official parquet conversion",
    "roles": "One 3-second admitted source per publisher speaker ID, no reused speakers across source triplets or languages",
    "eligibility": "Same language and declared gender with median periodic pitch spread <=4 semitones and >=0.8 s simultaneous activity on separate dry tracks",
    "conditioning": "−5 dB foreground-to-total-interference energy, center placement, compare dry and fixed linear short reflections, rotate the source roles within each triplet",
    "decisions": "Corroborated diarization. Compare its two-voice route with the raw count veto and frozen −24 dBFS zero-or-overlap veto from the prior room work.",
    "advance": "Require baseline mistakes from at least five independent triplets. Keep at least 80% of the two-voice controls that baseline routed, reduce three-voice mistaken routes by half, and permit at most one triplet with a remaining three-voice mistake.",
    "limit": "Controlled speech counting only. No authenticity score is measured or changed.",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def median_pitch(signal: np.ndarray) -> float | None:
    """Return a semitone coordinate for qualification only, not a voice ID."""
    frames = np.lib.stride_tricks.sliding_window_view(signal, 800)[::320].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frames = frames[rms >= max(.001, np.max(rms) * .15)]
    if len(frames) < 20:
        return None
    low, high = 46, 230
    length = frames.shape[1] - high
    lead = frames[:, :length]
    cross = np.fft.irfft(np.fft.rfft(frames, 2048) * np.fft.rfft(lead, 2048).conj(), 2048)[:, low:high + 1]
    sums = np.pad(np.cumsum(frames ** 2, axis=1), ((0, 0), (1, 0)))
    delayed_energy = sums[:, low + length:high + length + 1] - sums[:, low:high + 1]
    norms = np.linalg.norm(lead, axis=1)[:, None] * np.sqrt(np.maximum(delayed_energy, 0))
    scores = cross / np.maximum(norms, 1e-12)
    peaks = (scores[:, 1:-1] >= scores[:, :-2]) & (scores[:, 1:-1] >= scores[:, 2:])
    strength = np.where(peaks, scores[:, 1:-1], 0)
    best = strength.max(axis=1)
    chosen = (strength >= np.maximum(.58, .92 * best[:, None])).argmax(axis=1) + low + 1
    notes = 69 + 12 * np.log2(diarization.RATE / chosen[best >= .58] / 440)
    return float(np.median(notes)) if len(notes) >= 20 else None


def reflected(signal: np.ndarray) -> np.ndarray:
    result = np.zeros_like(signal)
    for offset, gain in ECHOES:
        result[offset:] += signal[:len(signal) - offset] * gain
    return result


def mix(voices: list[np.ndarray]) -> dict[tuple[int, str, str], bytes]:
    if len(voices) != 3 or any(source.shape != (diarization.SAMPLES,) or np.linalg.norm(source) == 0 for source in voices):
        raise ValueError("Expected three admitted 48,000-sample sources")
    variants = {}
    for delivery, clips in (("dry", voices), ("reflective", [reflected(s) for s in voices])):
        normalized = [source / np.linalg.norm(source) for source in clips]
        for rotation in range(3):
            a, b, c = normalized[rotation:] + normalized[:rotation]
            for name, third_db in DB_CONDITIONS.items():
                interference = b if third_db is None else b + c * 10 ** (third_db / 20)
                variants[rotation, delivery, name] = a + 10 ** .25 * interference / np.linalg.norm(interference)
    gain = .92 * 32768 / max(np.max(np.abs(x)) for x in variants.values())
    return {key: np.rint(x * gain).astype("<i2").tobytes() for key, x in variants.items()}


def read_clip(audio: bytes, activity):
    import soundfile as sf
    from scipy.signal import resample_poly

    wave, rate = sf.read(io.BytesIO(audio), dtype="float64", always_2d=True)
    if len(wave) > rate * 120:
        return None
    mono = wave.mean(axis=1)
    if rate != diarization.RATE:
        factor = math.gcd(rate, diarization.RATE)
        mono = resample_poly(mono, diarization.RATE // factor, rate // factor)
    if len(mono) < diarization.SAMPLES or not np.isfinite(mono).all():
        return None
    start = (len(mono) - diarization.SAMPLES) // 2
    pcm = np.rint(np.clip(mono[start:start + diarization.SAMPLES], -1, 32767 / 32768) * 32768).astype("<i2")
    samples = pcm.astype("float64") / 32768
    active = np.zeros(diarization.SAMPLES, dtype=bool)
    for a, b in activity.spans(samples.astype("float32")):
        active[round(a * diarization.RATE):round(b * diarization.RATE)] = True
    if active.sum() < 28000:
        return None
    pitch = median_pitch(samples)
    return None if pitch is None else (samples, active, pitch, start, pcm.tobytes())


def speakers(archive: Path, activity):
    import pyarrow.parquet as pq

    source = pq.ParquetFile(archive)
    planned = defaultdict(list)
    for row in source.read(columns=["audio_id", "speaker_id", "gender"]).to_pylist():
        if row["speaker_id"] and row["gender"] in {"female", "male"}:
            planned[row["speaker_id"]].append(row["audio_id"])
    def order(name):
        return digest(f"chipmunk.hard-overlap/1:{name}".encode())
    shortlist = {name for alternatives in planned.values() for name in sorted(alternatives, key=order)[:5]}
    qualified = {}
    for batch in source.iter_batches(batch_size=8, columns=["audio_id", "speaker_id", "gender", "audio"]):
        for row in batch.to_pylist():
            name, identity = row["audio_id"], row["speaker_id"]
            if name not in shortlist or identity in qualified and order(qualified[identity]["name"]) <= order(name):
                continue
            encoded = row["audio"]["bytes"]
            clip = read_clip(encoded, activity) if encoded else None
            if clip is not None:
                qualified[identity] = {"name": name, "identity": identity, "gender": row["gender"],
                                       "encoded_sha256": digest(encoded), "clip": clip}
    return len(planned), qualified


def triplets(qualified: dict, limit: int, used: set[str]):
    for kind in ("female", "male"):
        pool = sorted((row for row in qualified.values() if row["gender"] == kind and row["identity"] not in used),
                      key=lambda row: (row["clip"][2], row["identity"]))
        while len(pool) >= 3 and limit:
            a, b, c = pool[:3]
            common = int(np.count_nonzero(a["clip"][1] & b["clip"][1] & c["clip"][1]))
            if c["clip"][2] - a["clip"][2] > 4 or common < 12800:
                pool.pop(0)
                continue
            del pool[:3]
            used.update(item["identity"] for item in (a, b, c))
            limit -= 1
            yield (a, b, c), common


def prepare(folder: Path, vad: Path):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity

    if diarization.sha(vad) != count_model.VAD_SHA:
        raise ValueError("Exact prior speech activity weights required")
    archives = {lang: folder / f"{lang}-pinned.parquet" for lang in LANGUAGE_QUOTAS}
    intake = {"revision": REVISION, "protocol": METHOD, "sources": {k: diarization.sha(p) for k, p in archives.items()},
              "vad_sha256": diarization.sha(vad), "implementation": diarization.sha(Path(__file__))}
    diarization.write(folder / "source-intake.json", intake)
    torch.set_num_threads(2)
    activity = SpeechActivity(vad)
    for directory in (folder / "pcm", folder / "stems"):
        directory.mkdir()
    rows, groups, coverage, used = [], [], {}, set()
    for lang, limit in LANGUAGE_QUOTAS.items():
        proposed, pool = speakers(archives[lang], activity)
        qualified = list(triplets(pool, limit, used))
        coverage[lang] = {"planned_speakers": proposed, "qualified_speakers": len(pool), "used_triplets": len(qualified)}
        for batch, common in qualified:
            idx = len(groups)
            source_info, source_waves = [], []
            for slot, item in enumerate(batch):
                samples, _, pitch, start, raw = item["clip"]
                source_info.append({"id": item["identity"], "utterance": item["name"], "encoded_sha256": item["encoded_sha256"],
                                    "pcm_sha256": digest(raw), "start_sample_at_16k": start, "median_pitch_semitones": pitch})
                (folder / "stems" / f"{idx:03}-{slot}.pcm").write_bytes(raw)
                source_waves.append(samples)
            groups.append({"group": idx, "language": lang, "joint_samples": common, "sources": source_info})
            for (rotation, delivery, case), payload in mix(source_waves).items():
                row_id = len(rows)
                (folder / "pcm" / f"{row_id:03}.pcm").write_bytes(payload)
                rows.append({"id": row_id, "group": idx, "language": lang, "role_rotation": rotation, "delivery": delivery,
                             "condition": case, "expected_voices": 2 if case == "two" else 3, "pcm_sha256": digest(payload)})
        print(f"{lang}: {proposed} speakers, {len(qualified)} disjoint qualifying triplets", flush=True)
    diarization.write(folder / "manifest.json", {"rows": rows, "groups": groups, "coverage": coverage, "source_intake_sha256": diarization.sha(folder / "source-intake.json")})


def model_plan(folder: Path, assets: Path):
    manifest = diarization.read(folder / "manifest.json")
    intake = diarization.read(folder / "source-intake.json")
    if (intake["protocol"] != METHOD or intake["implementation"] != diarization.sha(Path(__file__))
            or manifest["source_intake_sha256"] != diarization.sha(folder / "source-intake.json") or len(manifest["groups"]) < 8):
        raise ValueError("Expected a sufficiently sized frozen disjoint source panel")
    if {name: diarization.sha(assets / name) for name in count_model.ASSETS} != count_model.ASSETS:
        raise ValueError("Exact prior independently licensed count assets required")
    manifest_hash = diarization.sha(folder / "manifest.json")
    plan = {"schema": "chipmunk.hard-overlap-transfer-freeze/1", "method": METHOD, "revision": REVISION,
            "sources": intake["sources"], "manifest_sha256": manifest_hash, "planned_rows": len(manifest["rows"]),
            "implementation_sha256": intake["implementation"], "count_assets": count_model.ASSETS,
            "diarization_model_sha256": diarization.MODEL_SHA, "diarization_runtime_sha256": diarization.RUNTIME_ARCHIVE_SHA}
    return manifest, plan


def summarize(rows, saved, counts):
    tallies = defaultdict(Counter)
    leaked = defaultdict(set)
    kept = defaultdict(set)
    for row, dia, count in zip(rows, saved, counts, strict=True):
        if row["id"] != dia["id"] or row["id"] != count["id"] or row["pcm_sha256"] != count["pcm_sha256"]:
            raise ValueError("Mismatched frozen audio response")
        base = diarization.routes(dia["measurements"])["corroborated"]
        unscaled, normalized = (count_model.count(count["probabilities"][name]) for name in ("original", "minus24"))
        decisions = {"diarization": True, "raw_overlap_veto": unscaled < 3,
                     "normalized_overlap_veto": normalized < 3, "normalized_zero_or_overlap_veto": 0 < normalized < 3}
        for arm, allows in decisions.items():
            admitted = base == "single_interferer" and allows
            three = row["expected_voices"] == 3
            for name in ("all", f"{row['delivery']}:{row['condition']}", f"language:{row['language']}"):
                stats = tallies[arm, name]
                stats["planned_three" if three else "planned_two"] += 1
                stats["incorrect_three_routes" if three else "correct_two_routes"] += int(admitted)
                stats["recorded_diarization_overlap"] += int(base == "overlap_detected")
                stats["diarization_unassessed"] += int(base == "unassessed")
                stats["count_vetoes"] += int(base == "single_interferer" and not allows)
            if admitted:
                (leaked if three else kept)[arm].add(row["group"])
    candidate = "normalized_zero_or_overlap_veto"
    base = tallies["diarization", "all"]
    result = tallies[candidate, "all"]
    passed = (len(leaked["diarization"]) >= 5 and len(leaked[candidate]) <= 1
              and 2 * result["incorrect_three_routes"] <= base["incorrect_three_routes"]
              and base["correct_two_routes"] > 0 and 5 * result["correct_two_routes"] >= 4 * base["correct_two_routes"])
    return {"counts": {f"{arm}/{condition}": count for (arm, condition), count in tallies.items()},
            "groups_with_incorrect_three_route": {arm: len(leaked[arm]) for arm, _ in tallies},
            "groups_with_correct_two_route": {arm: len(kept[arm]) for arm, _ in tallies}, "advance": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "count", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.action == "prepare":
        return prepare(folder, args.assets)
    manifest, plan = model_plan(folder, args.assets)
    rows = manifest["rows"]
    path = folder / "overlap-freeze.json"
    if args.action == "freeze":
        if [row["id"] for row in rows] != list(range(len(rows))):
            raise ValueError("Expected ordered unique row IDs")
        for row in rows:
            diarization.pcm(folder, row)
        return diarization.write(path, plan)
    if diarization.read(path) != plan:
        raise ValueError("Exact frozen protocol and inputs required")
    if args.action == "count":
        model = count_model.CountNet(args.assets)
        model.torch.set_num_threads(2)
        recorded = []
        for row in rows:
            audio = count_model.decode(diarization.pcm(folder, row))
            normalized, _, _ = level_replay.level(audio, -24)
            probabilities = {name: model.score(count_model.padded(x, count_model.POSITIONS["center"]))
                             for name, x in (("original", audio), ("minus24", normalized))}
            recorded.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "probabilities": probabilities})
        return diarization.write(folder / "counts-private.json", {"freeze_sha256": diarization.sha(path), "rows": recorded})
    measured = diarization.read(folder / "measured.json")
    counts = diarization.read(folder / "counts-private.json")
    if (counts["freeze_sha256"] != diarization.sha(path) or measured["freeze_sha256"] != diarization.sha(folder / "freeze.json")
            or diarization.read(folder / "freeze.json")["manifest_sha256"] != plan["manifest_sha256"]):
        raise ValueError("Incompatible inference freeze receipts")
    aggregate = summarize(rows, measured["rows"], counts["rows"])
    diarization.write(folder / "aggregate.json", {"schema": "chipmunk.hard-overlap-transfer/1", **aggregate, "plan": plan,
                     "coverage": manifest["coverage"], "freeze_sha256": diarization.sha(path), "count_forwards": len(rows) * 2,
                     "diarization_forwards": measured["calls"], "count_receipt_sha256": diarization.sha(folder / "counts-private.json"),
                     "diarization_receipt_sha256": diarization.sha(folder / "measured.json")})


if __name__ == "__main__":
    main()
