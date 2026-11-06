"""Prepare all1080 frozen foreground/background mixtures, without neural inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from scripts.audio_confirmation_v7 import CONDITIONS, PREPARED_SHA256
from scripts.prepare_audio_confirmation_v7 import digest, stable


def mixtures(clean, background_views):
    values = np.asarray(clean, np.float64)
    if values.shape != (64000,) or not np.isfinite(values).all():
        raise ValueError("Exactly four seconds of finite foreground required")
    rms = float(np.sqrt(np.mean(values**2)))
    if rms < 1e-4:
        raise ValueError("Silent fixed foreground; preserve selection")
    values = values * (0.1 / rms)
    result = {"clean": values}
    for kind in ("noise", "music", "speech"):
        bg = np.asarray(background_views[kind], np.float64)
        level = float(np.sqrt(np.mean(bg**2)))
        if bg.shape != values.shape or not np.isfinite(bg).all() or level < 1e-5:
            raise ValueError("Silent/invalid fixed background interval; do not select a different offset")
        for snr in (5, 15) if kind == "music" else (0, 5, 15):
            result[f"{kind}_{snr}"] = values + bg * (0.1 / level / 10 ** (snr / 20))
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(v))) for v in result.values()))
    return {name: (result[name] * gain).astype(np.float32) for name in CONDITIONS}, gain


def prepare(root):
    path = root / "prepared-sources.json"
    if digest(path) != PREPARED_SHA256:
        raise ValueError("Prepared source selection changed")
    if (root / "variants.json").exists():
        raise ValueError("Keep existing exact variants")
    body = json.loads(path.read_text())
    cache = {}
    for row in body["sources"] + body["backgrounds"]:
        source = root / row["path"]
        if digest(source) != row["sha256"]:
            raise ValueError("Original source bytes changed")
        audio = decode_audio(source, source_sha256=row["sha256"])
        if row.get("production_pcm_sha256") and audio.mapping["pcm_sha256"] != row["production_pcm_sha256"]:
            raise ValueError("Production decoding changed")
        cache[row["sha256"]] = audio.pcm
    pools = {k: [r for r in body["backgrounds"] if r["kind"] == k] for k in ("noise", "music")}
    pools["speech"] = [
        {**r, "kind": "speech", "source_id": r["id"]} for r in body["sources"] if r["role"] == "background"
    ]
    for pool in pools.values():
        pool.sort(key=lambda r: stable(r["source_id"]))
    foreground = sorted(
        [r for r in body["sources"] if r["role"] == "foreground"], key=lambda r: stable(r["id"])
    )
    counters, jobs = Counter(), []
    for row in foreground:
        jobs.append((row, counters[row["label"]]))
        counters[row["label"]] += 1
    (root / "inputs").mkdir(exist_ok=True)

    def one(job):
        source, ordinal = job
        clean = cache[source["sha256"]][:64000]
        selected, views = {}, {}
        for kind, pool in pools.items():
            bg = pool[ordinal % len(pool)]
            full = cache[bg["sha256"]]
            if len(full) < 64000:
                raise ValueError("Fixed background shorter than four seconds")
            start = int(stable(source["id"] + kind + ":offset")[:8], 16) % (len(full) - 64000 + 1)
            selected[kind] = (bg, start)
            views[kind] = full[start : start + 64000]
        variants, gain = mixtures(clean, views)
        rows, decoded_clean = [], None
        for condition, pcm in variants.items():
            payload_sha = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
            raw = wav_bytes(pcm)
            target = root / "inputs" / (hashlib.sha256(raw).hexdigest() + ".wav")
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError("Existing generated waveform differs")
            else:
                with target.open("xb") as stream:
                    stream.write(raw)
            sha = digest(target)
            decoded = decode_audio(target, source_sha256=sha)
            if len(decoded.pcm) != 64000:
                raise ValueError("Variant production sample clock changed")
            if condition == "clean":
                decoded_clean = decoded.pcm
            kind = condition.split("_")[0]
            background, start = selected[kind] if kind != "clean" else (None, None)
            measured = (
                None
                if background is None
                else float(
                    20
                    * np.log10(
                        np.sqrt(np.mean(decoded_clean.astype(float) ** 2))
                        / np.sqrt(np.mean((decoded.pcm.astype(float) - decoded_clean) ** 2))
                    )
                )
            )
            rows.append(
                {
                    "id": stable(source["id"] + condition),
                    "base_id": source["id"],
                    "group": body["assignments"][source["id"]],
                    "label": source["label"],
                    "generator": source["generator"],
                    "dataset": source["dataset"],
                    "condition": condition,
                    "source_sha256": source["sha256"],
                    "source_prefix_pcm_sha256": source["source_prefix_pcm_sha256"],
                    "source_interval_samples": [0, 64000],
                    "path": str(target.relative_to(root)),
                    "sha256": sha,
                    "payload_pcm_sha256": payload_sha,
                    "pcm_sha256": decoded.mapping["pcm_sha256"],
                    "production_input": decoded.mapping,
                    "samples": 64000,
                    "sample_rate": 16000,
                    "seconds": 4.0,
                    "common_peak_gain": gain,
                    "background_id": background["source_id"] if background else None,
                    "background_kind": background["kind"] if background else None,
                    "background_sha256": background["sha256"] if background else None,
                    "background_start_sample": start,
                    "background_interval_pcm_sha256": hashlib.sha256(
                        views[kind].astype("<f4").tobytes()
                    ).hexdigest()
                    if background
                    else None,
                    "snr_db": int(condition.rsplit("_", 1)[1]) if background else None,
                    "measured_snr_db": measured,
                }
            )
        return rows

    with ThreadPoolExecutor(4) as pool:
        variants = [row for rows in pool.map(one, jobs) for row in rows]
    if len(variants) != 1080 or len({r["id"] for r in variants}) != 1080:
        raise ValueError("Every planned source/condition must be present")
    result = {
        "schema": "chipmunk.audio-fresh-exact-variants/7",
        "prepared_sources_sha256": PREPARED_SHA256,
        "preparation_code_sha256": digest(Path(__file__)),
        "variants": variants,
        "source_root": str(root.resolve()),
        "detector_inference_run": False,
        "deployment_eligible": False,
    }
    out = root / "variants.json"
    with out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "variants_sha256": digest(out),
                "inputs": len(variants),
                "source_groups": len(body["groups"]),
                "detector_inference_run": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.root)
