"""Bind all four-second source/noise/music/speech variants before fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from scripts.audio_confirmation_v7 import CONDITIONS
from scripts.plan_audio_multisource_v8 import digest, stable
from scripts.prepare_audio_confirmation_variants_v7 import mixtures
from scripts.prepare_audio_guard_v7 import write
from scripts.prepare_audio_multisource_v8 import ROLES, pcm_hash


def background_pools(backgrounds):
    pools = {}
    for role in ROLES:
        for kind, expected in (("noise", 8), ("music", 4), ("speech", 6 if role == "confirmation" else 4)):
            rows = sorted(
                [r for r in backgrounds if r["role"] == role and r["kind"] == kind],
                key=lambda r: stable(r["id"]),
            )
            if len(rows) != expected:
                raise ValueError("Every role requires its complete fixed background pool")
            pools[role, kind] = rows
    if len({r["sha256"] for r in backgrounds}) != len(backgrounds):
        raise ValueError("Background source bytes repeated across pools")
    return pools


def prepare(root):
    prepared_path = root / "prepared-sources.json"
    prepared = json.loads(prepared_path.read_text())
    if prepared["preparation_code_sha256"] != digest(
        Path(__file__).with_name("prepare_audio_multisource_v8.py")
    ):
        raise ValueError("Source preparation implementation changed")
    outputs = {name: root / name / "variants.json" for name in ("development", "confirmation")}
    if any(p.exists() for p in outputs.values()):
        raise ValueError("Preserve existing exact variants")
    pools, cache = background_pools(prepared["backgrounds"]), {}
    for row in prepared["backgrounds"]:
        path = Path(row["source_path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Fixed background bytes changed")
        audio = decode_audio(path, source_sha256=row["sha256"])
        if audio.mapping["pcm_sha256"] != row["production_pcm_sha256"]:
            raise ValueError("Fixed background decoding changed")
        cache[row["sha256"]] = audio.pcm
    counters, jobs = Counter(), []
    sources = sorted(prepared["sources"], key=lambda r: (ROLES.index(r["role"]), stable(r["id"])))
    for row in sources:
        key = (row["role"], row["label"])
        jobs.append((row, counters[key]))
        counters[key] += 1
    (root / "inputs").mkdir(exist_ok=True)

    def one(job):
        source, ordinal = job
        path = Path(source["source_path"])
        if digest(path) != source["sha256"]:
            raise ValueError("Fixed foreground bytes changed")
        audio = decode_audio(path, source_sha256=source["sha256"])
        clean = audio.pcm[:64000]
        if (
            audio.mapping["pcm_sha256"] != source["production_pcm_sha256"]
            or pcm_hash(clean) != source["prefix_pcm_sha256"]
        ):
            raise ValueError("Fixed foreground decoding/prefix changed")
        selected, views = {}, {}
        for kind in ("noise", "music", "speech"):
            pool = pools[source["role"], kind]
            bg = pool[ordinal % len(pool)]
            full = cache[bg["sha256"]]
            start = int(stable(source["id"] + kind + ":offset")[:8], 16) % (len(full) - 64000 + 1)
            selected[kind] = (bg, start)
            views[kind] = full[start : start + 64000]
        variants, gain = mixtures(clean, views)
        rows, decoded_clean = [], None
        for condition, pcm in variants.items():
            raw = wav_bytes(pcm)
            sha = hashlib.sha256(raw).hexdigest()
            target = root / "inputs" / (sha + ".wav")
            if target.exists():
                if digest(target) != sha:
                    raise ValueError("Existing exact variant changed")
            else:
                with target.open("xb") as stream:
                    stream.write(raw)
            decoded = decode_audio(target, source_sha256=sha)
            if len(decoded.pcm) != 64000:
                raise ValueError("Variant sample clock changed")
            if condition == "clean":
                decoded_clean = decoded.pcm
            kind = condition.split("_")[0]
            bg, start = selected[kind] if kind != "clean" else (None, None)
            measured = (
                None
                if bg is None
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
                    "id": stable(source["id"] + ":" + condition),
                    "base_id": source["id"],
                    "group": source["group"],
                    "role": source["role"],
                    "label": source["label"],
                    "generator": source["generator"],
                    "dataset": source["dataset"],
                    "condition": condition,
                    "source_sha256": source["sha256"],
                    "source_prefix_pcm_sha256": source["prefix_pcm_sha256"],
                    "source_interval_samples": [0, 64000],
                    "path": str(target.relative_to(root)),
                    "sha256": sha,
                    "payload_pcm_sha256": pcm_hash(pcm),
                    "pcm_sha256": decoded.mapping["pcm_sha256"],
                    "production_input": decoded.mapping,
                    "samples": 64000,
                    "sample_rate": 16000,
                    "seconds": 4.0,
                    "common_peak_gain": gain,
                    "background_id": bg["source_id"] if bg else None,
                    "background_role": bg["role"] if bg else None,
                    "background_kind": kind if bg else None,
                    "background_sha256": bg["sha256"] if bg else None,
                    "background_start_sample": start,
                    "background_interval_pcm_sha256": pcm_hash(views[kind]) if bg else None,
                    "snr_db": int(condition.rsplit("_", 1)[1]) if bg else None,
                    "measured_snr_db": measured,
                }
            )
        return rows

    variants = []
    with ThreadPoolExecutor(4) as pool:
        for index, rows in enumerate(pool.map(one, jobs), 1):
            variants.extend(rows)
            if index % 50 == 0 or index == len(jobs):
                print(json.dumps({"foregrounds_prepared": index, "planned": len(jobs)}), flush=True)
    if len(variants) != 5364 or len({r["id"] for r in variants}) != 5364:
        raise ValueError("Every reserved source/condition required")
    if Counter(r["role"] for r in variants) != {
        "fit": 1971,
        "cal": 891,
        "development": 1422,
        "confirmation": 1080,
    }:
        raise ValueError("Frozen role counts changed")
    for name, output in outputs.items():
        selected = [r for r in variants if (r["role"] == "confirmation") == (name == "confirmation")]
        output.parent.mkdir(exist_ok=True)
        body = {
            "schema": "chipmunk.audio-multisource-exact-variants/8",
            "panel": name,
            "prepared_sources_sha256": digest(prepared_path),
            "preparation_code_sha256": digest(Path(__file__)),
            "mixture_code_sha256": digest(
                Path(__file__).with_name("prepare_audio_confirmation_variants_v7.py")
            ),
            "variants": selected,
            "source_root": str(root.resolve()),
            "conditions": list(CONDITIONS),
            "detector_inference_run": False,
            "deployment_eligible": False,
        }
        write(output, body)
        print(
            json.dumps(
                {
                    "panel": name,
                    "variants_sha256": digest(output),
                    "inputs": len(selected),
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
