"""Rebuild public audio-review inputs offline from separately acquired source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from app.detectors.audio.decode import decode_audio
from scripts.audio_quality_v5 import digest
from scripts.evaluate_audio_candidates import decode as background_decode
from scripts.export_audio_development_panel import wav_metadata

CONDITIONS = {"clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15"}


def restore_container_timestamp(path, timestamp):
    blob = bytearray(path.read_bytes())
    before = wav_metadata(blob)
    offset = 12
    while offset + 8 <= len(blob):
        kind, size = struct.unpack_from("<4sI", blob, offset)
        if kind == b"PEAK":
            struct.pack_into("<I", blob, offset + 12, timestamp)
            break
        offset += 8 + size + size % 2
    after = wav_metadata(blob)
    if (
        after["container_peak_timestamp"] != timestamp
        or before["float_payload_sha256"] != after["float_payload_sha256"]
    ):
        raise ValueError("Container timestamp restoration altered the sample payload")
    path.write_bytes(blob)


def rebuild(catalogue_path, catalogue_sha256, source_dir, output):
    if digest(catalogue_path) != catalogue_sha256:
        raise ValueError("Public catalogue differs from its expected hash")
    m = json.loads(catalogue_path.read_text())
    if m["schema"] != "chipmunk.public-audio-development-panel/2":
        raise ValueError("Unsupported public catalogue")
    if (
        output.exists()
        or source_dir.resolve() == output.resolve()
        or source_dir.resolve() in output.resolve().parents
    ):
        raise ValueError("Use a new output directory outside the source cache")
    originals = {r["id"]: r for r in m["sources"]}
    backgrounds = {r["sha256"]: r for r in m["backgrounds"]}
    groups = defaultdict(list)
    for row in m["inputs"]:
        if not re.fullmatch(r"[a-f0-9]{64}", row["id"]):
            raise ValueError("Input identifiers must be SHA-256 values")
        groups[row["base_id"]].append(row)
    if len({r["id"] for r in m["inputs"]}) != len(m["inputs"]):
        raise ValueError("Duplicate planned input")
    if any(len(v) != 6 or {r["condition"] for r in v} != CONDITIONS for v in groups.values()):
        raise ValueError("Every original needs all six declared conditions")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    receipts, failures, decoded_bg = {}, {}, {}
    total = len(m["inputs"])
    for source_id, variants in groups.items():
        started = time.monotonic()
        try:
            original = originals[source_id]
            source = source_dir / (original["sha256"] + ".audio")
            clean = decode_audio(source, source_sha256=original["sha256"])
            if clean.mapping["pcm_sha256"] != original["production_pcm_sha256"]:
                raise ValueError("Original production decoding differs from its public hash")
            n = variants[0]["samples"]
            if any(
                r["samples"] != n or r["sample_rate"] != 16000 or r["source_interval_samples"] != [0, n]
                for r in variants
            ):
                raise ValueError("Unsupported or inconsistent original sample coordinates")
            values = clean.pcm[:n].astype(np.float64)
            if len(values) != n or np.sqrt(np.mean(values**2)) < 1e-4:
                raise ValueError("The original speech interval is unavailable or silent")
            if (
                hashlib.sha256(clean.pcm[:n].astype("<f4").tobytes()).hexdigest()
                != original["source_prefix_pcm_sha256"]
            ):
                raise ValueError("The original speech prefix differs from its public hash")
            values *= 0.1 / np.sqrt(np.mean(values**2))
            mixtures = {}
            for row in variants:
                if row["condition"] == "clean":
                    mixtures[row["id"]] = values
                    continue
                sha = row["background_sha256"]
                bg = backgrounds[sha]
                if bg["source_id"] != row["background_id"]:
                    raise ValueError("Background identity mismatch")
                if sha not in decoded_bg:
                    p = source_dir / (sha + ".audio")
                    if digest(p) != sha:
                        raise ValueError("Background source bytes changed")
                    raw = background_decode(p)
                    if hashlib.sha256(raw.astype("<f4").tobytes()).hexdigest() != bg["decoded_pcm_sha256"]:
                        raise ValueError("Background decoding differs from its public hash")
                    decoded_bg[sha] = raw.astype(np.float64)
                start = row["background_start_sample"]
                segment = decoded_bg[sha][start : start + n]
                if len(segment) != n or np.sqrt(np.mean(segment**2)) < 1e-5:
                    raise ValueError("The declared background interval is unavailable or silent")
                mixtures[row["id"]] = values + segment * (
                    0.1 / np.sqrt(np.mean(segment**2)) / 10 ** (row["snr_db"] / 20)
                )
            gain = min(1.0, 0.98 / max(np.max(np.abs(x)) for x in mixtures.values()))
            if any(abs(row["common_peak_gain"] - gain) > 1e-12 for row in variants):
                raise ValueError("The shared clipping-safe gain differs from its public value")
            for row in variants:
                p = output / "inputs" / (row["id"] + ".wav")
                sf.write(p, (mixtures[row["id"]] * gain).astype(np.float32), 16000, subtype="FLOAT")
                if wav_metadata(p.read_bytes())["float_payload_sha256"] != row["float_payload_sha256"]:
                    raise ValueError(
                        "Independently recomputed float samples differ from the public payload hash"
                    )
                restore_container_timestamp(p, row["container_peak_timestamp"])
                sha = digest(p)
                decoded = decode_audio(p, source_sha256=sha)
                if sha != row["sha256"] or decoded.mapping["pcm_sha256"] != row["pcm_sha256"]:
                    raise ValueError("Recreated file or production PCM differs from its public hash")
                receipts[row["id"]] = {
                    "id": row["id"],
                    "status": "ok",
                    "input_sha256": sha,
                    "pcm_sha256": decoded.mapping["pcm_sha256"],
                    "samples": len(decoded.pcm),
                }
        except Exception as error:
            failures[source_id] = f"{type(error).__name__}: {error}"
            for row in variants:
                receipts.setdefault(
                    row["id"], {"id": row["id"], "status": "failed", "reason": failures[source_id]}
                )
        print(
            json.dumps(
                {
                    "completed": len(receipts),
                    "planned": total,
                    "source_seconds": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    result = {
        "schema": "chipmunk.audio-public-reconstruction/1",
        "catalogue_sha256": catalogue_sha256,
        "purpose": "Offline local reproduction of consumed inputs; no detection or independent accuracy claim",
        "planned_inputs": total,
        "source_groups": len(groups),
        "backgrounds": len(backgrounds),
        "implementation_sha256": digest(Path(__file__)),
        "status": "passed" if not failures else "failed",
        "source_failures": failures,
        "receipts": [receipts[r["id"]] for r in m["inputs"]],
    }
    with (output / "reconstruction.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("receipts", "source_failures")}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalogue", type=Path)
    p.add_argument(
        "source_dir", type=Path, help="Separate acquired originals/backgrounds, named <sha256>.audio"
    )
    p.add_argument("output", type=Path, help="New directory; refuses overwrite")
    p.add_argument("--catalogue-sha256", required=True)
    a = p.parse_args()
    rebuild(a.catalogue, a.catalogue_sha256, a.source_dir, a.output)
