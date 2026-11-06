"""Independently rebuild every frozen mixture from original production audio.

This does not import the variant generator or a neural model. The published
method/input lock remains unchanged. Outputs contain only digest references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes

ROLES = ("fit", "cal", "development", "confirmation")
CONDITIONS = (
    "clean",
    "noise_0",
    "noise_5",
    "noise_15",
    "music_5",
    "music_15",
    "speech_0",
    "speech_5",
    "speech_15",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def order(value):
    return hashlib.sha256(("chipmunk-audio-multisource-v8-20261106\0" + value).encode()).hexdigest()


def pcm_hash(values):
    return hashlib.sha256(np.asarray(values, "<f4").tobytes()).hexdigest()


def components(clean, intervals):
    clean = np.asarray(clean, np.float64)
    base = clean * (0.1 / float(np.sqrt(np.mean(clean**2))))
    result = {"clean": base}
    for kind in ("noise", "music", "speech"):
        bg = np.asarray(intervals[kind], np.float64)
        level = float(np.sqrt(np.mean(bg**2)))
        for snr in (5, 15) if kind == "music" else (0, 5, 15):
            result[f"{kind}_{snr}"] = base + bg * (0.1 / level / 10 ** (snr / 20))
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(value))) for value in result.values()))
    return {name: (result[name] * gain).astype(np.float32) for name in CONDITIONS}, gain


def audit(root, output):
    if output.exists():
        raise ValueError("Keep previous audit evidence")
    prepared_path = root / "prepared-sources.json"
    prepared = json.loads(prepared_path.read_text())
    variants, manifest_hashes = [], {}
    for name in ("development", "confirmation"):
        path = root / name / "variants.json"
        body = json.loads(path.read_text())
        if body["prepared_sources_sha256"] != digest(prepared_path):
            raise ValueError("Variant source catalogue changed")
        variants.extend(body["variants"])
        manifest_hashes[name] = digest(path)
    if len(variants) != 5364 or len({r["id"] for r in variants}) != 5364:
        raise ValueError("Every frozen variant required")
    cache, source_records = {}, []
    for row in prepared["sources"] + prepared["backgrounds"]:
        path = Path(row["source_path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Original source bytes changed")
        audio = decode_audio(path, source_sha256=row["sha256"])
        if (
            audio.mapping["pcm_sha256"] != row["production_pcm_sha256"]
            or len(audio.pcm) != row["full_samples"]
        ):
            raise ValueError("Original production decode changed")
        if row.get("prefix_pcm_sha256") and pcm_hash(audio.pcm[:64000]) != row["prefix_pcm_sha256"]:
            raise ValueError("Original four-second prefix changed")
        cache[row["sha256"]] = audio.pcm
        source_records.append(
            {
                "source_sha256": row["sha256"],
                "pcm_sha256": audio.mapping["pcm_sha256"],
                "samples": len(audio.pcm),
                "passed": True,
            }
        )
    pools = {
        (role, kind): sorted(
            [r for r in prepared["backgrounds"] if r["role"] == role and r["kind"] == kind],
            key=lambda r: order(r["id"]),
        )
        for role in ROLES
        for kind in ("noise", "music", "speech")
    }
    selected = {
        min(
            r["id"]
            for r in variants
            if r["role"] == role and r["condition"] == condition and r["label"] == label
        )
        for role in ROLES
        for condition in CONDITIONS
        for label in (0, 1)
    }
    by_source = defaultdict(list)
    for row in variants:
        by_source[row["base_id"]].append(row)
    counters, records = Counter(), []
    maximum_quantization, maximum_snr_error = 0.0, 0.0
    originals = sorted(prepared["sources"], key=lambda r: (ROLES.index(r["role"]), order(r["id"])))
    for index, source in enumerate(originals, 1):
        key = (source["role"], source["label"])
        ordinal = counters[key]
        counters[key] += 1
        intervals, backgrounds = {}, {}
        for kind in ("noise", "music", "speech"):
            pool = pools[source["role"], kind]
            bg = pool[ordinal % len(pool)]
            full = cache[bg["sha256"]]
            start = int(order(source["id"] + kind + ":offset")[:8], 16) % (len(full) - 64000 + 1)
            intervals[kind] = full[start : start + 64000]
            backgrounds[kind] = (bg, start)
        expected, gain = components(cache[source["sha256"]][:64000], intervals)
        rows = by_source[source["id"]]
        if [r["condition"] for r in rows] != list(CONDITIONS):
            raise ValueError("Missing/reordered condition")
        clean_decoded = None
        for row in rows:
            if (
                any(row[k] != source[k] for k in ("role", "group", "label", "generator", "dataset"))
                or row["source_sha256"] != source["sha256"]
            ):
                raise ValueError("Source/role/truth binding changed")
            pcm = expected[row["condition"]]
            if (
                not math.isclose(row["common_peak_gain"], gain, abs_tol=1e-12, rel_tol=1e-12)
                or pcm_hash(pcm) != row["payload_pcm_sha256"]
            ):
                raise ValueError("Independent mixture/common gain differs")
            path = root / row["path"]
            raw = path.read_bytes()
            if raw != wav_bytes(pcm) or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Canonical sample/file payload differs")
            decoded = read_wav(raw)
            if len(decoded) != 64000 or pcm_hash(decoded) != row["pcm_sha256"]:
                raise ValueError("Stored WAV PCM changed")
            maximum_quantization = max(maximum_quantization, float(np.max(np.abs(decoded - pcm))))
            if row["condition"] == "clean":
                clean_decoded = decoded
            else:
                kind = row["background_kind"]
                bg, start = backgrounds[kind]
                if (
                    row["background_sha256"] != bg["sha256"]
                    or row["background_role"] != source["role"]
                    or row["background_start_sample"] != start
                    or row["background_interval_pcm_sha256"] != pcm_hash(intervals[kind])
                ):
                    raise ValueError("Deterministic original background interval differs")
                snr = float(
                    20
                    * np.log10(
                        np.sqrt(np.mean(clean_decoded.astype(float) ** 2))
                        / np.sqrt(np.mean((decoded.astype(float) - clean_decoded) ** 2))
                    )
                )
                if not math.isclose(snr, row["measured_snr_db"], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError("Recorded measured SNR changed")
                maximum_snr_error = max(maximum_snr_error, abs(snr - row["snr_db"]))
            native = row["id"] in selected
            if native:
                actual = decode_audio(path, source_sha256=row["sha256"])
                if (
                    not np.array_equal(actual.pcm, decoded)
                    or actual.mapping["pcm_sha256"] != row["pcm_sha256"]
                ):
                    raise ValueError("Preselected native production decode differs")
            records.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "passed": True,
                    "preselected_native_production_decode": native,
                }
            )
        if index % 60 == 0 or index == len(originals):
            print(json.dumps({"original_foregrounds_rebuilt": index, "planned": len(originals)}), flush=True)
    result = {
        "schema": "chipmunk.audio-v8-complete-input-audit/1",
        "passed": True,
        "code_sha256": digest(Path(__file__)),
        "prepared_sources_sha256": digest(prepared_path),
        "variants_sha256": manifest_hashes,
        "originals": source_records,
        "cases": records,
        "native_decode_cases": len(selected),
        "maximum_pcm16_quantization_error": maximum_quantization,
        "maximum_measured_snr_error_db": maximum_snr_error,
        "detector_inference_run": False,
        "scope": "Local actual source decoding and independent complete recipe/file/PCM reconstruction. Native decoder selection fixed by role/class/condition before response inspection. Not learned-model or outside validation.",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "originals": len(source_records),
                "inputs": len(records),
                "native_decode_cases": len(selected),
                "audit_sha256": digest(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
