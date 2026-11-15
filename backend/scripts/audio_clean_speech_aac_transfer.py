# SPDX-License-Identifier: Apache-2.0
"""Compare a clean earlier WAV against PCM and AAC deliveries under competing speech."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
SCHEMA = "chipmunk.audio-clean-speech-aac-transfer/1"
SCHEMA48 = "chipmunk.audio-clean-speech-aac48-transfer/1"
SOURCE_PANELS = {"icsi": "audio-short-edit-detector-training-2026-11-14.json",
                 "ami": "audio-random-clock-stock-2026-11-13.json"}
RATES = (32, 64)
SAMPLES = 96_000
METHOD = {"condition": "The earlier file is the exact uninterrupted dry original. Submitted audio has competing speech 5 dB louder.",
          "comparison": "Use the frozen dense speech measurement and frequency-balance guard. Compare PCM with native 16 kHz mono AAC-LC at 32 and 64 kbps.",
          "location": "Cover half of the known interval and achieve at least .30 temporal intersection over union.",
          "controls": "Count genuine splices separately from ordinary processing, isolated background and an unrelated earlier reader."}
CARRIERS48 = {"aac48_mono": (64, 1), "aac48_stereo": (96, 2)}
METHOD48 = METHOD | {"comparison": "Replay the frozen dense speech measurement on 48 kHz AAC-LC at 64 kbps mono or 96 kbps stereo. "
    "The stereo right channel uses a 0.55 scaled copy plus a 0.20 copy delayed by 1.5 ms. The left channel is the original mixture."}


def read(path):
    return json.loads(path.read_bytes())


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hash(bank):
    report = read(PANELS / SOURCE_PANELS[bank])
    return report["incumbent"]["source_sha256"] if bank == "icsi" else report["native_freeze_sha256"]


def icsi_cases(folder):
    from scripts import audio_short_edit_detector_training as retained

    plan, digest = retained.load_plan(folder)
    require(digest == source_hash("icsi"), "The ICSI source plan differs")
    placements = {row["group"]: row for row in plan["sources"] if row["role"] == "held"}
    selected = [row for row in plan["rows"] if row["role"] == "held"]

    def pcm(row):
        key = row["pcm_sha256"]
        raw = (folder / "pcm" / f"{key}.pcm").read_bytes()
        require(sha(raw) == key and len(raw) == 2 * SAMPLES, "ICSI input PCM differs")
        return raw

    earlier = {row["group"]: pcm(row) for row in selected if row["kind"] == "untouched" and row["condition"] == "dry"}
    for row in selected:
        if row["condition"] != "competing_speech":
            continue
        group, kind = row["group"], row["kind"]
        submitted = pcm(row)
        truth = placements[group]["truth"] if kind == "genuine_splice" else row["truth"]
        yield group, kind, truth, earlier[group], submitted
        if kind == "untouched":
            other = next(key for key in earlier if placements[key]["reader_sha256"] != placements[group]["reader_sha256"])
            yield group, "unrelated", None, earlier[other], submitted


def ami_cases(folder):
    import numpy as np

    frozen = folder / "placements-freeze.json"
    require(sha(frozen.read_bytes()) == source_hash("ami"), "The AMI source plan differs")
    rows = read(frozen)["native_manifest_rows"]
    with np.load(folder / "placements-native.npz", allow_pickle=False) as storage:
        lane = [(row, a.tobytes(), b.tobytes()) for row, a, b in zip(rows, storage["dry"], storage["native"], strict=True)
                if row["snr"] == -5]
    originals = {}
    for row, dry, noisy in lane:
        require(sha(dry) == row["dry_pcm_sha256"] and sha(noisy) == row["pcm_sha256"]
                and len(noisy) == len(dry) == 2 * SAMPLES, "AMI input PCM differs")
        if row["kind"] == "untouched":
            originals[row["group"]] = dry
    for row, _, noisy in lane:
        i, kind = row["group"], row["kind"]
        yield i, kind, row["edit_clock"] if kind in {"piper", "kitten"} else row["ordinary_splice_clock"], originals[i], noisy
        if kind == "untouched":
            yield i, "unrelated", None, originals[(i + 1) % len(originals)], noisy


def encode(pcm, path, rate):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "16000",
                    "-ac", "1", "-i", "pipe:0", "-map_metadata", "-1", "-threads", "1", "-c:a", "aac",
                    "-profile:a", "aac_low", "-b:a", f"{rate}k", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(path)],
                   input=pcm, check=True, stdout=subprocess.DEVNULL, timeout=25)


def encode48(pcm, path, carrier):
    import numpy as np

    rate, channels = CARRIERS48[carrier]
    if channels == 2:
        left = np.frombuffer(pcm, dtype="<i2")
        delayed = np.pad(left[:-24], (24, 0))
        right = np.rint(0.55 * left + 0.20 * delayed).astype("<i2")
        pcm = np.column_stack((left, right)).astype("<i2", copy=False).tobytes()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", str(channels),
                    "-i", "pipe:0", "-ar", "48000", "-map_metadata", "-1", "-threads", "1", "-c:a", "aac",
                    "-profile:a", "aac_low", "-b:a", f"{rate}k", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(path)],
                   input=pcm, check=True, stdout=subprocess.DEVNULL, timeout=25)


def compact(result):
    return {"status": result["status"], "reason": result["reason"], "admitted": result["match_support"]["admitted"],
            "range": result["range"]["samples"] if "range" in result else None,
            "presentation": result.get("aac_presentation")}


def run(bank, folder, *, aac48=False):
    import wave

    from app.evidence.audio_reference import SCHEMA_WITH_AAC_CLEAN_SPEECH, compare

    rows = []
    source = icsi_cases if bank == "icsi" else ami_cases
    with tempfile.TemporaryDirectory(prefix="chipmunk-reference-aac-") as directory:
        earlier, pcm, aac = (Path(directory) / name for name in ("earlier.wav", "submitted.wav", "submitted.m4a"))
        for group, kind, truth, dry, noisy in source(folder):
            for path, raw in ((earlier, dry), (pcm, noisy)):
                with wave.open(str(path), "wb") as file:
                    file.setparams((1, 2, 16000, 0, "NONE", ""))
                    file.writeframes(raw)
            options = dict(include_clean_speech=True, use_dense_clean_speech=True, use_clean_speech_eq_guard=True)
            lossless, _ = compare(earlier, pcm, **options)
            results = {"pcm": compact(lossless["clean_speech"])}
            encodes = {}
            for key in (CARRIERS48 if aac48 else (f"aac{rate}" for rate in RATES)):
                if aac48:
                    encode48(noisy, aac, key)
                else:
                    encode(noisy, aac, int(key[3:]))
                measured, _ = compare(earlier, aac, **options, include_clean_speech_aac=True)
                require(measured["schema"] == SCHEMA_WITH_AAC_CLEAN_SPEECH, "AAC presentation was not verified")
                results[key] = compact(measured["clean_speech"])
                encodes[key] = sha(aac.read_bytes())
            rows.append({"group": group, "kind": kind, "truth": truth, "earlier_pcm_sha256": sha(dry),
                         "submitted_pcm_sha256": sha(noisy), "encoded_sha256": encodes, "outcomes": results})
    receipt = {"schema": SCHEMA48 if aac48 else SCHEMA, "bank": bank, "method": METHOD48 if aac48 else METHOD,
               "source_sha256": source_hash(bank), "rows": rows}
    receipt["summary"] = review(receipt)
    return receipt


def location(span, truth):
    if not span or not truth:
        return False
    start, stop = span
    a, b = truth
    intersection = max(0, min(b, stop) - max(a, start))
    union = max(b, stop) - min(a, start)
    return 2 * intersection >= b - a and 10 * intersection >= 3 * union


def review(receipt):
    bank = receipt["bank"]
    high = receipt["schema"] == SCHEMA48
    require(receipt["schema"] in (SCHEMA, SCHEMA48) and receipt["method"] == (METHOD48 if high else METHOD)
            and receipt["source_sha256"] == source_hash(bank), "Different source freeze or method")
    group_count, case_count = (24, 8) if bank == "icsi" else (8, 6)
    rows = receipt["rows"]
    require(len(rows) == group_count * case_count and len({(r["group"], r["kind"]) for r in rows}) == len(rows)
            and Counter(Counter(r["kind"] for r in rows).values()) == {group_count: case_count},
            "The frozen case roster is incomplete")
    report = {}
    for row in rows:
        kind = row["kind"]
        synthetic = kind.startswith("edited/") or kind in {"piper", "kitten"}
        splice = kind == "genuine_splice" or kind.endswith("reader_splice")
        bucket = kind if synthetic or kind in {"unrelated", "background"} else "genuine_splice" if splice else "ordinary_processing"
        if row["truth"] is not None:
            a, b = row["truth"]
            require(0 <= a < b <= SAMPLES, "The known range exceeds the source")
        require(not (synthetic or splice) or row["truth"] is not None, "A replacement lacks its known range")
        carriers = set(CARRIERS48) if high else {f"aac{rate}" for rate in RATES}
        require(set(row["outcomes"]) == {"pcm", *carriers}, "A carrier is missing")
        for carrier, result in row["outcomes"].items():
            span = result["range"]
            proof = result["presentation"]
            if carrier != "pcm":
                require(proof["side"] == "submitted" and proof["presentation_end_exclusive_pts"] == SAMPLES
                        and proof["presentation_start_pts"] == 0 and proof["sample_rate"] == 16000
                        and proof["time_base"] == "1/16000" and 0 <= proof["decoded_samples"] - SAMPLES <= (342 if high else 1023)
                        and proof["excluded_trailing_samples"] == proof["decoded_samples"] - SAMPLES, "AAC presentation differs")
                if high:
                    require(proof.get("native_clock") == {"sample_rate": 48_000, "channels": CARRIERS48[carrier][1],
                        "time_base": "1/48000", "presentation_end_exclusive_pts": SAMPLES * 3}, "Native AAC clock differs")
                else:
                    require("native_clock" not in proof, "Unexpected native clock")
            else:
                require(proof is None, "Unexpected AAC presentation reading")
            status = result["status"]
            require((span is not None) == (status == "localized_difference")
                    and (not span or 0 <= span[0] < span[1] <= SAMPLES), "A proposed interval exceeds the source")
            key = report.setdefault(carrier, {}).setdefault(bucket, Counter())
            key.update(files=1, admitted=int(result["admitted"]), flags=int(span is not None),
                       useful=int(location(span, row["truth"]) and (synthetic or splice)),
                       unassessed=int(status == "unassessed"), broad=int(status == "broad_processing_difference"))
    counts = {key: {bucket: dict(value) for bucket, value in cells.items()} for key, cells in report.items()}
    require("summary" not in receipt or receipt["summary"] == counts, "Summary does not reproduce")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("icsi", "ami", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--aac48", action="store_true")
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(read(args.receipt)), sort_keys=True, indent=2))
    else:
        if args.sources is None:
            parser.error("Native scoring requires --sources")
        result = run(args.phase, args.sources, aac48=args.aac48)
        args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
