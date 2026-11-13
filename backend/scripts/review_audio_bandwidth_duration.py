"""Replay a fixed-duration transfer of the upper-frequency evidence check."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

from scripts.audio_bandwidth_confirmation import VARIANTS, file_sha
from scripts.review_audio_formant_local_transfer import canonical, require, useful

SCHEMA = "chipmunk.audio-bandwidth-duration-transfer/1"
RATE = 16000
DURATIONS = (6, 8, 12)
CAPTURES = {
    "ES2010d": "27a6d3c29086ba8ed6ae5fd916af466c14603eeec133d713852417fc7249bb9e",
    "IS1007a": "5de35f96abc94296fe792830895dbc73727699f4c7fa05604a36eac06e099b63",
}
REPO = Path(__file__).resolve().parents[2]
MEASUREMENT = REPO / "backend/app/evidence/audio_bandwidth_confirmation.py"
BANDS = REPO / "backend/app/evidence/audio_bandwidth_return.py"
METHOD = {
    "sources": "Sixteen clips per retained AMI file, chosen by digest from 24-second grid positions in equal strata at least 24 seconds from the file edges. Reuse neither prior bandwidth meeting. Read channel one without mixing.",
    "duration": "Six and eight seconds are exact nested slices within twelve seconds. Start the six-second crop at +3 seconds and the eight at +2; preserve one recording and one intervention.",
    "controls": "Untouched, global low-pass, local gain, and a local ordinary cut from the next twelve seconds. Two local low-pass cutoffs are 3400 and 3800 Hz. Use 25 ms cosine edges and choose the exact edit from the unmodified twelve-second PCM hash.",
    "point": "Keep the same 16 dB point and compare every planned native sample range. No recentering or re-fit.",
    "advance": "For each added duration and cutoff, require at least as many useful locations as the paired six-second arm overall, at least one useful location from each source, and no more than one extra ordinary control flag overall relative to six seconds. Whole-recording low-pass must never flag. Useful means 0.30 IoU and half edit coverage.",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(paths):
    require(len(paths) == len(CAPTURES), "Supply both pinned parent WAVs once")
    by_capture = {path.name.split(".")[0]: path for path in paths}
    require(set(by_capture) == CAPTURES.keys(), "Expected the two published meeting names")
    rows = []
    width = 24 * RATE
    for capture, digest in CAPTURES.items():
        path = by_capture[capture]
        require(file_sha(path) == digest, "AMI source digest differs")
        with wave.open(str(path), "rb") as audio:
            channels, width_bytes, rate = audio.getparams()[:3]
            require(channels in (1, 2) and (width_bytes, rate) == (2, RATE), "Expected 16 kHz PCM16")
            extent = audio.getnframes()
            for ordinal in range(16):
                lo = max(width, extent * ordinal // 16)
                hi = min(extent - width, extent * (ordinal + 1) // 16)
                starts = range((lo + width - 1) // width * width, hi - width + 1, width)
                start = min(starts, key=lambda t: sha(f"{SCHEMA}/{capture}/{t}".encode()))
                audio.setpos(start)
                native = audio.readframes(width)
                data = native if channels == 1 else b"".join(native[i:i+2] for i in range(0, len(native), 4))
                require(len(data) == width * 2, "Native excerpt is incomplete")
                rows.append(({"capture": capture, "ordinal": ordinal, "parent_samples": extent,
                              "start": start, "pcm_sha256": sha(data)}, data))
    return {"schema": SCHEMA, "method": METHOD, "captures": CAPTURES, "measurement_sha256": file_sha(MEASUREMENT),
            "bands_sha256": file_sha(BANDS), "inputs": [item for item, _ in rows]}, rows


def edit_bounds(digest):
    start = 58000 + int(digest[:4], 16) % 57000
    return start, start + 12800 + int(digest[4:8], 16) % 3201


def scan(roster):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.evidence.audio_bandwidth_confirmation import locate

    rows = []
    for source, data in roster:
        recording, neighbor = np.frombuffer(data, dtype="<i2").astype(np.float64).reshape(2, -1) / 32768
        start, end = edit_bounds(source["pcm_sha256"])
        filter_3400, filter_3800 = (sosfiltfilt(butter(10, cutoff, fs=RATE, output="sos"), recording) for cutoff in (3400, 3800))
        edge = .5 - .5 * np.cos(np.linspace(0, np.pi, 400))
        blend = np.r_[edge, np.ones(end - start - 800), edge[::-1]]
        replacements = (recording, filter_3400, filter_3400, filter_3800, recording * .38, neighbor)
        for name, replacement in zip(VARIANTS, replacements, strict=True):
            delivery = recording.copy() if name.startswith("local_") else replacement
            if name.startswith("local_"):
                delivery[start:end] += (replacement[start:end] - delivery[start:end]) * blend
            pcm = np.clip(np.rint(delivery * 32768), -32768, 32767).astype("<i2")
            for seconds in DURATIONS:
                offset = (12 - seconds) * RATE // 2
                segment = pcm[offset:offset + seconds * RATE]
                count, best = locate(segment.astype(np.float64) / 32768)
                rows.append({"capture": source["capture"], "ordinal": source["ordinal"], "variant": name, "seconds": seconds,
                             "pcm_sha256": sha(segment.tobytes()), "assessed": count, "best": best})
    return rows


def summarize(lock, rows):
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["captures"] == CAPTURES,
            "Frozen source rule differs")
    keys = ("capture", "ordinal")
    sources = {tuple(item[k] for k in keys): item for item in lock["inputs"]}
    planned = {(capture, ordinal) for capture in CAPTURES for ordinal in range(16)}
    require(len(lock["inputs"]) == 32 and sources.keys() == planned, "Invalid capture roster")
    for (capture, ordinal), item in sources.items():
        width = 24 * RATE
        lo = max(width, item["parent_samples"] * ordinal // 16)
        hi = min(item["parent_samples"] - width, item["parent_samples"] * (ordinal + 1) // 16)
        starts = range((lo + width - 1) // width * width, hi - width + 1, width)
        require(item["start"] == min(starts, key=lambda t: sha(f"{SCHEMA}/{capture}/{t}".encode())), "Unexpected source timestamp")
    coverage = {(capture, ordinal, name, seconds) for capture, ordinal in planned for name in VARIANTS for seconds in DURATIONS}
    totals = {str(s): {name: Counter() for name in VARIANTS} for s in DURATIONS}
    captures = {(s, capture, name): 0 for s in DURATIONS for capture in CAPTURES for name in VARIANTS}
    for row in rows:
        capture, ordinal, name, seconds = (row[k] for k in (*keys, "variant", "seconds"))
        coverage.remove((capture, ordinal, name, seconds))
        counter = totals[str(seconds)][name]
        counter["planned"] += 1
        best = row["best"]
        max_ranges = sum(seconds * RATE // 1280 - 10 - n + 1 for n in (7, 9, 11, 13, 15))
        require(type(row["assessed"]) is int and 0 <= row["assessed"] <= max_ranges
                and (best is None) == (row["assessed"] == 0), "Incorrect assessment count")
        counter["unassessed"] += best is None
        flagged = bool(best and best["drop_db"] > 16)
        if best:
            a, b = best["range"]
            require(6400 <= a < b <= seconds * RATE - 6400 and a % 1280 == b % 1280 == 0
                    and (b - a) // 1280 in (7, 9, 11, 13, 15), "Measured interval differs")
            left, middle, right = best["relative_band_db"]
            drops = [min(l, r) - m for l, m, r in zip(left, middle, right, strict=True)]
            require(len(drops) == 3 and abs(drops[0]) <= 7.000002 and abs(min(drops[1:]) - best["drop_db"]) <= .000002,
                    "Measured bands differ")
        counter["flags"] += flagged
        if name.startswith("local_lowpass"):
            offset = (12 - seconds) * RATE // 2
            truth = [value - offset for value in edit_bounds(sources[capture, ordinal]["pcm_sha256"])]
            hit = flagged and useful(best["range"], truth)
            counter["useful"] += hit
            captures[seconds, capture, name] += hit
    require(not coverage, "Missing planned deliveries")
    ordinary = ("untouched", "local_gain", "local_natural_cut")
    base = totals["6"]
    positives = [name for name in VARIANTS if name.startswith("local_lowpass")]
    advanced = {}
    for seconds in DURATIONS[1:]:
        target = totals[str(seconds)]
        advanced[str(seconds)] = (all(target[name]["useful"] >= base[name]["useful"] for name in positives)
                                  and all(captures[seconds, source, name] > 0 for source in CAPTURES for name in positives)
                                  and sum(target[name]["flags"] - base[name]["flags"] for name in ordinary) <= 1
                                  and not target["global_lowpass"]["flags"])
    return {"totals": totals, "advances": advanced,
            "useful_by_capture": [[s, capture, name, value] for (s, capture, name), value in captures.items() if name in positives]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("sources", type=Path, nargs="*")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = json.loads(args.run.read_bytes())
        result = summarize(receipt["freeze"], receipt["rows"])
        require(receipt["summary"] == result, "Recorded totals differ")
    else:
        lock, roster = freeze(args.sources)
        frozen = args.run / "freeze.json"
        if args.phase == "freeze":
            args.run.mkdir(mode=0o700)
            with frozen.open("xb") as stream:
                stream.write(canonical(lock))
            return
        require(frozen.read_bytes() == canonical(lock), "Freeze differs")
        rows = scan(roster)
        result = summarize(lock, rows)
        with (args.run / "receipt.json").open("xb") as stream:
            stream.write(canonical({"freeze": lock, "rows": rows, "summary": result}))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
