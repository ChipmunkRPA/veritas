"""Compare acoustic periodicity measures on fixed mixed-sound excerpts."""
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "docs/evaluation-panels/audio-background-persistence-2026-11-13.json"
METHOD = {
    "source": "All 116 deliveries in the retained noise-confirmation ledger. Use exact samples 20800–27200 (1.300–1.700 s).",
    "comparison": "Raw decoded waveform versus fourth-order Butterworth high-pass at 100 Hz or 180 Hz, both applied "
                  "in both directions to the complete three-second waveform. All views use the same existing block rule.",
    "outcome": "Recover sustained periodicity in noisy pairs that had it on their matched clean original. Keep "
               "clean-concordance and isolated background outcomes separately.",
    "advance": "At least six added noisy locations on six original files, at most one newly flagged noise-only excerpt, "
               "and preservation of at least 90% of the raw clean flags. This diagnostic has zero confidence weight.",
}
SCHEMA = "chipmunk.video-mouth-hold-noise-probe/1"
VIEWS = ("raw", "highpass_100", "highpass_180")
SCENES = ("clean", "street", "washroom")
SAMPLE_COUNT = 6_400
SAMPLE_RANGE = (20_800, 27_200)


def sha(data):
    return sha256(data).hexdigest()


def summary(rows):
    groups = {}
    controls = Counter()
    for row in rows:
        readings = row["periodic_blocks"]
        assert set(readings) == set(VIEWS)
        assert all(type(n) is int and 0 <= n <= 19 for n in readings.values())
        flags = {view: 3 * count >= 2 * 19 for view, count in readings.items()}
        group = row["group"]
        if group is None:
            for view in VIEWS:
                controls[view] += flags[view]
                controls[f"{view}_added"] += flags[view] and not flags["raw"]
            continue
        group_rows = groups.setdefault(group, {})
        assert row["scene"] not in group_rows
        group_rows[row["scene"]] = flags
    assert len(groups) == 36 and all(set(record) == set(SCENES) for record in groups.values())
    result = {}
    for view in VIEWS:
        counts = Counter()
        newly_supported_groups = set()
        for group, record in groups.items():
            if not record["clean"]["raw"]:
                counts["new_clean"] += record["clean"][view]
                continue
            counts["clean_retained"] += record["clean"][view]
            counts["clean_expected"] += 1
            for scene in SCENES[1:]:
                flags = record[scene]
                counts[f"{scene}_recovered"] += flags[view]
                counts["noisy_recovered"] += flags[view]
                if flags[view] and not flags["raw"]:
                    counts["new_noisy"] += 1
                    newly_supported_groups.add(group)
                counts["lost_noisy"] += flags["raw"] and not flags[view]
        result[view] = dict(counts) | {"new_noisy_groups": len(newly_supported_groups),
                                      "background_only_flags": controls[view], "new_background_only_flags": controls[f"{view}_added"]}
    for record in result.values():
        record["qualifies"] = (record["new_noisy_groups"] >= 6 and record["new_background_only_flags"] <= 1
                               and record["clean_retained"] * 10 >= 9 * record["clean_expected"])
    return result


def read_prior():
    data = PRIOR.read_bytes()
    return sha(data), json.loads(data)["holdout"]["rows"]


def identity(row):
    group = row.get("source")
    return {"id": row["id"], "scene": row["scene"], "source_wav_sha256": row["wav_sha256"],
            "group": None if group is None else group["source_pcm_sha256"]}


def review(record):
    prior_sha, prior = read_prior()
    assert record["schema"] == SCHEMA and record["method"] == METHOD and record["prior_sha256"] == prior_sha
    assert len(record["rows"]) == len(prior) == 116
    for row, original in zip(record["rows"], prior, strict=True):
        assert identity(original) == {k: row[k] for k in identity(original)}
    result = summary(record["rows"])
    assert record["summary"] == result
    return result


def run(source, output):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.video.mouth_hold_sound import measure

    prior_sha, prior = read_prior()
    review_window = {"schema": "chipmunk.video-mouth-local-hold/2", "status": "matched",
                     "match": {"frames": [0, 1]}, "pts": list(SAMPLE_RANGE)}
    mapping = {"timeline_time_base": "1/16000", "samples": 48_000,
               "timeline": [{"start_sample": 0, "end_sample": 48_000, "source_pts": 0}]}
    filters = {name: butter(4, cutoff, btype="highpass", fs=16000, output="sos")
               for name, cutoff in zip(VIEWS[1:], (100, 180), strict=True)}
    rows = []
    for row in prior:
        path = source / f"{row['id']:03d}.wav"
        assert sha(path.read_bytes()) == row["wav_sha256"]
        with wave.open(str(path), "rb") as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels(), wav.getnframes()) == (16_000, 2, 1, 48_000)
            pcm = np.frombuffer(wav.readframes(48_000), dtype="<i2").astype(np.float64) / 32768.0
        signals = {"raw": pcm, **{name: sosfiltfilt(coefficients, pcm) for name, coefficients in filters.items()}}
        counts = {}
        for name, signal in signals.items():
            found = measure(signal, mapping, review_window, "1/16000")
            assert found["samples"] == list(SAMPLE_RANGE) and found["blocks"] == 19
            counts[name] = found["periodic_blocks"]
        rows.append(identity(row) | {"periodic_blocks": counts})
    record = {"schema": SCHEMA, "method": METHOD, "prior_sha256": prior_sha, "rows": rows, "summary": summary(rows)}
    review(record)
    with output.open("x") as file:
        file.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source", type=Path, help="Retained WAV directory. Omit for scalar replay.")
    args = parser.parse_args()
    report = run(args.source, args.receipt) if args.source else review(json.loads(args.receipt.read_bytes()))
    print(json.dumps(report, indent=2, sort_keys=True))
