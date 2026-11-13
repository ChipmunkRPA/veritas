"""Confirm local upper-band loss on source-disjoint AMI recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

if __package__:
    from .review_audio_formant_local_transfer import canonical, require, useful
else:
    from review_audio_formant_local_transfer import canonical, require, useful

SCHEMA = "chipmunk.audio-bandwidth-confirmation/1"
VARIANTS = ("untouched", "global_lowpass", "local_lowpass_3400", "local_lowpass_3800", "local_gain", "local_natural_cut")
CAPTURES = ("ES2007c", "IS1004c", "ES2012b", "IS1001b")
METHOD = {
    "sources": "Four AMI recordings from source meetings excluded from both initial bandwidth banks. Per session choose eight six-second excerpts, one per eighth of meeting duration, by content-independent SHA-256 ordering among nonoverlapping twelve-second grid starts away from meeting edges. Read channel 01 signed 16-bit mono 16 kHz without scaling.",
    "positive": "10-pole zero-phase whole-mix lowpass at 3400 or 3800 Hz during an 0.8-1.0 second interval chosen by the original PCM digest with 25 ms cosine edges. Both flanks and all controls derive from the same frozen recording.",
    "control": "Untouched, full-file 3400 Hz, a 0.38 gain reduction, and a time-shifted local ordinary cut from the following six seconds of the same capture. All local interventions use identical 25 ms fades.",
    "point": "Freeze before held scoring. Based on initial development, use median flanks, 65th percentile interior, a -80 dBFS minimum upper-band level, 12 dB maximum flank disagreement, 7 dB limit on middle-band change and a strict 16 dB minimum drop in both upper bands.",
    "gate": "Find at least 80% useful locations on each lowpass cutoff and at least one useful location in each capture, while flagging at most one untouched or ordinary/gain control. Whole-file-filter controls should carry no local report; explicitly count unassessed. Compare prior two-edge transition outcomes. A useful span covers at least half the intervention and reaches 0.30 IoU.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_source(path, session):
    rate, extent = 16_000, 96_000
    with wave.open(str(path), "rb") as audio:
        require(audio.getparams()[:3] == (1, 2, rate), "AMI source is not native 16 kHz mono PCM16")
        limit = audio.getnframes()
        for block in range(8):
            begin = max(60 * rate, block * limit // 8)
            end = min(limit - 60 * rate, (block + 1) * limit // 8)
            grid = range(((begin + 2 * extent - 1) // (2 * extent)) * 2 * extent, end - 2 * extent + 1, 2 * extent)
            tick = min(grid, key=lambda value: sha(f"chipmunk-bandwidth-20261113:{session}:{value}".encode()))
            audio.setpos(tick)
            raw, neighbor = audio.readframes(extent), audio.readframes(extent)
            require(len(raw) == len(neighbor) == 2 * extent, "AMI native excerpt is truncated")
            yield {"id": f"{session}-{block}", "session": session, "start_sample": tick,
                   "pcm_sha256": sha(raw), "neighbor_sha256": sha(neighbor)}, raw, neighbor


def report(rows):
    output = {"variants": {}, "captures": {}}
    for row in rows:
        count = output["variants"].setdefault(row["variant"], Counter())
        count["planned"] += 1
        count["unassessed"] += row["best"] is None
        flagged = row["best"] is not None and row["best"]["drop_db"] > 16
        count["flags"] += flagged
        truth = row["truth"]
        if truth is None:
            continue
        hit = flagged and useful(row["best"]["range"], truth)
        count["useful"] += hit
        count["baseline_two_edges"] += (any(abs(tick - truth[0]) <= 4000 for tick in row["existing_transitions"])
                                        and any(abs(tick - truth[1]) <= 4000 for tick in row["existing_transitions"]))
        output["captures"].setdefault(row["id"].rsplit("-", 1)[0], Counter())[row["variant"]] += hit
    variants = output["variants"]
    positives = [variants[arm] for arm in VARIANTS if arm.startswith("local_lowpass_")]
    output["gate_pass"] = (all(5 * arm["useful"] >= 4 * arm["planned"] for arm in positives)
                           and all(all(values.values()) for values in output["captures"].values())
                           and sum(variants[arm]["flags"] for arm in ("untouched", "local_gain", "local_natural_cut")) <= 1
                           and not variants["global_lowpass"]["flags"])
    return output


def check(receipt):
    rows, freeze = receipt["rows"], receipt["freeze"]
    require(freeze["schema"] == SCHEMA and freeze["method"] == METHOD and len(freeze["sources"]) == 32, "Unexpected method or source extent")
    require(len(rows) == 192 and len({(row["id"], row["variant"]) for row in rows}) == 192, "Unexpected variant extent")
    sources = {row["id"]: row for row in freeze["sources"]}
    for row in rows:
        source = sources[row["id"]]
        variant, digest, truth, best = (row[key] for key in ("variant", "sha256", "truth", "best"))
        require(variant in VARIANTS and len(digest) == 64, "Invalid variant or native hash")
        original = source["pcm_sha256"]
        first = 16_000 + int(original[:4], 16) % 56_000
        duration = 12_800 + int(original[4:8], 16) % 3_201
        require(truth == ([first, first + duration] if variant.startswith("local_lowpass") else None), "Wrong authored source interval")
        if variant == "untouched":
            require(digest == original, "Untouched source waveform changed")
        require(0 <= row["candidates"] <= 285 and bool(row["candidates"]) == (best is not None), "Invalid measured coverage")
        require(all(tick % 4000 == 0 and 8000 <= tick <= 88000 for tick in row["existing_transitions"]), "Wrong comparison clock")
        if best is None:
            continue
        a, b = best["range"]
        require(6400 <= a < b <= 89600 and a % 1280 == 0 and (b-a) // 1280 in (7, 9, 11, 13, 15)
                and b % 1280 == 0, "New interval differs from the 80 ms clock")
        left, interior, right = best["relative_band_db"]
        drop = [min(l, r) - m for l, m, r in zip(left, interior, right, strict=True)]
        require(len(drop) == 3 and abs(drop[0]) <= 7.000002 and abs(min(drop[1:]) - best["drop_db"]) <= .000002
                and max(abs(l-r) for l, r in zip(left[1:], right[1:], strict=True)) <= 12.000002, "Band support differs")
    result = report(rows)
    require(result == receipt["summary"], "Recorded counts differ")
    return result


def measure(roster):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.evidence.audio_bandwidth_confirmation import locate
    from app.evidence.audio_continuity import candidates, select

    rows = []
    for source, raw, neighbor in roster:
        signal = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
        first = 16_000 + int(source["pcm_sha256"][:4], 16) % 56_000
        duration = 12_800 + int(source["pcm_sha256"][4:8], 16) % 3_201
        end = first + duration
        rise = .5 - .5 * np.cos(np.linspace(0, np.pi, 400))
        blend = np.r_[rise, np.ones(duration - 800), rise[::-1]]
        filtered = {rate: sosfiltfilt(butter(10, rate, fs=16000, output="sos"), signal) for rate in (3400, 3800)}
        remote = np.frombuffer(neighbor, dtype="<i2").astype(np.float64) / 32768
        for variant, replacement in zip(VARIANTS, (signal, filtered[3400], filtered[3400], filtered[3800], signal * .38, remote), strict=True):
            samples = signal.copy() if variant.startswith("local_") else replacement
            if variant.startswith("local_"):
                samples[first:end] += (replacement[first:end] - samples[first:end]) * blend
            rendered = np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")
            samples = rendered.astype(np.float64) / 32768
            count, best = locate(samples)
            rows.append({"id": source["id"], "variant": variant, "sha256": sha(rendered.tobytes()),
                         "truth": [first, end] if variant.startswith("local_lowpass") else None,
                         "best": best, "candidates": count,
                         "existing_transitions": [row["boundary_sample"] for row in select(candidates(samples))]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("sources", nargs="*", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(check(json.loads(args.run.read_bytes())), sort_keys=True))
        return
    indexed = {source.name.split(".", 1)[0]: source for source in args.sources}
    require(set(indexed) == set(CAPTURES), "Supply one publisher WAV for each specified AMI capture")
    roster = [row for name in CAPTURES for row in select_source(indexed[name], name)]
    repo = Path(__file__).resolve().parents[2]
    frozen = {"schema": SCHEMA, "method": METHOD, "source_pins": {name: file_sha(indexed[name]) for name in CAPTURES},
              "sources": [row[0] for row in roster], "runner_sha256": file_sha(Path(__file__)),
              "method_sha256": file_sha(repo / "backend/app/evidence/audio_bandwidth_confirmation.py"),
              "bands_sha256": file_sha(repo / "backend/app/evidence/audio_bandwidth_return.py")}
    lock = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        lock.write_bytes(canonical(frozen))
        print(f"Froze {len(roster)} source excerpts before measurement")
        return
    require(lock.read_bytes() == canonical(frozen), "Frozen clock, source, or method changed")
    rows = measure(roster)
    receipt = {"freeze": frozen, "rows": rows, "summary": report(rows)}
    check(receipt)
    output = args.run / "public.json"
    if args.phase == "score":
        with output.open("xb") as stream:
            stream.write(canonical(receipt))
    else:
        require(output.read_bytes() == canonical(receipt), "Native replay differs")
    print(json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
