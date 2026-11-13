# SPDX-License-Identifier: Apache-2.0
"""Replay exact channel timing on licensed speech and measured stereo rooms."""

import argparse
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from app.evidence.audio_exact_channel_timing import locate
from scripts import audio_blind_huddersfield_transfer as rooms
from scripts import evaluate_short_lip_release_timing as inventory

SCHEMA = "chipmunk.stereo-channel-timing-challenge/1"
SOURCE_RECEIPT = "83c333f306d21bb7acfc16c3db9189db5c7cdc603d2603bef044184583dcb4ee"


def source_wave(root, row):
    _, pcm, _ = inventory.source_pcm(inventory.prior.source(root, row, "audio"))
    return pcm.astype(np.float64)


def build(sources, archive):
    ref, _ = rooms.impulses(archive)
    rooms.require(
        rooms.file_hash(sources / "source-receipt.json") == SOURCE_RECEIPT, "Publisher roster differs"
    )
    by = {}
    for row in inventory.sources(sources):
        by.setdefault(row["group"], []).append(row)
    groups = sorted(by)
    for index, group in enumerate(groups):
        masker_group = groups[(index + 5) % len(groups)]
        row = next(x for x in by[group] if x["condition"] == "p")
        noise = next(x for x in by[masker_group] if x["condition"] == "l")
        src = source_wave(sources, row)
        other = source_wave(sources, noise)
        n = min(len(src), len(other))
        mix = rooms.render_stem(src[:n], ref["S01"])[:n]
        inter = rooms.render_stem(other[:n], ref[("S03", "S07")[index % 2]])[:n]
        inter *= np.linalg.norm(mix.mean(axis=1)) * 10**0.25 / np.linalg.norm(inter.mean(axis=1))
        full = mix + inter
        scale = 22000 / max(np.max(np.abs(full)), np.max(np.abs(inter)))
        full = np.rint(full * scale).astype("<i2")
        only = np.rint(inter * scale).astype("<i2")
        yield index + 1, full, only


def variants(index, first, only):
    n = len(first)
    start = n // 2 - 3500 + index * 13
    for ch in (0, 1):
        for delta in (2049 + index * 33, -2893 - index * 41):
            stop = start + 6400 + index * 113
            changed = first.copy()
            changed[start:stop, ch] = first[start - delta : stop - delta, ch]
            if (index + ch) % 2:
                # ordinary nonexact crossfade while keeping an exact interior
                for at in (start, stop - 80):
                    ramp = np.linspace(0.05, 0.95, 80)
                    if at > start:
                        ramp = ramp[::-1]
                    changed[at : at + 80, ch] = np.rint(
                        changed[at : at + 80, ch] * ramp + first[at : at + 80, ch] * (1 - ramp)
                    ).astype("<i2")
            yield "shift", first, changed, {ch: (delta, start, stop)}
    yield "unchanged", first, first, {}
    yield "background_only", only, only, {}
    altered = first.copy()
    altered[start : start + 6500, index % 2] = np.rint(altered[start : start + 6500, index % 2] * 0.72)
    yield "gain", first, altered, {}
    altered = first.copy()
    altered[start : start + 6500, index % 2] = only[start : start + 6500, index % 2]
    yield "background_replace", first, altered, {}
    # An exact excerpt may match several copies of the repeated background.
    periodic = first.copy()
    periodic[:, 0] = np.resize(first[:2400, 0], n)
    altered = periodic.copy()
    altered[start : start + 6500, 0] = periodic[start - 2650 : start + 3850, 0]
    yield "ambiguous", periodic, altered, {}


def wav(path, signal):
    with wave.open(str(path), "wb") as f:
        f.setparams((2, 2, 16000, 0, "NONE", ""))
        f.writeframes(signal.tobytes())


def replay(sources, archive, fixtures=None):
    if fixtures:
        fixtures.mkdir(mode=0o700)
    ledger = []
    for index, first, only in build(sources, archive):
        for role, a, b, truth in variants(index, first, only):
            result = [locate(a[:, j], b[:, j]) for j in range(2)]
            expected = {str(channel): list(bounds) for channel, bounds in truth.items()}
            ok = all(
                r is None
                if j not in truth
                else (
                    r is not None
                    and r["shift_samples"] == truth[j][0]
                    and truth[j][1] <= r["submitted_samples"][0] < r["submitted_samples"][1] <= truth[j][2]
                )
                for j, r in enumerate(result)
            )
            ledger.append(
                {"group": index, "role": role, "expected": expected, "result": result, "passed": ok}
            )
            if fixtures and index == 1 and role == "shift" and 0 in truth and truth[0][0] > 0:
                wav(fixtures / "earlier.wav", a)
                wav(fixtures / "analyzed.wav", b)
            if fixtures and index == 2 and role == "shift" and 1 in truth and truth[1][0] < 0:
                wav(fixtures / "earlier-2.wav", a)
                wav(fixtures / "analyzed-2.wav", b)
    counts = Counter((row["role"], row["passed"]) for row in ledger)
    rooms.require(
        len(ledger) == 108 and all(row["passed"] for row in ledger),
        "Unexpected signed offset or control result",
    )
    return {
        "schema": SCHEMA,
        "source_receipt_sha256": SOURCE_RECEIPT,
        "counts": {role: count for (role, _), count in sorted(counts.items())},
        "results": ledger,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("room_archive", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = replay(args.sources, args.room_archive, args.fixtures)
    if args.verify:
        rooms.require(result == rooms.read(args.receipt), "Native replay differs")
    else:
        rooms.write(args.receipt, result)
    print(result["counts"])


if __name__ == "__main__":
    main()
