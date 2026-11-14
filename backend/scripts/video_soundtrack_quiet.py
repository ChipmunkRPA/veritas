# SPDX-License-Identifier: Apache-2.0
"""Recreate hash-bound soundtrack controls and test quieter waveform matches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import wav_bytes
from app.evidence import video_reference_soundtrack as waveform
from app.evidence.audio_reference import read_track
from scripts import review_video_soundtrack_quiet as rule
from scripts import video_soundtrack_cadence as cadence
from scripts.video_common_sources_v89 import read, require, sha, write

PARENT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/video-soundtrack-cadence-2026-11-14.json"


def extract(source, offset, *, transfer):
    raw = cadence.parent.execute("ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(source),
                                 "-map", "0:a:0", "-t", "4.25" if transfer else "4", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                                 "-f", "s16le", "pipe:1")
    samples = np.frombuffer(raw, dtype="<i2")
    require(64000 <= len(samples) <= (69024 if transfer else 65024), "The frozen four-second source has incomplete audio")
    return samples[:64000].astype(np.float32) / 32768


def decoded(wave, folder, carrier):
    source = folder / "clip.wav"
    source.write_bytes(wav_bytes(wave))
    if carrier == "AAC":
        target = folder / "clip.m4a"
        cadence.parent.execute("ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", "aac", "-b:a", "64k",
                               "-threads", "1", "-y", str(target))
        source = target
    status, values, mapping = read_track(source, {"sha256": sha(source)}, max_seconds=6, video=True)
    require(status == "measured" and mapping["relative_start"] == "0", "A soundtrack lost its continuous source clock")
    digest = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
    require(digest == mapping["pcm_sha256"], "PCM does not match the exact decoded input")
    return values, digest


def variations(original, donor, start, end, *, transfer, group):
    weights = np.ones(end - start, dtype=np.float32)
    weights[:80] = np.arange(80) / 80
    weights[-80:] = weights[:80][::-1]

    def mixed(other):
        target = original.copy()
        target[start:end] += weights * (other[start:end] - original[start:end])
        return target

    advanced, delayed = (cadence.parent.shift(original, shift) for shift in (-3200, 3200))
    require(np.array_equal(advanced[start:end], original[start + 3200:end + 3200])
            and np.array_equal(delayed[start:end], original[start - 3200:end - 3200]), "Signed edit input changed")
    filtered = sosfiltfilt(butter(4, 2500, fs=16000, output="sos"), original)
    cases = {"earlier": mixed(advanced), "later": mixed(delayed), "untouched": original,
             "gain": mixed(original * .55), "filter": mixed(filtered), "constant": delayed, "alternate_speech": mixed(donor)}
    if transfer:
        quiet = original * np.float32(10 ** (-12 / 20))
        bed = np.random.default_rng(4879 + group).normal(0., .0014, 3200).astype(np.float32) * np.hanning(3200)
        bed_record = original.copy()
        bed_record[start:end] *= .02
        bed_record[start:start + 3200] += bed
        bed_record[start + 3200:start + 6400] += bed
        cases.update(silent_interval=mixed(np.zeros_like(original)), repeat_bed=bed_record, quiet_gain=mixed(original * .13))
        return quiet, {role: wave * np.float32(10 ** (-12 / 20)) for role, wave in cases.items()}
    synchronized = mixed(cadence.parent.shift(original, 3333))
    cases.update(synchronized_picture=synchronized, synchronized_mouth=synchronized, picture_only=original)
    return original, cases


def freeze(folders):
    parent = read(PARENT)
    sources = parent["freeze"]["sources"]
    offsets = {}
    for source in sources:
        path = cadence.locate(folders, source["name"], source["sha256"])
        duration = float(cadence.parent.execute("ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                "-of", "csv=p=0", str(path)))
        offsets[str(source["group"])] = source["offset"] + (5 if source["offset"] + 9 <= duration else -5)
    return {"schema": rule.SCHEMA, "methods": rule.METHODS, "settings": rule.SETTINGS,
            "parent_sha256": sha(PARENT), "transfer_offsets": offsets, "bindings": {Path(name).name: sha(name) for name in (
                __file__, rule.__file__, cadence.__file__)},
            "waveform_sha256": sha(waveform.__file__)}


def score(lock, folders):
    require(lock == freeze(folders), "The sources or implementation changed after freeze")
    parent = read(PARENT)
    sources = parent["freeze"]["sources"]
    plans = {row["group"]: row for row in parent["rows"] if row["rate"] == 24}
    result = []
    for transfer in (False, True):
        offsets = {s["group"]: lock["transfer_offsets"][str(s["group"])] if transfer else s["offset"] for s in sources}
        originals = {s["group"]: extract(cadence.locate(folders, s["name"], s["sha256"]), offsets[s["group"]], transfer=transfer)
                     for s in sources}
        for index, source in enumerate(sources):
            group = source["group"]
            donor = originals[sources[(index + 1) % len(sources)]["group"]]
            truth = plans[group]["edit_samples"]
            reference, versions = variations(originals[group], donor, *truth, transfer=transfer, group=group)
            row = {"panel": "transfer" if transfer else "prior", "group": group, "offset": offsets[group],
                   "source_sha256": source["sha256"], "edit_samples": truth, "samples": len(reference), "cases": []}
            pinned = {(c["role"], c["carrier"]): c for c in plans[group]["cases"]}
            for carrier in rule.CARRIERS:
                with TemporaryDirectory(prefix="soundtrack-quiet-") as scratch:
                    folder = Path(scratch)
                    first, first_digest = decoded(reference, folder, carrier)
                    require(row["samples"] <= len(first) <= row["samples"] + 1024,
                            "Delivery does not cover the planned four seconds")
                    for role, wave in versions.items():
                        second, second_digest = decoded(wave, folder, carrier)
                        require(len(second) == len(first), "The two delivered audio clocks differ")
                        comparator = pinned.get((role, carrier), {})
                        if not transfer:
                            require((first_digest, second_digest) == (comparator["earlier_pcm_sha256"],
                                                                     comparator["submitted_pcm_sha256"]),
                                    "A recreated PCM input differs from the frozen video delivery")
                        qualified = transfer or all(value <= rule.POINTS[metric]
                            for grid in comparator["picture"].values() for metric, value in grid.items())
                        arms = {}
                        for name, (window, step, rms) in rule.METHODS.items():
                            best, count = waveform.measure(first, second, window=window, step=step, min_rms=rms)
                            if best and max(best["submitted_samples"][1], best["earlier_samples"][1]) * 24 > 95 * 16000:
                                best = None
                            arms[name] = [best, count]
                        row["cases"].append({"role": role, "carrier": carrier, "qualified": qualified, "samples": len(first),
                                             "earlier_pcm_sha256": first_digest, "submitted_pcm_sha256": second_digest,
                                             "arms": arms})
            result.append(row)
            print(f"{row['panel']} source {group} verified", flush=True)
    return parent, {"freeze": lock, "rows": result}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score"))
    cli.add_argument("output", type=Path)
    cli.add_argument("sources", type=Path, nargs="+")
    args = cli.parse_args()
    pin = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(pin, freeze(args.sources))
    else:
        parent, measured = score(read(pin), args.sources)
        measured["outcome"] = rule.review(measured, parent)
        write(args.output / "receipt.json", measured)
        print(json.dumps(measured["outcome"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
