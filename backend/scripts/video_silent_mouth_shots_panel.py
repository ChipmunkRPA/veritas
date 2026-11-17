# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen near-silence check on exact post-cut soundtrack intervals."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from scripts.video_lip_multishot_panel import CARRIERS, RATE, digest, ffmpeg, mux, overlap, read, save


def variants(pcm, truth, scope, variant):
    import numpy as np

    lo, hi = scope
    center = sum(truth) // 2
    count = (3840, 5760, 8320)[variant % 3]
    length = count // 2
    start = max(lo + 2560, min(center - length, hi - 2560 - count)) // 80 * 80
    stop = start + count
    if not lo + 1920 <= start < stop <= hi - 1920:
        raise ValueError("The staged edit will not fit completely inside the target shot")
    for name in ("original", "muted", "muted_faded", "lower_level", "across_cut"):
        values = pcm.copy()
        label = None
        if name == "muted":
            values[start:stop] = 0
            label = [start, stop]
        elif name == "muted_faded":
            fade = 120
            values[start:start + fade] = np.rint(values[start:start + fade] * np.linspace(1., 0., fade))
            values[start + fade:stop - fade] = 0
            values[stop - fade:stop] = np.rint(values[stop - fade:stop] * np.linspace(0., 1., fade))
            label = [start, stop]
        elif name == "lower_level":
            values[start:stop] = np.rint(values[start:stop] * .10)
        elif name == "across_cut":
            values[lo - 2560:lo + 3520] = 0
        if label is not None and not (np.array_equal(pcm[:start], values[:start]) and np.array_equal(pcm[stop:], values[stop:])):
            raise ValueError("The staged mute changed samples outside its label")
        yield name, values, label


def counts(rows):
    result = defaultdict(Counter)
    for row in rows:
        truth, matches = row["truth"], row["matches"]
        found = sum(truth is not None and overlap(truth, span) >= .4 for span in matches)
        result[f"{row['carrier']}/{row['role']}"].update(files=1, earlier_eligible=row["earlier_eligible"],
                                                        assessed=row["assessed"], flagged=bool(matches),
                                                        correct=bool(found), misplaced=len(matches) - found)
    return result


def measure(montages, assets, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from app.detectors.video.viseme_review import sound_levels, supported, supported_shots
    from app.detectors.video.viseme_worker import capture_mouth_frames
    from app.evidence import video_lip_agreement as agreement
    from app.evidence import video_silent_mouth as silent
    from app.evidence import video_viseme_binding as binding

    original = read(montages / "result.json.gz")
    before = {(row["group"], row["role"]): row for row in original["rows"] if row["carrier"] == "pcm"}
    output.mkdir(mode=0o700)
    sys.path.insert(0, str(assets / "runtime"))
    results, unrecognized = [], []
    for group, role in sorted(before):
        if role != "original":
            continue
        row = before[group, role]
        target = before[group, "different_words"]["truth"]
        path = montages / row["delivery"]
        pin = digest(path)
        if pin != row["input_sha256"]:
            raise ValueError("The new mute trial differs from the verified montage input")
        table = probe_frames(path)
        with closing(decode_stream(path, table)) as stream:
            shot_scan = detect_shots(stream, table).as_json()
        shots = shot_scan["ranges"]
        if shots != row["observed_shots"] or shot_scan["frames_scanned"] != len(table):
            raise ValueError("The original shot boundaries differ from the prior readback")
        if not row["exact_cuts"]:
            unrecognized.append({"group": group, "source_sha256": pin, "planned_shots": row["planned_shots"], "observed_shots": shots})
            continue
        frames, _, _ = capture_mouth_frames(path, table, assets / "face_landmarker.task", motion=False, shots=shots)
        decoded = decode_audio(path, source_sha256=pin)
        pcm = np.rint(decoded.pcm * 32768).astype("<i2")
        last = agreement.shot_audio_ranges(table.pts, table.time_base, len(pcm), shots)[-1]
        video_hash = ffmpeg("-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1")
        for name, edited, truth in variants(pcm, target, last, group - 1):
            for carrier in CARRIERS:
                media = output / f"group-{group:02d}-{carrier}-{name}.mov"
                mux(path, edited, carrier, media)
                source_hash = digest(media)
                native = probe_frames(media)
                if (native.pts != table.pts or native.time_base != table.time_base
                        or ffmpeg("-i", str(media), "-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1") != video_hash):
                    raise ValueError("A sound-only edit changed the picture stream")
                sound = decode_audio(media, source_sha256=source_hash)
                mapping = sound.mapping
                samples = binding.presented_samples(mapping, max_samples=agreement.MAX_MULTISHOT_SAMPLES)
                if (samples != len(pcm) or not supported_shots(table, mapping, shots)
                        or (carrier == "pcm" and not np.array_equal(edited, np.rint(sound.pcm * 32768)))):
                    raise ValueError("The delivered samples differ from the planned common clock")
                measured = binding.measured(
                    {"frames": frames, "phones": [], silent.SHOT_KEY: shots, "sound_levels": sound_levels(sound.pcm, samples)},
                    timestamps=table.pts, time_base=table.time_base, source={"width": table.width, "height": table.height}, mapping=mapping,
                )
                matches = [exhibit["witness"]["audio_source_pts"] for _, exhibit in binding.exhibits(measured)]
                results.append({"group": group, "carrier": carrier, "role": name, "delivery": media.name, "input_sha256": source_hash,
                                "source_sha256": pin, "truth": truth, "matches": matches, "coverage": measured["shot_coverage"],
                                "earlier_eligible": bool(supported(table, mapping, shots)),
                                "assessed": measured["status"] in {"matched", "no_match"}})
        print(f"Verified shots, picture hashes and soundtrack ranges for group {group}", flush=True)
    data = counts(results)
    save(output / "result.json.gz", {"schema": "chipmunk.silent-mouth-shots-panel/1", "rows": results, "counts": data,
                                     "montage_comparison_sha256": digest(montages / "result.json.gz"), "sample_rate": RATE,
                                     "unrecognized_shots": unrecognized})
    print(json.dumps(data, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    trial = modes.add_parser("measure")
    for name in ("montages", "assets", "output"):
        trial.add_argument(name, type=Path)
    modes.add_parser("review").add_argument("receipt", type=Path)
    args = vars(parser.parse_args())
    if args.pop("action") == "measure":
        measure(**args)
    else:
        result = read(args["receipt"])
        data = counts(result["rows"])
        if result["schema"] != "chipmunk.silent-mouth-shots-panel/1" or result["counts"] != data:
            raise ValueError("The near-silence results do not recount")
        print(json.dumps(data, sort_keys=True))
