# SPDX-License-Identifier: Apache-2.0
"""Test timing comparisons across two audio streams in one uploaded clip."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.video_short_soundtrack_transfer import OFFSETS as EDIT_CLOCK_OFFSETS
from scripts.video_short_soundtrack_transfer import RATE, boundary, sha, sources

OFFSETS = (43, 51)
CARRIERS = {"pcm": ("mov", "pcm_s16le"), "aac": ("mp4", "aac")}
VARIANTS = ("earlier", "later", "earlier_rain", "later_train", "same", "gain", "filter", "constant", "different_speech",
            "rain_only", "train_only", "short", "clock_offset", "same_local_shift", "one_track", "three_tracks")
LAGS = {"earlier": -3200, "later": 3200, "earlier_rain": -3200, "later_train": 3200}
SCHEMA = "chipmunk.video-soundtrack-pair-transfer/1"


def execute(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)], check=True, timeout=20)


def freeze(bank):
    plan = sources()
    for source in plan:
        if sha(bank / source["file"]) != source["sha256"]:
            raise ValueError("A publisher recording differs from its source commitment")
    return {"schema": SCHEMA, "offsets": list(OFFSETS), "sources": plan, "variants": list(VARIANTS), "lags": LAGS,
            "acceptance": "Preserve every useful range with at most two additional ordinary locations per carrier and "
                          "zero new alerts on whole-track delay, different speech, or matching local shifts."}


def review(receipt):
    if receipt["freeze"]["schema"] != SCHEMA:
        raise ValueError("Unexpected receipt schema")
    keys = [(r["group"], r["offset"], r["carrier"], r["role"]) for r in receipt["rows"]]
    expected = [(i, at, carrier, role) for i in range(1, 6) for at in OFFSETS for carrier in CARRIERS for role in VARIANTS]
    if keys != expected:
        raise ValueError("Missing or reordered delivery")
    results = {}
    for carrier in CARRIERS:
        totals = {name: Counter() for name in ("guarded", "unrestricted")}
        for row in (row for row in receipt["rows"] if row["carrier"] == carrier):
            truth = row["truth"]
            expected_truth = None if row["role"] not in LAGS else {
                "samples": boundary(row["group"], EDIT_CLOCK_OFFSETS[OFFSETS.index(row["offset"])]),
                "shift": LAGS[row["role"]]}
            if truth != expected_truth:
                raise ValueError("Known edit differs from its planned interval")
            for arm, counter in totals.items():
                reading = row[arm]
                if reading["status"] not in {"unsupported_clock", "insufficient_same_time_support", "local_shift", "no_local_shift"}:
                    raise ValueError("Unexpected status")
                span = reading.get("range")
                if (reading["status"] == "local_shift") != (span is not None):
                    raise ValueError("Timing status differs from the recorded span")
                counter[reading["status"]] += 1
                counter["planned_edits" if truth else "controls"] += 1
                if not span:
                    continue
                a, b = span["submitted_samples"]
                c, d = span["earlier_samples"]
                shift = span["shift_samples"]
                if not (0 <= min(a, c) < max(b, d) <= 65024 and b > a and a-c == b-d == shift
                        and 1920 <= abs(shift) <= 5760 and span["supporting_windows"] >= 4):
                    raise ValueError("Invalid sound interval")
                if truth:
                    start, stop = truth["samples"]
                    shared = max(0, min(stop, b) - max(start, a))
                    useful = 2*shared >= max(stop, b)-min(start, a) and abs(shift-truth["shift"]) <= 16
                    counter["useful" if useful else "misplaced_edits"] += 1
                    if useful:
                        counter[f"useful_{row['role']}"] += 1
                else:
                    counter["control_flags"] += 1
                    counter[f"flag_{row['role']}"] += 1
        guarded = totals["guarded"]
        results[carrier] = {**{name: dict(counts) for name, counts in totals.items()},
                            "passes": guarded["useful"] >= totals["unrestricted"]["useful"] and guarded["useful"] >= 8
                            and guarded["control_flags"] <= 2 and not any(guarded[f"flag_{case}"] for case in
                                                                         ("constant", "different_speech", "same_local_shift"))}
    return results


def run(bank, sounds, output, examples, *, plan=None, callback=None):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.audio.decode import decode_audio, wav_bytes
    from app.evidence import video_soundtrack_pair as pair
    from scripts.audio_reference_variable_window import read_noise
    from scripts.video_lipsync_local_delay_transfer import shift

    plan = plan or freeze(bank)
    sounds = read_noise(sounds)
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "freeze.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = []
    for group, item in enumerate(plan["sources"], 1):
        for ordinal, offset in enumerate(item.get("offsets", OFFSETS)):
            with TemporaryDirectory(prefix="soundtrack-pair-") as scratch:
                work = Path(scratch)
                picture = work / "base.mov"
                execute("-ss", offset, "-i", bank / item["file"], "-t", 4, "-map", "0:v:0", "-map", "0:a:0",
                        "-vf", "fps=25,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", 22, "-pix_fmt", "yuv420p", "-threads", 1,
                        "-c:a", "pcm_s16le", "-ar", RATE, "-ac", 1, picture)
                decoded = decode_audio(picture, source_sha256=sha(picture), max_seconds=5)
                original = decoded.pcm[:4 * RATE]
                if len(decoded.mapping["timeline"]) != 1 or len(original) < 4 * RATE - 16:
                    raise ValueError(f"Original clip lacks four continuous seconds ({len(original)} decoded samples)")
                start, stop = boundary(group, EDIT_CLOCK_OFFSETS[ordinal])
                weights = np.ones(stop-start)
                weights[:160] = np.arange(160) / 160
                weights[-160:] = weights[:160][::-1]

                def local(wave, original=original, start=start, stop=stop, weights=weights):
                    edited = original.copy()
                    edited[start:stop] += weights * (wave[start:stop] - edited[start:stop])
                    return edited

                beds = {}
                for name, cells in sounds.items():
                    ambient = cells[(group + ordinal) % 4][:len(original)].astype(np.float64) / 32768
                    beds[name] = ambient * (.45 * np.linalg.norm(original) / np.linalg.norm(ambient))
                delayed = local(shift(original, 3200))
                advanced = local(shift(original, -3200))
                choose = {"earlier": advanced, "later": delayed, "earlier_rain": advanced + beds["rain"],
                          "later_train": delayed + beds["train"], "same": original, "gain": original * .55,
                          "filter": sosfiltfilt(butter(2, 2300, fs=RATE, output="sos"), original), "constant": shift(original, 3200),
                          "different_speech": np.r_[original[32000:], original[:32000]], "rain_only": beds["rain"],
                          "train_only": beds["train"], "short": original[:-3200], "clock_offset": original,
                          "same_local_shift": delayed, "one_track": original, "three_tracks": original}
                primary_tracks = {"same_local_shift": delayed}
                if callback:
                    choose, primary_tracks = callback.variants(original, advanced, delayed, beds, choose, primary_tracks)
                for carrier, (suffix, codec) in CARRIERS.items():
                    for role, variant in choose.items() if callback else ((role, choose[role]) for role in VARIANTS):
                        base, alternate = work / "primary.wav", work / "alternate.wav"
                        base.write_bytes(wav_bytes(np.clip(primary_tracks.get(role, original), -1., 32767 / 32768)))
                        alternate.write_bytes(wav_bytes(np.clip(variant, -1., 32767 / 32768)))
                        output_clip = work / f"delivery.{suffix}"
                        other_input = ["-itsoffset", "0.12"] if role == "clock_offset" else []
                        mapping = ["-map", "0:v:0", "-map", "1:a:0"]
                        if role != "one_track":
                            mapping += ["-map", "2:a:0"]
                        if role == "three_tracks":
                            mapping += ["-map", "2:a:0"]
                        execute("-i", picture, "-i", base, *other_input, "-i", alternate, *mapping,
                                "-c:v", "copy", "-c:a", codec, "-threads", 1, output_clip)
                        digest = sha(output_clip)
                        first = decode_audio(output_clip, source_sha256=digest, max_seconds=pair.DECODE_SECONDS)
                        readings = {arm: {"status": "unsupported_clock"} for arm in ("guarded", "unrestricted")}
                        second = None
                        if len(first.mapping.get("audio_stream_indices", [])) == 2:
                            second = pair.decode(output_clip, digest)
                            readings = {"guarded": pair.measure(first, second),
                                        "unrestricted": pair.measure(first, second, require_shared_track=False)}
                        lags = callback.LAGS if callback else LAGS
                        truth = {"samples": [start, stop], "shift": lags[role]} if role in lags else None
                        row = {"group": group, "offset": offset, "carrier": carrier, "role": role,
                               "truth": truth, "sha256": digest, **readings}
                        rows.append(row)
                        if callback:
                            callback.score(row, first, second, output_clip, output, examples)
                        if examples and group == 1 and ordinal == 0 and role in {"later", "same", "constant", "different_speech"}:
                            (output / f"{carrier}-{role}.{suffix}").write_bytes(output_clip.read_bytes())
            print(group, offset, len(rows), flush=True)
    receipt = {"freeze": plan, "rows": rows}
    receipt["results"] = (callback.review if callback else review)(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["results"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    if args.phase == "run":
        if not (args.bank and args.sounds):
            parser.error("Run requires both source recordings and background sound")
        run(args.bank, args.sounds, args.path, args.examples)
    else:
        receipt = json.loads(args.path.read_text())
        if review(receipt) != receipt["results"]:
            raise ValueError("Recomputed outcomes differ")
        print(json.dumps(receipt["results"], indent=2, sort_keys=True))
