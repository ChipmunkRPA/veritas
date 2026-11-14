# SPDX-License-Identifier: Apache-2.0
"""Compare short soundtrack replacements and matched ordinary controls."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_short_soundtrack_transfer as sources
from scripts.audio_vocal_event_mixtures import digest, read, require, save, sha

SCHEMA = "chipmunk.video-sound-difference-guard/1"
OFFSETS = (23, 37)
RATE = 16000
CARRIERS = ("PCM", "AAC")
SCENES = ("clean", "rain", "train", "competing")
ROLES = ("same_speaker", "other_speaker", "unchanged", "local_gain", "local_filter", "global_gain", "constant_shift",
         "local_shift", "background_only", "unrelated")
EDITS = ROLES[:2]
ARMS = ("current", "strong_flanks_first", "check_for_delay")
PEOPLE = ("ioanna", "james", "mckensie", "paison", "piotr", "priscilla")
GATE = {"useful_gain": 3, "distinct_groups": 2, "max_added_control_flags_per_carrier": 2,
        "max_priority_control_flags": 0}
BASELINE_SHA256 = "8d07b2a70ca0c3a41412a1b757f580f39678096a42ff64217ed5b64bc5ab8723"


def boundary(group, offset, offsets):
    width = (8, 12, 16, 20)[(group + offsets.index(offset)) % 4] * 640
    start = (34 + (group % 4) * 3) * 640
    return [start, start + width]


def plan():
    method = Path(__file__).with_name("video_sound_difference_guard.py")
    parent = sources.ROOT / "docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json"
    roster = {item["name"]: item for item in read(parent)["source_plan"]["rows"]}
    selected = [{"file": name + ".webm", "sha256": roster[name]["source_sha256"], "credit": roster[name]["credit"],
                 "license": roster[name]["license"], "page": roster[name]["commons"]} for name in PEOPLE]
    return {"schema": SCHEMA, "sources": selected, "offsets": list(OFFSETS), "gate": GATE,
            "method_sha256": sha(method), "baseline_sha256": BASELINE_SHA256, "sources_sha256": sha(parent),
            "method": "For two copies of the same clip, compare the current select-then-check rule with applying "
                      "the 0.84 flank floor before ranking, and with a lower 0.52 floor plus a local timing alternative. "
                      "Each arm still yields at most one range after the existing short timing check."}


def overlap(a, b):
    return 2 * (min(a[1], b[1]) - max(a[0], b[0])) >= max(a[1], b[1]) - min(a[0], b[0])


def review(receipt):
    freeze = receipt["freeze"]
    require(freeze == plan(), "The frozen comparison changed")
    cases = [(scene, carrier, role) for scene in SCENES for carrier in CARRIERS for role in ROLES]
    expected = [(group, source, offset) for group, source in enumerate(freeze["sources"], 1) for offset in OFFSETS]
    totals, comparisons, gains = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for (group, source, offset), row in zip(expected, receipt["rows"], strict=True):
        require((row["group"], row["offset"], row["source_sha256"]) == (group, offset, source["sha256"])
                and [(c["scene"], c["carrier"], c["role"]) for c in row["cases"]] == cases, "Incomplete matched roster")
        for case in row["cases"]:
            scene, carrier, role = (case[k] for k in ("scene", "carrier", "role"))
            truth = boundary(group, offset, OFFSETS) if role in EDITS else None
            require(case["truth"] == truth and set(case["arms"]) == set(ARMS)
                    and all(len(case[k]) == 64 and all(ch in "0123456789abcdef" for ch in case[k])
                            for k in ("earlier_pcm_sha256", "submitted_pcm_sha256")), "Source commitment or ground truth changed")
            success, flags = {}, {}
            cell = totals[carrier, scene]
            cell["edits" if truth else "controls"] += 1
            for arm, reading in case["arms"].items():
                if reading:
                    a, b = reading["samples"]
                    floor = .52 if arm == "check_for_delay" else .84
                    require(case["timing"] is None and 0 < a < b < 64000 and b - a in (5120, 7680, 10240, 12800)
                            and 2 <= len(reading["bands"]) <= 3 and math.isfinite(reading["strength"])
                            and sum(min(x["correlation"][0], x["correlation"][2]) >= floor for x in reading["bands"]) >= 2,
                            "A proposed range lacks flank support or conflicts with the timing reading")
                success[arm] = bool(reading and truth and overlap(reading["samples"], truth))
                flags[arm] = bool(reading)
                cell[f"{arm}_useful"] += success[arm]
                cell[f"{arm}_wrong_location"] += bool(truth) and flags[arm] and not success[arm]
                cell[f"{arm}_control_flags"] += truth is None and flags[arm]
            for arm in ARMS[1:]:
                outcome = comparisons[arm, carrier]
                added = success[arm] and not success["current"]
                outcome["added"] += added
                outcome["lost"] += success["current"] and not success[arm]
                outcome["added_noisy"] += added and scene != "clean"
                flag = truth is None and flags[arm] and not flags["current"]
                outcome["new_flags"] += flag
                outcome[f"new_flags_{role}"] += flag
                outcome["priority_flags"] += flag and role in ("constant_shift", "unrelated")
                if added:
                    gains[arm, carrier].add(group)
    outcomes = {}
    for key, counts in comparisons.items():
        groups = len(gains[key])
        advancement = (counts["added"] >= GATE["useful_gain"] and not counts["lost"] and groups >= GATE["distinct_groups"]
                       and counts["new_flags"] <= GATE["max_added_control_flags_per_carrier"]
                       and counts["priority_flags"] <= GATE["max_priority_control_flags"])
        outcomes["/".join(key)] = dict(counts) | {"source_groups": groups, "advances": advancement}
    return {"by_scene": {"/".join(k): dict(v) for k, v in sorted(totals.items())}, "comparison": outcomes}


def score(freeze, bank, sounds, output):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.evidence import video_reference_sound_difference as baseline
    from app.evidence import video_reference_soundtrack as timing
    from scripts import video_sound_difference_guard as guard
    from scripts import video_soundtrack_cadence as media
    from scripts.audio_reference_variable_window import read_noise

    noise = read_noise(sounds)
    clips = {}
    offsets = freeze["offsets"]
    with TemporaryDirectory(prefix="sound-difference-trial-") as tmp:
        root = Path(tmp)
        for group, source in enumerate(freeze["sources"], 1):
            path = bank / source["file"]
            require(sha(path) == source["sha256"], "Publisher file differs from its pinned digest")
            for offset in offsets:
                clip = root / f"source-{group}-{offset}.mov"
                media.parent.execute("ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(path),
                    "-map", "0:v:0", "-map", "0:a:0", "-t", "4", "-vf",
                    "fps=25,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", "-threads", "1",
                    "-video_track_timescale", "16000", "-af", "aresample=16000:async=1:first_pts=0,apad=whole_dur=4",
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(clip))
                samples = media._sound(clip)[0][:4 * RATE]
                require(len(samples) == 4 * RATE, "A frozen publisher excerpt lacks continuous audio")
                clips[group, offset] = (clip, samples)
        rows = []
        for group, item in enumerate(freeze["sources"], 1):
            for offset, donor_offset in zip(offsets, reversed(offsets), strict=True):
                clip, original = clips[group, offset]
                donor = clips[group % len(freeze["sources"]) + 1, donor_offset][1]
                same = clips[group, donor_offset][1]
                background = {name: beds[(group + offsets.index(offset)) % len(beds)][:len(original)] / 32768.
                              for name, beds in noise.items()} | {"competing": donor, "clean": np.zeros_like(original)}
                a, b = boundary(group, offset, offsets)
                fade = np.ones(b - a)
                fade[:120] = np.arange(120) / 120
                fade[-120:] = fade[:120][::-1]
                base = original[a:b]

                def replace(edited, *, original=original, a=a, b=b, fade=fade, base=base):
                    result = original.copy()
                    result[a:b] += fade * (edited[a:b] - base)
                    return result

                def adjust(edited, *, base=base, a=a, b=b):
                    return edited * np.std(base) / max(.0001, np.std(edited[a:b]))

                locally_filtered = sosfiltfilt(butter(4, 1100, fs=RATE, output="sos"), original)
                shifted = media.parent.shift(original, 3200)
                changes = {"same_speaker": replace(adjust(same)), "other_speaker": replace(adjust(donor)),
                           "unchanged": original, "local_gain": replace(.35 * original), "local_filter": replace(locally_filtered),
                           "global_gain": .55 * original, "constant_shift": shifted, "local_shift": replace(shifted),
                           "background_only": original, "unrelated": donor}
                cases = []
                for scene in SCENES:
                    bed = background[scene]
                    bed = bed * (10 ** .25 * np.std(original) / max(.0001, np.std(bed)))
                    # The reference stays clean. Bound both deliveries without clipping.
                    scale = min(1., .985 / max(float(np.max(np.abs(change + bed))) for change in changes.values()))
                    for carrier in CARRIERS:
                        suffix = "mp4" if carrier == "AAC" else "mov"
                        earlier = root / f"earlier.{suffix}"
                        earlier.unlink(missing_ok=True)
                        media._remux_wave(clip, original * scale, earlier, carrier)
                        first = media._sound(earlier)[0][:4 * RATE]
                        for role, change in changes.items():
                            submitted = root / f"submitted.{suffix}"
                            submitted.unlink(missing_ok=True)
                            if role == "background_only":
                                change = replace(original + bed)
                                mixture = change
                            else:
                                mixture = change + bed
                            media._remux_wave(clip, mixture * scale, submitted, carrier)
                            second = media._sound(submitted)[0][:4 * RATE]
                            truth = [a, b] if role in EDITS else None
                            shift, matches = timing.measure(first, second, window=timing.SHORT_WINDOW,
                                                            step=timing.SHORT_STEP, min_rms=timing.QUIET_RMS)
                            shifted_samples = shift["submitted_samples"] if shift else None
                            first_pcm, second_pcm = ((audio * 32768).astype("<i2") for audio in (first, second))
                            result = dict.fromkeys(ARMS) if shift else guard.inspect(first_pcm, second_pcm)
                            incumbent = None
                            if shift is None:
                                _, best = baseline.measure(first_pcm, second_pcm)
                                incumbent = best if best and baseline.strong_flanks(best) else None
                            require(result["current"] == incumbent, "The runner did not reproduce the incumbent")
                            cases.append({"scene": scene, "carrier": carrier, "role": role, "truth": truth,
                                          "earlier_pcm_sha256": digest(first_pcm.tobytes()), "submitted_pcm_sha256": digest(second_pcm.tobytes()),
                                          "timing": shifted_samples, "timing_support": matches, "arms": result})
                rows.append({"group": group, "offset": offset, "source_sha256": item["sha256"], "cases": cases})
                print(f"Completed {len(rows)}/{len(clips)} source excerpts", flush=True)
    receipt = {"freeze": freeze, "rows": rows}
    receipt["summary"] = review(receipt)
    save(output, receipt)
    print(json.dumps(receipt["summary"]["comparison"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.output)
        report = review(receipt)
        require(report == receipt["summary"], "Saved outcomes do not replay")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    implementation = Path(__file__).resolve().parents[1] / "app/evidence/video_reference_sound_difference.py"
    lock = {"plan": plan(), "runner_sha256": sha(Path(__file__)), "implementation_sha256": sha(implementation)}
    if args.output.exists():
        parser.error("Use a new path")
    if args.phase == "freeze":
        save(args.output, lock)
    else:
        if not (args.freeze and args.bank and args.sounds):
            parser.error("Supply the frozen design, publisher recordings and noise recordings")
        require(lock == read(args.freeze), "Code or input plan changed after freeze")
        score(lock["plan"], args.bank, args.sounds, args.output)


if __name__ == "__main__":
    main()
