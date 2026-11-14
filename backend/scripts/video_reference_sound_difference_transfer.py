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

SCHEMA = "chipmunk.video-reference-sound-difference-transfer/1"
OFFSETS = (45, 80)
RATE = 16000
CARRIERS = ("PCM", "AAC")
SCENES = ("clean", "rain", "train", "competing")
ROLES = ("same_speaker", "other_speaker", "unchanged", "local_gain", "local_filter", "global_gain", "constant_shift",
         "local_shift", "background_only", "unrelated")
EDITS = ROLES[:2]
GATE = {"minimum_useful_aac": 36, "max_ordinary_flags_per_carrier": 5, "max_constant_or_unrelated_flags": 0}
DEVELOPMENT_METHOD = "707b1e686b88201ccc0739eae69d865f04d1856300b0ff0a686d6cd947234f0a"


def boundary(group, offset, offsets):
    width = (8, 12, 16, 20)[(group + offsets.index(offset)) % 4] * 640
    start = (34 + (group % 4) * 3) * 640
    return [start, start + width]


def plan(transfer=False):
    method = Path(__file__).resolve().parents[1] / "app/evidence/video_reference_sound_difference.py"
    comparison = read(sources.ROOT / "docs/evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json")
    roster = [row | {"file": row["id"] + ".webm"} for row in comparison["sources"]] if transfer else sources.sources()
    gate = {"clean_aac_locations": 8, "minimum_source_groups": 3, "max_ordinary_flags_per_carrier": 1,
            "max_constant_or_unrelated_flags": 0} if transfer else GATE
    return {"schema": SCHEMA, "sources": roster, "offsets": [31, 46] if transfer else list(OFFSETS), "gate": gate,
            "method_sha256": sha(method) if transfer else DEVELOPMENT_METHOD,
            "method": "Compare paired windows of the two soundtracks on the same picture clock. "
              "Use the existing signed timing method first. Count acoustic loss in the absence of an explainable local delay. "
              "Require temporal IoU of 0.50 with the authored substitution. No model or transcription is used."} | (
            {"transfer": True, "clean_flank_correlation": .84} if transfer else {})


def overlap(a, b):
    return 2 * (min(a[1], b[1]) - max(a[0], b[0])) >= max(a[1], b[1]) - min(a[0], b[0])


def review(receipt):
    freeze = receipt["freeze"]
    transfer = freeze.get("transfer", False)
    require(freeze == plan(transfer), "The frozen plan changed")
    offsets = freeze["offsets"]
    cases = [(s, c, r) for s in SCENES for c in CARRIERS for r in ROLES]
    expected = [(group, source, offset) for group, source in enumerate(freeze["sources"], 1) for offset in offsets]
    totals = defaultdict(Counter)
    by_role = defaultdict(Counter)
    recovered_groups = set()
    for (group, source, offset), row in zip(expected, receipt["rows"], strict=True):
        require(row["group"] == group and row["offset"] == offset and row["source_sha256"] == source["sha256"]
                and [(c["scene"], c["carrier"], c["role"]) for c in row["cases"]] == cases,
                "The frozen source or condition roster differs")
        for case in row["cases"]:
            scene, carrier, role = (case[k] for k in ("scene", "carrier", "role"))
            truth = boundary(group, offset, offsets) if role in EDITS else None
            kind = "edits" if truth else "controls"
            status, best, shift = (case[k] for k in ("status", "difference", "timing"))
            require(case["truth"] == truth and all(len(case[k]) == 64 for k in ("earlier_pcm_sha256", "submitted_pcm_sha256")),
                    "Pair commitments or edit range changed")
            require(status in {"no_local_difference", "matching_flanks_unavailable", "localized_difference", "explained_by_timing"}
                    and bool(best) == (status == "localized_difference") and (not shift or status == "explained_by_timing"),
                    "Local disposition and evidence disagree")
            if best:
                a, b = best["samples"]
                require(0 < a < b < 4 * RATE and b - a in (5120, 7680, 10240, 12800) and math.isfinite(best["strength"])
                        and 2 <= len(best["bands"]) <= 3, "Invalid measured acoustic support")
            cell = totals[carrier, scene]
            group_cell = by_role[carrier, role]
            good = bool(best and truth and overlap(best["samples"], truth))
            if good and carrier == "AAC" and scene == "clean":
                recovered_groups.add(group)
            good_prior = bool(shift and truth and overlap(shift, truth))
            cell.update({kind: 1, f"prior_assessed_{kind}": case["timing_support"] >= 8,
                         f"candidate_assessed_{kind}": status != "matching_flanks_unavailable"})
            cell.update(prior_useful=good_prior, candidate_useful=good, added_useful=good and not good_prior,
                        misplaced=bool(best and truth and not good), ordinary_flags=bool(best and not truth),
                        priority_control_flags=bool(best and role in ("constant_shift", "unrelated")))
            group_cell.update(files=1, flagged=bool(best), useful=good, timing=bool(shift))
    gate = freeze["gate"]
    if transfer:
        accepts = (totals["AAC", "clean"]["candidate_useful"] >= gate["clean_aac_locations"]
                   and len(recovered_groups) >= gate["minimum_source_groups"])
    else:
        accepts = sum(c["candidate_useful"] for (carrier, _), c in totals.items() if carrier == "AAC") >= gate["minimum_useful_aac"]
    accepts &= all(sum(c["ordinary_flags"] for (fmt, _), c in totals.items() if fmt == carrier)
                   <= gate["max_ordinary_flags_per_carrier"] for carrier in CARRIERS)
    accepts &= sum(c["priority_control_flags"] for c in totals.values()) <= gate["max_constant_or_unrelated_flags"]
    return {"results": {"/".join(k): dict(v) for k, v in sorted(totals.items())},
            "roles": {"/".join(k): dict(v) for k, v in sorted(by_role.items())}, "passes": bool(accepts)} | (
                {"clean_aac_source_groups": len(recovered_groups)} if transfer else {})


def score(freeze, bank, sounds, output):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.evidence import video_reference_sound_difference as difference
    from app.evidence import video_reference_soundtrack as timing
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
                    "-video_track_timescale", "16000", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(clip))
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
                            status, selection = (("explained_by_timing", None) if shift else difference.measure(first_pcm, second_pcm))
                            if selection is not None and freeze.get("transfer") and not difference.strong_flanks(
                                selection, freeze["clean_flank_correlation"]
                            ):
                                status, selection = "no_local_difference", None
                            cases.append({"scene": scene, "carrier": carrier, "role": role, "truth": truth,
                                          "earlier_pcm_sha256": digest(first_pcm.tobytes()), "submitted_pcm_sha256": digest(second_pcm.tobytes()),
                                          "timing": shifted_samples, "timing_support": matches, "status": status, "difference": selection})
                rows.append({"group": group, "offset": offset, "source_sha256": item["sha256"], "cases": cases})
                print(f"Completed {len(rows)}/{len(clips)} source excerpts", flush=True)
    receipt = {"freeze": freeze, "rows": rows}
    receipt["summary"] = review(receipt)
    save(output, receipt)
    print(json.dumps(receipt["summary"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--transfer", action="store_true")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.output)
        report = review(receipt)
        require(report == receipt["summary"], "Saved outcomes do not replay")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    lock = {"plan": plan(args.transfer), "runner_sha256": sha(Path(__file__))}
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
