# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay an independent-speaker test of a compact timing model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.video_short_soundtrack_transfer import sha, sources

ROOT = Path(__file__).resolve().parents[2]
PINNED = ROOT / "docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json"
ROLES = ("original", "gain", "lowpass", "noise", "global_early", "local_early", "global_late", "local_late",
         "matched_picture_early", "matched_picture_late")
SCHEMA = "chipmunk.video-local-alignment-transfer/1"
GATE = {"added_locations": 4, "sources_with_gain": 3, "max_added_ordinary_flags": 2,
        "max_added_wrong_edit_ranges": 2, "max_matched_or_global_flags": 0}
FIT = ("diego", "ioanna", "jan", "mckensie", "piotr", "regina", "simon")


def plan():
    rows = json.loads(PINNED.read_bytes())["source_plan"]["rows"]
    local = [{"file": row["name"] + ".webm", "sha256": row["source_sha256"], "page": row["commons"],
              "license": row["license"], "attribution": row["credit"], "split": "fit" if row["name"] in FIT else "point",
              "offsets": [row["offset_s"] + x for x in ((2, 8, 14, 20) if row["name"] in FIT else (6, 13))]}
             for row in rows]
    held = [item | {"split": "held", "offsets": [72, 86]} for item in sources()]
    return {"schema": SCHEMA, "sources": local + held, "roles": list(ROLES), "gate": GATE,
            "method": "Balanced logistic ranking with bilinear features from 12 whitened components in each modality, "
                      "five-tick local input and six nearby negative lags, trained on aligned fit excerpts only. "
                      "Point is the strongest ordinary result on four other people plus 0.03. "
                      "Positive requires 0.40 temporal IoU and signed offset within 80 ms."}


def useful(span, signed, truth, lag):
    a, b = span
    c, d = truth
    return signed is not None and abs(signed - lag) <= 2 and 5 * (min(b, d) - max(a, c)) >= 2 * (max(b, d) - min(a, c))


def point(rows):
    return round(max((case["score"]["score"] for row in rows for case in row.get("cases", [])
                      if case["score"] is not None and case["truth"] is None), default=0.) + .03, 6)


def review(doc):
    freeze = plan()
    if doc["freeze"] != freeze:
        raise ValueError("Evaluation design differs from the published plan")
    expected = [(s["file"], group, offset) for group, s in enumerate(freeze["sources"], 1)
                if s["split"] != "fit" for offset in s["offsets"]]
    if [(row["file"], row["group"], row["offset"]) for row in doc["rows"]] != expected:
        raise ValueError("The receipt does not cover all frozen excerpts")
    roles = {source["file"]: source["split"] for source in freeze["sources"]}
    chosen = point([row for row in doc["rows"] if roles[row["file"]] == "point"])
    if doc["point"] != chosen:
        raise ValueError("The operating point differs from the non-held controls")
    result, gains = Counter(), set()
    for row in doc["rows"]:
        if row.get("unavailable") is None and [x["role"] for x in row["cases"]] != list(ROLES):
            raise ValueError("A matched control or edit is absent")
        held = roles[row["file"]] == "held"
        if held:
            result.update(excerpts=1, edits=2, controls=8)
        for case in row.get("cases", []):
            proposal = case["score"]
            truth, lag = case["truth"], case["lag"]
            if truth is not None:
                start = (row["ticks"] - 16) // 2 + row["group"] % 5 - 2
                if (truth != [start, start + (16, 22)[row["group"] % 2]] or not case["role"].startswith("local_")
                        or lag != (5, 7)[row["group"] % 2] * (-1 if case["role"].endswith("early") else 1)):
                    raise ValueError("Edited interval differs from the frozen clock")
            elif case["role"].startswith("local_"):
                raise ValueError("Missing edit truth")
            if proposal:
                a, b = proposal["ticks"]
                ref, moved = proposal["reference_lag"], proposal["changed_lag"]
                if (not case["brackets"] or not 13 <= a < b <= row["ticks"] - 13 or b - a not in (14, 16, 18, 20, 22)
                        or abs(ref) > 4 or abs(moved) > 9 or not 4 <= abs(moved - ref) <= 8
                        or len(proposal["gains"]) != 3 or not all(math.isfinite(x) and x > 0 for x in proposal["gains"])
                        or proposal["score"] != min(proposal["gains"])):
                    raise ValueError("Reported timing hypothesis violates its search rules")
            if not held:
                continue
            flagged = proposal is not None and proposal["score"] >= chosen
            old = case["current"]["ranges"]
            result["current_measured"] += bool(case["current"]["starts"])
            result["candidate_measured"] += bool(case["brackets"])
            if truth is not None:
                good_old = any(useful((a / 2, b / 2), offset, truth, lag) for _, a, b, offset in old)
                good_new = proposal is not None and useful(proposal["ticks"], proposal["changed_lag"] - proposal["reference_lag"], truth, lag)
                result["current_useful"] += good_old
                result["candidate_useful_before_point"] += good_new
                result["candidate_useful"] += flagged and good_new
                result["added_useful"] += flagged and good_new and not good_old
                result["candidate_wrong"] += flagged and not good_new
                result["added_wrong"] += flagged and not good_new and not old
                if flagged and good_new and not good_old:
                    gains.add(row["file"])
            else:
                result["current_control_flags"] += bool(old)
                result["candidate_control_flags"] += flagged
                result["added_control_flags"] += flagged and not old
                if flagged and case["role"].startswith(("global_", "matched_picture_")):
                    result["global_or_matched_flags"] += 1
    passes = (result["added_useful"] >= GATE["added_locations"] and len(gains) >= GATE["sources_with_gain"]
              and result["added_control_flags"] <= GATE["max_added_ordinary_flags"]
              and result["added_wrong"] <= GATE["max_added_wrong_edit_ranges"]
              and result["global_or_matched_flags"] <= GATE["max_matched_or_global_flags"])
    return dict(result) | {"sources_with_gain": len(gains), "passes": passes}


def run(cache, held, output):
    import subprocess
    from dataclasses import replace
    from tempfile import TemporaryDirectory

    import cv2
    import numpy as np
    import sklearn

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import lipsync, xcorr
    from scripts import video_lipsync_signed_transitions as prior
    from scripts import video_lipsync_window_schedule as native
    from scripts import video_local_alignment_model as model

    freeze = plan()
    runtime = {"numpy": np.__version__, "opencv": cv2.__version__, "sklearn": sklearn.__version__,
               "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
    bindings = {Path(p).name: sha(Path(p)) for p in (__file__, model.__file__, native.__file__, native.FACE_FILE, prior.__file__)}
    if bindings[native.FACE_FILE.name] != native.FACE_SHA:
        raise ValueError("Face weights differ from the pinned license-reviewed asset")
    for source in freeze["sources"]:
        folder = held if source["split"] == "held" else cache
        if sha(folder / source["file"]) != source["sha256"]:
            raise ValueError("Original source differs from its published commitment")
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "freeze.json").write_text(json.dumps({"plan": freeze, "bindings": bindings, "runtime": runtime}, indent=2) + "\n")
    cv2.setNumThreads(1)
    detector = load_yunet(native.FACE_FILE)
    fitted = None
    observations, results, qualified = [], [], []
    with TemporaryDirectory(prefix="av-alignment-") as tmp:
        for split in ("fit", "point", "held"):
            folder = held if split == "held" else cache
            if split == "held":
                threshold = point(results)
                (output / "point.json").write_text(json.dumps({"point": threshold, "fit_observations": fitted[2], "rows": results}, indent=2) + "\n")
            for group, item in enumerate(freeze["sources"], 1):
                if item["split"] != split:
                    continue
                for offset in item["offsets"]:
                    row = {"file": item["file"], "offset": offset, "group": group}
                    try:
                        grid, table, origin, pcm, _, faces, patches, frames = native._observe(
                            item | {"offset_s": offset, "bank": "conversation"}, cache, folder, detector, Path(tmp))
                    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        row["unavailable"] = str(error).split("\n", 1)[0] if isinstance(error, ValueError) else type(error).__name__
                    else:
                        row.update(ticks=grid.ticks, source_audio_start=str(origin), picture_time_base=table.time_base,
                                   picture_count=len(frames), mouth_ticks=len(patches))
                        if grid.ticks < 75:
                            row["unavailable"] = "short_clock"
                        elif split == "fit":
                            observations.append((patches, grid, faces, pcm))
                        else:
                            cases = []
                            interventions = {role: (wave, truth, lag, indices)
                                             for role, wave, truth, lag, indices in prior.variants(pcm, grid, group)}
                            for role in ROLES:
                                wave, truth, lag, indices = interventions[role]
                                updated = replace(grid, frames=tuple(grid.frames[i] for i in indices))
                                mouth = {i: patches[j] for i, j in enumerate(indices) if j in patches}
                                vad = lipsync.vad(wave.astype(np.float32) / 32768.)
                                count, best = model.locate(model.profile(fitted, mouth, updated.frames, wave), vad.speech)
                                aperture = {tick: xcorr.aperture(patch) for tick, patch in mouth.items()}
                                cases.append({"role": role, "truth": [i // 640 for i in truth] if truth else None, "lag": lag,
                                              "pcm_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                                              "current": prior.current(faces, mouth, aperture, vad, updated), "score": best, "brackets": count})
                            row["cases"] = cases
                    (qualified if split == "fit" else results).append(row)
                    print(split, group, offset, row.get("unavailable", row.get("mouth_ticks")), flush=True)
            if split == "fit":
                fitted = model.fit(observations)
                print("Fitted aligned ticks", fitted[2], flush=True)
    doc = {"freeze": freeze, "runtime": runtime, "bindings": bindings, "training": qualified,
           "fit_observations": fitted[2], "point": threshold, "rows": results}
    doc["results"] = review(doc)
    (output / "receipt.json").write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(doc["results"], flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("run", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("--cache", type=Path)
    cli.add_argument("--held", type=Path)
    args = cli.parse_args()
    if args.phase == "run":
        if not args.held or not args.cache:
            cli.error("Native replay needs both pinned source directories")
        run(args.cache, args.held, args.path)
    else:
        doc = json.loads(args.path.read_bytes())
        totals = review(doc)
        if totals != doc["results"]:
            raise ValueError("Evaluation results do not recount")
        print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
