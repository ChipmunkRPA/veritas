"""Compare kept and removed stereo speech before changing a voice decision."""

from __future__ import annotations

import argparse
import math
import resource
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.detectors.audio import calibration
from scripts import audio_paired_nuisance_transfer as binding
from scripts import audio_spatial_interference as spatial

require, read, write, digest = binding.require, binding.read, binding.write, binding.digest
PANELS = {"selection": "archived", "confirmation": "transfer"}
ARMS = ("current", "projection_only", "removed_source")
FLOOR = math.log(2)
MINIMUM_RATIO = .1
METHOD = {
    "source": "Preserve the existing submitted-mono and centered-projection readouts. Measure the exact component subtracted by that projection.",
    "support": "Require distinct channels, admitted raw and projected scores, removed-to-mono RMS ratio at least 0.10, and speech admission on the removed view. "
               "Normalize the removed view towards mono RMS, capped at 6x amplification and a PCM peak of 31,129.",
    "rescue": "If the incumbent is inconclusive, the projected view leans AI, and its raw log odds beat the removed view by more than the rescue point, flag AI.",
    "suppress": "If the incumbent leans AI, the removed view leans AI, and its raw log odds beat the projected view by more than the suppress point, withhold to inconclusive.",
    "points": "Use the higher of ln(2) and the strongest unsafe eligible transition in the independent selection room cohort for each action. "
              "Protect genuine sources from new AI and synthetic sources from lost AI. Use strict excess.",
    "reference": "The projection-only comparator replays the historical raw/conditioned decisions and ln(2) drop requirement.",
    "scope": "Stereo centered target only. Offset and untouched room controls quantify stress and target ambiguity. No unassessed or original direction can change.",
    "advance": "On centered mixtures recover at least two synthetic flags and clear at least two genuine false flags, retain current synthetic count, "
               "add at most one genuine false flag and lose at most one true flag, with no new false AI flags on clean or native controls. "
               "No more than one additional offset false flag or lost true flag.",
}


def evidence(args):
    pub = binding.PANEL / "audio-spatial-interference-2026-11-11.json"
    source_lock, rows, old_rows = spatial.bound(args.parent, args.archived)
    score_path = args.parent / "scores.json"
    source_result = read(score_path)
    public = read(pub)
    require(digest(args.parent / "freeze.json") == public["freeze_sha256"] == source_result["freeze_sha256"]
            and digest(score_path) == public["scores_sha256"], "Exact earlier centered-projection and raw response bindings required")
    require(digest(calibration.PATH) == source_lock["calibration_sha256"] == calibration.SHA256,
            "Pinned incumbent calibration changed")
    old_lock = read(args.archived / "freeze.json")
    require(not set(old_lock["rooms"]) & set(source_lock["publisher_mics"]), "Overlapping captured rooms")
    clean = [{row["group"]: row["mono_sha256"] for row in rows if row["role"] == value and row["placement"] == "clean"}
             for value in PANELS.values()]
    require([len(x) for x in clean] == [17, 16] and not set(clean[0]) & set(clean[1])
            and not set(clean[0].values()) & set(clean[1].values()), "Exact clean foreground roles overlap")
    scores = source_result["rows"]
    require(len(rows) == len(scores) == 217 and all((r["id"], r["mono_sha256"]) == (s["id"], s["mono_sha256"])
            for r, s in zip(rows, scores, strict=True)), "Exact delivery or baseline mismatch")
    lock = {"schema": "chipmunk.audio-removed-source/1", "method": METHOD, "code_sha256": digest(Path(__file__)),
            "source_freeze_sha256": public["freeze_sha256"], "source_scores_sha256": public["scores_sha256"],
            "source_public_receipt_sha256": digest(pub), "calibration_sha256": calibration.SHA256,
            "assets": source_lock["incumbent_assets"], "views": {k: sum(row["role"] == v for row in rows) for k, v in PANELS.items()},
            "source_groups": [len(x) for x in clean], "recorded_rooms_by_panel": [2, 2]}
    return lock, list(zip(rows, scores, strict=True)), old_rows


def normalize(values, mono):
    norm = np.linalg.norm(values)
    total = np.linalg.norm(mono)
    if norm == 0 or total == 0 or norm / total < MINIMUM_RATIO:
        return None, 0. if total == 0 else float(norm / total)
    gain = min(6., total / norm, 31_129 / np.max(np.abs(values)))
    return spatial.pcm_bytes(gain * values), float(norm / total)


def residual_view(args, row, reference, old_rows):
    stereo, mono = spatial.audio_bytes(args.parent, args.archived, row, old_rows)
    if reference["treatment"] != "conditioned" or reference["conditioned"]["raw"] is None:
        return None, None
    projected = spatial.apply(stereo, spatial.fit(stereo))
    require(spatial.sha(spatial.pcm_bytes(projected)) == reference["conditioned_sha256"], "Earlier projected waveform changed")
    return normalize(stereo.mean(axis=1) - projected, mono)


def score(args, panel, lock, rows, old_rows):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    spatial.parent.INCUMBENT.network_free()
    model = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    detector, bundle = PellaVoiceDetector(), calibration.load()
    started, output = time.monotonic(), []
    for row, reference in rows:
        if row["role"] != PANELS[panel]:
            continue
        residual, ratio = residual_view(args, row, reference, old_rows)
        value, state = None, "unassessed"
        if residual is not None:
            audio = np.frombuffer(residual, "<i2").astype(np.float32) / 32768
            ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: SimpleNamespace(obj=activity)),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", audio, spatial.COUNT)))
            windows = detector._windows(ctx)
            if windows:
                require(len(windows) == 1 and windows[0][0] == 0 and np.array_equal(windows[0][1], audio), "Removed-view window differs")
                value, = model.predict([audio])
                require(calibration.valid(value), "Invalid removed-component response")
                state = calibration.initial_decision(calibration.probability(value, bundle), bundle)
        output.append({"id": row["id"], "input_sha256": row["mono_sha256"], "removed_pcm_sha256": spatial.sha(residual) if residual else None,
                       "removed_rms_fraction": ratio, "removed_score": value, "removed_direction": state})
        if len(output) % 20 == 0:
            print(panel, len(output), "delivered views", flush=True)
    write(args.output / f"{panel}.json", {"freeze_sha256": digest(args.output / "freeze.json"), "rows": output,
          "forwards": sum(r["removed_score"] is not None for r in output), "activity_forwards": gate.calls,
          "elapsed_s": time.monotonic() - started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def panel_data(args, panel, rows):
    path = args.output / f"{panel}.json"
    scored = read(path)
    require(scored["freeze_sha256"] == digest(args.output / "freeze.json"), "Removed-view freeze changed")
    output = []
    for (row, reference), obs in zip((pair for pair in rows if pair[0]["role"] == PANELS[panel]), scored["rows"], strict=True):
        require((row["id"], row["mono_sha256"]) == (obs["id"], obs["input_sha256"]), "Removed-view observation does not bind delivered audio")
        output.append((row, reference, obs))
    return output


def margins(reference, observation):
    if observation["removed_score"] is None:
        return None, None
    baseline, target = reference["baseline"], reference["conditioned"]
    contrast = calibration.logit(target["raw"]) - calibration.logit(observation["removed_score"])
    rescue = contrast if baseline["decision"] == "inconclusive" and target["decision"] == "ai" else None
    suppress = -contrast if baseline["decision"] == observation["removed_direction"] == "ai" else None
    return rescue, suppress


def fit(data):
    support = [[], []]
    for row, ref, obs in data:
        rescue, suppress = margins(ref, obs)
        index = int(row["family"] != "genuine")
        value = (rescue, suppress)[index]
        if value is not None:
            support[index].append((value, row["group"]))
    return {key: {"point": max([FLOOR, *(value for value, _ in eligible)]), "protected_views": len(eligible),
                  "protected_groups": len({group for _, group in eligible})}
            for key, eligible in zip(("rescue", "suppress"), support, strict=True)}


def outcomes(ref, obs, points):
    baseline, target = ref["baseline"], ref["conditioned"]
    current = baseline["decision"]
    projection = current
    if current == "inconclusive" and target["decision"] == "ai":
        projection = "ai"
    elif current == "ai" and target["raw"] is not None and target["decision"] != "ai" and (
            calibration.logit(baseline["raw"]) - calibration.logit(target["raw"]) > FLOOR):
        projection = "inconclusive"
    source = current
    rescue, suppress = margins(ref, obs)
    if rescue is not None and rescue > points["rescue"]["point"]:
        source = "ai"
    elif suppress is not None and suppress > points["suppress"]["point"]:
        source = "inconclusive"
    return dict(zip(ARMS, (current, projection, source), strict=True))


def result(data, points):
    cells, subgroups = {}, {}
    rooms = {room: f"room_{i}" for i, room in enumerate(sorted({r["room"] for r, _, _ in data if r["room"]}))}
    for row, ref, obs in data:
        placement, mode = row["placement"], row["mode"] or "none"
        family = "genuine" if row["family"] == "genuine" else "synthetic"
        room = rooms.get(row["room"], "none")
        section = cells.setdefault(placement, Counter())
        local = subgroups.setdefault(f"{placement}/{mode}/{room}/{row['family']}", Counter())
        for bucket in (section, local):
            bucket[f"{family}_planned"] += 1
            bucket["assessed_removed"] += obs["removed_score"] is not None
            current = ref["baseline"]["decision"]
            for arm, state in outcomes(ref, obs, points).items():
                bucket[f"{family}_{arm}_{state}"] += 1
                bucket[f"{family}_{arm}_added"] += current != "ai" and state == "ai"
                bucket[f"{family}_{arm}_lost"] += current == "ai" and state != "ai"
    return {"cells": dict(cells), "room_mode_family": dict(subgroups)}


def qualification(held):
    center, offset = (held["cells"][key] for key in ("center", "offset"))
    return {"center_true_gains": center["synthetic_removed_source_added"] >= 2,
            "center_false_flags_cleared": center["genuine_removed_source_lost"] >= 2,
            "center_limits": center["genuine_removed_source_added"] <= 1 and center["synthetic_removed_source_lost"] <= 1
            and center["synthetic_removed_source_ai"] >= center["synthetic_current_ai"],
            "native_and_clean_no_added_false": all(held["cells"][cell]["genuine_removed_source_added"] == 0 for cell in ("clean", "native")),
            "offset_limits": offset["genuine_removed_source_added"] <= 1 and offset["synthetic_removed_source_lost"] <= 1}


def run(args):
    lock, rows, old_rows = evidence(args)
    path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(path, lock)
        return
    require(read(path) == lock, "Study inputs or algorithm changed after freeze")
    if args.stage == "score":
        if args.panel == "confirmation":
            points = read(args.output / "points.json")
            require(points["selection_sha256"] == digest(args.output / "selection.json") and points["freeze_sha256"] == digest(path),
                    "The selection-derived points must be frozen before held scoring")
        score(args, args.panel, lock, rows, old_rows)
    elif args.stage == "fit":
        write(args.output / "points.json", {"points": fit(panel_data(args, "selection", rows)), "freeze_sha256": digest(path),
                                            "selection_sha256": digest(args.output / "selection.json")})
    else:
        point_path = args.output / "points.json"
        points = read(point_path)
        require(points["freeze_sha256"] == digest(path) and points["selection_sha256"] == digest(args.output / "selection.json"),
                "Frozen points or selection observations changed")
        data = {panel: result(panel_data(args, panel, rows), points["points"]) for panel in PANELS}
        advances = qualification(data["confirmation"])
        write(args.output / "result.json", {"results": data, "advances": all(advances.values()), "qualification": advances,
              "freeze_sha256": digest(path), "points_sha256": digest(point_path),
              "panels_sha256": {panel: digest(args.output / f"{panel}.json") for panel in PANELS}})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "score", "fit", "report"))
    for field in ("output", "parent", "archived", "weights"):
        cli.add_argument(field, type=Path)
    cli.add_argument("--panel", choices=PANELS, default="selection")
    run(cli.parse_args())
