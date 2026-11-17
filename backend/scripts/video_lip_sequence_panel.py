"""Evaluate a local speech-motion comparison on speaker-separated source clips."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.bench.independent_locations import _matches
from scripts import video_lip_articulation_transfer as prior
from scripts import video_lip_correspondence as panel
from scripts.export_video_lip_correspondence import make_soundtracks, waveform

SCHEMA = "chipmunk.video-lip-sequence-panel/1"
ARMS = ("geometry", "mouth_texture")
DELIVERIES = ("pcm", "aac48")
LEVELS = {"strict": 0, "wider": 2}
SEED = 28
STUDY_METHOD = {
    "sources": "Use preassigned original-only fit speakers and calibration speakers. Select previously unused talks for held "
               "pairs. Use the same speaker, condition and rate-matched donor for different or re-recorded same words.",
    "fit": "Match short ordered mouth and log-mel-derived acoustic sequences on the original clock. Train positives and "
           "mismatches using synchronized and shifted observations of the same genuine utterance.",
    "scope": "The local score compares a candidate range with both surrounding ranges at a shared sound-picture offset. "
             "The measured mismatch carries no AI-origin weight.",
    "selection": "Freeze both points using distinct per-speaker calibration maxima and only previously assigned calibration "
                 "locations. Compare paired held clips with their ordinary, same-word and whole-track-delay controls.",
}


def freeze(sources, stereo_path, output):
    from scripts.lombard_mouth_source_intake import index_zip

    with TemporaryDirectory(prefix="lip-seq-plan-") as directory:
        previous = Path(directory) / "previous.json"
        prior.frozen_inputs(sources, previous)
        parent = json.loads(previous.read_bytes())
        parent_sha = panel.sha(previous)
    stereo = json.loads(stereo_path.read_bytes())
    if stereo["retained_plan_sha256"] != parent_sha:
        raise ValueError("The earlier speaker assignments differ from their committed plan")
    assignment = {entry["speaker"]: entry["split"] for entry in parent["rows"]}
    fit_rows = [row for row in parent["rows"] if row["split"] == "fit"]
    excluded = [row[key] for table in (parent, stereo) for row in table["rows"]
                for key in ("target", "same_words", "different_words") if key in row]
    seen_audio = {record["audio_sha256"] for record in excluded}
    seen_picture = {record["front_sha256"] for record in excluded if "front_sha256" in record}
    seen_ids = {record["id"] for record in excluded}
    inventory = defaultdict(dict)
    rows = list(fit_rows)
    with (ZipFile(sources / "publisher-json.zip") as labels, ZipFile(sources / "alignment.zip") as alignment,
          ZipFile(sources / "lombardgrid_front.zip") as pictures, ZipFile(sources / "lombardgrid_audio.zip") as audio):
        eye = index_zip(pictures, ".mov")
        ear = index_zip(audio, ".wav")
        bounds = index_zip(alignment, ".json")
        for metadata in index_zip(labels, ".json").values():
            for row in json.loads(labels.read(metadata)):
                speaker = prior.keyed(row["SPKR"])
                condition = row["COND"]
                uid = f"{row['SPKR']}_{condition}_{row['UTTERANCE']}"
                if (condition not in {"p", "l"} or row["STATUS"] != "CORRECT" or uid in seen_ids
                        or not all(uid in index for index in (eye, ear, bounds))):
                    continue
                record = alignment.read(bounds[uid])
                if span := prior.parent.word_bounds(json.loads(record)[uid]):
                    inventory[speaker, condition][uid] = span, record
        for (speaker, condition), cases in sorted(inventory.items()):
            role = assignment.get(speaker, "held")
            if role == "fit":
                additional = prior.fit_records(cases, audio, pictures, ear, eye, seen_audio, seen_picture)
                rows.extend({"speaker": speaker, "split": role, "condition": condition, "target": source}
                            for source in additional)
            elif record := prior.matched_records(cases, audio, pictures, ear, eye, seen_audio, seen_picture):
                rows.append({"speaker": speaker, "split": role, "condition": condition, **record})
    roster = {"schema": SCHEMA, "method": STUDY_METHOD, "previous_sha256": parent_sha,
              "stereo_roster_sha256": panel.sha(stereo_path), "publisher": parent["publisher"], "rows": rows}
    counts = {role: {"recordings": sum(row["split"] == role for row in rows),
                     "speakers": len({row["speaker"] for row in rows if row["split"] == role})}
              for role in ("fit", "calibration", "held")}
    if counts["held"]["speakers"] < 12 or counts["calibration"]["speakers"] < 6:
        raise ValueError("The fixed speaker partitions do not supply enough matched source groups")
    panel.save(output, roster | {"counts": counts})
    print(json.dumps(counts, sort_keys=True), flush=True)


def _serial(value):
    return {key: array.tolist() for key, array in value.items()} if value is not None else None


def capture(plan, sources, assets, phase, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, wav_bytes
    from app.detectors.video.decode import probe_frames
    from app.evidence.video_reference_timing import _landmarker
    from app.evidence.video_viseme_timing import SCORE_UNITS
    from scripts import video_lip_delivery_transfer as incumbent
    from scripts import video_lip_sequence_features as features
    from scripts.video_lip_crop_correspondence import native_texture

    roster = panel.read(plan)
    if roster["schema"] != SCHEMA or roster["method"] != STUDY_METHOD:
        raise ValueError("The frozen source plan differs from the study method")
    sys.path.insert(0, str((assets / "runtime").resolve()))
    landmarker = assets / "face_landmarker.task"
    if panel.sha(landmarker) != prior.study.parent.FACE_SHA256:
        raise ValueError("The local facial landmark asset differs from the pinned weights")
    reader, device = prior.audio_reader(assets / "checkpoint") if phase != "fit" else (None, "not_needed")
    rows = []
    with (TemporaryDirectory(prefix="lip-sequence-") as directory,
          ZipFile(sources / "lombardgrid_front.zip") as archive, ZipFile(sources / "lombardgrid_audio.zip") as audio,
          _landmarker(landmarker) as (mp, task)):
        picture = Path(directory) / "original.mov"
        wav = Path(directory) / "track.wav"
        carrier = Path(directory) / "track.m4a"
        for work in roster["rows"]:
            if work["split"] != phase:
                continue
            target = work["target"]
            picture.write_bytes(archive.read(target["front"]))
            if panel.sha(picture) != target["front_sha256"]:
                raise ValueError("The publisher picture differs from the source plan")
            original = waveform(audio, target)
            geometry = prior.parent.pictures(picture, len(original), mp, task, with_shape=True)
            frames = native_texture(picture, geometry)
            aligned = {key: work[key] for key in ("speaker", "condition", "split")} | {
                "picture_sha256": target["front_sha256"], "audio_sha256": target["audio_sha256"],
                "pts": [row["pts"] for row in frames], "times": [row["sample"] for row in frames],
                "time_base": geometry["time_base"], "samples": len(original), "truth": target["span"],
                "visual": {arm: _serial(features.visual(frames, texture=arm == "mouth_texture")) for arm in ARMS},
            }
            if phase == "fit":
                aligned["audio"] = _serial(features.acoustic(original))
            else:
                table = probe_frames(picture, cap=120, timeout_s=18)
                geometry["pts"] = aligned["pts"]
                geometry["frames"] = [{k: v for k, v in frame.items() if k not in {"gap", "texture", "sample"}}
                                      | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * SCORE_UNITS)}
                                      for frame in frames]
                trials = make_soundtracks(audio, work)
                aligned["cases"] = {name: {} for name in DELIVERIES}
                for treatment, values in trials.items():
                    wav.write_bytes(wav_bytes(values.astype(np.float32) / 32_768))
                    lossless = decode_audio(wav, source_sha256=panel.sha(wav))
                    aac, mapping, codec = incumbent.qualified_48k(values, carrier)
                    pcm = np.rint(lossless.pcm * 32_768).astype("<i2")
                    if not np.array_equal(values, pcm):
                        raise ValueError("The PCM carrier differs from the known authored samples")
                    for name, track, clock, current, carrier_sha in (
                        ("pcm", pcm, lossless.mapping, "both", panel.sha(wav)),
                        ("aac48", aac, mapping, "articulation", codec["file_sha256"]),
                    ):
                        measured = incumbent.compare(geometry, reader(track), clock, {"width": table.width, "height": table.height})
                        witness = measured[current]["witness"]
                        aligned["cases"][name][treatment] = {"audio": _serial(features.acoustic(track)),
                            "source_pcm_sha256": prior.phone.digest(track.tobytes()), "carrier_sha256": carrier_sha,
                            "current_status": measured[current]["state"],
                            "current_span": witness["audio_source_pts"] if witness else None}
            rows.append(aligned)
            if len(rows) % 8 == 0:
                print(f"{phase} {len(rows)} recordings decoded", flush=True)
    if len(rows) != roster["counts"][phase]["recordings"]:
        raise ValueError("The capture did not include every selected recording")
    panel.save(output, {"schema": SCHEMA, "phase": phase, "plan_sha256": panel.sha(plan), "device": device, "rows": rows})


def _arrays(row):
    import numpy as np

    def decode(value):
        return {key: np.asarray(array) for key, array in value.items()} if value is not None else None

    return row | {"times": np.asarray(row["times"]), "visual": {arm: decode(data) for arm, data in row["visual"].items()},
                  "audio": decode(row.get("audio"))}


def train(rows, arm):
    import numpy as np
    import torch
    from torch import nn

    from scripts import video_lip_sequence_features as features

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    visual, audio, truth = features.fit_pairs(rows, arm, SEED)
    parameters = {}
    for name, array in (("visual", visual), ("audio", audio)):
        mean, scale = array.mean(axis=0), array.std(axis=0).clip(.05, None)
        parameters[name + "_mean"] = mean.tolist()
        parameters[name + "_scale"] = scale.tolist()
        array[:] = np.clip((array - mean) / scale, -5., 5.)
    model = nn.ModuleDict({"visual": nn.Linear(visual.shape[1], 28), "audio": nn.Linear(audio.shape[1], 28),
                           "joint": nn.Linear(56, 24), "output": nn.Linear(24, 1)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    x, y, labels = (torch.from_numpy(item) for item in (visual, audio, truth))
    for _ in range(25):
        for indices in torch.randperm(len(labels)).split(192):
            left = model["visual"](x[indices]).tanh()
            right = model["audio"](y[indices]).tanh()
            joint = model["joint"](torch.cat((left * right, (left - right).abs()), dim=-1)).tanh()
            response = model["output"](joint).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(response, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    for name, layer in model.items():
        parameters[name + "_weight"] = layer.weight.detach().tolist()
        parameters[name + "_bias"] = layer.bias.detach().tolist()
    print(f"{arm} {len(labels)} paired training inputs", flush=True)
    return parameters


def _reads(capture, fits):
    import numpy as np

    from scripts import video_lip_sequence_features as features

    results = []
    for saved in capture["rows"]:
        row = _arrays(saved)
        for delivery, cases in row["cases"].items():
            for role, case in cases.items():
                record = {key: row[key] for key in ("speaker", "picture_sha256", "condition", "truth", "time_base", "pts")}
                raw = case["audio"]
                audio = {key: np.asarray(arr) for key, arr in raw.items()} if raw is not None else None
                record.update(delivery=delivery, role=role, **{key: value for key, value in case.items() if key != "audio"})
                record["candidate"] = {arm: features.locate(row["times"], features.lane_scores(row, audio, fit, arm))
                                       for arm, fit in fits.items()}
                results.append(record)
    return results


def totals(rows, selected):
    arm, point = selected["arm"], selected["point"]
    counts = defaultdict(Counter)
    for row in rows:
        proposal = row["candidate"][arm]
        prior_span = row["current_span"]
        next_span = proposal["span"] if proposal is not None and proposal["score"] > point else None
        combined = prior_span or next_span
        goal = [row["truth"]] if row["role"] == "different_words" else []
        stats = counts[f"{row['delivery']}/{panel.ROLES[row['role']]}"]
        stats.update(files=1, current_flags=bool(prior_span), candidate_flags=bool(next_span), combined_flags=bool(combined),
                     current_useful=_matches(goal, [prior_span] if prior_span else [], .40, 0.),
                     candidate_useful=_matches(goal, [next_span] if next_span else [], .40, 0.),
                     combined_useful=_matches(goal, [combined] if combined else [], .40, 0.),
                     candidate_unassessed=proposal is None,
                     current_unassessed=row["current_status"] in {"insufficient_events", "incomplete_context", "unsupported"})
    return {key: dict(value) for key, value in sorted(counts.items())}


def calibrate(training, controls, output):
    fit_rows, calibration = panel.read(training), panel.read(controls)
    if (fit_rows["phase"], calibration["phase"]) != ("fit", "calibration") or fit_rows["plan_sha256"] != calibration["plan_sha256"]:
        raise ValueError("Calibration requires separate source roles from the same plan")
    speakers = {row["speaker"] for row in fit_rows["rows"]}
    if any(row["speaker"] in speakers for row in calibration["rows"]):
        raise ValueError("Calibration reused a fit speaker")
    fits = {arm: train([_arrays(row) for row in fit_rows["rows"]], arm) for arm in ARMS}
    scores = _reads(calibration, fits)
    points = []
    for arm in ARMS:
        maxima = dict.fromkeys({row["speaker"] for row in scores}, 0.)
        for row in scores:
            if row["role"] != "different_words" and row["current_span"] is None and row["candidate"][arm] is not None:
                maxima[row["speaker"]] = max(maxima[row["speaker"]], row["candidate"][arm]["score"])
        maxima = sorted(maxima.values(), reverse=True)
        for level, rank in LEVELS.items():
            point = {"arm": arm, "level": level, "point": maxima[rank]}
            points.append(point | {"outcomes": totals(scores, point)})

    def rank(item):
        useful = extra = 0
        for key, count in item["outcomes"].items():
            if key.endswith("/content"):
                useful += count["combined_useful"] - count["current_useful"]
            else:
                extra += count["combined_flags"] - count["current_flags"]
        return useful, -extra, item["level"] == "strict", -ARMS.index(item["arm"])

    chosen = max(points, key=rank)
    panel.save(output, {"schema": SCHEMA, "method": STUDY_METHOD, "plan_sha256": calibration["plan_sha256"],
                        "fit_sha256": panel.sha(training), "calibration_sha256": panel.sha(controls),
                        "fit_speakers": sorted(speakers), "fits": fits, "choices": points, "selected": chosen,
                        "calibration_speakers": sorted({row["speaker"] for row in scores})})
    print(json.dumps(chosen, sort_keys=True), flush=True)


def assess(freeze_path, held, output):
    settings, capture = panel.read(freeze_path), panel.read(held)
    seen = set(settings["fit_speakers"]) | set(settings["calibration_speakers"])
    if (capture["phase"] != "held" or capture["plan_sha256"] != settings["plan_sha256"]
            or any(row["speaker"] in seen for row in capture["rows"])):
        raise ValueError("Held recordings share fit or calibration speakers, or their source plan differs")
    rows = _reads(capture, settings["fits"])
    result = {"schema": SCHEMA, "method": STUDY_METHOD, "freeze": settings, "freeze_sha256": panel.sha(freeze_path),
              "held_sha256": panel.sha(held), "rows": rows, "outcomes": totals(rows, settings["selected"])}
    panel.save(output, result)
    print(json.dumps(verify(result), sort_keys=True), flush=True)


def verify(result):
    rows, freeze = result["rows"], result["freeze"]
    if result["schema"] != SCHEMA or totals(rows, freeze["selected"]) != result["outcomes"]:
        raise ValueError("Reported counts differ from paired source ranges")
    seen = set(freeze["fit_speakers"]) | set(freeze["calibration_speakers"])
    if seen & {row["speaker"] for row in rows}:
        raise ValueError("The held measurements reuse speakers from calibration or fitting")
    return result["outcomes"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="action", required=True)
    programs = {
        "freeze": (freeze, ("sources", "stereo_path", "output")),
        "capture": (capture, ("plan", "sources", "assets", "phase", "output")),
        "calibrate": (calibrate, ("training", "controls", "output")),
        "assess": (assess, ("freeze_path", "held", "output")),
        "verify": (None, ("receipt",)),
    }
    for command, (_, operands) in programs.items():
        parser = modes.add_parser(command)
        for name in operands:
            if name == "phase":
                parser.add_argument(name, choices=("fit", "calibration", "held"))
            else:
                parser.add_argument(name, type=Path)
    args = vars(cli.parse_args())
    action = args.pop("action")
    if action == "verify":
        print(json.dumps(verify(panel.read(args["receipt"])), sort_keys=True))
    else:
        programs[action][0](**args)


if __name__ == "__main__":
    main()
