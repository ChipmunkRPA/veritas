"""Rebuild the independent lip-sync benchmark as verified playable videos."""
from __future__ import annotations

import argparse
import json
import subprocess
import wave
from bisect import bisect_left
from fractions import Fraction
from math import lcm
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import video_lip_correspondence as measures
from scripts import video_lip_crop_correspondence as trial

ROSTER = "chipmunk.video-lip-correspondence-inputs/1"
SAMPLE_RATE = 16_000
PROFILES = ("pcm", "aac48")
ORDER = ("original", "gain", "same_words", "different_words", "global_early", "global_late")
LABELS = {
    "original": "unchanged speech", "gain": "local gain", "same_words": "same-word splice",
    "different_words": "different-word splice", "global_early": "whole-track early", "global_late": "whole-track late",
}


def json_file(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


def make_roster(full, output):
    source = json.loads(full.read_bytes())
    scenario = [{key: row[key] for key in ("speaker", "condition", "target", "same_words", "different_words")}
                for row in source["rows"] if row["split"] == "new_speaker"]
    if len(scenario) != 19 or len({row["target"]["front_sha256"] for row in scenario}) != len(scenario):
        raise ValueError("Missing the distinct source clips in the approved roster")
    json_file(output, {"schema": ROSTER, "publisher": "https://zenodo.org/records/3736465",
                       "parent_sha256": measures.sha(full), "cases": scenario})


def waveform(zip_file, item):
    from scripts.video_lip_articulation_transfer import phone

    blob = zip_file.read(item["audio"])
    if phone.digest(blob) != item["audio_sha256"]:
        raise ValueError("Publisher audio differs from its commitment")
    return phone.pcm(blob)


def make_soundtracks(audio, source):
    import numpy as np

    from scripts.video_lip_articulation_transfer import parent, study

    original = waveform(audio, source["target"])
    span = source["target"]["span"]
    tracks = {role: values for role, values in study.parent.treatments(original, span).items()
              if role in {"original", "gain", "global_early", "global_late"}}
    for role in ("same_words", "different_words"):
        donor = waveform(audio, source[role])
        edited = parent.splice(original, span, donor, source[role]["span"])
        start, stop = span
        if not (np.array_equal(edited[:start], original[:start]) and np.array_equal(edited[stop:], original[stop:])):
            raise ValueError("Staged recording differs outside its sample interval")
        tracks[role] = edited
    if set(tracks) != set(ORDER):
        raise ValueError("Staged soundtrack set is incomplete")
    return tracks


def visual_hash(path):
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
                             "-c", "copy", "-f", "hash", "-hash", "sha256", "pipe:1"],
                            check=True, capture_output=True, timeout=20)
    return result.stdout.decode("ascii").strip().removeprefix("SHA256=")


def make_mux(source, audio, target, table, expected_picture):
    from app.detectors.video.decode import probe_frames

    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-i", str(audio), "-map", "0:v:0",
                    "-map", "1:a:0", "-c", "copy", "-map_metadata", "-1", "-map_chapters", "-1",
                    "-fflags", "+bitexact", "-video_track_timescale", str(table.tb.denominator),
                    "-movie_timescale", str(lcm(48_000, table.tb.denominator)), str(target)],
                   check=True, timeout=20)
    readback = probe_frames(target, cap=120, timeout_s=20, max_pixels=1280 * 720)
    if (readback.truncated or readback.interpolated or
            tuple(pts * readback.tb for pts in readback.pts) != tuple(pts * table.tb for pts in table.pts)
            or visual_hash(target) != expected_picture):
        raise ValueError("Delivery changed the original picture stream or its clock")


def verify_sound(path, profile, committed, extent):
    import hashlib

    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.evidence.video_viseme_binding import audio_origin, presented_samples

    decoded = decode_audio(path, source_sha256=measures.sha(path))
    length = presented_samples(decoded.mapping)
    if length != extent or audio_origin(decoded.mapping) != 0:
        raise ValueError("Rendered audio presentation does not match the scored sample clock")
    if (decoded.mapping["source_sample_rate"], decoded.mapping["source_channels"]) != (
        48_000 if profile == "aac48" else 16_000, 1
    ):
        raise ValueError("Independent benchmark codec geometry differs")
    pcm = np.rint(decoded.pcm[:length] * 32768).astype("<i2")
    if hashlib.sha256(pcm.tobytes()).hexdigest() != committed:
        raise ValueError("Rendered audio samples differ from the original scored input")


def frame_range(span, times):
    first, end = (bisect_left(times, sample) for sample in span)
    if first >= end:
        raise ValueError("No native picture timestamp falls in the scoring interval")
    return [first, end]


def records(roster, report, sources, folder):
    from app.detectors.video.decode import probe_frames
    from scripts.video_lip_articulation_transfer import phone
    from scripts.video_lip_delivery_transfer import qualified_48k

    clip_path = folder / "clips"
    original_path = folder / "source"
    clip_path.mkdir()
    original_path.mkdir()
    by_profile = {profile: {item["picture_sha256"]: item for item in report["transfer"][profile]["rows"]}
                  for profile in PROFILES}
    with (TemporaryDirectory(prefix="benchmark-sound-", dir=folder) as work,
          ZipFile(sources / "lombardgrid_audio.zip") as source_audio, ZipFile(sources / "lombardgrid_front.zip") as source_video):
        wavefile = Path(work) / "mono.wav"
        compressed = Path(work) / "mono.m4a"
        for i, case in enumerate(roster["cases"], 1):
            item = case["target"]
            identity = item["front_sha256"]
            raw = source_video.read(item["front"])
            if phone.digest(raw) != identity:
                raise ValueError("Publisher picture bytes differ from their pinned identity")
            native = original_path / f"source-{i:02d}.mov"
            native.write_bytes(raw)
            table = probe_frames(native, cap=120, timeout_s=20, max_pixels=1280 * 720)
            if table.truncated or table.interpolated or table.rotation or not table.pts:
                raise ValueError("Publisher picture clock is unsupported")
            times = [Fraction(pts - table.pts[0]) * table.tb * SAMPLE_RATE for pts in table.pts]
            picture_sha = visual_hash(native)
            tracks = make_soundtracks(source_audio, case)
            for profile in PROFILES:
                scored = by_profile[profile][identity]
                if scored["truth"] != item["span"] or scored["samples"] != len(tracks["original"]):
                    raise ValueError("Transfer receipt expectations do not match the source roster")
                if str(table.tb) != scored["time_base"]:
                    raise ValueError("Transfer picture clock differs from the released video")
                for role in ORDER:
                    tested = scored["cases"][role]
                    data = tracks[role]
                    if profile == "pcm":
                        with wave.open(str(wavefile), "wb") as out:
                            out.setparams((1, 2, SAMPLE_RATE, 0, "NONE", ""))
                            out.writeframes(data.tobytes())
                        audio = wavefile
                    else:
                        _, _, clock = qualified_48k(data, compressed)
                        if clock != tested["clock"]:
                            raise ValueError("AAC carrier differs from the scored delivery")
                        audio = compressed
                    target = clip_path / f"clip-{i:02d}-{profile}-{ORDER.index(role):02d}.mov"
                    make_mux(native, audio, target, table, picture_sha)
                    verify_sound(target, profile, tested["pcm_sha256"], len(data))
                    yield case, role, profile, tested, times, native, target
            print(f"Verified native pictures and both sound formats for source {i:02d}/{len(roster['cases'])}", flush=True)


def prepare(roster_path, receipt, sources, output):
    from app.detectors.audio.phone_events import WEIGHT_PIN
    from app.evidence.video_reference_timing import MODEL_SHA256

    roster = json.loads(roster_path.read_bytes())
    report = measures.read(receipt)
    if (roster["schema"] != ROSTER or report["schema"] != trial.SCHEMA
            or roster["parent_sha256"] != report["roster_sha256"]):
        raise ValueError("Source roster and scored clips are from different intakes")
    output.mkdir()
    cases, targets, index = [], [], []
    systems = {"current": [], "candidate": []}
    choice = report["freeze"]["selected"]
    for source, role, profile, scored, times, original, submitted in records(roster, report, sources, output):
        digest = measures.sha(submitted)
        case_id = submitted.stem
        target = frame_range(source["target"]["span"], times) if role == "different_words" else []
        speaker_id = f"lombard-grid/{source['speaker']}"
        cases.append({"id": case_id, "split": "confirmation", "modality": "video", "group": speaker_id,
                      "label": int(bool(target)), "family": LABELS[role], "condition": f"{profile}/{LABELS[role]}",
                      "source_sha256": source["target"]["front_sha256"], "input_sha256": digest, "lineage": [speaker_id],
                      "license": {"id": "CC BY 4.0 (Lombard Grid)", "evaluation_permitted": True, "redistribution_permitted": True},
                      "localization_truth": {"unit": "frames", "source_units": len(times),
                                             "target_units": target[1] - target[0] if target else 0}})
        targets.append({"id": case_id, "input_sha256": digest, "extent": [len(times)], "regions": [target] if target else []})
        index.append({"id": case_id, "input": submitted.relative_to(output).as_posix(), "source": original.relative_to(output).as_posix()})
        for system, predictions in systems.items():
            candidate = scored.get(choice["arm"]) if system == "candidate" else None
            marked = scored["incumbent"]
            if not marked and candidate and candidate["status"] == "measured" and candidate["score"] > choice["point"]:
                marked = candidate
            checked = (scored["incumbent_assessed"] or
                       (candidate and candidate["status"] in {"measured", "no_supported_change"}))
            row = {"id": case_id, "input_sha256": digest}
            if marked or checked:
                # Benchmark localization uses the audio interval expressed on the source picture grid.
                regions = [frame_range(marked["span"], times)] if marked else []
                row.update(status="ok", response=int(bool(regions)), localization={"status": "ok", "regions": regions})
            else:
                reason = "No usable speech and mouth support at the required times"
                row.update(status="not_applicable", response=None, reason=reason,
                           localization={"status": "not_applicable", "reason": reason})
            predictions.append(row)
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate different-word splices from the same speaker on native pictures.",
                "publisher": roster["publisher"], "cases": cases}
    if len(cases) != len(roster["cases"]) * len(PROFILES) * len(ORDER):
        raise ValueError("Benchmark inputs are incomplete")
    binding = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    truth = binding | {"schema": TRUTH_SCHEMA, "cases": targets}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": index}
    checked = verify_inputs(manifest, file_index, output)
    for name, doc in (("manifest", manifest), ("truth", truth), ("file-index", file_index), ("input-verification", checked)):
        json_file(output / f"{name}.json", doc)
    for system, predictions in systems.items():
        identity = [report["prior_sha256"], system]
        if system == "candidate":
            identity.extend([measures.sha(Path(measures.__file__)), choice, report["freeze"]["heads"][choice["arm"]]])
        response = binding | {"system": system, "implementation_sha256": independent.digest(identity),
                              "method_kind": "weighted", "weight_sha256": [WEIGHT_PIN, MODEL_SHA256],
                              "score_kind": "raw", "response_units": "Selected local mismatch (1 or 0)", "predictions": predictions}
        events = count_events(manifest, truth, response, minimum_iou=.40, minimum_recall=.40)
        json_file(output / f"{system}.json", response)
        json_file(output / f"{system}-events.json", events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, fields in (("roster", ("full", "out")), ("prepare", ("roster", "receipt", "sources", "out"))):
        sub = commands.add_parser(command)
        for name in fields:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "roster":
        make_roster(args.full, args.out)
    else:
        prepare(args.roster, args.receipt, args.sources, args.out)


if __name__ == "__main__":
    main()
