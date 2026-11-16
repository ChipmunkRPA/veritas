# SPDX-License-Identifier: MIT
"""Compare lip-and-speech checks at a new soundtrack rate and picture cadence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_ranked as ranked
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_articulation_transfer as articulation
from scripts import video_lip_viseme_transfer as alternate

phone = articulation.phone
ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
GRID_REFERENCE = PANEL / "video-lip-articulation-pcm-2026-11-15.json.gz"
GRID_AAC_REFERENCE = PANEL / "video-lip-articulation-aac-2026-11-15.json.gz"
SCHEMA = "chipmunk.video-lip-delivery-transfer/1"
ARMS = {"timing": (False, False), "opening": (True, False), "articulation": (False, True), "both": (True, True)}
POINTS = [binding.POINT, ranked.POINT, binding.ARTICULATION_POINT]
GRID_ROLES = {"original", "gain", "global_early", "global_late", "same_words", "different_words"}
NATIVE_ROLES = {"original", "gain", "filtered", "global_early", "global_late", "early", "late"}
MP4_METHOD = ("Encode lossless H.264, retaining every VP6 source picture and timestamp, then mux each treated "
              "soundtrack as 16 kHz mono AAC. Use the production decoder on each MP4 and keep codec padding and "
              "coverage behavior as the production 16 kHz check would see them.")
METHOD = {
    "aac48": "Re-encode frozen held soundtrack cases as mono 48 kHz AAC at 96 kbps. Verify the presented sample extent "
             "and origin using the production decoder before running the frozen checks. Keep the original pictures, "
             "same-word controls, ordinary deliveries and uniform-delay controls.",
    "native_clock": "Keep the original source picture timestamps and continuous 16 kHz decoded soundtrack. Report the "
                    "24 earlier 29.97 fps actors separately from the 21 actors outside the earlier clock trial. Compare "
                    "untouched, filtered and gain-adjusted recordings, two local shifts and two whole-track delays.",
    "selection": "Use the fixed sparse timing, relative mouth opening, and articulation parameters from the app. "
                 "Keep an earlier timing witness before considering a fallback. Check source-frame brackets for every "
                 "result. Mark location at 0.40 temporal intersection over union. Check shift direction separately.",
}


def qualified_48k(pcm, path):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from scripts.video_lip_timing_aac48 import encode

    encode(pcm, path, 48_000)
    digest = phone.sha(path)
    decoded = decode_audio(path, source_sha256=digest)
    mapping = decoded.mapping
    extent = binding.presented_samples(mapping)
    if extent != len(pcm) or binding.audio_origin(mapping) != 0:
        raise ValueError("AAC readback changed the source extent or origin")
    values = np.rint(decoded.pcm[:extent] * 32768.).astype("<i2")
    expected, actual = pcm.astype(np.float64), values.astype(np.float64)
    best = max((float(np.sum(expected[64:-64] * actual[64 + lag:extent - 64 + lag])), lag)
               for lag in range(-16, 17))[1]
    if abs(best) > 1:
        raise ValueError("AAC readback shifted the continuous sample clock")
    return values, mapping, {"file_sha256": digest, "pcm_sha256": mapping["pcm_sha256"], "samples": mapping["samples"],
                             "presented": extent, "native_pts": mapping["source_duration_pts"], "alignment": best}


def compare(geometry, capture, mapping, source):
    extent = binding.presented_samples(mapping)
    origin = binding.audio_origin(mapping)
    # Replay the separately tested opening check on the same admitted
    # samples. Its production admission still requires a 16 kHz source.
    canonical = mapping | {"source_sample_rate": timing.RATE, "samples": extent,
                           "timeline": [{"start_sample": 0, "end_sample": extent, "source_pts": origin}]}
    frames = geometry["frames"]
    plain = [{k: v for k, v in frame.items() if k != "shape"} for frame in frames]
    results = {}
    for name, (opening, local) in ARMS.items():
        raw = {"phones": capture["standard"], "frames": frames if local else plain}
        if opening:
            raw["aperture_prediction"] = ranked.predict(capture["groups"])
        if local:
            raw["articulation_phones"] = capture["articulation"]
        measured = binding.measured(raw, timestamps=geometry["pts"], time_base=geometry["time_base"],
                                    source=source, mapping=canonical if opening else mapping)
        results[name] = {"state": measured["status"], "basis": measured.get("measurement_basis", "speech_events"),
                         "witness": measured["witness"]}
    return results


def bound_grid(plan_path, pictures_path):
    plan = json.loads(plan_path.read_bytes())
    capture = articulation.read(pictures_path)
    reference = articulation.read(GRID_REFERENCE)
    aac_reference = articulation.read(GRID_AAC_REFERENCE)
    phone.require(reference["held_sha256"] == phone.sha(pictures_path)
                  and reference["plan_sha256"] == capture["plan_sha256"] == phone.sha(plan_path)
                  and aac_reference["plan_sha256"] == phone.sha(plan_path)
                  and reference["points"]["aperture_classes"]["tight"] == binding.ARTICULATION_POINT,
                  "Published picture measurements or selected operating points changed")
    for item, saved, prior in zip([row for row in plan["rows"] if row["split"] == "held"],
                                 capture["rows"], aac_reference["cohort"], strict=True):
        target = item["target"]
        phone.require(target["front_sha256"] == saved["picture_sha256"] == prior["picture_sha256"]
                      and target["span"] == saved["truth"] and item["speaker"] == saved["speaker"],
                      "The retained source pairing differs")
        frames = [{k: v for k, v in frame.items() if k not in {"gap", "sample"}}
                  | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * timing.SCORE_UNITS)}
                  for frame in saved["frames"]]
        yield item, {"frames": frames, "pts": [frame["pts"] for frame in frames], "time_base": saved["time_base"]}


def grid(plan, pictures, archive, assets, out):
    reader, device = articulation.audio_reader(assets / "checkpoint")
    rows = []
    with (ZipFile(archive) as source, TemporaryDirectory(prefix="lip-delivery-") as temp):
        for item, geometry in bound_grid(plan, pictures):
            def samples(record):
                pcm = source.read(record["audio"])
                phone.require(phone.digest(pcm) == record["audio_sha256"], "The selected archival soundtrack changed")
                return phone.pcm(pcm)

            target = item["target"]
            truth = target["span"]
            original = samples(target)
            alternates = {role: pcm for role, pcm in alternate.parent.treatments(original, truth).items()
                          if role in {"original", "gain", "global_early", "global_late"}}
            for role in ("same_words", "different_words"):
                alternates[role] = articulation.parent.splice(original, truth, samples(item[role]), item[role]["span"])
            cases = {}
            for role, pcm in alternates.items():
                values, mapping, carrier = qualified_48k(pcm, Path(temp) / "carrier.m4a")
                cases[role] = {"carrier": carrier, "readings": compare(geometry, reader(values), mapping,
                                                                       {"width": 1280, "height": 720})}
            rows.append({"source": target["front_sha256"], "speaker": item["speaker"], "cohort": item["condition"],
                         "audio": 0, "truth": truth, "pts": geometry["pts"], "time_base": geometry["time_base"],
                         "samples": len(original), "cases": cases})
            print(f"Verified 48 kHz deliveries for source {len(rows)}/34", flush=True)
    save(out, "aac48", rows, device, {"plan": phone.sha(plan), "picture_capture": phone.sha(pictures),
                                      "earlier_aac16": phone.sha(GRID_AAC_REFERENCE)})


def video_carrier(source, path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(source), "-map", "0:v:0",
                    "-an", "-sn", "-dn", "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", "0",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "1/30000",
                    "-video_track_timescale", "30000", "-avoid_negative_ts", "disabled", str(path)], check=True, timeout=25)


def mux(pictures, pcm, path):
    from scripts.video_lip_timing_aac48 import encode

    audio = path.with_suffix(".m4a")
    encode(pcm, audio, timing.RATE)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(pictures), "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-video_track_timescale", "30000",
                    "-avoid_negative_ts", "disabled", str(path)], check=True, timeout=20)


def same_pictures(original, carrier):
    return ((original.width, original.height) == (carrier.width, carrier.height)
            and not carrier.interpolated and not carrier.truncated and not carrier.rotation
            and [point * original.tb for point in original.pts] == [point * carrier.tb for point in carrier.pts])


def native(audit, media, assets, out, *, repack=False):
    if repack:
        with TemporaryDirectory(prefix="native-lip-mp4-") as temp:
            native_capture(audit, media, assets, out, Path(temp))
    else:
        native_capture(audit, media, assets, out, None)


def native_capture(audit, media, assets, out, scratch):
    sys.path.insert(0, str((assets / "runtime").resolve()))
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import FrameTable, decode_stream, probe_frames
    from app.evidence.video_reference_timing import _landmarker, _mesh, available
    from scripts import video_lip_native_clock_transfer as earlier

    fresh = earlier.roster(audit, remaining=True)
    previous = earlier.roster(audit)
    geometry_asset = assets / "face_landmarker.task"
    phone.require(available(geometry_asset) and phone.sha(geometry_asset) == alternate.parent.FACE_SHA256,
                  "The pinned landmark asset is required")
    reader, device = articulation.audio_reader(assets / "checkpoint")
    rows = []
    for cohort, roster in (("earlier", previous), ("additional", fresh)):
        for row in roster:
            source = media / f"asset-{row['index']:03d}.flv"
            table = probe_frames(source, cap=121, timeout_s=20)
            phone.require(phone.sha(source) == row["input_sha256"] and table == FrameTable.from_json(row["frame_table"]),
                          "Audited source file or picture clock changed")
            result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                                     "-protocol_whitelist", "file", "-i", f"file:{source.resolve()}", "-map", "0:a:0",
                                     "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(timing.RATE), "-f", "s16le", "pipe:1"],
                                    capture_output=True, check=True, timeout=20)
            original = np.frombuffer(result.stdout, dtype="<i2")
            samples = len(original)
            phone.require(not result.stderr and samples == row["canonical_audio"]["samples"]
                          and phone.digest(result.stdout) == row["canonical_audio"]["pcm_sha256"], "Audited PCM changed")
            origin = round(Fraction(row["audio_clocks"]["frames"][0]["start"]) * timing.RATE)
            frames = []
            with _landmarker(geometry_asset) as (mp, task):
                for frame in decode_stream(source, table, max_side=None, timeout_s=20):
                    phone.require((frame.index, frame.pts, frame.pts_verified) == (len(frames), table.pts[len(frames)], True),
                                  "The native source picture differs from its exact clock")
                    values = {"index": frame.index, "pts": frame.pts, "gap_units": None}
                    if observed := _mesh(task, mp, frame.rgb):
                        features, box = observed
                        x, y, right, bottom = box
                        values.update(gap_units=round(float(median(features[:3])) * timing.SCORE_UNITS), mouth_box=box,
                                      shape=[round(float(value) * timing.SCORE_UNITS) for value in features],
                                      rgb_sha256=phone.digest(frame.rgb.tobytes()),
                                      mouth_crop_sha256=phone.digest(frame.rgb[y:bottom, x:right].tobytes()))
                    frames.append(values)
            phone.require(len(frames) == len(table), "The native picture scan is incomplete")
            geometry = {"frames": frames, "pts": list(table.pts), "time_base": table.time_base}
            if scratch is not None:
                pictures = scratch / "video-only.mp4"
                delivery = scratch / "case.mp4"
                video_carrier(source, pictures)
                encoded = probe_frames(pictures, cap=121, timeout_s=20)
                phone.require(same_pictures(table, encoded), "The MP4 picture clock differs from the native clock")
                expected_pixels = [phone.digest(frame.rgb.tobytes()) for frame in decode_stream(source, table, max_side=None, timeout_s=20)
                                   if frame.pts_verified]
                received_pixels = [phone.digest(frame.rgb.tobytes()) for frame in decode_stream(pictures, encoded, max_side=None, timeout_s=20)
                                   if frame.pts_verified]
                phone.require(len(expected_pixels) == len(table) and expected_pixels == received_pixels, "MP4 changed a source picture")
                geometry = {"frames": [frame | {"pts": pts} for frame, pts in zip(frames, encoded.pts, strict=True)],
                            "pts": list(encoded.pts), "time_base": encoded.time_base}
            start = (samples - earlier.EDIT_WIDTH) // 2
            truth = [start, start + earlier.EDIT_WIDTH]
            cases = {}
            for role, pcm in alternate.parent.treatments(original, truth).items():
                digest = phone.digest(pcm.tobytes())
                mapping = {"samples": samples, "source_sample_rate": timing.RATE, "source_channels": 1,
                           "timeline_time_base": "1/16000", "pcm_sha256": digest,
                           "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": origin}]}
                metadata = {}
                if scratch is not None:
                    mux(pictures, pcm, delivery)
                    received = probe_frames(delivery, cap=121, timeout_s=20)
                    phone.require(same_pictures(encoded, received), "Muxing changed the video presentation")
                    file_sha256 = phone.sha(delivery)
                    sound = decode_audio(delivery, source_sha256=file_sha256)
                    mapping = sound.mapping
                    phone.require(binding.audio_origin(mapping) == origin and mapping["source_sample_rate"] == timing.RATE
                                  and 0 <= mapping["samples"] - samples <= 1024, "Muxing changed the audio extent or origin")
                    pcm = np.rint(sound.pcm * 32768.0).astype("<i2")
                    metadata = {"file_sha256": file_sha256, "decoded_samples": len(pcm)}
                cases[role] = {"readings": compare(geometry, reader(pcm), mapping, {"width": table.width, "height": table.height}),
                               "pcm_sha256": mapping["pcm_sha256"], **metadata}
            rows.append({"source": row["input_sha256"], "speaker": phone.digest(str(row["group"]).encode()), "cohort": cohort,
                         "samples": samples, "audio": origin, "truth": truth, "pts": geometry["pts"], "time_base": geometry["time_base"],
                         **({"repacked_rgb_sha256": phone.digest("\n".join(received_pixels).encode())} if scratch is not None else {}),
                         "cases": cases})
            print(f"Verified native picture and audio clocks for {cohort} actor {len(rows)}/{len(fresh) + len(previous)}", flush=True)
    save(out, "native_mp4" if scratch is not None else "native_clock", rows, device,
         {p.name: phone.sha(p) for p in sorted(audit.glob("batch-??.jsonl"))})


def outcomes(packet):
    phone.require(packet["schema"] == SCHEMA and packet["method"] == METHOD
                  and packet["points"] == POINTS and packet["study"] in {"aac48", "native_clock", "native_mp4"},
                  "Frozen comparison differs")
    rows = packet["rows"]
    kind = packet["study"]
    if kind == "native_mp4":
        phone.require(packet["carrier_method"] == MP4_METHOD, "The frozen MP4 preparation differs")
    totals = defaultdict(Counter)
    seen = set()
    cohorts = Counter(row["cohort"] for row in rows)
    if kind == "aac48":
        phone.require(len(rows) == 34 and len({row["speaker"] for row in rows}) == 17,
                      "The 48 kHz speech-replacement panel is incomplete")
        expected_roles = GRID_ROLES
    else:
        phone.require(cohorts == {"earlier": 24, "additional": 21} and len({row["speaker"] for row in rows}) == len(rows),
                      "The 29.97 fps actor panels are incomplete")
        expected_roles = NATIVE_ROLES
    for row in rows:
        pts, origin = row["pts"], row["audio"]
        clock = Fraction(row["time_base"])
        phone.require(row["source"] not in seen and set(row["cases"]) == expected_roles
                      and 24 <= len(pts) <= 120 and all(a < b for a, b in zip(pts, pts[1:])),
                      "Repeated source, incomplete control set or invalid frame table")
        seen.add(row["source"])
        a, b = [origin + sample for sample in row["truth"]]
        phone.require(origin <= a < b <= origin + row["samples"], "The known edit is outside the source soundtrack")
        if kind != "aac48":
            phone.require(all(abs((point - pts[0]) * clock - Fraction(index * 1001, 30_000)) <= Fraction(3, 2000)
                              for index, point in enumerate(pts)),
                          "A picture falls outside the audited 29.97 fps clock")
        for role, case in row["cases"].items():
            group = ("local" if role in {"different_words", "early", "late"} else "same_word" if role == "same_words"
                     else "constant" if role.startswith("global_") else "ordinary")
            phone.require(set(case["readings"]) == set(ARMS), "Incomplete paired comparison")
            if kind == "aac48":
                carrier = case["carrier"]
                phone.require(carrier["native_pts"] // 3 == carrier["presented"] == row["samples"]
                              and 0 <= carrier["samples"] - row["samples"] <= 342 and abs(carrier["alignment"]) <= 1,
                              "Invalid readback duration or padding")
            if kind == "native_mp4":
                phone.require(0 <= case["decoded_samples"] - row["samples"] <= 1024 and len(row["repacked_rgb_sha256"]) == 64,
                              "Invalid readback duration, padding or frame commitment")
            baseline = case["readings"]["timing"]["witness"]
            for arm, result in case["readings"].items():
                hit = result["witness"]
                useful, exact_shift = False, False
                phone.require((result["state"] == "matched") == bool(hit) and result["state"] in binding.STATUS,
                              "Invalid result disposition")
                if hit:
                    lo, hi = hit["audio_source_pts"]
                    left, right = hit["picture_source_samples"]
                    phone.require(origin <= lo < hi <= origin + case.get("decoded_samples", row["samples"]),
                                  "Audio range is outside the decoded presentation")
                    photos = hit["picture_frames"]
                    be, du, af = [round(pts[photos[key]] * clock * timing.RATE) for key in binding.ROLES]
                    phone.require(be < left <= du < right <= af and (not baseline or hit == baseline),
                                  "The result is missing its source bracket or replaced an earlier finding")
                    useful = group == "local" and articulation.parent.overlap([a, b], [lo, hi])
                    shift = alternate.parent.EDITS.get(role)
                    exact_shift = bool(useful and shift and abs(hit["change_samples"] - shift) <= 1280
                                       and hit["change_samples"] * shift > 0)
                    if event := hit.get("articulation"):
                        phone.require(pts[event["picture_frame"]] == event["picture_pts"] and lo <= event["audio_sample"] < hi,
                                      "The strongest mismatch does not match the original clock")
                for cohort in ("all", row["cohort"]):
                    totals[f"{cohort}/{arm}/{group}"].update(files=1, flags=bool(hit), useful=useful, useful_and_shift=exact_shift,
                        new_flags=bool(hit and not baseline), misplaced=bool(hit and group == "local" and not useful),
                        unresolved=result["state"] not in {"matched", "no_match"})
    return {key: dict(value) for key, value in sorted(totals.items())}


def save(out, study, rows, device, parents):
    packet = {"schema": SCHEMA, "method": METHOD, "study": study, "device": device, "parents": parents,
              "points": POINTS, "rows": rows}
    if study == "native_mp4":
        packet["carrier_method"] = MP4_METHOD
    packet["summary"] = outcomes(packet)
    articulation.seal(out, packet)
    print(json.dumps({key: val for key, val in packet["summary"].items() if key.startswith(("all/", "additional/"))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, fields in (("aac48", ("plan", "pictures", "archive", "assets", "output")),
                            ("native", ("audit", "media", "assets", "output")), ("review", ("receipt",))):
        layout = commands.add_parser(command)
        for field in fields:
            layout.add_argument(field, type=Path)
        if command == "native":
            layout.add_argument("--repack-aac16", action="store_true")
    args = parser.parse_args()
    if args.command == "aac48":
        grid(args.plan, args.pictures, args.archive, args.assets, args.output)
    elif args.command == "native":
        native(args.audit, args.media, args.assets, args.output, repack=args.repack_aac16)
    else:
        packet = articulation.read(args.receipt)
        got = outcomes(packet)
        phone.require(got == packet["summary"], "Recorded results differ from the replay")
        print(json.dumps(got, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
