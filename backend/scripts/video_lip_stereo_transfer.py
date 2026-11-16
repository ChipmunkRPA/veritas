# SPDX-License-Identifier: MIT
"""Compare bounded local speech and mouth timing on 48 kHz stereo delivery."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.bench.independent_locations import _matches
from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_stereo as stereo
from scripts import video_lip_articulation_transfer as articulation
from scripts import video_lip_delivery_transfer as delivery

phone, parent = articulation.phone, articulation.parent
SCHEMA = "chipmunk.video-lip-stereo-transfer/1"
ROLES = ("original", "same_words", "different_words")
SCENES = {scene: ROLES for scene in ("centered", "panned_environment", "competing_speech")} | {"opposed": ("original",)}
METHOD = {
    "carrier": "Upsample stereo from the same 16 kHz utterance clock and encode both channels in 48 kHz AAC-LC at 160 kbps. "
               "Keep the source picture track and first audio stream clock.",
    "centered": "Deliver the same foreground in both channels.",
    "panned_environment": "Add a licensed environment from the full four-second release at −12 dB in its dominant channel "
                          "and one quarter of that amplitude on the other channel.",
    "competing_speech": "Add a verified meeting excerpt at +6 dB in its dominant channel and one tenth of that amplitude "
                        "on the other channel.",
    "opposed": "Reverse the polarity of one speech channel on the ordinary recording.",
    "score": "Compare stereo downmix without a guard and the same input with the frozen channel-agreement rule. "
             "Use the current mouth and speech fits without changing their operating points. Count a correct range at 0.40 IoU.",
}


def fresh_rows(sources, previous):
    from scripts.lombard_mouth_source_intake import index_zip

    seen = {row["speaker"] for row in previous["rows"]}
    inventory = defaultdict(dict)
    used_audio = {value["audio_sha256"] for row in previous["rows"] for key, value in row.items()
                  if key in {"target", "same_words", "different_words"}}
    used_pictures = {row["target"]["front_sha256"] for row in previous["rows"]}
    with (ZipFile(sources / "publisher-json.zip") as labels, ZipFile(sources / "alignment.zip") as align,
          ZipFile(sources / "lombardgrid_audio.zip") as audio, ZipFile(sources / "lombardgrid_front.zip") as front):
        eye, ear, bounds = index_zip(front, ".mov"), index_zip(audio, ".wav"), index_zip(align, ".json")
        for member in index_zip(labels, ".json").values():
            for row in json.loads(labels.read(member)):
                speaker = articulation.keyed(row["SPKR"])
                condition = row["COND"]
                uid = f"{row['SPKR']}_{condition}_{row['UTTERANCE']}"
                if speaker in seen or condition not in {"p", "l"} or row["STATUS"] != "CORRECT" or not all(
                    uid in index for index in (eye, ear, bounds)
                ):
                    continue
                raw = align.read(bounds[uid])
                if span := parent.word_bounds(json.loads(raw)[uid]):
                    inventory[speaker, condition][uid] = (span, raw)
        fresh = []
        for (speaker, condition), files in sorted(inventory.items()):
            matched = articulation.matched_records(files, audio, front, ear, eye, used_audio, used_pictures)
            if matched:
                fresh.append({"speaker": speaker, "condition": condition, "split": "new_speaker"} | matched)
        return fresh


def freeze(args):
    import numpy as np

    previous = json.loads(args.plan.read_bytes())
    saved = [row for row in previous["rows"] if row["split"] == "held"]
    assert len(saved) == 34 and len({row["speaker"] for row in saved}) == 17
    fresh = fresh_rows(args.sources, previous)
    environment = json.loads((args.environment / "environment-acquisition.json").read_bytes())["rows"]
    backgrounds = [{"file": f"{row['group']:02d}-real.wav", "sha256": row["wav_sha256"], "group": row["group"]}
                   for row in sorted(environment, key=lambda row: row["group"]) if row["lane"] == "real" and row["group"] >= 56]
    voices = [{"file": row["room"] + ".pcm", "sha256": row["pcm_sha256"], "index": 60_000, "source_url": row["url"]}
              for row in sorted(json.loads((args.rooms / "rooms.json").read_bytes())["rooms"], key=lambda row: row["room"])]
    assert len(backgrounds) == 24 and len(voices) == 6
    backgrounds = [item for item in backgrounds
                   if np.sqrt(np.mean(np.frombuffer(environment_bytes(args.environment / "environment", item),
                                                    dtype="<i2").astype(np.float64)[:20_000]**2)) > 5]
    for index, row in enumerate(saved + fresh):
        row["background"] = backgrounds[index % len(backgrounds)]
        row["competing"] = voices[index % len(voices)]
    packet = {"schema": SCHEMA, "method": METHOD, "retained_plan_sha256": phone.sha(args.plan),
              "guard_sha256": phone.sha(Path(stereo.__file__)), "rows": saved + fresh}
    phone.write(args.output, packet)
    print(json.dumps({"retained_links": len(saved), "new_speaker_sources": len(fresh),
                      "new_speakers": len({row["speaker"] for row in fresh})}), flush=True)


def environment_bytes(folder, item):
    blob = (folder / item["file"]).read_bytes()
    assert phone.digest(blob) == item["sha256"]
    with wave.open(io.BytesIO(blob), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 16_000, 64_000)
        return stream.readframes(stream.getnframes())


def fixture_inputs(row, audio, background_path, room_path):
    import numpy as np

    inputs = {}
    for key in ("target", "same_words", "different_words"):
        record = row[key]
        body = audio.read(record["audio"])
        assert phone.digest(body) == record["audio_sha256"]
        inputs[key] = phone.pcm(body)
    original = inputs.pop("target")
    truth = row["target"]["span"]
    edited = {role: parent.splice(original, truth, donor, row[role]["span"]) for role, donor in inputs.items()}
    assert all(np.array_equal(original[:truth[0]], take[:truth[0]]) and
               np.array_equal(original[truth[1]:], take[truth[1]:]) for take in edited.values())
    record = row["competing"]
    raw = (room_path / record["file"]).read_bytes()
    assert phone.digest(raw) == record["sha256"]
    start = record["index"]
    sounds = {"background": np.frombuffer(environment_bytes(background_path, row["background"]), dtype="<i2")[:len(original)],
              "competing": np.frombuffer(raw, dtype="<i2")[start:start + len(original)]}
    sounds = {kind: samples.astype(np.float64) for kind, samples in sounds.items()}
    assert all(len(samples) == len(original) and np.sqrt(np.mean(samples**2)) > 1 for samples in sounds.values())
    return {"original": original} | edited, sounds


def stereo_mix(voice, scene, sounds, ordinary):
    import numpy as np

    values = voice.astype(np.float64)
    if scene == "opposed":
        l, r = values, -values
    else:
        l = r = values
        blends = {"panned_environment": ("background", -12, .25), "competing_speech": ("competing", 6, .10)}
        if scene in blends:
            kind, db, spill = blends[scene]
            noise = sounds[kind]
            noise = noise * 10**(db / 20) * np.sqrt(np.mean(ordinary.astype(np.float64)**2) / np.mean(noise**2))
            l, r = values + noise, values + spill * noise
    return np.column_stack((l, r))


def encode(values, path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "2",
                    "-i", "pipe:0", "-ar", "48000", "-threads", "1", "-c:a", "aac", "-b:a", "160k",
                    "-movie_timescale", "48000", str(path)], input=values.tobytes(), check=True, timeout=15)


def assess_case(values, geometry, reader, path):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.viseme_stereo import capture

    encode(values, path)
    digest = phone.sha(path)
    decoded = decode_audio(path, source_sha256=digest, inspect_stereo=True, retain_stereo=True)
    mapping = decoded.mapping
    extent = binding.presented_samples(mapping)
    mono = mapping | {"source_channels": 1}
    if extent != len(values) or binding.audio_origin(mono) != 0 or (scope := capture(decoded)) is None:
        raise ValueError("The stereo delivery changed its audio extent, channels or presentation clock")
    lags = []
    for original, received in zip(values.T, decoded.stereo_pcm[:extent].T, strict=True):
        original = original[48:-48].astype(np.float64)
        received = received.astype(np.float64)
        best = max((float(np.dot(original, received[48 + lag:extent - 48 + lag])), lag) for lag in range(-12, 13))[1]
        if abs(best) > 1:
            raise ValueError("A stereo channel shifted by more than one analysis sample")
        lags.append(best)
    clipped = np.rint(decoded.pcm[:extent] * 32768).astype("<i2")
    phone_events = reader(clipped)
    readings = {"frames": geometry["frames"], "phones": phone_events["standard"], "articulation_phones": phone_events["articulation"]}
    metadata = {key: mapping[key] for key in ("samples", "timeline_time_base", "pcm_sha256", "source_channels", "source_sample_rate",
                "source_profile", "source_container", *binding.AAC_PRESENTATION_FIELDS)}
    metadata["timeline"] = [{"start_sample": 0, "end_sample": mapping["samples"], "source_pts": 0}]
    checked = binding.measured(readings, timestamps=geometry["pts"], time_base=geometry["time_base"],
                               source={"width": 1280, "height": 720}, mapping=mono)
    return {"mapping": metadata, "scope": scope, "alignment": lags, "audio_container_sha256": digest,
            "guard_passed": stereo.qualifies(scope, mapping, extent), "witness": checked["witness"],
            "state": checked["status"], "basis": checked["measurement_basis"], "input_sha256": phone.digest(values.tobytes())}


def cases(row, audio, background, rooms):
    import numpy as np

    content, sounds = fixture_inputs(row, audio, background, rooms)
    for scene, roles in SCENES.items():
        mixed = {role: stereo_mix(content[role], scene, sounds, content["original"]) for role in roles}
        gain = min(1., 28_000 / max(1., max(float(np.abs(value).max()) for value in mixed.values())))
        for role, value in mixed.items():
            yield scene, role, np.rint(gain * value).astype("<i2")


def geometry_from_frames(captured):
    rows = captured["frames"]
    return {"pts": [frame["pts"] for frame in rows], "time_base": captured["time_base"],
            "frames": [{k: v for k, v in frame.items() if k not in {"gap", "sample"}}
                       | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * delivery.timing.SCORE_UNITS)}
                       for frame in rows]}


def score(args):
    from app.evidence.video_reference_timing import _landmarker

    frozen = json.loads(args.freeze.read_bytes())
    assert frozen["guard_sha256"] == phone.sha(Path(stereo.__file__))
    assert frozen["retained_plan_sha256"] == phone.sha(args.plan) and frozen["method"] == METHOD
    previous = {row["target"]["front_sha256"]: geometry for row, geometry in delivery.bound_grid(args.plan, args.pictures)}
    sys.path.insert(0, str((args.assets / "runtime").resolve()))
    reader, device = articulation.audio_reader(args.assets / "checkpoint")
    output = []
    with (ZipFile(args.sources / "lombardgrid_audio.zip") as audio, ZipFile(args.sources / "lombardgrid_front.zip") as front,
          TemporaryDirectory(prefix="chipmunk-stereo-lips-") as scratch, _landmarker(args.assets / "face_landmarker.task") as (mp, task)):
        path = Path(scratch)
        for row in frozen["rows"]:
            target = row["target"]
            observed = previous.get(target["front_sha256"])
            if observed is None:
                picture, sound = front.read(target["front"]), audio.read(target["audio"])
                assert phone.digest(picture) == target["front_sha256"] and phone.digest(sound) == target["audio_sha256"]
                (path / "source.mov").write_bytes(picture)
                observed = geometry_from_frames(parent.pictures(path / "source.mov", len(phone.pcm(sound)), mp, task, with_shape=True))
            entries = [{"scene": scene, "role": role} | assess_case(values, observed, reader, path / "case.m4a")
                       for scene, role, values in cases(row, audio, args.environment / "environment", args.rooms)]
            output.append({"source": target["front_sha256"], "source_audio_sha256": target["audio_sha256"],
                           "speaker": row["speaker"], "split": row["split"], "truth": target["span"], "pts": observed["pts"],
                           "time_base": observed["time_base"], "cases": entries,
                           "background_sha256": row["background"]["sha256"], "competing_sha256": row["competing"]["sha256"]})
            print(f"Checked {len(output)}/{len(frozen['rows'])} source videos", flush=True)
    report = {"schema": SCHEMA, "method": METHOD, "freeze_sha256": phone.sha(args.freeze), "guard_sha256": frozen["guard_sha256"],
              "device": device, "rows": output, "results": summarize(output)}
    verify(report)
    args.output.write_bytes(gzip.compress(json.dumps(report, sort_keys=True, separators=(",", ":")).encode(), mtime=0))


def summarize(rows):
    counts = defaultdict(Counter)
    for row in rows:
        for case in row["cases"]:
            positive = case["role"] == "different_words"
            for check, assessed in (("downmix", True), ("agreement", case["guard_passed"])):
                record = counts[f"{row['split']}/{case['scene']}/{case['role']}/{check}"]
                witness = case["witness"] if assessed else None
                found = bool(positive and witness and _matches([row["truth"]], [witness["audio_source_pts"]], .40, 1e-9))
                record.update(total=1, admitted=assessed, useful=found, flags=bool(witness),
                              inconclusive=not assessed or case["state"] not in {"matched", "no_match"})
    return {key: dict(value) for key, value in sorted(counts.items())}


def verify(report):
    from fractions import Fraction

    assert report["schema"] == SCHEMA and report["method"] == METHOD
    assert report["guard_sha256"] == phone.sha(Path(stereo.__file__))
    expected = {(scene, role) for scene, roles in SCENES.items() for role in roles}
    rows = report["rows"]
    assert len({row["source"] for row in rows}) == len(rows) >= 34
    held = {row["speaker"] for row in rows if row["split"] == "held"}
    fresh = {row["speaker"] for row in rows if row["split"] == "new_speaker"}
    assert len(held) == 17 and not held & fresh
    for row in rows:
        assert len(row["cases"]) == len(expected) and {(case["scene"], case["role"]) for case in row["cases"]} == expected
        base = Fraction(row["time_base"])
        assert all(Fraction(1, 28) <= (b - a) * base <= Fraction(1, 22) for a, b in zip(row["pts"], row["pts"][1:]))
        for case in row["cases"]:
            mapping = case["mapping"]
            extent = binding.presented_samples(mapping)
            assert binding.audio_origin(mapping | {"source_channels": 1}) == 0 and extent is not None
            assert 0 <= row["truth"][0] < row["truth"][1] <= extent
            assert stereo.qualifies(case["scope"], mapping, extent) is case["guard_passed"]
            assert len(case["alignment"]) == 2 and all(abs(lag) <= 1 for lag in case["alignment"])
            if witness := case["witness"]:
                assert 0 <= witness["audio_source_pts"][0] < witness["audio_source_pts"][1] <= extent
                frames = [witness["picture_frames"][key] for key in ("before", "during", "after")]
                assert 0 <= frames[0] < frames[1] < frames[2] < len(row["pts"])
                if event := witness.get("articulation"):
                    assert event["picture_frame"] == frames[1] and event["picture_pts"] == row["pts"][frames[1]]
    assert summarize(rows) == report["results"]
    return {"videos": len(rows), "retained_speakers": len(held), "new_speakers": len(fresh), "deliveries": len(rows) * len(expected)}


def mono_totals(rows):
    results = defaultdict(Counter)
    for row in rows:
        for case in row["cases"]:
            for method in ("mono", "stereo"):
                reading = case[method]
                witness = reading["witness"]
                hit = bool(case["role"] == "different_words" and witness and
                           _matches([row["truth"]], [witness["audio_source_pts"]], .40, 1e-9))
                results[f"{case['role']}/{method}"].update(total=1, flags=bool(witness), useful=hit,
                    inconclusive=reading["state"] not in {"matched", "no_match"})
    return {key: dict(value) for key, value in sorted(results.items())}


def mono_check(args):
    from app.evidence.video_reference_timing import _landmarker

    packet = articulation.read(args.receipt)
    verify(packet)
    assert packet["freeze_sha256"] == phone.sha(args.freeze)
    fixed = json.loads(args.freeze.read_bytes())
    fresh = {row["source"]: row for row in packet["rows"] if row["split"] == "new_speaker"}
    sys.path.insert(0, str((args.assets / "runtime").resolve()))
    reader, device = articulation.audio_reader(args.assets / "checkpoint")
    output = []
    with (ZipFile(args.sources / "lombardgrid_audio.zip") as audio, ZipFile(args.sources / "lombardgrid_front.zip") as front,
          TemporaryDirectory(prefix="chipmunk-stereo-mono-compare-") as scratch,
          _landmarker(args.assets / "face_landmarker.task") as (mp, task)):
        path = Path(scratch)
        for row in fixed["rows"]:
            target = row["target"]
            if target["front_sha256"] not in fresh:
                continue
            prior = fresh[target["front_sha256"]]
            picture, sound = front.read(target["front"]), audio.read(target["audio"])
            assert phone.digest(picture) == target["front_sha256"] and phone.digest(sound) == target["audio_sha256"]
            (path / "source.mov").write_bytes(picture)
            observed = geometry_from_frames(parent.pictures(path / "source.mov", len(phone.pcm(sound)), mp, task, with_shape=True))
            assert observed["pts"] == prior["pts"] and observed["time_base"] == prior["time_base"]
            by_case = {case["role"]: case for case in prior["cases"] if case["scene"] == "centered"}
            checked = []
            for scene, role, values in cases(row, audio, args.environment / "environment", args.rooms):
                if scene != "centered":
                    continue
                prior_case = by_case[role]
                assert prior_case["guard_passed"] and phone.digest(values.tobytes()) == prior_case["input_sha256"]
                data, mapping, carrier = delivery.qualified_48k(values[:, 0], path / "case.m4a")
                compared = delivery.compare(observed, reader(data), mapping, {"width": 1280, "height": 720})
                checked.append({"role": role, "carrier": carrier, "mono": compared["articulation"],
                                "stereo": {key: prior_case[key] for key in ("state", "basis", "witness")}})
            output.append({"source": prior["source"], "truth": prior["truth"], "cases": checked})
    assert len(output) == len(fresh)
    record = {"schema": "chipmunk.video-lip-stereo-mono/1", "stereo_sha256": phone.sha(args.receipt),
              "device": device, "rows": output, "results": mono_totals(output)}
    args.output.write_bytes(gzip.compress(json.dumps(record, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    print(json.dumps(record["results"], sort_keys=True), flush=True)


def verify_mono(receipt, stereo_path):
    measured = articulation.read(stereo_path)
    verify(measured)
    assert receipt["schema"] == "chipmunk.video-lip-stereo-mono/1" and receipt["stereo_sha256"] == phone.sha(stereo_path)
    fresh = {row["source"]: row for row in measured["rows"] if row["split"] == "new_speaker"}
    assert {row["source"] for row in receipt["rows"]} == set(fresh) and len(receipt["rows"]) == len(fresh)
    for row in receipt["rows"]:
        previous = fresh[row["source"]]
        assert row["truth"] == previous["truth"] and sorted(case["role"] for case in row["cases"]) == sorted(ROLES)
        by_role = {item["role"]: item for item in previous["cases"] if item["scene"] == "centered"}
        for case in row["cases"]:
            variant = by_role[case["role"]]
            assert variant["guard_passed"] and all(variant[key] == value for key, value in case["stereo"].items())
            extent = case["carrier"]["presented"]
            assert extent == binding.presented_samples(variant["mapping"]) and abs(case["carrier"]["alignment"]) <= 1
            if witness := case["mono"]["witness"]:
                assert 0 <= witness["audio_source_pts"][0] < witness["audio_source_pts"][1] <= extent
                assert all(0 <= frame < len(previous["pts"]) for frame in witness["picture_frames"].values())
    assert mono_totals(receipt["rows"]) == receipt["results"]
    return {"sources": len(fresh), "paired_carriers": len(fresh) * len(ROLES)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for step, names in (("freeze", ("plan", "sources", "environment", "rooms", "output")),
                        ("score", ("freeze", "plan", "pictures", "sources", "environment", "rooms", "assets", "output")),
                        ("mono", ("receipt", "freeze", "sources", "environment", "rooms", "assets", "output")),
                        ("verify", ("receipt",)), ("verify-mono", ("receipt", "stereo"))):
        command = commands.add_parser(step)
        for name in names:
            command.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    elif args.phase == "mono":
        mono_check(args)
    elif args.phase == "verify-mono":
        print(json.dumps(verify_mono(articulation.read(args.receipt), args.stereo), sort_keys=True))
    else:
        print(json.dumps(verify(articulation.read(args.receipt)), sort_keys=True))
