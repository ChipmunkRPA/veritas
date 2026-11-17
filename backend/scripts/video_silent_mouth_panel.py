# SPDX-License-Identifier: Apache-2.0
"""Check near-silent soundtrack locations against known PCM edits and original video clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.video.decode import probe_frames
from app.detectors.video.viseme_review import sound_levels
from app.evidence import video_silent_mouth as quiet
from app.evidence import video_viseme_binding as binding
from app.evidence.video_reference_timing import _landmarker
from scripts import video_lip_articulation_transfer as prior
from scripts import video_lip_delivery_transfer as timing
from scripts import video_lip_sequence_panel as panel
from scripts.export_video_lip_correspondence import (
    make_mux,
    make_soundtracks,
    verify_sound,
    visual_hash,
    waveform,
)

SCHEMA = "chipmunk.video-silent-mouth-study/1"
RATE = quiet.RATE
CONTROLS = ("original", "gain", "same_words", "different_words", "global_early", "global_late", "reduced_gain")
PREPARED = ("muted", "muted_faded", "original")
WIDTHS = (3840, 5760, 8320)
FADE = 120


def truth_window(span, digest, duration):
    width = WIDTHS[int(digest[:8], 16) % len(WIDTHS)]
    middle = sum(span) // 2
    start = min(max(middle - width // 2, 4800), duration - width - 4800) // quiet.HOP * quiet.HOP
    return start, start + width


def render_tracks(ear, row):
    trials = make_soundtracks(ear, row)
    original = trials["original"]
    start, stop = truth_window(row["target"]["span"], row["target"]["front_sha256"], len(original))
    fades = np.ones(stop - start)
    fades[FADE:-FADE] = 0.
    fades[:FADE] = np.linspace(1, 0, FADE)
    fades[-FADE:] = np.linspace(0, 1, FADE)
    for role, gain in (("muted", 0.), ("muted_faded", fades), ("reduced_gain", .10)):
        copied = original.copy()
        copied[start:stop] = np.rint(original[start:stop] * gain)
        trials[role] = copied
        if not (np.array_equal(copied[:start], original[:start]) and np.array_equal(copied[stop:], original[stop:])):
            raise ValueError("Authored mute changed samples outside its truth interval")
    return trials, [start, stop]


def raw_geometry(geometry):
    rows = []
    for frame in geometry["frames"]:
        row = {key: frame[key] for key in ("index", "pts")}
        row["gap_units"] = None if frame["gap"] is None else round(frame["gap"] * binding.timing.SCORE_UNITS)
        if frame["gap"] is not None:
            row.update({key: frame[key] for key in ("mouth_box", "rgb_sha256", "mouth_crop_sha256", "shape")})
        rows.append(row)
    return rows


def measurement(frames, audio, mapping, reader, table, name):
    presented = binding.presented_samples(mapping)
    origin = binding.audio_origin(mapping)
    if origin is None or presented is None:
        return {"baseline": None, "added": None, "without_mouth": [], "status": "unsupported"}
    phonetic = reader(audio[:presented])
    raw = {"frames": frames, "phones": phonetic["standard"], "articulation_phones": phonetic["articulation"]}
    if name == "pcm":
        raw["aperture_prediction"] = binding.ranked.predict(phonetic["groups"])
    coordinates = {"timestamps": table.pts, "time_base": table.time_base,
                   "source": {"width": table.width, "height": table.height}, "mapping": mapping}
    before = binding.measured(raw, **coordinates)
    series = sound_levels(audio.astype(np.float32) / 32768., presented)
    measured = binding.measured(raw | {"sound_levels": series}, **coordinates)
    if before["witness"] is not None and before != measured:
        raise ValueError("This check changed an earlier evidence window")
    result = {"baseline": [v-origin for v in before["witness"]["audio_source_pts"]] if before["witness"] else None,
              "without_mouth": [row["span"] for row in quiet.quiet_ranges(series, presented)],
              "status": before["status"], "added": None}
    if measured["measurement_basis"] == quiet.KEY:
        result.update(added=measured["measurement"]["span"], frames=measured["witness"]["picture_frames"],
                      measurements=measured["measurement"])
    return result


def match(span, truth):
    return bool(span and 5 * max(0, min(span[1], truth[1]) - max(span[0], truth[0]))
                >= 2 * (max(span[1], truth[1]) - min(span[0], truth[0])))


def summary(rows):
    totals = defaultdict(lambda: dict.fromkeys(("views", "current", "combined", "new_flags", "sound_only"), 0))
    for row in rows:
        role, measurement = row["role"], row["reading"]
        counts = totals[f'{row["delivery"]}/{role}']
        counts["views"] += 1
        if role in CONTROLS:
            a, b = bool(measurement["baseline"]), bool(measurement["added"])
            counts["sound_only"] += bool(measurement["without_mouth"])
        else:
            a, b = (match(measurement[key], row["truth"]) for key in ("baseline", "added"))
            counts["sound_only"] += any(match(span, row["truth"]) for span in measurement["without_mouth"])
        counts["current"] += a
        counts["combined"] += a or b
        counts["new_flags"] += measurement["added"] is not None
    return dict(totals)


def execute(plan_path, sources, assets, output):
    roster = panel.panel.read(plan_path)
    work = [row for row in roster["rows"] if row["split"] == "held"]
    sys.path.insert(0, str((assets / "runtime").resolve()))
    task_asset = assets / "face_landmarker.task"
    if panel.panel.sha(task_asset) != prior.study.parent.FACE_SHA256:
        raise ValueError("The installed landmarks differ from the verified weights")
    reader, device = prior.audio_reader(assets / "checkpoint")
    rows = []
    output.mkdir(mode=0o700)
    deliveries = output / "playable"
    deliveries.mkdir(mode=0o700)
    with (ZipFile(sources / "lombardgrid_audio.zip") as ear, ZipFile(sources / "lombardgrid_front.zip") as eye,
          TemporaryDirectory(prefix="silent-mouth-") as directory, _landmarker(task_asset) as (mp, task)):
        directory = Path(directory)
        source = directory / "picture.mov"
        wave = directory / "input.wav"
        carrier = directory / "aac.m4a"
        for index, row in enumerate(work):
            committed = row["target"]["front_sha256"]
            source.write_bytes(eye.read(row["target"]["front"]))
            if panel.panel.sha(source) != committed:
                raise ValueError("Publisher video differs from its committed input")
            table = probe_frames(source, cap=120, timeout_s=12)
            original = waveform(ear, row["target"])
            native = prior.parent.pictures(source, len(original), mp, task, with_shape=True)
            frames = raw_geometry(native)
            identity = visual_hash(source)
            if [r["pts"] for r in frames] != list(table.pts):
                raise ValueError("Native mouth observations differ from the presentation clock")
            tracks, truth = render_tracks(ear, row)
            for role, samples in sorted(tracks.items()):
                wave.write_bytes(wav_bytes(samples.astype(np.float32) / 32768.))
                lossless = decode_audio(wave, source_sha256=panel.panel.sha(wave))
                aac, mapping, codec = timing.qualified_48k(samples, carrier)
                for carrier_name, audio, clock, container, info_sha in (
                    ("pcm", samples, lossless.mapping, wave, panel.panel.sha(wave)),
                    ("aac48", aac, mapping, carrier, codec["file_sha256"]),
                ):
                    read = measurement(frames, audio, clock, reader, table, carrier_name)
                    digest = hashlib.sha256(audio.tobytes()).hexdigest()
                    playable = None
                    if role in PREPARED:
                        playable = f"{index:02d}-{role}-{carrier_name}.mov"
                        make_mux(source, container, deliveries / playable, table, identity)
                        trimmed = audio[:binding.presented_samples(clock)]
                        verify_sound(deliveries / playable, carrier_name, hashlib.sha256(trimmed.tobytes()).hexdigest(), len(trimmed))
                    rows.append({"group": committed, "speaker": row["speaker"], "delivery": carrier_name, "role": role,
                                 "truth": truth if role not in CONTROLS else None,
                                 "pcm_sha256": digest, "audio_carrier_sha256": info_sha,
                                 "picture_timescale": table.time_base, "pts": list(table.pts),
                                 "playable": playable, "playable_sha256": panel.panel.sha(deliveries / playable) if playable else None,
                                 "reading": read})
            print(f"{index+1}/{len(work)} native source clips measured", flush=True)
    results = summary(rows)
    panel.panel.save(output / "result.json.gz", {"schema": SCHEMA, "source_plan_sha256": panel.panel.sha(plan_path),
                                                "device": device, "rows": rows, "totals": results})
    print(json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sources", "assets", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    execute(args.plan, args.sources, args.assets, args.output)
