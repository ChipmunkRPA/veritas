# SPDX-License-Identifier: Apache-2.0
"""Build local camera-cut trials with separate picture and soundtrack readbacks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

SCHEMA = "chipmunk.video-lip-multishot-panel/1"
WIDTH, HEIGHT, FPS, RATE = 480, 320, 25, 16_000
STEP = RATE // FPS
SLATE = 18
ROLES = ("original", "gain", "global_early", "global_late", "same_words", "different_words")
CARRIERS = ("pcm", "aac48")
CUT_POINTS = {"pcm": 6137394, "aac48": 6178780}


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def ffmpeg(*args, data=None):
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", *args],
                            input=data, capture_output=True, timeout=25)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace"))
    return result.stdout


def picture(path):
    import numpy as np

    pixels = ffmpeg("-i", str(path), "-map", "0:v:0", "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
                    "-c:v", "rawvideo", "-threads:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


def audio(path, count=None):
    import numpy as np

    data = ffmpeg("-i", str(path), "-map", "0:a:0", "-ar", str(RATE), "-ac", "1", "-f", "s16le", "pipe:1")
    pcm = np.frombuffer(data, dtype="<i2")
    if count is None:
        return pcm
    if count - len(pcm) >= STEP:
        raise ValueError("The treatment ends before its original soundtrack")
    return np.pad(pcm[:count], (0, max(0, count - len(pcm))))


def render_pictures(lead, target, path, *, slate=True):
    import numpy as np

    pieces = [lead]
    if slate:
        card = np.full((SLATE, HEIGHT, WIDTH, 3), (41, 93, 155), dtype=np.uint8)
        card[:, 148:152, 80:400, :] = 238
        pieces.append(card)
    pieces.append(target)
    pictures = np.concatenate(pieces)
    ffmpeg("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
           "-c:v", "libx264", "-threads:v", "2", "-crf", "18", "-pix_fmt", "yuv420p", str(path), data=pictures.tobytes())


def mux(pictures, pcm, carrier, delivery):
    sound = ("-c:a", "pcm_s16le") if carrier == "pcm" else ("-c:a", "aac", "-ar", "48000", "-b:a", "112k")
    ffmpeg("-i", str(pictures), "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *sound, str(delivery), data=pcm.tobytes())


def pinned_inputs(native):
    receipt = read(native / "review.json.gz")
    inputs = {row["id"]: native / row["input"] for row in receipt["file_index"]["cases"]}
    rows = {row["id"]: row for row in receipt["deliveries"]}

    def path(index, role, source_sha):
        case = f"seq-{index:02d}-pcm-{role}"
        value = inputs[case]
        if digest(value) != rows[case]["input_sha256"] or source_sha != rows[case]["source_sha256"]:
            raise ValueError("The short source delivery differs from its roster or readback")
        return value

    return path


def materialize(sources, plan, output):
    """Prepare original and matched speech variants from frozen publisher members."""
    import wave

    from app.detectors.video.decode import probe_frames
    from scripts.export_video_lip_correspondence import make_mux, make_soundtracks, visual_hash

    output.mkdir(mode=0o700)
    clips = output / "clips"
    clips.mkdir()
    inputs, names = [], []
    with (ZipFile(sources / "lombardgrid_front.zip") as front, ZipFile(sources / "lombardgrid_audio.zip") as sound,
          TemporaryDirectory(prefix="lip-shot-source-") as scratch):
        video, wav = (Path(scratch) / name for name in ("original.mov", "track.wav"))
        for index, case in enumerate(read(plan)["rows"], 1):
            target = case["target"]
            video.write_bytes(front.read(target["front"]))
            if digest(video) != target["front_sha256"]:
                raise ValueError("The original picture member differs from the frozen roster")
            table = probe_frames(video, cap=120, timeout_s=20, max_pixels=1280 * 720)
            expected_picture = visual_hash(video)
            for role, values in make_soundtracks(sound, case).items():
                name = f"seq-{index:02d}-pcm-{role}"
                with wave.open(str(wav), "wb") as handle:
                    handle.setparams((1, 2, RATE, 0, "NONE", ""))
                    handle.writeframes(values.tobytes())
                delivery = clips / f"{name}.mov"
                make_mux(video, wav, delivery, table, expected_picture)
                names.append({"id": name, "input": delivery.relative_to(output).as_posix()})
                inputs.append({"id": name, "source_sha256": target["front_sha256"], "input_sha256": digest(delivery)})
    save(output / "review.json.gz", {"source_plan_sha256": digest(plan), "file_index": {"cases": names}, "deliveries": inputs})


def save(path, record):
    with path.open("xb") as file:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        file.write(gzip.compress(payload, mtime=0))


def overlap(a, b):
    common = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return common / (max(a[1], b[1]) - min(a[0], b[0]))


def recount(rows):
    counts = defaultdict(Counter)
    for row in rows:
        matched = row["matches"]
        truth = row["truth"]
        correct = sum(truth is not None and overlap(span, truth) >= .4 for span in matched)
        counts[row["carrier"], row["role"]].update(
            files=1, earlier_eligible=row["earlier_eligible"], assessed=row["assessed"], exact_cuts=row["exact_cuts"],
            flagged=bool(matched), correctly_located=bool(correct), misplaced=len(matched) - correct,
        )
    return {f"{carrier}/{role}": dict(values) for (carrier, role), values in counts.items()}


def motion_shots(models, patches, valid, table, pcm, shots, point):
    import numpy as np

    from app.detectors.video.viseme_worker import _compare_motion
    from app.evidence import video_lip_agreement as agreement

    clocks = agreement.shot_audio_ranges(table.pts, table.time_base, len(pcm), shots)
    times = np.asarray([round(pts * table.tb * RATE) for pts in table.pts])
    readings, coverage, matches = [], [], []
    for (start, stop), (lo, hi) in zip(shots, clocks, strict=True):
        status, proposal = "outside_scope", None
        if agreement.eligible_shot(start, stop):
            status, proposal = _compare_motion(models, patches[start:stop], valid[start:stop], times[start:stop] - lo, pcm[lo:hi])
            if proposal is not None:
                proposal["span"] = [lo + sample for sample in proposal["span"]]
                for key in ("bracket_frames", "picture_span"):
                    proposal[key] = [start + index for index in proposal[key]]
                if proposal["score_units"] > point:
                    matches.append(proposal["span"])
        readings.append({"frames": [start, stop], "state": status, "proposal": proposal})
        coverage.append({"frames": [start, stop], "audio_samples": [lo, hi], "status": status})
    return readings, coverage, matches


def compare(native, plan, assets, output, *, limit=None, direct=False, earlier_points=False):
    import numpy as np
    import torch

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from app.detectors.video.viseme_review import supported, supported_shots
    from app.detectors.video.viseme_worker import _motion_models, capture_mouth_frames
    from app.evidence import video_lip_agreement as agreement
    from app.evidence import video_viseme_binding as binding

    torch.set_num_threads(1)
    sys.path.insert(0, str(assets / "video-lip-timing/runtime"))
    geometry = assets / "video-lip-timing/face_landmarker.task"
    models = _motion_models(assets / "video-lip-agreement")
    output.mkdir(mode=0o700)
    cases = read(plan)["rows"]
    source = pinned_inputs(native)
    rows = []
    for index, case in enumerate(cases[:limit], 1):
        intro = index % len(cases) + 1
        lead = source(intro, "original", cases[intro - 1]["target"]["front_sha256"])
        target_sha = case["target"]["front_sha256"]
        target = source(index, "original", target_sha)
        # Keep only the common picture and soundtrack extent, then pad at
        # most one partial picture at the end. Source speech time stays fixed.
        lead_frames = picture(lead)[:(len(audio(lead)) + STEP - 1) // STEP]
        target_frames = picture(target)[:(len(audio(target)) + STEP - 1) // STEP]
        if case["target"]["span"][1] > len(target_frames) * STEP:
            raise ValueError("The planned speech replacement lies beyond the delivered pictures")
        slate = 0 if direct else SLATE
        target_start = len(lead_frames) + slate
        end = target_start + len(target_frames)
        intended = [[0, len(lead_frames)], [len(lead_frames), target_start], [target_start, end]] if slate else [[0, target_start], [target_start, end]]
        video = output / f"group-{index:02d}-pictures.mov"
        render_pictures(lead_frames, target_frames, video, slate=not direct)
        table = probe_frames(video, cap=agreement.MAX_MULTISHOT_FRAMES + 1)
        if (len(table) != end or not all(float(pts * table.tb) == i / FPS for i, pts in enumerate(table.pts))):
            raise ValueError("The rendered pictures do not use the planned frame clock")
        with closing(decode_stream(video, table)) as frames:
            observed = detect_shots(frames, table).as_json()
        shots = observed["ranges"]
        if observed["frames_scanned"] != end:
            raise ValueError("Shot scan did not read every rendered picture")
        picture_hash = ffmpeg("-i", str(video), "-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1")
        _, patches, valid = capture_mouth_frames(video, table, geometry, motion=True, shots=shots)
        first = audio(lead, len(lead_frames) * STEP)
        for role in ROLES:
            late = source(index, role, target_sha)
            mixed = np.concatenate((first, np.zeros(slate * STEP, dtype="<i2"), audio(late, len(target_frames) * STEP)))
            for carrier in CARRIERS:
                file = output / f"group-{index:02d}-{carrier}-{role}.mov"
                mux(video, mixed, carrier, file)
                actual = probe_frames(file, cap=agreement.MAX_MULTISHOT_FRAMES + 1)
                if actual.pts != table.pts or actual.time_base != table.time_base or (
                    ffmpeg("-i", str(file), "-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1") != picture_hash
                ):
                    raise ValueError("The muxed pictures differ from the verified montage")
                pin = digest(file)
                sound = decode_audio(file, source_sha256=pin)
                clock = sound.mapping
                sample_count = binding.presented_samples(clock, max_samples=agreement.MAX_MULTISHOT_SAMPLES)
                if sample_count != len(mixed) or (carrier == "pcm" and not np.array_equal(np.rint(sound.pcm * 32768), mixed)):
                    raise ValueError("The muxed soundtrack differs from the planned presentation")
                readings, coverage, matches = None, None, []
                if supported_shots(table, clock, shots):
                    point = agreement.POINTS[carrier][0] if earlier_points else CUT_POINTS[carrier]
                    values = np.rint(sound.pcm[:sample_count] * 32768).astype("<i2")
                    readings, coverage, matches = motion_shots(models, patches, valid, table, values, shots, point)
                truth = [target_start * STEP + pos for pos in case["target"]["span"]] if role == "different_words" else None
                rows.append({"group": index, "speaker": case["speaker"], "lead_speaker": cases[intro - 1]["speaker"],
                             "source_sha256": digest(late), "lead_sha256": digest(lead), "input_sha256": pin,
                             "delivery": file.name, "carrier": carrier, "role": role, "truth": truth, "matches": matches,
                             "exact_cuts": shots == intended, "planned_shots": intended, "observed_shots": shots,
                             "earlier_eligible": bool(supported(table, clock, shots)), "coverage": coverage, "readings": readings,
                             "assessed": bool(coverage and any(row["status"] in {"measured", "no_local_dip", "whole_track_delay"}
                                                             for row in coverage))})
        print(f"Compared montage {index}/{len(cases[:limit])}, planned {intended}, observed {shots}", flush=True)
    counts = recount(rows)
    result = {"schema": SCHEMA, "counts": counts, "rows": rows, "direct_cuts": direct, "source_plan_sha256": digest(plan),
              "source_replay_sha256": digest(native / "review.json.gz"), "recipe_sha256": digest(Path(__file__)),
              "weights": {name: agreement.WEIGHT_PINS[name] for name in ("vision", "shared")}}
    save(output / "result.json.gz", result)
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    run = modes.add_parser("compare")
    for arg in ("native", "plan", "assets", "output"):
        run.add_argument(arg, type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--direct", action="store_true")
    run.add_argument("--earlier-points", action="store_true")
    build = modes.add_parser("materialize")
    for arg in ("sources", "plan", "output"):
        build.add_argument(arg, type=Path)
    modes.add_parser("review").add_argument("receipt", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "compare":
        compare(**args)
    elif action == "materialize":
        materialize(**args)
    else:
        result = read(args["receipt"])
        counts = recount(result["rows"])
        if result["schema"] != SCHEMA or result["counts"] != counts:
            raise ValueError("The measured results do not recount")
        print(json.dumps(counts, sort_keys=True))
