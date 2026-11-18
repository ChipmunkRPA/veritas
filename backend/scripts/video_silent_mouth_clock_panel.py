# SPDX-License-Identifier: Apache-2.0
"""Compare the existing dropout measurement on genuine 29.97 fps pictures and a 30 fps delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.acquire_video_common_sources_v91 import lfs_pointer
from scripts.video_common_sources_v89 import CREMA, CREMA_REV
from scripts.video_lip_multishot_panel import CARRIERS, RATE, digest, ffmpeg, overlap, read, save

SCHEMA = "chipmunk.video-silent-mouth-clock-panel/1"
SOURCE_INDEX_SHA = "d88cb50e84bbc30b2e159cdfc25e1cf87777ad97c7c41d6ab8f75d3383b8f740"
LICENSE_SHA = "e76747a4efa4b338914bb3f930b5d3d3c3ac15b886bbabf897f6a19731f987f8"
GEOMETRY_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
SOURCE_COUNT = 32
MONTAGE_COUNT = 12
TITLE_FRAMES = 21
FRAME_PERIOD = Fraction(1001, 30_000)
VIDEO_ENCODING = ("-c:v", "libx264", "-threads:v", "2", "-preset", "fast", "-crf", "0", "-pix_fmt", "yuv420p",
                  "-fps_mode", "passthrough", "-enc_time_base", "1/30000", "-video_track_timescale", "30000")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def order(value):
    return sha(f"{SCHEMA}\0{value}".encode())


def fetch(url, size):
    with urlopen(Request(url, headers={"User-Agent": "Chipmunk-Verify-source-review"}), timeout=45) as response:
        return response.read(size + 1)


def acquire(index, license_file, earlier, output):
    if digest(index) != SOURCE_INDEX_SHA or digest(license_file) != LICENSE_SHA:
        raise ValueError("Publisher metadata differs from the reviewed revision")
    previous = {row["speaker"] for row in read(earlier)["rows"]}
    pool = defaultdict(list)
    with index.open(newline="") as source:
        for row in csv.DictReader(source):
            clip = row["Filename"]
            if clip.endswith("_NEU_XX"):
                actor = clip.split("_")[0]
                speaker = sha(f"CREMA-D:{actor}".encode())
                if speaker not in previous:
                    pool[speaker].append(clip + ".flv")
    if len(pool) < SOURCE_COUNT:
        raise ValueError("The prespecified panel requires more distinct recording participants")
    selection = [{"speaker": speaker, "name": min(pool[speaker], key=order)}
                 for speaker in sorted(pool, key=order)[:SOURCE_COUNT]]
    output.mkdir(mode=0o700, exist_ok=True)
    plan = {"schema": SCHEMA, "earlier_panel_sha256": digest(earlier), "source_index_sha256": SOURCE_INDEX_SHA,
            "license_sha256": LICENSE_SHA, "publisher_revision": CREMA_REV, "selection": selection}
    with (output / "plan.json").open("x") as file:
        json.dump(plan, file, sort_keys=True)

    def materialize(entry):
        name = entry["name"]
        pointer = fetch(f"{CREMA}/VideoFlash/{name}", 1024)
        expected, size = lfs_pointer(pointer)
        if size > 2 * 1024**2:
            raise ValueError("Unexpected acted speech clip size")
        media = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{CREMA_REV}/VideoFlash/{name}"
        data = fetch(media, size)
        if len(data) != size or sha(data) != expected:
            raise ValueError(f"Publisher object differs from its pinned pointer for {name}")
        with (output / name).open("xb") as file:
            file.write(data)
        return entry | {"source_sha256": expected, "bytes": size}

    with ThreadPoolExecutor(max_workers=5) as workers:
        rows = list(workers.map(materialize, selection))
    with (output / "sources.json").open("x") as file:
        json.dump(plan | {"selection": rows}, file, sort_keys=True)
    print(f"Verified publisher digests for {len(rows)} source recordings", flush=True)


def summarize(rows):
    counts = defaultdict(Counter)
    for row in rows:
        truth, observed = row["truth"], row["matches"]
        hits = sum(truth is not None and overlap(truth, span) >= .4 for span in observed)
        counts[f"{row['clock']}/{row['carrier']}/{row['role']}"].update(
            files=1, assessed=row["assessed"], flagged=bool(observed), localized=bool(hits),
            missed=truth is not None and not hits, other_findings=len(observed) - hits,
        )
    return dict(counts)


def pixels(path, table):
    from app.detectors.video.decode import decode_stream

    with closing(decode_stream(path, table, max_side=None)) as stream:
        frames = [sha(frame.rgb.tobytes()) for frame in stream if frame.pts_verified]
    if len(frames) != len(table):
        raise ValueError("The exact frame scan did not finish")
    return frames


def source_sound(sound, table):
    import numpy as np

    timeline = sound.mapping["timeline"]
    origin = timeline[0]["source_pts"]
    shift = round(table.pts[0] * table.tb * RATE) - origin
    if (sound.mapping["source_channels"] != 1 or not 0 <= shift <= 1600
            or any(abs(row["source_pts"] - row["start_sample"] - origin) > 16 for row in timeline)):
        raise ValueError("The source recording has an unsupported mono soundtrack clock")
    return np.rint(sound.pcm[shift:] * 32768).astype("<i2"), shift


def fit_sound(pcm, extent):
    import numpy as np

    if abs(len(pcm) - extent) > 1600:
        raise ValueError("The picture and soundtrack spans disagree by more than 100 ms")
    return np.pad(pcm[:extent], (0, max(0, extent - len(pcm))))


def treatments(pcm, target_start, number):
    """Choose a speech-active window using amplitude and source position only."""
    import numpy as np

    width = (3840, 5760, 8320)[number % 3]
    words = pcm[target_start:].astype(np.float64)
    cumulative_energy = np.r_[0, np.cumsum(words**2)]
    margin = 2560
    starts = range(margin, len(words) - width - margin + 1, 80)

    def rms(lo, hi):
        return ((cumulative_energy[hi] - cumulative_energy[lo]) / (hi - lo)) ** .5

    active = [i for i in starts if min(rms(i - 1920, i), rms(i + width, i + width + 1920)) >= 192
              and rms(i, i + width) >= 384]
    center = (len(words) - width) // 2
    start = (target_start + min(active or starts, key=lambda i: abs(i - center))) // 80 * 80
    stop = start + width
    yield "original", pcm, None
    yield "lower_level", np.r_[pcm[:start], np.rint(.1 * pcm[start:stop]).astype("<i2"), pcm[stop:]], None
    yield "gain", np.rint(.53 * pcm).astype("<i2"), None
    edit = pcm.copy()
    edit[start:stop] = 0
    yield "muted", edit, [start, stop]
    faded = edit.copy()
    ramp = np.linspace(1., 0., 120)
    faded[start:start + 120] = np.rint(pcm[start:start + 120] * ramp)
    faded[stop - 120:stop] = np.rint(pcm[stop - 120:stop] * ramp[::-1])
    yield "muted_faded", faded, [start, stop]
    if target_start:
        across_cut = pcm.copy()
        across_cut[target_start - 2560:target_start + 3520] = 0
        yield "across_cut", across_cut, None


def mux(pictures, pcm, carrier, delivery):
    sound = ("-c:a", "pcm_s16le") if carrier == "pcm" else ("-c:a", "aac", "-ar", "48000", "-b:a", "112k")
    ffmpeg("-i", str(pictures), "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *sound, "-video_track_timescale", "30000",
           "-movie_timescale", "240000", str(delivery), data=pcm.tobytes())


def measure_deliveries(pictures, table, planned_shots, pcm, target_start, clock, number, assets, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from app.detectors.video.viseme_review import sound_levels, supported, supported_shots
    from app.detectors.video.viseme_worker import capture_mouth_frames
    from app.evidence import video_lip_agreement as agreement
    from app.evidence import video_silent_mouth as silent
    from app.evidence import video_viseme_binding as binding

    with closing(decode_stream(pictures, table)) as stream:
        scan = detect_shots(stream, table).as_json()
    group = f"{clock}-{number:02d}"
    shots = scan["ranges"]
    if scan["frames_scanned"] != len(table) or shots != planned_shots:
        return [], {"group": group, "planned_shots": planned_shots, "observed_shots": shots,
                    "scanned_pictures": scan["frames_scanned"]}
    frames, _, _ = capture_mouth_frames(pictures, table, assets, motion=False, shots=shots)
    video_stream = ("-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1")
    encoded = ffmpeg("-i", str(pictures), *video_stream)
    rows = []
    for role, edited, truth in treatments(pcm, target_start, number):
        for carrier in CARRIERS:
            delivery = output / f"{group}-{carrier}-{role}.mov"
            mux(pictures, edited, carrier, delivery)
            native = probe_frames(delivery)
            if ((native.pts, native.time_base) != (table.pts, table.time_base)
                    or ffmpeg("-i", str(delivery), *video_stream) != encoded):
                raise ValueError("Soundtrack encoding changed a picture or its source time")
            pin = digest(delivery)
            sound = decode_audio(delivery, source_sha256=pin)
            mapping = sound.mapping
            samples = binding.presented_samples(mapping, max_samples=agreement.MAX_MULTISHOT_SAMPLES)
            if (samples != len(pcm) or not supported_shots(native, mapping, shots) or supported(native, mapping, shots)
                    or (carrier == "pcm" and not np.array_equal(edited, np.rint(sound.pcm * 32768)))):
                raise ValueError("A delivery differs from the planned picture scope or audio clock")
            bound = binding.measured(
                {"frames": frames, "phones": [], "sound_levels": sound_levels(sound.pcm, samples), silent.SHOT_KEY: shots},
                timestamps=native.pts, time_base=native.time_base,
                source={"width": table.width, "height": table.height}, mapping=mapping,
            )
            matches = [exhibit["witness"]["audio_source_pts"] for _, exhibit in binding.exhibits(bound)]
            rows.append({"group": group, "clock": clock, "role": role, "carrier": carrier, "delivery": delivery.name,
                         "input_sha256": pin, "decoded_pcm_sha256": mapping["pcm_sha256"], "truth": truth,
                         "matches": matches, "coverage": bound["shot_coverage"], "status": bound["status"],
                         "assessed": bound["status"] in {"matched", "no_match"}})
    return rows, None


def measure(sources, assets, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import probe_frames

    geometry = assets / "face_landmarker.task"
    if digest(geometry) != GEOMETRY_SHA:
        raise ValueError("The reviewed landmark asset is required")
    sys.path.insert(0, str(assets / "runtime"))
    acquired = json.loads((sources / "sources.json").read_text())
    selection = acquired.pop("selection")
    output.mkdir(mode=0o700)
    rows, clocks, unresolved, retained = [], [], [], []
    for i, entry in enumerate(selection, start=1):
        source = sources / entry["name"]
        source_hash = digest(source)
        if source_hash != entry["source_sha256"]:
            raise ValueError("A source recording differs from the frozen publisher digest")
        native = probe_frames(source)
        if (native.truncated or native.rotation or native.interpolated or not 40 <= len(native) <= 120
                or any(abs((pts - native.pts[0]) * native.tb - j * FRAME_PERIOD) > Fraction(3, 2000)
                       for j, pts in enumerate(native.pts))):
            unresolved.append({"source_sha256": source_hash, "reason": "Source picture clock outside the comparison"})
            continue
        pcm, shift = source_sound(decode_audio(source, source_sha256=source_hash), native)
        original_pixels = pixels(source, native)
        for clock, pts_formula, ticks in (
            ("native-29.97", "PTS-STARTPTS", [round((pts - native.pts[0]) * native.tb * 30_000) for pts in native.pts]),
            ("exact-30", "N*1000", [j * 1000 for j in range(len(native))]),
        ):
            pictures = output / f"{clock}-{i:02d}-pictures.mp4"
            packet_duration = 1001 if clock == "native-29.97" else 1000
            ffmpeg("-i", str(source), "-an", "-map", "0:v:0", "-vf", f"settb=1/30000,setpts={pts_formula}",
                   *VIDEO_ENCODING, "-bsf:v", f"setts=duration={packet_duration}", str(pictures))
            table = probe_frames(pictures)
            if table.tb != Fraction(1, 30_000) or list(table.pts) != ticks or pixels(pictures, table) != original_pixels:
                raise ValueError("A staged delivery changed the planned timestamps or source picture pixels")
            period = FRAME_PERIOD if clock == "native-29.97" else Fraction(1, 30)
            extent = round((table.pts[-1] * table.tb + period) * RATE)
            fitted = fit_sound(pcm, extent)
            observed, skipped = measure_deliveries(pictures, table, [[0, len(table)]], fitted, 0, clock, i, geometry, output)
            key = f"{clock}-{i:02d}"
            clocks.append({"group": key, "source_sha256": source_hash, "speaker": entry["speaker"],
                           "source_start_pts": native.pts[0], "source_time_base": native.time_base,
                           "trimmed_leading_audio_samples": shift, "source_tail_adjustment_samples": extent - len(pcm),
                           "ticks": ticks, "sample_extent": extent, "rgb_sha256": sha("\n".join(original_pixels).encode())})
            rows.extend(observed)
            if skipped:
                unresolved.append(skipped)
            if clock == "exact-30":
                retained.append((i, pictures, table, fitted, original_pixels))
        print(f"Verified both picture clocks for source {i}/{len(selection)}", flush=True)

    for (previous, lead, a, lead_pcm, left), (i, target, b, target_pcm, right) in zip(
        retained[0:2 * MONTAGE_COUNT:2], retained[1:2 * MONTAGE_COUNT:2], strict=True,
    ):
        clock = "montage-30"
        path = output / f"{clock}-{i:02d}-pictures.mp4"
        title = f"color=c=0x315e95:s={a.width}x{a.height}:r=30:d=0.7"
        ffmpeg("-i", str(lead), "-f", "lavfi", "-i", title, "-i", str(target), "-an", "-filter_complex",
               "[0:v][1:v][2:v]concat=n=3:v=1:a=0,settb=1/30000,setpts=N*1000[out]", "-map", "[out]",
               *VIDEO_ENCODING, "-bsf:v", "setts=duration=1000", str(path))
        table = probe_frames(path)
        joined = pixels(path, table)
        start = len(a) + TITLE_FRAMES
        cuts = [[0, len(a)], [len(a), start], [start, start + len(b)]]
        if (list(table.pts) != [j * 1000 for j in range(start + len(b))] or table.tb != Fraction(1, 30_000)
                or joined[:len(a)] != left or joined[start:] != right):
            raise ValueError("The montage changed a retained recording frame or planned clock")
        title_pcm = np.rint(400 * np.sin(np.arange(TITLE_FRAMES * RATE // 30) * (2 * np.pi * 60 / RATE))).astype("<i2")
        pcm = np.r_[lead_pcm, title_pcm, target_pcm]
        observed, skipped = measure_deliveries(path, table, cuts, pcm, len(lead_pcm) + len(title_pcm), clock, i, geometry, output)
        rows.extend(observed)
        clocks.append({"group": f"{clock}-{i:02d}", "components": [f"exact-30-{previous:02d}", f"exact-30-{i:02d}"],
                       "sample_extent": len(pcm), "rgb_sha256": sha("\n".join(joined).encode())})
        if skipped:
            unresolved.append(skipped)
        print(f"Checked source boundaries of 30 fps montage {i:02d}", flush=True)
    summary = summarize(rows)
    save(output / "result.json.gz", {**acquired, "geometry_sha256": GEOMETRY_SHA, "clocks": clocks, "rows": rows,
                                     "unresolved": unresolved, "summary": summary})
    print(json.dumps(summary, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_subparsers(dest="action", required=True)
    source = mode.add_parser("acquire")
    for name in ("index", "license_file", "earlier", "output"):
        source.add_argument(name, type=Path)
    trial = mode.add_parser("measure")
    for name in ("sources", "assets", "output"):
        trial.add_argument(name, type=Path)
    panel = mode.add_parser("review")
    panel.add_argument("receipt", type=Path)
    args = vars(cli.parse_args())
    action = args.pop("action")
    if action == "acquire":
        acquire(**args)
    elif action == "measure":
        measure(**args)
    else:
        receipt = read(args["receipt"])
        summary = summarize(receipt["rows"])
        if receipt["schema"] != SCHEMA or summary != receipt["summary"]:
            raise ValueError("The trial results do not recount")
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
