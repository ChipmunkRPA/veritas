# SPDX-License-Identifier: Apache-2.0
"""Build and replay matched 48 kHz stereo timing-change tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.audio_native_alignment_panel import digest, iou, read, save, save_wav, segment

SCHEMA = "chipmunk.stereo-timing-change-panel/1"
SAMPLES = 180_000
STARTS = {
    "development": (13, 43, 78, 112, 151, 177, 202, 239),
    "confirmation": (22, 54, 81, 108, 134, 171, 207, 243),
}
VOICES = {"development": ("ES2004a", "TS3004a"), "confirmation": ("IS1006b", "ES2011b")}
DELAYS = ((14, -15, 19, 21, -17, 7, -13, 10), (-17, 4, -11, -8, 12, -8, 8, -15))
PCM = "pcm"
CARRIERS = (PCM, "aac96", "aac160")
GENERATORS = ("tts", "voice_conversion")


def _rms(values):
    import numpy as np

    return float(np.sqrt(np.mean(values.astype(float) ** 2)))


def _fade(start, stop, edge):
    import numpy as np

    alpha = np.zeros(SAMPLES)
    alpha[start:stop] = 1.
    alpha[start:start + edge] = np.linspace(0., 1., edge)
    alpha[stop - edge:stop] = np.linspace(1., 0., edge)
    return alpha[:, None]


def _render(room, voice, alternates, base, level_db):
    import numpy as np

    from scripts.audio_channel_alignment_panel import delayed

    factor = _rms(room) / max(1e-4, _rms(voice)) * 10 ** (level_db / 20)
    voices = [voice * factor, *(part * factor for part in alternates)]
    ceiling = max(float(np.max(np.abs(x))) for x in voices) + float(np.max(np.abs(room)))
    scale = min(1., 24_000. / max(1., ceiling))

    def stereo(value, offset):
        return np.column_stack((value, .93 * delayed(value, offset)))

    def render(foreground):
        return np.rint((foreground + room) * scale).astype("<i2")

    return stereo(voices[0], base), voices, stereo, render


def treatments(room, voice, donor, variant):
    import numpy as np

    base, inside = (values[variant] for values in DELAYS)
    start = 36621 + 2531 * variant
    stop = start + (12500, 19200, 25440, 29760)[variant % 4]
    alpha = _fade(start, stop, 480 + 240 * (variant % 3))
    donor *= _rms(voice) / max(1e-4, _rms(donor))
    level_db = (8, 13)[variant % 2]
    original, (original_voice, alternate_voice), stereo, render = _render(room, voice, (donor,), base, level_db)
    difference = stereo(original_voice, inside)
    new_voice = stereo(alternate_voice, inside)
    same_delay = stereo(alternate_voice, base)
    times = np.arange(SAMPLES)
    movement = np.clip((times - start + 5760) / 11520, 0, 1) * np.clip((stop + 5760 - times) / 11520, 0, 1)
    lag = base + (inside - base) * movement
    slow_turn = np.column_stack((original_voice, .93 * np.interp(times - lag, times, original_voice)))
    for role, audio, changed in (
        ("delay_change", original * (1 - alpha) + difference * alpha, True),
        ("alternate_voice_with_delay_change", original * (1 - alpha) + new_voice * alpha, True),
        ("ordinary", original, False),
        ("alternate_voice_same_delay", original * (1 - alpha) + same_delay * alpha, False),
        ("right_level_change", original * (1 - alpha * [0., .25]), False),
        ("whole_track_different_delay", difference, False),
        ("gradual_turn", slow_turn, False),
    ):
        yield role, render(audio), [start, stop] if changed else None, [base, inside] if changed else None, level_db
    yield "recorded_environment", room, None, None, None


def scene_cases(folders, speakers, split):
    import numpy as np
    from scipy.signal import resample_poly

    recording = read(folders / "recorded-sources.json")
    foreground = {name: speakers / f"{name}.Array1-05.wav" for name in VOICES[split]}
    for position, desc in enumerate(recording):
        scene = desc["scene"]
        files = [folders / scene / Path(item["member"]).name for item in desc["members"]]
        if any(digest(p) != item["sha256"] for p, item in zip(files, desc["members"], strict=True)):
            raise ValueError("Background channels differ from the pinned publisher archive")
        names = VOICES[split][position % 2], VOICES[split][(position + 1) % 2]
        for index, start in enumerate(STARTS[split]):
            room = np.column_stack([segment(path, start, SAMPLES, 48000) for path in files])
            voice_start, donor_start = start + position * 10 + 11, start + position * 10 + 23
            voice, donor = (resample_poly(segment(foreground[name], offset, SAMPLES // 3, 16000).astype(float), 3, 1)
                            for name, offset in zip(names, (voice_start, donor_start), strict=True))
            for role, audio, truth, offsets, level in treatments(room, voice, donor, index):
                yield {"group": scene, "index": index, "role": role, "truth": truth, "offsets": offsets,
                       "background_start_seconds": start, "speech_sources": list(names),
                       "speech_starts_seconds": [voice_start, donor_start], "speech_over_background_db": level}, audio


def natural_cases(speakers):
    import numpy as np
    from scipy.signal import resample_poly

    for name in VOICES["confirmation"]:
        files = [speakers / f"{name}.Array1-{channel}.wav" for channel in ("05", "08")]
        for position, start in enumerate((17, 63, 92, 128, 155, 188, 221, 261)):
            stereo = np.column_stack([segment(path, start, SAMPLES // 3, 16000) for path in files])
            prepared = np.rint(resample_poly(stereo.astype(float), 3, 1, axis=0)).clip(-32768, 32767).astype("<i2")
            yield {"group": name, "index": position, "role": "natural_stereo_conversation", "truth": None,
                   "offsets": None, "speech_starts_seconds": [start]}, prepared


def synthetic_cases(archive, rooms):
    import numpy as np
    from scipy.signal import resample_poly

    source = read(archive / "freeze.json")
    kinds = {"untouched": "genuine", "edited/ChatterboxTTS": "tts", "edited/ChatterboxVC": "voice_conversion"}
    rows = {(row["group"], row["kind"]): row for row in source["cases"] if row["panel"] == "vc_dry"}
    scenes = read(rooms / "recorded-sources.json")
    for index, desc in enumerate(source["sources"]):
        scene = scenes[index % len(scenes)]["scene"]
        start = 44 + 19 * index
        room = np.column_stack([segment(rooms / scene / f"ch{channel}.wav", start, SAMPLES, 48000)
                                for channel in ("01", "08")])
        one, two = desc["truth"]
        crop = (one + two) // 2 - SAMPLES // 6
        variants, sources = {}, {}
        for kind, name in kinds.items():
            sha = rows[index, kind]["pcm_sha256"]
            src = archive / "pcm" / f"{sha}.pcm"
            raw = src.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Synthetic test source differs from its published commitment")
            data = np.frombuffer(raw, dtype="<i2")[crop:crop + SAMPLES // 3].astype(float)
            if len(data) != SAMPLES // 3:
                raise ValueError("Synthetic test crop goes outside its source")
            variants[name] = resample_poly(data, 3, 1)
            sources[name] = sha
        low, high = (3 * (s - crop) for s in (one, two))
        alpha = _fade(low, high, 480)
        base, inside = (values[index] for values in DELAYS)
        original, voices, stereo, render = _render(room, variants["genuine"], [variants[key] for key in GENERATORS], base, 11)
        info = {"group": desc["reader_sha256"], "index": index, "synthetic_parent_sha256": sources,
                "background_scene": scene, "background_start_seconds": start, "crop_sample_16k": crop}
        for name, value in zip(GENERATORS, voices[1:], strict=True):
            # Keep the outer channel timing identical across these variants.
            control = original * (1 - alpha) + stereo(value, base) * alpha
            suspect = original * (1 - alpha) + stereo(value, inside) * alpha
            yield info | {"role": name, "truth": [low, high], "offsets": [base, inside]}, render(suspect)
            yield info | {"role": f"{name}_same_delay", "truth": None, "offsets": None}, render(control)
        yield info | {"role": "genuine_same_words", "truth": None, "offsets": None}, render(original)


def _deliver(work, base, carrier, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.evidence.audio_stereo_timing_change import locate

    path = base
    if carrier != PCM:
        path = output / f"{base.stem}-{carrier}.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(base), "-map_metadata", "-1",
                        "-c:a", "aac", "-b:a", f"{carrier[3:]}k", str(path)], check=True, timeout=20)
    sha = digest(path)
    decoded = decode_audio(path, source_sha256=sha)
    presented = decode_native_stereo(path, decoded.mapping, aac=carrier != PCM)
    if presented is None or len(presented) != 4 * SAMPLES or decoded.mapping["timeline"][0]["source_pts"] != 0:
        raise ValueError("The native delivery did not preserve its stereo sample extent")
    array = np.frombuffer(presented, dtype="<i2").reshape(-1, 2)
    assessed, measured = locate(array)
    return work | {"carrier": carrier, "delivery": path.name, "input_sha256": sha, "parent_sha256": digest(base),
                   "presented_pcm_sha256": hashlib.sha256(presented).hexdigest(),
                   "assessed": assessed, "measurement": measured}


def totals(rows):
    table = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        measured, truth, offsets = (row[k] for k in ("measurement", "truth", "offsets"))
        correct = bool(measured and truth and iou((measured["start_sample"], measured["end_sample"]), truth) >= .40
                       and abs(measured["before_offset_samples"] - offsets[0]) <= 3
                       and abs(measured["after_offset_samples"] - offsets[0]) <= 3
                       and abs(measured["inside_offset_samples"] - offsets[1]) <= 3)
        table[row["carrier"]][row["role"]].update(n=1, assessed=row["assessed"], flagged=bool(measured), correct=correct)
    return table


def measure(rooms, speakers, split, output, *, generators=None):
    from app.evidence import audio_stereo_timing_change

    output.mkdir(mode=0o700)
    groups = list(scene_cases(rooms, speakers, split))
    if split == "confirmation":
        groups.extend(natural_cases(speakers))
        if generators:
            groups.extend(synthetic_cases(generators, rooms))
    carriers = CARRIERS if split == "confirmation" else (PCM,)
    prepared = []
    for work, samples in groups:
        key = work["group"][:10]
        base = output / f"{key}-{work['index']}-{work['role']}.wav"
        save_wav(base, samples)
        prepared.extend((work, base, carrier, output) for carrier in carriers)
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(lambda fields: _deliver(*fields), prepared):
            results.append(row)
            if len(results) % 96 == 0:
                print(len(results), "deliveries measured", flush=True)
    metadata = {"schema": SCHEMA, "split": split, "rows": results, "totals": totals(results),
                "backgrounds": read(rooms / "recorded-sources.json"),
                "speech": {path.name: digest(path) for path in sorted(speakers.glob("*.Array1-*.wav"))
                           if path.name.split(".")[0] in VOICES[split]},
                "synthetic_plan_sha256": digest(generators / "freeze.json") if generators else None,
                "implementation_sha256": digest(Path(audio_stereo_timing_change.__file__)),
                "recipe_sha256": digest(Path(__file__)),
                "encoder": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
    save(output / "result.json.gz", metadata)


def review(receipt):
    data = read(receipt)
    counts = totals(data["rows"])
    if data["schema"] != SCHEMA or data["totals"] != counts:
        raise ValueError("Published stereo timing counts could not be reproduced")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    trial = actions.add_parser("measure")
    trial.add_argument("rooms", type=Path)
    trial.add_argument("speakers", type=Path)
    trial.add_argument("split", choices=VOICES)
    trial.add_argument("output", type=Path)
    trial.add_argument("--generators", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    options = vars(cli.parse_args())
    globals()[options.pop("action")](**options)
