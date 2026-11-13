# SPDX-License-Identifier: Apache-2.0
"""Evaluate source-clock speech localization with a frozen background-reduced view."""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from scripts import audio_nemotron_noise_transfer as previous
from scripts import audio_partial_publisher_transfer as publisher
from scripts import review_audio_reduced_background_localization as review

common, require, save = previous.common, previous.require, previous.save
RATE, WINDOW = review.RATE, review.WINDOW
PARENT = common.PUBLIC / "audio-nemotron-noise-transfer-2026-11-12.json"
PARTIAL = common.PUBLIC / "audio-partial-publisher-transfer-2026-11-12.json"
BACKGROUNDS = {
    "construction": ("7b69cda9671d635bd20ed17f1f7db728fbca3ba6a67cc7ce2512777a8e02534a", (5, 30, 55, 80)),
    "leicester": ("42d2f7a9989748fe3f190b39665e7519bc205b336b3237c0788e2f36c6bab707", (4, 21, 38, 55)),
}


class SplitArchive:
    """Read the two committed contiguous ranges as one bounded uncompressed tar."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def read(self, count):
        data = self.first.read(count)
        return data + self.second.read(count - len(data)) if len(data) < count else data


def positioned_tar(file):
    """Find the first verified 512-byte WAV header within an aligned archive slice."""
    for offset in range(0, 2 * 1024**2, tarfile.BLOCKSIZE):
        block = file.read(tarfile.BLOCKSIZE)
        if not block.startswith(b"R01TTS.0.a/"):
            continue
        try:
            member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
        except tarfile.HeaderError:
            continue
        if member.isfile() and publisher.MEMBER.fullmatch(member.name) and 44 < member.size < 2 * 1024**2:
            file.seek(offset)
            return offset
    raise ValueError("Publisher slice has no verified WAV header")


def source_candidates(first, second, labels_path, slices=()):
    require(first.stat().st_size == 192 * 1024**2 and common.sha(first) == previous.ARCHIVE
            and second.stat().st_size == 160 * 1024**2
            and common.sha(labels_path) == publisher.PINS["part-a.labels.txt"], "Publisher inputs differ")
    older_partial = common.read(PARTIAL)["roster"]
    older_enhancement = common.read(PARENT)["held"]
    exclude = {row["source"]["speaker"] for row in older_partial if row["source"]}
    exclude.update(row["origin"]["group"] for row in older_enhancement if row["label"] != "isolated")
    labels = {name: parts for name, *parts in map(str.split, labels_path.read_text().splitlines())}
    require(len(labels) == 76_228, "Unexpected label roster")
    candidates = defaultdict(list)
    names = set()

    def collect(file):
        try:
            with tarfile.open(fileobj=file, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    match = publisher.MEMBER.fullmatch(member.name)
                    require(match and 44 < member.size < 2 * 1024**2, "Unexpected speech member")
                    name, speaker = match.groups()
                    require(name in labels and name not in names, "Missing or repeated publisher label")
                    with archive.extractfile(member) as file:
                        body = file.read()
                    require(len(body) == member.size, "Incomplete native WAV")
                    names.add(name)
                    if not 6 <= Decimal(labels[name][0]) <= 14:
                        continue
                    raw = publisher.native(body)
                    samples = len(raw) // 2
                    clock = publisher.labels_for(labels[name], samples)
                    edits = [[a, b] for a, b, label in clock if label == "spoof"]
                    label = "genuine" if not edits else "full" if edits == [[0, samples]] else "partial"
                    if label == "partial" and not (
                        .10 * samples <= sum(b - a for a, b in edits) <= .60 * samples
                        and any(9600 <= b - a <= 38400 for a, b in edits)
                    ):
                        continue
                    if not 6 * RATE <= samples <= 14 * RATE:
                        continue
                    family = None if label == "genuine" else name.split("-")[1]
                    row = {"publisher_id": name, "group": speaker, "label": label, "family": family, "samples": samples,
                           "prior_group_exposure": speaker in exclude,
                           "edits": edits, "clock": clock, "source_wav_sha256": common.digest(body),
                           "source_pcm_sha256": common.digest(raw)}
                    candidates[label, family].append((row, raw))
        except (tarfile.ReadError, EOFError) as error:
            require(isinstance(error, EOFError) or "end of data" in str(error), "Unexpected tar prefix error")

    with first.open("rb") as head, second.open("rb") as tail:
        collect(SplitArchive(head, tail))
    for path in slices:
        require(path.stat().st_size == 96 * 1024**2, "Publisher slice is incomplete")
        with path.open("rb") as file:
            positioned_tar(file)
            collect(file)
    require(len(names) > 1500, "Archive prefix is incomplete")
    return candidates, len(names), len(exclude)


def roster(candidates):
    roles = [("partial", family) for family in publisher.GENERATORS for _ in range(3)]
    roles += [("genuine", None)] * 12 + [("full", family) for family in publisher.GENERATORS]
    options = {}
    for role in set(roles):
        # Prefer previously unused publisher IDs within each role. The
        # matching step can move earlier assignments to retain every role.
        ranked = sorted(candidates[role], key=lambda pair: (
            pair[0]["prior_group_exposure"], common.digest(f"{review.SCHEMA}/{pair[0]['publisher_id']}".encode())))
        unique = {}
        for pair in ranked:
            unique.setdefault(pair[0]["group"], pair)
        options[role] = unique
    choices, assigned = {}, {}

    def assign(slot, visited):
        for group, pair in options[roles[slot]].items():
            if group in visited:
                continue
            visited.add(group)
            if group not in assigned or assign(assigned[group], visited):
                choices[slot] = pair
                assigned[group] = slot
                return True
        return False

    for slot in sorted(range(len(roles)), key=lambda slot: (len(options[roles[slot]]), slot)):
        require(assign(slot, set()), f"Insufficient distinct speakers for {roles[slot]}")
    chosen = [choices[slot] for slot in range(len(roles))]
    require(len({row["source_pcm_sha256"] for row, _ in chosen}) == len(chosen), "Repeated PCM across source groups")
    return chosen


def prepare(run, archive, extension, labels, backgrounds, slices):
    candidates, scanned, excluded = source_candidates(archive, extension, labels, slices)
    chosen = roster(candidates)
    sounds = {}
    for scene, (pin, _) in BACKGROUNDS.items():
        path = backgrounds / f"{scene}.ogg"
        require(common.sha(path) == pin, "Held sound recording differs")
        sounds[scene] = previous.prior.enhancement.float_pcm(common.decode(path))
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    rows = []

    def keep(values, scene, source, offset):
        raw = previous.prior.enhancement.pcm(values)
        row = {"id": len(rows), "scene": scene, "source": source, "offset": offset, "samples": len(values),
               "pcm_sha256": common.digest(raw)}
        (run / "pcm" / f"{row['id']:03}.pcm").write_bytes(raw)
        rows.append(row)

    ranks = Counter()
    for row, raw in chosen:
        rank = ranks[row["label"]]
        ranks[row["label"]] += 1
        offsets = {scene: times[rank % len(times)] * RATE for scene, (_, times) in BACKGROUNDS.items()}
        clips = {scene: sounds[scene][start:start + row["samples"]] for scene, start in offsets.items()}
        mixes = previous.prior.match_views(previous.prior.enhancement.float_pcm(raw), clips, -5)
        for scene, values in mixes.items():
            keep(values, scene, row, offsets.get(scene))
    for scene, (_, offsets) in BACKGROUNDS.items():
        for start in offsets:
            keep(sounds[scene][start * RATE:(start + 8) * RATE], scene, None, start * RATE)
    require(len(rows) == 116 and len({row["pcm_sha256"] for row in rows}) == len(rows), "Frozen delivery roster differs")
    save(run / "roster.json", rows)
    lock = {"schema": review.SCHEMA, "method": review.METHOD, "runner_sha256": common.sha(Path(__file__)),
            "reviewer_sha256": common.sha(Path(review.__file__)), "roster_sha256": common.sha(run / "roster.json"),
            "publisher": publisher.PUBLISHER, "prefix_sha256": common.sha(archive), "extension_sha256": common.sha(extension),
            "archive_slices_sha256": {path.name: common.sha(path) for path in slices},
            "labels_sha256": common.sha(labels), "complete_members": scanned, "prior_study_speakers_screened": excluded,
            "excluded_partials_receipt_sha256": common.sha(PARTIAL), "enhancement_receipt_sha256": common.sha(PARENT),
            "backgrounds": BACKGROUNDS, "ratio_db": -5, "voice_assets": common.assets(),
            "enhancer_assets": previous.ASSETS, "enhancer_revision": previous.REVISION,
            "runtime": {key: value for key, value in previous.RUNTIME.items() if key != "onnxruntime"},
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0],
            "enhanced_point": common.read(PARENT)["result"]["points"]["new_enhancer"],
            "known_model_responses_at_freeze": 0}
    save(run / "freeze.json", lock)
    print("Frozen", dict(ranks), len(rows), "deliveries", common.sha(run / "freeze.json"), flush=True)


def locked(run):
    lock, rows = (common.read(run / name) for name in ("freeze.json", "roster.json"))
    require(lock["schema"] == review.SCHEMA and lock["method"] == review.METHOD
            and lock["runner_sha256"] == common.sha(Path(__file__))
            and lock["reviewer_sha256"] == common.sha(Path(review.__file__))
            and lock["roster_sha256"] == common.sha(run / "roster.json"), "Frozen protocol or roster changed")
    return lock, rows


def assets_for_scores(lock):
    from app.core.config import settings
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    require(settings.audio_detector_mode == "candidate", "Voice calibration mode changed")
    assets = {name: pin for name, pin in lock["voice_assets"].items() if name not in common.SOUND_ASSETS}
    return load_voice({"weights": str(common.WEIGHTS), "assets": assets}), SpeechActivity(common.WEIGHTS / "silero_vad.jit")


def primary(run):
    from app.detectors.audio import calibration
    from app.detectors.audio.short_voice_review import scan as short_scan
    from app.engine.boundary import Deadline
    from scripts.audio_radar_transfer import adapter_result

    lock, rows = locked(run)
    require(calibration.SHA256 == review.METHOD["primary_calibration_sha256"], "Primary calibration changed")
    voice, vad = assets_for_scores(lock)
    measured = []
    for row in rows:
        raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row["pcm_sha256"], "Frozen input changed")
        pcm = previous.prior.enhancement.float_pcm(raw)
        sparse, dense = review.schedules(len(pcm))
        checks = []
        for a, b in sorted(set(map(tuple, sparse + dense))):
            result = adapter_result(pcm[a:b], voice, vad)
            require((result["scope"] and result["windows"] == [[0, WINDOW]]) or result["decision"] == "unassessed",
                    "Unexpected native speech window")
            checks.append({"span": [a, b], "direction": result["decision"], "pcm_sha256": common.digest(raw[2*a:2*b])})
        short = None
        if len(pcm) == 6 * RATE and adapter_result(pcm, voice, vad)["decision"] != "unassessed":
            short = short_scan(pcm, vad.spans(pcm), voice, Deadline(), codec="pcm_s16le")["selected"]
        measured.append({"id": row["id"], "checks": checks, "short_selected": short})
        if len(measured) % 12 == 0 or len(measured) == len(rows):
            print("Primary:", len(measured), "/", len(rows), flush=True)
    save(run / "primary.json", {"freeze_sha256": common.sha(run / "freeze.json"), "rows": measured})


def enhance(run, model, overlay):
    import importlib.metadata
    import sys

    lock, rows = locked(run)
    native = common.read(run / "primary.json")
    require(native["freeze_sha256"] == common.sha(run / "freeze.json"), "Primary input is not frozen")
    installed = {dist.metadata["Name"].lower(): dist.version
                 for dist in importlib.metadata.distributions(path=[str(overlay)])}
    require(all(installed.get(name) == ver for name, ver in lock["runtime"].items())
            and all(common.sha(model / name) == pin for name, pin in lock["enhancer_assets"].items()),
            "Exact runtime and enhancement files required")
    sys.path.insert(0, str(overlay))
    import mlx.core as mx
    from mlx_audio.sts.utils import load_model

    mx.set_cache_limit(256 * 1024**2)
    worker = load_model(str(model))
    directory = run / "processed"
    directory.mkdir(mode=0o700)
    measured = []
    for row, native_row in zip(rows, native["rows"], strict=True):
        raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row["pcm_sha256"] and row["id"] == native_row["id"], "Frozen input changed")
        _, dense = review.schedules(row["samples"])
        for check in native_row["checks"]:
            a, b = check["span"]
            if check["span"] not in dense or check["direction"] not in review.RESCUE:
                continue
            clip = raw[2*a:2*b]
            require(common.digest(clip) == check["pcm_sha256"], "Primary and processed sample geometry differs")
            wave = previous.prior.enhancement.float_pcm(clip)
            out = np.asarray(worker.generate(wave, sample_rate=RATE, chunk_seconds=None).audio, dtype=np.float32)
            require(out.shape == wave.shape and np.isfinite(out).all(), "Processed audio lost its native clock")
            body = previous.prior.enhancement.pcm(out)
            (directory / f"{row['id']:03}-{a}.pcm").write_bytes(body)
            measured.append({"id": row["id"], "span": check["span"], "processed_pcm_sha256": common.digest(body)})
        if row["id"] % 12 == 11 or row["id"] + 1 == len(rows):
            print("Enhanced:", row["id"] + 1, "/", len(rows), "with", len(measured), "processed windows", flush=True)
    save(run / "enhancement.json", {"primary_sha256": common.sha(run / "primary.json"), "rows": measured})


def score_processed(run):
    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech

    lock, _ = locked(run)
    enhancement = common.read(run / "enhancement.json")
    require(enhancement["primary_sha256"] == common.sha(run / "primary.json"), "Processed audio source is not pinned")
    voice, vad = assets_for_scores(lock)
    measured = []
    for row in enhancement["rows"]:
        a, _ = row["span"]
        raw = (run / "processed" / f"{row['id']:03}-{a}.pcm").read_bytes()
        require(len(raw) == 2 * WINDOW and common.digest(raw) == row["processed_pcm_sha256"], "Processed PCM differs")
        wave = previous.prior.enhancement.float_pcm(raw)
        admitted = window_in_speech(vad.spans(wave), 0.0, 3.0)
        score = calibration.logit(voice.predict([wave])[0]) if admitted else None
        measured.append({**row, "admitted": admitted, "logit": None if score is None else score.hex()})
        if len(measured) % 40 == 0 or len(measured) == len(enhancement["rows"]):
            print("Processed:", len(measured), "/", len(enhancement["rows"]), flush=True)
    save(run / "processed.json", {"enhancement_sha256": common.sha(run / "enhancement.json"), "rows": measured})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("prepare", "primary", "enhance", "processed"))
    cli.add_argument("run", type=Path)
    for flag in ("archive", "extension", "labels", "backgrounds", "model", "overlay"):
        cli.add_argument("--" + flag, type=Path)
    cli.add_argument("--slice", dest="slices", action="append", type=Path, default=[])
    args = cli.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.archive, args.extension, args.labels, args.backgrounds, args.slices)
    elif args.stage == "enhance":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        enhance(args.run, args.model, args.overlay)
    else:
        {"primary": primary, "processed": score_processed}[args.stage](args.run)


if __name__ == "__main__":
    main()
