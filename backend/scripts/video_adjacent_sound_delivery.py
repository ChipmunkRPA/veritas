# SPDX-License-Identifier: Apache-2.0
"""Check adjacent soundtrack evidence separately on public speech excerpts and AAC."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.audio.decode import wav_bytes
from scripts import review_video_adjacent_sound_delivery as review
from scripts import video_adjacent_sound_transfer as transfer
from scripts import video_lipsync_repeat_audio as video
from scripts import video_lipsync_repeat_public_speech as speech
from scripts import video_lipsync_window_schedule as parent
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels/video-lipsync-repeat-public-speech-2026-11-13.json"
WITH_NOISE = ("local_early", "local_late", "matched_picture_early", "matched_picture_late", "original")


def freeze(source_dir):
    publisher = read(speech.SOURCES)["sources"]
    qualified = read(PANEL)["qualification"]
    rows = []
    for source, result in zip(publisher, qualified, strict=True):
        path = source_dir / f'{source["id"]}.webm'
        require(sha(path) == result["source_sha256"] == source["sha256"], "A public conversation source differs")
        rows.append({"id": source["id"], "group": result["group"], "sha256": source["sha256"], "offset": result["selected"]})
    bindings = {Path(path).name: sha(path) for path in (__file__, review.__file__, speech.SOURCES, PANEL,
                transfer.__file__, transfer.method.__file__, parent.FACE_FILE)}
    return {"schema": review.SCHEMA, "sources": rows, "bindings": bindings,
            "runtime": {"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                        "opencv": cv2.__version__, "numpy": np.__version__},
            "design": {"carrier": "Actual 48 kbps AAC in MP4 with copied H.264 pictures, continuous decoded audio. "
                                  "PCM comparator reuses original adapted picture and single spoken excerpt.",
                       "controls": "Prior schedule plus same-clock picture and sound, repeated bed, "
                                   "and matched -5 dB independent speech added to both shifts, the synchronized controls and untouched speech. "
                                   "Synchronized pictures are retimed, encoded, decoded and measured on their delivery clock.",
                       "decision": "Extra evidence is justified for a later product trial only with at least two added clean AAC pairs "
                                   "from distinct recordings, no added ordinary flags in either carrier and no synchronized or global "
                                   "flags in the candidate. No coverage or adoption inference is drawn from the three unseen speakers."}}


def _decode_aac(wave, picture, scratch, table):
    source = scratch / "audio.wav"
    path = scratch / "delivery.mp4"
    source.write_bytes(wav_bytes(wave.astype(np.float32) / 32768))
    if path.exists():
        path.unlink()
    video.delivery.remux(picture, source, path, ("-c:a", "aac", "-b:a", "48k"))
    compared = video.probe_frames(path, timeout_s=8)
    require(len(compared) == len(table)
            and [(x - compared.pts[0]) * compared.tb for x in compared.pts]
            == [(x - table.pts[0]) * table.tb for x in table.pts], "AAC remux changed the picture clock")
    from app.evidence.audio_reference import read_track

    measured = read_track(path, {"sha256": sha(path)}, max_seconds=5, video=True, origin=compared.pts[0] * compared.tb)
    require(measured[0] == "measured" and measured[2]["relative_start"] == "0" and len(measured[1]) >= len(wave),
            "Decoded AAC does not cover the intended continuous clock")
    return np.asarray(measured[1][:len(wave)] * 32768, dtype="<i2")


def _with_competing_voice(wave, original, donor):
    voice = original.astype(np.float64)
    background = donor.astype(np.float64)
    scale = 10 ** (.25) * np.std(voice) / max(1., np.std(background))
    mix = wave + scale * background
    peak = float(np.max(np.abs(mix)))
    gain = min(1., 32760. / max(peak, 1.))
    return np.rint(mix * gain).astype("<i2")


def score(lock, source_dir):
    require(lock == freeze(source_dir), "Sources or method changed after freeze")
    require(sha(parent.FACE_FILE) == parent.FACE_SHA, "Face detector differs from its pinned weights")
    cv2.setNumThreads(1)
    detector = video.load_yunet(parent.FACE_FILE)
    rows = []
    with TemporaryDirectory(prefix="adjacent-aac-") as private:
        plans = []
        for row in lock["sources"]:
            prepared = Path(private) / f'{row["id"]}.mov'
            source = source_dir / f'{row["id"]}.webm'
            adapted = speech.prepare(source, prepared, row["offset"])
            require(adapted is not None, "The pinned original failed excerpt qualification")
            table, _, patches, _, pictures = video.video_inputs(prepared, detector)
            plans.append((row, prepared, table, adapted[1], patches, pictures, adapted[0]))
        for index, (row, prepared, table, pcm, patches, pictures, pixels) in enumerate(plans):
            donor = plans[(index + 1) % len(plans)][3]
            cases = []
            synchronized = {}
            for role, wave, truth, frames, crops in transfer.variants(pcm, len(table), row["group"],
                                                                     tuple(range(len(table))), patches):
                picture, clock = prepared, table
                if role.startswith("matched_picture_"):
                    picture = Path(private) / f'{row["id"]}-{role}.mov'
                    require(min(frames) >= 0, "Synchronized control leaves the observed picture clock")
                    video.picture(pixels[list(frames)], picture)
                    clock, _, crops, _, changed = video.video_inputs(picture, detector)
                    frames = tuple(range(len(clock)))
                    synchronized[role] = [item["rgb_sha256"] for item in changed]
                conditions = [("clean", wave)]
                if role in WITH_NOISE:
                    conditions.append(("competing", _with_competing_voice(wave, pcm, donor)))
                for condition, samples in conditions:
                    deliveries = {"PCM": samples, "AAC": _decode_aac(samples, picture, Path(private), clock)}
                    for carrier, decoded in deliveries.items():
                        cases.append({"role": role, "condition": condition, "carrier": carrier, "truth": truth,
                                      "decoded_pcm_sha256": transfer._hash(decoded),
                                      "arms": transfer.assess(decoded, frames, crops)})
            rows.append({"id": row["id"], "sha256": row["sha256"], "audio_samples": len(pcm), "cases": cases,
                         "picture_time_base": table.time_base,
                         "synchronized_picture_sha256": synchronized,
                         "pictures": [{"index": item["index"], "pts": item["pts"], "rgb_sha256": item["rgb_sha256"]}
                                      for item in pictures]})
            print("verified and measured", row["id"], flush=True)
    return json.loads(json.dumps({"freeze": lock, "rows": rows}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score"))
    cli.add_argument("sources", type=Path)
    cli.add_argument("run", type=Path)
    args = cli.parse_args()
    lock = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(lock, freeze(args.sources))
    else:
        result = score(read(lock), args.sources)
        result["outcome"] = review.review(result)
        write(args.run / "receipt.json", result)
        print(json.dumps(result["outcome"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
