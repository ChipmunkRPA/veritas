"""Qualify the fixed RUEG recordings using an activity-ranked crop fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

from scripts import audio_rueg_genuine_v246 as source
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE = source.ROOT, source.BASE
PANEL = ROOT / "docs/evaluation-panels"
TRIAL = BASE / "audio-rueg-activity-crop-v252"
CACHE = TRIAL / "prepared"
PUBLIC = PANEL / "audio-rueg-activity-crop-freeze-v252.json"
ORIGINAL = PANEL / "audio-rueg-genuine-complete-v246.json"
ORIGINAL_SHA = "f6540363409aa0f2b02e39f0853a8c1556ae3c7856bc16e11b741cffdace0a2e"
NO_PAUSE = "No pause-bounded three-to-four-second clip meets the activity proxy"
read, save, sha, require = source.read, source.save, source.digest, source.require
activity_only = scoped(source.crop.select_crop, {"MIN_QUIET_FRAMES_AT_BOUNDARY": 0})


def select_crop(pcm):
    try:
        return source.crop.select_crop(pcm)
    except ValueError as error:
        if str(error) != NO_PAUSE:
            raise
    bounds, activity = activity_only(pcm)
    return bounds, {**activity, "selection": "activity_ranked_crop"}


crop = SimpleNamespace(
    MIN_NATIVE_RATE=source.crop.MIN_NATIVE_RATE,
    MAX_NATIVE_RATE=source.crop.MAX_NATIVE_RATE,
    RATE=source.crop.RATE,
    pcm_sha=source.crop.pcm_sha,
    sample_range=source.crop.sample_range,
    select_crop=select_crop,
)
prepare = scoped(source.prepare_clip, {"crop": crop})


def context():
    original = source.readout()
    require(
        sha(ORIGINAL) == ORIGINAL_SHA
        and read(ORIGINAL) == original
        and original["counts"] == {"verified_member": 24, "qualified_pcm": 9, "qualification_unavailable": 15}
        and all(original["terminal_stages"].values()),
        "Exact completed RUEG intake and prior outcomes required",
    )
    lock, members = source.verify()
    return {
        "schema": "chipmunk.audio-rueg-activity-crop-freeze/252",
        "original_result_sha256": ORIGINAL_SHA,
        "original_freeze_sha256": lock,
        "original_receipt_set_sha256": original["receipt_set_sha256"],
        "code_sha256": {
            name: sha(ROOT / name)
            for name in (
                "backend/scripts/audio_rueg_activity_crop_v252.py",
                "backend/tests/test_audio_rueg_activity_crop_v252.py",
                "docs/audio-rueg-activity-crop-v252-method.md",
            )
        },
        "planned": len(members),
        "fallback_on_exact_no_pause_error": True,
        "fallback_quiet_frames_required": 0,
        "model_calls_permitted": False,
    }, members


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier preparation attempts")
    lock, _ = context()
    TRIAL.mkdir(mode=0o700)
    CACHE.mkdir(mode=0o700)
    save(TRIAL / "freeze.json", lock)
    save(PUBLIC, lock)
    return lock


def verify():
    lock, members = context()
    require(read(TRIAL / "freeze.json") == read(PUBLIC) == lock, "Exact fallback freeze required")
    return sha(PUBLIC), members


def stage_receipt(stage, slot):
    require(stage == "qualify" and 1 <= slot <= source.SLOTS, "Exact qualification slot required")
    return TRIAL / f"qualify-{slot:02d}.json"


def prior_case(slot, member):
    acquired, prior = (source.stage_receipt(stage, slot) for stage in ("acquire", "qualify"))
    original = read(acquired)
    path = source.verified_original(slot, member, original)
    return path, original["source_sha256"], prior


def qualify():
    source.headroom()
    locked, members = verify()
    save(TRIAL / "started.json", {"freeze_sha256": locked})
    seen = set(read(source.HISTORY)["keys"])
    for slot, member in enumerate(members, 1):
        path, checksum, prior = prior_case(slot, member)
        previous = read(prior)
        row = {"slot": slot, "freeze_sha256": locked, "prior_qualification_sha256": sha(prior)}
        try:
            wav, facts, identities = prepare(path, checksum, seen)
        except (ValueError, OSError, EOFError, source.wave.Error):
            require(previous["status"] == "unavailable", "Previously qualified input no longer qualifies")
            row.update(status="unavailable", reason="source_or_activity_unqualified")
        else:
            if previous["status"] == "qualified_pcm":
                require(
                    wav == (source.CACHE / f"slot-{slot:02d}.crop.wav").read_bytes()
                    and all(previous[key] == value for key, value in facts.items()),
                    "Previously qualified crop changed",
                )
            else:
                require(facts["selection"] == "activity_ranked_crop", "Recovery must use the fixed fallback")
            with (CACHE / f"slot-{slot:02d}.crop.wav").open("xb") as stream:
                stream.write(wav)
            seen.update(identities)
            row.update(facts, status="qualified_pcm")
        save(stage_receipt("qualify", slot), row)
    save(TRIAL / "complete.json", {"freeze_sha256": locked})
    return readout()


def readout():
    locked, members = verify()
    require(
        read(TRIAL / "complete.json") == {"freeze_sha256": locked}, "Terminal fallback preparation required"
    )
    counts, receipts = Counter(), []
    for slot in range(1, len(members) + 1):
        prior = source.stage_receipt("qualify", slot)
        path = stage_receipt("qualify", slot)
        previous, row = read(prior), read(path)
        require(
            row["slot"] == slot
            and row["freeze_sha256"] == locked
            and row["prior_qualification_sha256"] == sha(prior),
            "Exact crop qualification receipt required",
        )
        receipts.append(sha(path))
        clip = CACHE / f"slot-{slot:02d}.crop.wav"
        if row["status"] == "unavailable":
            require(previous["status"] == "unavailable" and not clip.exists(), "Invalid unavailable crop")
            counts["qualification_unavailable"] += 1
            continue
        require(
            row["status"] == "qualified_pcm" and clip.is_file() and not clip.is_symlink(),
            "Exact crop required",
        )
        body = clip.read_bytes()
        pcm = source.read_wav(body)
        bounds = row["source_range"]
        require(
            source.crop.MIN_SAMPLES <= len(pcm) == row["samples"] <= source.crop.MAX_SAMPLES
            and hashlib.sha256(body).hexdigest() == row["wav_sha256"]
            and crop.pcm_sha(pcm) == row["pcm_sha256"]
            and bounds
            == crop.sample_range(
                bounds["start_sample"], bounds["end_sample"], row["full_source_range"]["source_start_pts"]
            )
            and bounds["end_sample"] - bounds["start_sample"] == len(pcm),
            "Prepared crop waveform or source clock differs",
        )
        if previous["status"] == "qualified_pcm":
            require(row["wav_sha256"] == previous["wav_sha256"], "Previous crop bytes differ")
            counts["preserved_original_crops"] += 1
        else:
            require(row["selection"] == "activity_ranked_crop", "Recovered crop lacks fallback evidence")
            counts["recovered_activity_crops"] += 1
        counts["qualified_pcm"] += 1
    return {
        "schema": "chipmunk.audio-rueg-activity-crop-result/252",
        "status": "complete",
        "freeze_sha256": locked,
        "original_result_sha256": ORIGINAL_SHA,
        "planned": len(members),
        "counts": dict(counts),
        "terminal_stages": {"qualify": True},
        "qualification_receipt_set_sha256": hashlib.sha256(json.dumps(receipts).encode()).hexdigest(),
        "model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"freeze": freeze, "qualify": qualify, "readout": readout}
    parser.add_argument("command", choices=commands)
    print(json.dumps(commands[parser.parse_args().command](), indent=2, sort_keys=True, allow_nan=False))
