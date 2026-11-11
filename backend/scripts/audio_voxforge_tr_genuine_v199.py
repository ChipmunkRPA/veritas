"""Prepare new genuine Turkish contributor groups with predeclared short crops."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import types
import wave
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, source_ranges, wav_bytes
from scripts import audio_voxforge_tr_genuine_v188 as source
from scripts import audio_voxforge_tr_genuine_v192 as rights
from scripts import video_appearance_runtime_v108 as runtime
from scripts.audio_history_v58 import text_keys
from scripts.video_appearance_corrected_v112 import scoped

NAME = "audio-voxforge-tr-genuine-v199"
MODULE = "scripts.audio_voxforge_tr_genuine_v199"
ROOT, BASE, RATE = source.ROOT, source.BASE, source.SAMPLE_RATE
TRIAL = BASE / NAME
STAGE = TRIAL / "bind-00"
PUBLIC = ROOT / "docs/evaluation-panels/audio-voxforge-tr-genuine-freeze-v199.json"
OUTPUT = STAGE / "output"
LISTING = source.TRIAL / "publisher-index.html"
GROUPS, SESSIONS_PER_GROUP = 24, 2
MIN_SAMPLES, MAX_SAMPLES = 3 * RATE, 4 * RATE
MAX_SOURCE_SECONDS = 20
MIN_NATIVE_RATE, MAX_NATIVE_RATE = 8000, 192000
FRAME_SAMPLES = RATE // 50
CROP_FRAME_COUNTS = (175, 170, 180, 165, 185, 160, 190, 155, 195, 150, 200)
ACTIVITY_FLOOR_DBFS, ACTIVITY_BELOW_P90_DB = -48, 25
MIN_ACTIVE_PERCENT = 60
MIN_QUIET_FRAMES_AT_BOUNDARY = 2
MAX_CLIPS_PER_SESSION = 24
read, write, sha, require, digest = source.read, source.write, source.sha, source.require, source.digest


def rank(value):
    return digest(f"chipmunk.audio-voxforge-tr/199:{source.LISTING_SHA}:{value}".encode())


def source_plan(raw, excluded):
    listing = source.IndexLinks()
    listing.feed(raw.decode("utf-8"))
    require(len(listing.links) == len(set(listing.links)), "Publisher index includes repeated session URLs")
    groups = defaultdict(list)
    for name in listing.links:
        alias = source.SESSION.fullmatch(name)[1].casefold()
        if not re.search(r"anon|unknown|guest", alias) and alias not in excluded:
            groups[alias].append(name)
    require(len(groups) >= GROUPS, "24 fresh named contributor slots required")
    plan = [
        {
            "slot": f"genuine-{index:02d}",
            "contributor": alias,
            "sessions": sorted(groups[alias], key=lambda item: rank(f"session:{item}"))[:SESSIONS_PER_GROUP],
        }
        for index, alias in enumerate(sorted(groups, key=lambda item: rank(f"speaker:{item}"))[:GROUPS])
    ]
    return plan, len(groups)


def prior_metadata():
    prior_path = source.TRIAL / "private-plan.json"
    prior = source.verify()
    require(
        len(prior) == len({row["contributor"] for row in prior}) == source.GROUPS == 12,
        "Exact 12 prior named contributor exclusions required",
    )
    receipt_path, cases_path = rights.OUTPUT / "result.json", rights.OUTPUT / "private-cases.json"
    receipt, rows = read(receipt_path), read(cases_path)
    require(
        receipt["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/192"
        and receipt["status"] == "complete"
        and receipt["freeze_sha256"] == sha(rights.PUBLIC)
        and receipt["private_cases_sha256"] == sha(cases_path)
        and [row["slot"] for row in rows] == [row["slot"] for row in prior]
        and receipt["holdout_media_opened"] is receipt["holdout_scored"] is False,
        "Exact prior preparation metadata required for prompt and PCM exclusions",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    return prior, qualified, [source.PUBLIC, prior_path, rights.PUBLIC, receipt_path, cases_path]


def context(raw):
    require(digest(raw) == source.LISTING_SHA, "Exact saved publisher index required")
    history = read(source.HISTORY)
    prior, _, references = prior_metadata()
    historical = {
        key.casefold().removeprefix("voxforge:")
        for key in history["voice_keys"]
        if key.casefold().startswith("voxforge:")
    }
    plan, available = source_plan(raw, historical | {row["contributor"] for row in prior})
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "Production audio decoder tools required")
    code = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_voxforge_tr_genuine_v199.py",
        ROOT / "docs/audio-voxforge-tr-genuine-v199-method.md",
        Path(source.__file__),
        Path(rights.__file__),
        Path(runtime.__file__),
        ROOT / "backend/scripts/audio_history_v58.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
    )
    return {
        "schema": "chipmunk.audio-voxforge-tr-genuine-freeze/199",
        "publisher_index_url": source.ORIGIN,
        "publisher_index_sha256": source.LISTING_SHA,
        "prior_metadata_sha256": {path.name: sha(path) for path in references},
        "historical_identity_catalogue_sha256": sha(source.HISTORY),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "decoder_tool_sha256": {name: sha(Path(path).resolve()) for name, path in tools.items()},
        "excluded_prior_v188_named_groups": len(prior),
        "recorded_v58_voxforge_voice_keys": len(historical),
        "named_groups_after_exclusions": available,
        "planned_cases": GROUPS,
        "planned_session_archives": len(source.session_list(plan)),
        "maximum_sessions_per_group": SESSIONS_PER_GROUP,
        "maximum_clips_per_session": MAX_CLIPS_PER_SESSION,
        "maximum_archive_bytes": source.MAX_ARCHIVE,
        "maximum_unpacked_tar_bytes": source.MAX_UNPACKED,
        "maximum_member_bytes": source.MAX_WAVE,
        "native_rate_bounds": [MIN_NATIVE_RATE, MAX_NATIVE_RATE],
        "maximum_native_duration_seconds": MAX_SOURCE_SECONDS,
        "production_sample_rate": RATE,
        "prepared_sample_bounds": [MIN_SAMPLES, MAX_SAMPLES],
        "crop_frame_samples": FRAME_SAMPLES,
        "crop_frame_counts_in_preference_order": list(CROP_FRAME_COUNTS),
        "activity_threshold_floor_dbfs": ACTIVITY_FLOOR_DBFS,
        "activity_threshold_db_below_p90_frame_rms": ACTIVITY_BELOW_P90_DB,
        "minimum_active_frame_percent": MIN_ACTIVE_PERCENT,
        "quiet_frames_on_each_cut_side": MIN_QUIET_FRAMES_AT_BOUNDARY,
        "language": "tr",
        "origin": "genuine",
        "model_calls_permitted": False,
        "holdout_access_permitted": False,
    }, plan


def preflight():
    lock, _ = context(LISTING.read_bytes())
    return lock


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier preparation attempt")
    raw = LISTING.read_bytes()
    lock, plan = context(raw)
    TRIAL.mkdir(mode=0o700)
    with (TRIAL / "publisher-index.html").open("xb") as stream:
        stream.write(raw)
    path = TRIAL / "private-plan.json"
    write(path, plan)
    lock["private_plan_sha256"] = sha(path)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return lock


def verify():
    lock, plan = context((TRIAL / "publisher-index.html").read_bytes())
    path = TRIAL / "private-plan.json"
    lock["private_plan_sha256"] = sha(path)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == lock and read(path) == plan,
        "Exact prospective fresh contributor and session plan required",
    )
    return plan


def acquire():
    scoped(source.acquire, {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "verify": verify})()


def acquired(plan):
    return scoped(source.acquired, {"TRIAL": TRIAL, "PUBLIC": PUBLIC})(plan)


def native_header(data, extension):
    if extension == ".wav":
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate, channels, width, frames = (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getnframes(),
            )
            require(
                channels == 1 and width in {1, 2, 3, 4} and len(wav.readframes(frames)) == frames * width,
                "Uncompressed mono WAV with its full declared sample data required",
            )
        bits = width * 8
    else:
        require(
            extension == ".flac"
            and len(data) >= 42
            and data.startswith(b"fLaC")
            and (data[4] & 0x7F) == 0
            and int.from_bytes(data[5:8], "big") == 34,
            "Native FLAC STREAMINFO required",
        )
        info = int.from_bytes(data[18:26], "big")
        rate, channels, bits, frames = (
            info >> 44,
            ((info >> 41) & 7) + 1,
            ((info >> 36) & 31) + 1,
            info & ((1 << 36) - 1),
        )
    require(
        channels == 1
        and MIN_NATIVE_RATE <= rate <= MAX_NATIVE_RATE
        and 4 <= bits <= 32
        and 3 * rate <= frames <= MAX_SOURCE_SECONDS * rate,
        "Expected supported native mono PCM/FLAC with three to twenty seconds of samples",
    )
    return rate, bits, frames


def select_crop(pcm):
    """Use independent frame RMS and pauses, retaining the natural source sample clock."""
    require(pcm.ndim == 1 and len(pcm) >= MIN_SAMPLES and np.isfinite(pcm).all(), "Invalid source PCM")
    frames = len(pcm) // FRAME_SAMPLES
    values = pcm[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES).astype(np.float64)
    rms = np.sqrt(np.mean(values**2, axis=1))
    threshold = max(
        10 ** (ACTIVITY_FLOOR_DBFS / 20), float(np.percentile(rms, 90)) * 10 ** (-ACTIVITY_BELOW_P90_DB / 20)
    )
    active = rms >= threshold
    cumulative = np.r_[0, np.cumsum(active)]
    if len(pcm) <= MAX_SAMPLES:
        require(
            100 * cumulative[-1] >= MIN_ACTIVE_PERCENT * frames,
            "Natural short clip is below activity proxy minimum",
        )
        return (0, len(pcm)), activity_receipt(threshold, int(cumulative[-1]), frames, "natural_short_full")

    # Two quiet frames adjacent to the cut on each side, within the source extent.
    quiet = np.r_[0, np.cumsum(~active)]
    cuts = np.arange(frames + 1)
    left = np.maximum(cuts - MIN_QUIET_FRAMES_AT_BOUNDARY, 0)
    right = np.minimum(cuts + MIN_QUIET_FRAMES_AT_BOUNDARY, frames)
    quiet_cut = quiet[right] - quiet[left] == right - left
    for width in CROP_FRAME_COUNTS:
        active_counts = cumulative[width:] - cumulative[:-width]
        valid = np.flatnonzero(
            quiet_cut[:-width] & quiet_cut[width:] & (100 * active_counts >= MIN_ACTIVE_PERCENT * width)
        )
        if len(valid):
            # The first occurrence wins a tie after maximizing activity at the fixed preferred width.
            start = int(valid[np.argmax(active_counts[valid])])
            return (start * FRAME_SAMPLES, (start + width) * FRAME_SAMPLES), activity_receipt(
                threshold, int(active_counts[start]), width, "pause_bounded_crop"
            )
    raise ValueError("No pause-bounded three-to-four-second clip meets the activity proxy")


def activity_receipt(threshold, active, frames, method):
    return {
        "selection": method,
        "activity_threshold_dbfs": 20 * math.log10(threshold),
        "active_frames": active,
        "whole_frames": frames,
        "active_frame_share": active / frames,
    }


def sample_range(start, end, first_pts=0):
    return {
        "start_sample": start,
        "end_sample": end,
        "source_start_pts": first_pts + start,
        "source_end_pts": first_pts + end,
        "time_base": "1/16000",
    }


def pcm_sha(pcm):
    return digest(pcm.astype("<f4").tobytes())


def production_crop(payload, extension, scratch, excluded):
    rate, bits, frames = native_header(payload, extension)
    original = digest(payload)
    require(f"bytes:{original}" not in excluded, "Known original byte identity")
    path = scratch / f"original{extension}"
    path.write_bytes(payload)
    result = decode_audio(path, source_sha256=original, max_seconds=MAX_SOURCE_SECONDS + 1)
    pcm, mapping = result.pcm, result.mapping
    total = len(pcm)
    require(
        mapping["source_sha256"] == original
        and mapping["sample_rate"] == RATE
        and mapping["channels"] == 1
        and mapping["sample_format"] == "s16le"
        and mapping["source_sample_rate"] == rate
        and mapping["source_channels"] == 1
        and (
            mapping["source_codec"] == "flac"
            if extension == ".flac"
            else mapping["source_codec"].startswith("pcm_")
        )
        and mapping["samples"] == total
        and abs(total - frames * RATE / rate) <= 1
        and MIN_SAMPLES <= total <= MAX_SOURCE_SECONDS * RATE
        and mapping["pcm_sha256"] == pcm_sha(pcm)
        and np.array_equal(read_wav(result.wav), pcm),
        "Exact complete production PCM disagrees with the original audio header",
    )
    full_ranges = source_ranges(mapping, 0, total)
    require(len(full_ranges) == 1, "Expected continuous original source clock")
    require(f"pcm:{mapping['pcm_sha256']}" not in excluded, "Known full production PCM identity")
    (start, end), activity = select_crop(pcm)
    crop = pcm[start:end]
    crop_ranges = source_ranges(mapping, start, end)
    require(
        crop_ranges == [sample_range(start, end, full_ranges[0]["source_start_pts"])],
        "Crop source range differs from the continuous original sample clock",
    )
    crop_sha = pcm_sha(crop)
    identities = {f"bytes:{original}", f"pcm:{mapping['pcm_sha256']}", f"pcm:{crop_sha}"}
    require(not identities & excluded, "Known crop or original byte/PCM identity")
    wav = wav_bytes(crop)
    require(np.array_equal(read_wav(wav), crop), "Exact zero-origin WAV round trip required")
    return (
        wav,
        {
            "source_sha256": original,
            "source_bytes": len(payload),
            "native_samples": frames,
            "native_rate": rate,
            "native_bits": bits,
            "native_codec": mapping["source_codec"],
            "full_samples": total,
            "full_pcm_sha256": mapping["pcm_sha256"],
            "full_source_range": full_ranges[0],
            "samples": len(crop),
            "sample_rate": RATE,
            "pcm_sha256": crop_sha,
            "source_range": crop_ranges[0],
            "wav_range": sample_range(0, len(crop)),
            **activity,
        },
        identities,
    )


def prompt_exclusions(archive, files, expected):
    for names in ({"prompts-original", "prompts-original.txt"}, {"prompts", "prompts.txt"}):
        matches = [member for path, member in files.items() if path.name.casefold() in names]
        if matches:
            break
    require(len(matches) == 1, "Exact session prompt manifest required")
    payload = source.member_bytes(archive, matches[0], source.MAX_TEXT)
    require(digest(payload) == expected, "Verified prompt manifest changed")
    text = {}
    for line in payload.decode("utf-8-sig", errors="replace").splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and "�" not in fields[1]:
            text[Path(fields[0]).stem.casefold()] = text_keys(fields[1])
    return text


def prepare_slot(group, records, prompts_used, fingerprints, scratch, prepared, attempts):
    slot = group["slot"]
    verify_archive = scoped(source.verify_archive, {"TRIAL": TRIAL})
    for row in records:
        candidate = {"slot": slot, "archive": row["archive"]}
        if row["status"] != "downloaded":
            attempts.append({**candidate, "status": "archive_unavailable", "reason": row["reason"]})
            continue
        try:
            with verify_archive(row) as archive:
                files = source.regular_members(archive)
                prompts, permissions = rights.session_metadata(archive, files, group["contributor"])
                text = prompt_exclusions(archive, files, permissions["prompt_manifest_sha256"])
                clips = sorted(
                    (path for path in files if path.suffix.casefold() in {".wav", ".flac"}),
                    key=lambda path: rank(f"clip:{row['archive']}:{path.as_posix()}"),
                )
                require(clips, "No native WAV or FLAC members")
                attempts.append(
                    {
                        **candidate,
                        "status": "session_qualified",
                        "archive_sha256": row["sha256"],
                        **permissions,
                        "available_members": len(clips),
                    }
                )
                for path in clips[:MAX_CLIPS_PER_SESSION]:
                    attempt = {**candidate, "member": path.as_posix()}
                    key = path.stem.casefold()
                    prompt = prompts.get(key)
                    if not prompt or prompt in prompts_used or not text.get(key) or text[key] & fingerprints:
                        attempts.append({**attempt, "status": "prompt_unavailable_or_duplicate"})
                        continue
                    try:
                        payload = source.member_bytes(archive, files[path], source.MAX_WAVE)
                        wav, receipt, identities = production_crop(
                            payload, path.suffix.casefold(), scratch, fingerprints
                        )
                    except (OSError, ValueError, EOFError, wave.Error) as error:
                        attempts.append(
                            {**attempt, "status": "clip_unavailable", "private_reason": str(error)}
                        )
                        continue
                    target = prepared / f"{slot}.wav"
                    with target.open("xb") as stream:
                        stream.write(wav)
                    prompts_used.add(prompt)
                    fingerprints.update(identities | text[key])
                    attempts.append({**attempt, "status": "selected"})
                    return {
                        **attempt,
                        **permissions,
                        **receipt,
                        "status": "qualified",
                        "language": "tr",
                        "label": 0,
                        "archive_sha256": row["sha256"],
                        "prompt_sha256": prompt,
                        "wav_sha256": sha(target),
                    }
        except (OSError, ValueError, EOFError, tarfile.TarError) as error:
            attempts.append({**candidate, "status": "session_unavailable", "private_reason": str(error)})
    return {"slot": slot, "status": "unavailable", "language": "tr", "label": 0}


def aggregation(rows, attempts, plan, archive_rows):
    require(
        [row["slot"] for row in rows] == [group["slot"] for group in plan]
        and len(rows) == GROUPS
        and all(
            row["status"] in {"qualified", "unavailable"} and row["language"] == "tr" and row["label"] == 0
            for row in rows
        ),
        "Every frozen named contributor slot must be accounted for",
    )
    qualified = [row for row in rows if row["status"] == "qualified"]
    archived = {
        (row["slot"], row["archive"]): row["sha256"] for row in archive_rows if row["status"] == "downloaded"
    }
    require(
        all(archived.get((row["slot"], row["archive"])) == row["archive_sha256"] for row in qualified)
        and all(
            row["license"] == "GPL-3.0-or-later"
            and row["stock_gpl_reference"]["sha256"] in rights.STOCK_LICENSES
            for row in qualified
        )
        and all(
            len({row[key] for row in qualified}) == len(qualified)
            for key in ("prompt_sha256", "source_sha256", "pcm_sha256", "full_pcm_sha256")
        ),
        "Exact attributed archive, license, prompt and PCM evidence required for qualified cases",
    )
    return {
        "planned_cases": len(rows),
        "case_outcomes": dict(Counter(row["status"] for row in rows)),
        "attempt_outcomes": dict(Counter(row["status"] for row in attempts)),
        "selection_outcomes": dict(Counter(row["selection"] for row in qualified)),
        "qualified_licenses": dict(Counter(row["license"] for row in qualified)),
    }


def prepare_worker():
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded data-only preparation required",
    )
    plan = verify()
    archives, acquisition = acquired(plan)
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    _, older, _ = prior_metadata()
    prompts = {row["prompt_sha256"] for row in older}
    fingerprints = set(read(source.HISTORY)["keys"])
    fingerprints.update(f"bytes:{row['source_sha256']}" for row in older)
    fingerprints.update(f"pcm:{row['pcm_sha256']}" for row in older)
    attempts = []
    with TemporaryDirectory(prefix="voxforge-genuine-v199-") as directory:
        rows = [
            prepare_slot(
                group,
                [row for row in archives if row["slot"] == group["slot"]],
                prompts,
                fingerprints,
                Path(directory),
                prepared,
                attempts,
            )
            for group in plan
        ]
    cases, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    write(cases, rows)
    write(journal, attempts)
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-voxforge-tr-genuine-result/199",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_acquisition_sha256": acquisition["private_acquisition_sha256"],
            "private_cases_sha256": sha(cases),
            "private_attempts_sha256": sha(journal),
            **aggregation(rows, attempts, plan, archives),
            "language": "tr",
            "origin": "genuine",
            "new_neural_calls": 0,
            "holdout_media_opened": False,
            "holdout_scored": False,
        },
    )


def prepare():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact monitored v108 data-only invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "prepare-worker"], **kwargs)

    def ready(*_):
        if not (TRIAL / "acquired/result.json").is_file():
            return False
        acquired(verify())
        return True

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy, "prerequisites": ready, "__file__": __file__},
    )("bind", 0)


def read_prepared_pcm(row):
    size = row["samples"]
    full, crop = row["full_source_range"], row["source_range"]
    start, end = crop["start_sample"], crop["end_sample"]
    require(
        row["status"] == "qualified"
        and re.fullmatch(r"genuine-[0-9]{2}", row["slot"])
        and row["sample_rate"] == RATE
        and MIN_SAMPLES <= size <= MAX_SAMPLES
        and 0 <= start < end <= row["full_samples"]
        and end - start == size
        and full == sample_range(0, row["full_samples"], full["source_start_pts"])
        and crop == sample_range(start, end, full["source_start_pts"])
        and row["wav_range"] == sample_range(0, size)
        and row["whole_frames"] == size // FRAME_SAMPLES
        and row["active_frame_share"] == row["active_frames"] / row["whole_frames"]
        and MIN_ACTIVE_PERCENT / 100 <= row["active_frame_share"] <= 1,
        "Exact prepared crop and original sample mapping required",
    )
    if row["selection"] == "natural_short_full":
        require(
            start == 0 and size == row["full_samples"], "Natural short clip must retain the complete source"
        )
    else:
        require(
            row["selection"] == "pause_bounded_crop"
            and row["full_samples"] > MAX_SAMPLES
            and start % FRAME_SAMPLES == size % FRAME_SAMPLES == 0
            and size // FRAME_SAMPLES in CROP_FRAME_COUNTS,
            "Pause-bounded crop must match a frozen frame extent",
        )
    path = OUTPUT / "prepared" / f"{row['slot']}.wav"
    require(path.resolve(strict=True) == path and path.is_file(), "Direct private WAV required")
    body = path.read_bytes()
    require(digest(body) == row["wav_sha256"], "Prepared WAV bytes differ")
    pcm = read_wav(body)
    require(len(pcm) == size and pcm_sha(pcm) == row["pcm_sha256"], "Prepared WAV PCM differs")
    return pcm


def readout():
    plan = verify()
    archive_rows, acquisition = acquired(plan)
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    cases, journal = OUTPUT / "private-cases.json", OUTPUT / "private-attempts.json"
    result, profile = read(result_path), read(profile_path)
    rows, attempts = read(cases), read(journal)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and result["schema"] == "chipmunk.audio-voxforge-tr-genuine-result/199"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_acquisition_sha256"] == acquisition["private_acquisition_sha256"]
        and result["private_cases_sha256"] == sha(cases)
        and result["private_attempts_sha256"] == sha(journal)
        and all(
            result[key] == value for key, value in aggregation(rows, attempts, plan, archive_rows).items()
        )
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False,
        "Exact completed offline data preparation and provenance required",
    )
    for row in rows:
        if row["status"] == "qualified":
            read_prepared_pcm(row)
    return {
        **result,
        "archive_outcomes": acquisition["archive_outcomes"],
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "freeze", "acquire", "prepare", "prepare-worker", "readout")
    )
    command = parser.parse_args().command
    result = {
        "preflight": preflight,
        "freeze": freeze,
        "acquire": acquire,
        "prepare": prepare,
        "prepare-worker": prepare_worker,
        "readout": readout,
    }[command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
