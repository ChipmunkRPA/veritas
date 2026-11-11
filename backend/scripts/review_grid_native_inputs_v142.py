"""Independent saved-bank content/clock replay; never decode MPG or run a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import types
import wave
import zlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PANEL = ROOT / "docs/evaluation-panels"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
BANK = ARCHIVE / "grid-native-inputs-v137"
TRIAL = BASE / "grid-native-input-review-v142"
PUBLIC = PANEL / "grid-native-input-review-freeze-v142.json"
CODE = (
    "backend/scripts/review_grid_native_inputs_v142.py",
    "backend/tests/test_review_grid_native_inputs_v142.py",
    "docs/grid-native-input-review-v142-method.md",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/video_appearance_corrected_v112.py",
)
CONTEXTS = {
    131: ("grid-transfer-acquisition-v131", "grid-transfer-acquisition-freeze-v131.json"),
    134: ("grid-media-preflight-v134", "grid-media-preflight-freeze-v134.json"),
    135: ("grid-acquisition-continuation-v135", "grid-acquisition-continuation-freeze-v135.json"),
    136: ("grid-clip-plan-v136", "grid-clip-plan-freeze-v136.json"),
    137: ("grid-native-inputs-v137", "grid-native-inputs-freeze-v137.json"),
}
UNAVAILABLE = {
    "unavailable_video_probe",
    "unavailable_video_clock_or_bounds",
    "unavailable_video_pixels",
    "video_only_no_embedded_audio",
    "unavailable_audio_clock_or_bounds",
    "unavailable_canonical_pcm",
}
AVAILABLE = "native_video_and_audio_available"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def positive_int(value):
    return type(value) is int and value > 0


def fraction(value):
    require(type(value) in (int, str), "Exact rational clock required; no float clock")
    return Fraction(value)


def stage(folder, freeze_sha, caller, batch, bindings, *, monitor=False):
    """Only receipt/resource/JSON hashing here; never read saved media at admission."""
    receipt_path, profile_path = folder / "output/result.json", folder / "profile.json"
    result, profile = read(receipt_path), read(profile_path)
    journal = folder / "resources.jsonl"
    require(
        result["status"] == "complete"
        and (monitor or result["freeze_sha256"] == freeze_sha)
        and result["new_neural_calls"] == 0
        and (monitor or result["defaults_changed"] is False)
        and profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == []
        and profile["resource_limit_observations"] == []
        and profile["command"] == ("monitor" if monitor else "bind")
        and profile["batch"] == batch
        and profile["new_neural_calls"] == 0
        and profile["worker_receipt_sha256"] == sha(receipt_path)
        and profile["resource_journal_sha256"] == sha(journal)
        and profile["runtime_sha256"] == sha(ROOT / "backend/scripts" / caller),
        "Actual complete bound data stage required",
    )
    resources = lines(journal)
    require(
        resources and all(not row["violations"] for row in resources),
        "Complete clean resource journal required",
    )
    for i, row in enumerate(resources):
        reserve = (3 if i == 0 else 1.5) * 1024**3
        require(
            type(row["elapsed_s"]) in (int, float)
            and math.isfinite(row["elapsed_s"])
            and 0 <= row["elapsed_s"] <= 900
            and (i == 0 or resources[i - 1]["elapsed_s"] <= row["elapsed_s"])
            and all(
                type(row[k]) is int and row[k] >= 0
                for k in (
                    "tree_rss_bytes",
                    "ram_available_bytes",
                    "disk_free_bytes",
                    "archive_free_disk_bytes",
                )
            )
            and 0 <= row["tree_rss_bytes"] <= 1536 * 1024**2
            and all(
                row[k] >= reserve
                for k in ("ram_available_bytes", "disk_free_bytes", "archive_free_disk_bytes")
            ),
            "Original data-stage resources exceeded",
        )
    require(
        type(profile["duration_s"]) in (int, float)
        and math.isfinite(profile["duration_s"])
        and resources[-1]["elapsed_s"] <= profile["duration_s"] <= 900
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(r["tree_rss_bytes"] for r in resources)
        and profile["minimum_available_ram_bytes"] == min(r["ram_available_bytes"] for r in resources),
        "Profile differs from actual resource samples",
    )
    for path in (receipt_path, profile_path, journal):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    return result


def metadata():
    """Bind every actual source, directory, selection and native receipt, without media I/O."""
    bindings, contexts, freezes = {}, {}, {}

    def bound(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    monitor = stage(
        BASE / "video-appearance-native-v108/monitor",
        None,
        "video_appearance_runtime_v108.py",
        None,
        bindings,
        monitor=True,
    )
    require(
        monitor["owned_child_exit_code"] == 0
        and monitor["owned_descendant_sampled"] is True
        and monitor["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "Actual unchanged owned-descendant monitor qualification required",
    )

    for version, (name, public) in CONTEXTS.items():
        contexts[version] = BASE / name
        frozen = bound(contexts[version] / "freeze.json")
        require(frozen == bound(PANEL / public), "Public/private source freeze differs")
        for relative, digest in frozen["code_sha256"].items():
            require(sha(ROOT / relative) == digest, "Historical executed source code changed")
        freezes[version] = frozen
    a, p, c, s, n = (contexts[v] for v in (131, 134, 135, 136, 137))
    af, pf, cf, sf, nf = (freezes[v] for v in (131, 134, 135, 136, 137))
    original, preflight, continuation = bound(a / "plan.json"), bound(p / "plan.json"), bound(c / "plan.json")
    original_journal = a / "bind-00/output/archives.jsonl"
    stopped = bound(a / "bind-00/profile.json")
    bindings[str(original_journal.relative_to(ROOT))] = sha(original_journal)
    first = lines(original_journal)
    require(
        stopped["child_exit_code"] == -15
        and stopped["guard"] == "resource_limit"
        and stopped["sample_errors"] == 0
        and stopped["resource_limit_observations"]
        and all(r["violations"] == ["elapsed_time_limit"] for r in stopped["resource_limit_observations"])
        and len(first) == 1
        and first[0]["ordinal"] == 0
        and first[0]["status"] == "publisher_archive_verified"
        and original["selected"] == continuation["selected"]
        and len(original["selected"]) == 16
        and [r["ordinal"] for r in original["selected"]] == list(range(16))
        and preflight["archive"] == continuation["first_complete"] == first[0],
        "Preserve stopped v131 and its actual complete first source; require all original slots",
    )
    require(
        af["private_plan_sha256"] == sha(a / "plan.json")
        and pf["private_plan_sha256"] == sha(p / "plan.json")
        and cf["private_plan_sha256"] == sha(c / "plan.json")
        and pf["acquisition_freeze_sha256"]
        == cf["original_acquisition_freeze_sha256"]
        == sha(a / "freeze.json")
        and pf["stopped_acquisition_profile_sha256"]
        == cf["original_stopped_profile_sha256"]
        == sha(a / "bind-00/profile.json")
        and pf["private_archive_journal_sha256"]
        == cf["original_archive_journal_sha256"]
        == sha(original_journal)
        and sf["acquisition_freeze_sha256"]
        == nf["original_acquisition_continuation_freeze_sha256"]
        == sha(c / "freeze.json")
        and sf["private_source_plan_sha256"] == sha(c / "plan.json")
        and nf["clip_planning_freeze_sha256"] == sha(s / "freeze.json"),
        "Exact historical freeze/plan chain required",
    )
    first_result = stage(p / "bind-00", sha(p / "freeze.json"), "grid_media_preflight_v134.py", 0, bindings)
    preflight_records = p / "bind-00/output/private-records.json"
    require(
        first_result["private_records_sha256"] == sha(preflight_records)
        and cf["independent_first_source_preflight_sha256"] == sha(p / "bind-00/output/result.json"),
        "Actual independent first archive receipt/journal binding required",
    )
    bound(preflight_records)
    archives = [first[0]]
    for ordinal in range(1, 16):
        folder = c / f"source-{ordinal:02d}/bind-00"
        result = stage(folder, sha(c / "freeze.json"), "grid_acquisition_continuation_v135.py", 0, bindings)
        journal = folder / "output/archives.jsonl"
        require(result["private_archive_journal_sha256"] == sha(journal), "Acquisition journal changed")
        bindings[str(journal.relative_to(ROOT))] = sha(journal)
        records = lines(journal)
        require(
            len(records) == 1 and result["archives"] == 1,
            "One complete acquisition per original slot required",
        )
        record = records[0]
        require(
            result["verified_archive_bytes"] == record["bytes"]
            and result["real_video_members_inventoried"] == 1000,
            "Acquisition aggregate differs from actual archive record",
        )
        archives.append(record)
    for ordinal, record in enumerate(archives):
        require(
            all(record[k] == v for k, v in original["selected"][ordinal].items())
            and record["status"] == "publisher_archive_verified"
            and len(record["real_video_members"])
            == len({r["name"] for r in record["real_video_members"]})
            == 1000,
            "All sixteen full publisher archive records required",
        )
    directories = []
    for batch in (0, 1):
        folder = s / f"bind-{batch:02d}"
        result = stage(folder, sha(s / "freeze.json"), "grid_clip_plan_v136.py", batch, bindings)
        journal = folder / "output/directories.jsonl"
        require(result["private_directory_journal_sha256"] == sha(journal), "Directory journal changed")
        bindings[str(journal.relative_to(ROOT))] = sha(journal)
        rows = lines(journal)
        require(
            len(rows) == result["source_archives"] == 8, "Both complete eight-source directories required"
        )
        directories.extend(rows)
    selected_result = stage(s / "bind-02", sha(s / "freeze.json"), "grid_clip_plan_v136.py", 2, bindings)
    selection = s / "bind-02/output/selected-clips.json"
    selected = bound(selection)
    require(
        selected_result["private_selected_clip_plan_sha256"]
        == nf["private_selected_clips_sha256"]
        == sha(selection)
        and nf["clip_planning_result_sha256"] == sha(s / "bind-02/output/result.json")
        and selected_result["private_directory_journal_sha256"]
        == [sha(s / f"bind-{b:02d}/output/directories.jsonl") for b in (0, 1)],
        "Native bank must use exact preselected full32 cohort",
    )
    validate_selection(selected, directories, archives)
    require(
        [r["member"] for r in selected[:2]] == preflight["clips"], "Original first-source choices changed"
    )
    banks = []
    for ordinal in range(16):
        folder = n / f"source-{ordinal:02d}/bind-00"
        result = stage(folder, sha(n / "freeze.json"), "grid_native_inputs_v137.py", 0, bindings)
        path = folder / "output/native-inputs.json"
        records = bound(path)
        require(
            result["private_native_inputs_sha256"] == sha(path)
            and result["source_archives"] == 1
            and result["planned_clips"] == result["retained_clips"] == 2
            and [r["planned"] for r in records] == selected[ordinal * 2 : ordinal * 2 + 2]
            and result["availability"] == dict(Counter(r["status"] for r in records))
            and result["native_video_frames"]
            == sum(len(r.get("native_video_table", {}).get("pts", [])) for r in records)
            and result["retained_native_pictures"] == sum(len(r.get("pictures", [])) for r in records)
            and result["observed_mono16k_samples"]
            == sum(r.get("pcm_mapping", {}).get("samples", 0) for r in records),
            "Native stage must retain both exact selected inputs and matching counts",
        )
        require(
            all(r["status"] in {AVAILABLE, *UNAVAILABLE} for r in records),
            "Unknown availability cannot be dropped",
        )
        banks.extend(records)
    require(len(banks) == 32, "All32 observations, including unavailable inputs, required")
    return bindings, banks


def validate_selection(selected, directories, archives):
    require(len(directories) == len(archives) == 16 and len(selected) == 32, "Complete fixed cohort required")
    expected, used = [], set()
    for ordinal, (directory, archive) in enumerate(zip(directories, archives, strict=True)):
        require(
            directory["source"]["ordinal"] == ordinal
            and directory["source"] == {k: archive[k] for k in directory["source"]}
            and directory["full_archive_checksum_verified_by_this_stage"] is False
            and directory["media"] == archive["real_video_members"],
            "Directory/archive source correspondence differs",
        )
        ordered = sorted(
            directory["media"],
            key=lambda m: (
                hashlib.sha256(
                    ("chipmunk.grid-clips/134:" + PurePosixPath(m["name"]).stem).encode()
                ).hexdigest(),
                m["name"],
            ),
        )
        chosen = []
        for member in ordered:
            path = PurePosixPath(member["name"])
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in member["name"]
                and path.suffix.lower() == ".mpg"
                and "__MACOSX" not in path.parts
                and not path.name.startswith("._")
                and 1000 <= member["bytes"] <= 2 * 1024**2
                and type(member["crc32"]) is int
                and 0 <= member["crc32"] < 2**32,
                "Safe bounded real member required",
            )
            if path.stem not in used:
                chosen.append(member)
                used.add(path.stem)
            if len(chosen) == 2:
                break
        require(len(chosen) == 2, "Two untouched selection stems required")
        for offset, member in enumerate(chosen):
            expected.append(
                {
                    "ordinal": len(expected),
                    "source_ordinal": ordinal,
                    "within_source_ordinal": offset,
                    "member": member,
                    "source": directory["source"],
                    "study_role": "external_transfer",
                    "media_qualified": False,
                }
            )
    require(
        selected == expected and len(used) == 32,
        "Independent fixed salted selection differs; no substitutions",
    )


def supported_video(table):
    pts, tb = table["pts"], fraction(table["time_base"])
    return bool(
        tb > 0
        and pts
        and len(pts) <= 500
        and all(type(p) is int for p in pts)
        and all(a < b for a, b in zip(pts, pts[1:]))
        and table["interpolated"] == []
        and table["truncated"] is False
        and all(type(table[k]) is int and 0 < table[k] <= 640 for k in ("width", "height"))
        and (pts[-1] - pts[0]) * tb <= 10
    )


def video_clock(table, pictures, complete=True):
    pts, tb = table["pts"], fraction(table["time_base"])
    require(
        tb > 0 and pts and all(type(p) is int for p in pts) and all(a < b for a, b in zip(pts, pts[1:])),
        "Ordered exact observed native video PTS required",
    )
    if complete:
        require(
            supported_video(table),
            "Complete supported source video clock required",
        )
    indices = [r["frame_index"] for r in pictures]
    require(
        indices == list(range(len(pictures)))
        and len(pictures) <= len(pts)
        and (not complete or len(pictures) == len(pts))
        and all(
            type(r["frame_index"]) is int and type(r["pts"]) is int and r["pts"] == pts[r["frame_index"]]
            for r in pictures
        ),
        "Saved picture index/PTS/table differs",
    )
    return pts[0] * tb


def native_audio(clock):
    tb, rate = fraction(clock["time_base"]), clock["sample_rate"]
    require(
        tb > 0 and positive_int(rate) and positive_int(clock["channels"]),
        "Positive native audio clock required",
    )
    frames, starts, ends = clock["frames"], [], []
    for row in frames:
        require(
            type(row["pts"]) is int and positive_int(row["samples"]), "Observed native sample frames required"
        )
        start = row["pts"] * tb
        end = start + Fraction(row["samples"], rate)
        require(
            fraction(row["start"]) == start and fraction(row["end"]) == end,
            "Native audio rational bounds differ",
        )
        starts.append(start)
        ends.append(end)
    tolerance = tb + Fraction(1, rate)
    gaps = sum(b - a > tolerance for a, b in zip(ends, starts[1:]))
    overlaps = sum(b - a < -tolerance for a, b in zip(ends, starts[1:]))
    ordered = all(a < b for a, b in zip(starts, starts[1:]))
    missing = clock["missing_frame_clocks"]
    require(
        type(missing) is int
        and missing >= 0
        and clock["gap_count"] == gaps
        and clock["overlap_count"] == overlaps
        and clock["clock_qualified"] == bool(frames and not missing and ordered and not overlaps)
        and clock["contiguous"] == bool(frames and not missing and ordered and not gaps and not overlaps),
        "Native audio availability/continuity differs from rational frames",
    )
    return (starts[0], ends[-1]) if frames else None


def mapping_ranges(mapping, samples):
    require(
        mapping["timeline_time_base"] == "1/16000"
        and mapping["sample_rate"] == 16000
        and mapping["channels"] == 1
        and mapping["sample_format"] == "s16le"
        and type(samples) is int
        and 0 < samples <= 160000
        and mapping["samples"] == samples,
        "Complete bounded canonical mono16k sample mapping required",
    )
    result, end_sample, source_end = [], 0, None
    for row in mapping["timeline"]:
        a, b, pts = (row[k] for k in ("start_sample", "end_sample", "source_pts"))
        require(
            all(type(v) is int for v in (a, b, pts)) and a == end_sample and a < b <= samples,
            "Canonical mapping has missing, repeated or unordered samples",
        )
        require(source_end is None or pts == source_end, "Canonical source timeline is discontinuous")
        result.append(
            {
                "start_sample": a,
                "end_sample": b,
                "source_start_pts": pts,
                "source_end_pts": pts + b - a,
                "time_base": "1/16000",
            }
        )
        end_sample, source_end = b, pts + b - a
    require(result and end_sample == samples, "Complete mapping through last PCM sample required")
    return result


def verify_member(path, member, digest):
    count, crc, observed = 0, 0, hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            count += len(block)
            crc = zlib.crc32(block, crc)
            observed.update(block)
    require(
        count == member["bytes"] and crc & 0xFFFFFFFF == member["crc32"] and observed.hexdigest() == digest,
        "Saved MPG SHA/size/selected ZIP CRC differs",
    )


def verify_png(path, record, size):
    from PIL import Image

    require(sha(path) == record["file_sha256"], "Saved PNG file changed")
    with Image.open(path) as image:
        require(image.format == "PNG" and image.size == size, "Saved native PNG shape/format differs")
        rgb = image.convert("RGB").tobytes()
    require(hashlib.sha256(rgb).hexdigest() == record["rgb_sha256"], "Independent Pillow RGB bytes differ")
    return rgb


def verify_pcm(row):
    import numpy as np

    require(
        sha(row["wav_file"]) == row["wav_file_sha256"] and sha(row["pcm_file"]) == row["pcm_file_sha256"],
        "Saved WAV/PCM files changed",
    )
    with wave.open(row["wav_file"], "rb") as source:
        require(
            source.getnchannels() == 1
            and source.getframerate() == 16000
            and source.getsampwidth() == 2
            and source.getcomptype() == "NONE"
            and 0 < source.getnframes() <= 160000,
            "Canonical WAV format differs",
        )
        count = source.getnframes()
        raw = source.readframes(count + 1)
    require(len(raw) == count * 2, "Complete exact WAV samples required")
    wave_pcm = np.frombuffer(raw, dtype="<i2").astype("<f4") / np.float32(32768)
    stored = np.load(row["pcm_file"], allow_pickle=False, mmap_mode="r")
    require(
        stored.dtype == np.dtype("<f4")
        and stored.shape == (count,)
        and np.isfinite(stored).all()
        and stored.tobytes() == wave_pcm.tobytes(),
        "Independent WAV and float32 PCM differ",
    )
    digest = hashlib.sha256(wave_pcm.astype("<f4").tobytes()).hexdigest()
    mapping, clock = row["pcm_mapping"], row["native_audio_clock"]
    require(
        mapping["pcm_sha256"] == digest and mapping["source_sha256"] == row["source_sha256"],
        "PCM/source content commitment differs",
    )
    ranges = mapping_ranges(mapping, count)
    require(
        row["whole_pcm_source_ranges"] == ranges, "Saved whole-PCM ranges differ from independent mapping"
    )
    native_bounds = native_audio(clock)
    require(
        native_bounds is not None and clock["contiguous"] and native_bounds[1] - native_bounds[0] <= 10,
        "Supported continuous native source audio required",
    )
    require(
        mapping["source_sample_rate"] == clock["sample_rate"]
        and mapping["source_channels"] == clock["channels"]
        and fraction(mapping["source_time_base"]) == fraction(clock["time_base"]),
        "Canonical/native source audio format differs",
    )
    origin, end = (
        Fraction(ranges[0]["source_start_pts"], 16000),
        Fraction(ranges[-1]["source_end_pts"], 16000),
    )
    tolerance = fraction(clock["time_base"]) + Fraction(1, clock["sample_rate"]) + Fraction(1, 16000)
    require(
        fraction(row["audio_origin_seconds"]) == origin
        and abs(origin - native_bounds[0]) <= tolerance
        and abs(end - native_bounds[1]) <= tolerance,
        "Canonical/native observed audio origin or endpoint differs",
    )
    return count, digest


def components(rows):
    """Private source-slot and exact content links; no inferred person identities."""
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    groups, duplicates = {}, {}
    for field in ("source_slot", "source_sha256", "video_content_sha256", "pcm_content_sha256"):
        linked = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get(field) is not None:
                linked[row[field]].append(i)
        groups[field] = list(linked.values())
        duplicates[field] = [v for v in linked.values() if len(v) > 1]
        for values in linked.values():
            for i in values[1:]:
                parent[root(i)] = root(values[0])
    joined = defaultdict(list)
    for i in range(len(rows)):
        joined[root(i)].append(i)
    return {
        "groups": groups,
        "duplicate_groups": duplicates,
        "components": list(joined.values()),
        "meaning": "Source slots and exactly equal saved content only; not people, unknown ancestry or pretraining independence.",
    }


def identity():
    bindings, rows = metadata()
    return {
        "schema": "chipmunk.grid-native-input-review-freeze/142",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "completed_metadata_sha256": bindings,
        "source_slots": 16,
        "selected_clips": 32,
        "availability": dict(Counter(r["status"] for r in rows)),
        "saved_picture_records": sum(len(r.get("pictures", [])) for r in rows),
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "Pillow", "psutil")},
        "limits": {
            "elapsed_s": 900,
            "sampled_descendant_rss_bytes": 1536 * 1024**2,
            "initial_ram_and_disk_bytes": 3 * 1024**3,
            "running_reserves_bytes": 1536 * 1024**2,
        },
        "scope": "Independent saved PNG/WAV/float32 PCM/source-member byte and rational-clock replay; no MPG re-decode, model, source-authenticity or person/pretraining independence proof.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","source_slots":16,"selected_clips":32,"new_neural_calls":0}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged complete saved-bank review contract required",
    )


def review(output):
    _bindings, records = metadata()
    private, counters = [], Counter()
    with (output / "reviewed-inputs.jsonl").open("x") as journal:
        for ordinal, row in enumerate(records):
            plan = row["planned"]
            folder = BANK / f"source-{plan['source_ordinal']:02d}/clip-{plan['within_source_ordinal']:02d}"
            require(
                Path(row["source_file"]).resolve() == (folder / "source.mpg").resolve()
                and row["study_role"] == "external_transfer"
                and row["source_license"] == "CC-BY-4.0",
                "Exact private selected bank/source role required",
            )
            verify_member(row["source_file"], plan["member"], row["source_sha256"])
            available = row["status"] == AVAILABLE
            result = {
                "ordinal": ordinal,
                "source_slot": plan["source_ordinal"],
                "within_source_ordinal": plan["within_source_ordinal"],
                "status": row["status"],
                "source_sha256": row["source_sha256"],
                "video_content_sha256": None,
                "pcm_content_sha256": None,
            }
            pictures, table = row.get("pictures", []), row.get("native_video_table")
            if row["status"] == "unavailable_video_clock_or_bounds":
                require(
                    table is not None and not supported_video(table) and not pictures,
                    "Unavailable video bounds must preserve an actually unsupported table",
                )
            elif table is not None:
                complete_video = row["status"] != "unavailable_video_pixels"
                origin = video_clock(table, pictures, complete=complete_video)
                if available:
                    require(
                        fraction(row["video_origin_seconds"]) == origin,
                        "Recorded video origin differs from native PTS",
                    )
                pixels = hashlib.sha256()
                pixels.update(json.dumps([table["width"], table["height"], len(pictures)]).encode())
                for picture in pictures:
                    expected = folder / f"frame-{picture['frame_index']:04d}.png"
                    require(
                        Path(picture["file"]).resolve() == expected.resolve(),
                        "Picture belongs to another bank/position",
                    )
                    rgb = verify_png(expected, picture, (table["width"], table["height"]))
                    pixels.update(rgb)
                    counters["verified_pngs"] += 1
                if len(pictures) == len(table["pts"]):
                    result["video_content_sha256"] = pixels.hexdigest()
            else:
                require(not pictures and not available, "Pictures cannot lose their source clock")
            if available:
                require(
                    Path(row["pcm_file"]).resolve() == (folder / "pcm-f32.npy").resolve()
                    and Path(row["wav_file"]).resolve() == (folder / "source-mono16k.wav").resolve(),
                    "PCM belongs to another input bank",
                )
                count, digest = verify_pcm(row)
                counters["verified_mono16k_samples"] += count
                result["pcm_content_sha256"] = digest
            elif "native_audio_clock" in row:
                native_audio(row["native_audio_clock"])
            private.append(result)
            journal.write(json.dumps(result, allow_nan=False) + "\n")
            journal.flush()
            print(
                json.dumps({"reviewed_clips": ordinal + 1, "verified_pngs": counters["verified_pngs"]}),
                flush=True,
            )
    require(
        len(private) == 32 and counters["verified_pngs"] == sum(len(r.get("pictures", [])) for r in records),
        "No planned clip or saved picture may disappear",
    )
    links = components(private)
    write(output / "private-components.json", links)
    result = {
        "schema": "chipmunk.grid-native-input-review-result/142",
        "status": "complete",
        "source_slots": 16,
        "planned_clips": 32,
        "reviewed_clips": len(private),
        "availability": dict(Counter(r["status"] for r in records)),
        **counters,
        "exact_full_video_duplicate_groups": len(links["duplicate_groups"]["video_content_sha256"]),
        "exact_pcm_duplicate_groups": len(links["duplicate_groups"]["pcm_content_sha256"]),
        "source_and_exact_content_components": len(links["components"]),
        "private_reviewed_inputs_sha256": sha(output / "reviewed-inputs.jsonl"),
        "private_components_sha256": sha(output / "private-components.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "mpg_decoder_calls": 0,
        "defaults_changed": False,
        "scope": "Independent saved-bank file/RGB/PCM/CRC and rational-clock consistency only. Prior acquisition checks are receipt-bound, not whole-ZIP replayed; original MPG is not decoded. Unavailable stays unavailable, not original. No source-authenticity, people/unknown-ancestry/pretraining independence or detection-accuracy claim.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    expected = TRIAL / "bind-00/output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only exact exclusive data-only saved-bank review stage permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.review_grid_native_inputs_v142", "review", str(expected)], **kwargs
    )


def run():
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda _c, _b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "review"))
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None, "No extra freeze argument")
        freeze()
    elif args.command == "run":
        require(args.output is None, "No extra stage argument")
        run()
    else:
        verify()
        require(
            args.output is not None and Path(args.output).resolve() == (TRIAL / "bind-00/output").resolve(),
            "Exact exclusive worker output required",
        )
        target = Path(args.output)
        target.mkdir(exist_ok=False)
        review(target)
