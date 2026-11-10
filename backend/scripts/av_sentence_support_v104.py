"""Sentence-connected AV roles and rational native support; metadata only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PRIOR = BASE / "video-common-sources-v89"
NATIVE = BASE / "video-native-sources-v92"
ACQUIRED = BASE / "video-common-acquisition-v91"
TRIAL = BASE / "av-sentence-support-v104"
PUBLIC = ROOT / "docs/evaluation-panels/av-sentence-support-freeze-v104.json"
CODE = (
    "backend/scripts/av_sentence_support_v104.py",
    "backend/tests/test_av_sentence_support_v104.py",
    "docs/av-sentence-support-v104-method.md",
)
NAME = re.compile(r"(10\d{2})_([A-Z]{3})_[A-Z]{3}_[A-Z]{2}\.flv")
HISTORY_NAME = re.compile(r"(?<!\d)(10\d{2})_([A-Z]{3})_[A-Z]{3}_[A-Z]{2}(?:\.flv)?")
RESERVED = ("confirmation", "confidence_cal", "selection")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assign_roles(rows, historical_sentences):
    """Join sentence, actor and known ancestry before assigning an entire component."""
    require(rows and len({r["id"] for r in rows}) == len(rows), "Unique source records required")
    parent = {r["id"]: r["id"] for r in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in rows:
        for key in ("sentence", "actor", "ancestry"):
            value = row[key]
            require(type(value) is str and value, "Explicit sentence, actor and ancestry required")
            identity = (key, value)
            if identity in seen:
                a, b = root(row["id"]), root(seen[identity])
                parent[max(a, b)] = min(a, b)
            seen[identity] = row["id"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[root(row["id"])].append(row)
    fitting = {
        key for key, group in grouped.items() if any(r["sentence"] in historical_sentences for r in group)
    }
    require(fitting, "This study requires its recorded historical fitting contexts")
    fresh = sorted(
        set(grouped) - fitting,
        key=lambda k: hashlib.sha256(("chipmunk.av-sentence-roles/104\0" + k).encode()).digest(),
    )
    require(6 <= len(fresh) <= 10, "Six to ten bounded reserved components required, at least two per role")
    best = None
    for assignment in itertools.product(range(3), repeat=len(fresh)):
        contexts = Counter(assignment)
        if any(contexts[r] < 2 for r in range(3)):
            continue
        counts = [
            sum(len(grouped[k]) for k, role in zip(fresh, assignment, strict=True) if role == r)
            for r in range(3)
        ]
        score = (
            max(counts) - min(counts),
            sum(n * n for n in counts),
            max(contexts.values()) - min(contexts.values()),
            assignment,
        )
        if best is None or score < best[0]:
            best = (score, assignment)
    require(best is not None, "No source-count balanced sentence-disjoint roles available")
    roles = dict.fromkeys(fitting, "fusion_fit")
    roles.update({key: RESERVED[role] for key, role in zip(fresh, best[1], strict=True)})
    output = []
    for row in rows:
        result = copy.deepcopy(row)
        result.update(
            study_role=roles[root(row["id"])],
            study_component=root(row["id"]),
            historical_sentence=row["sentence"] in historical_sentences,
        )
        output.append(result)
    return output


def clock_support(table, audio):
    require(not table["truncated"] and not table["interpolated"], "Observed video clocks required")
    require(
        audio and audio["clock_qualified"] and audio["contiguous"],
        "Qualified contiguous audio clocks required",
    )
    pts = table["pts"]
    require(pts and all(type(p) is int for p in pts), "Actual integer video timestamps required")
    tb = Fraction(table["time_base"])
    require(tb > 0, "Positive video time base required")
    times = [p * tb for p in pts]
    require(all(a < b for a, b in zip(times, times[1:])), "Strictly ordered video clock required")
    rate = audio["sample_rate"]
    require(
        type(rate) is int and rate > 0 and audio["frames"], "Native audio sample rate and frames required"
    )
    atb = Fraction(audio["time_base"])
    require(atb > 0, "Positive audio time base required")
    intervals = []
    for frame in audio["frames"]:
        start, end = Fraction(frame["start"]), Fraction(frame["end"])
        require(
            type(frame["pts"]) is int
            and type(frame["samples"]) is int
            and frame["samples"] > 0
            and start == frame["pts"] * atb
            and end == start + Fraction(frame["samples"], rate),
            "Native audio interval contradicts its timestamp or samples",
        )
        intervals.append((start, end))
    tolerance = atb + Fraction(1, rate)
    require(
        all(a[0] < b[0] and abs(b[0] - a[1]) <= tolerance for a, b in zip(intervals, intervals[1:])),
        "Audio clock discontinuity exceeds native clock precision",
    )
    origin, end = intervals[0][0], intervals[-1][1]
    cumulative, deviations = 0, []
    for frame, (start, _) in zip(audio["frames"], intervals, strict=True):
        deviations.append(abs(start - (origin + Fraction(cumulative, rate))))
        cumulative += frame["samples"]
    require(max(deviations) <= tolerance, "Cumulative audio sample lattice drifts beyond clock precision")
    lattice_end = origin + Fraction(cumulative, rate)
    indices = [i for i, t in enumerate(times) if origin <= t < min(end, lattice_end)]
    require(
        indices and indices == list(range(indices[0], indices[-1] + 1)),
        "Common observed picture support required",
    )
    mapping = []
    for i in indices:
        candidates = [
            {"audio_frame_index": k, "sample_offset_fraction": str((times[i] - a) * rate)}
            for k, (a, b) in enumerate(intervals)
            if a <= times[i] < b
        ]
        mapping.append(
            {
                "video_frame_index": i,
                "video_pts": pts[i],
                "source_time_seconds": str(times[i]),
                "audio_relative_seconds": str(times[i] - origin),
                "audio_frame_candidates": candidates,
                "uniform_sample_lattice_offset_fraction": str((times[i] - origin) * rate),
            }
        )
    n = len(indices)
    edit = None
    reason = (
        "Fewer than 24 observed native pictures for a 20-frame edit and two pictures of context on each side."
    )
    if n >= 24:
        a = indices[0] + (n - 20) // 2
        b = a + 20
        used = [m for m in mapping if a - 2 <= m["video_frame_index"] < b + 2]
        require(len(used) == 24, "Complete native context support required")
        edit = {
            "frames_half_open": [a, b],
            "source_frame_center_seconds": [str(times[a]), str(times[b - 1])],
            "first_after_frame_seconds": str(times[b]),
            "representatives": {"before": a - 1, "during": (a + b - 1) // 2, "after": b},
            "audio_relative_first_edit_seconds": str(times[a] - origin),
        }
        reason = None
    return {
        "video_time_base": str(tb),
        "audio_time_base": str(atb),
        "audio_sample_rate": rate,
        "source_video_origin_seconds": str(times[0]),
        "source_audio_origin_seconds": str(origin),
        "source_audio_frame_end_seconds": str(end),
        "conditional_sample_lattice_end_seconds": str(lattice_end),
        "native_audio_samples_declared": cumulative,
        "audio_minus_video_origin_seconds": str(origin - times[0]),
        "native_audio_clock_quantization_allowance_seconds": str(tolerance),
        "maximum_cumulative_audio_frame_deviation_seconds": str(max(deviations)),
        "uniform_sample_lattice_scope": "Conditional on contiguous native sample order. Native PCM sample count and resampler phase have not been independently decoded by this preparation.",
        "common_frame_indices": indices,
        "common_frame_center_support_seconds": [str(times[indices[0]]), str(times[indices[-1]])],
        "common_frame_center_span_seconds": str(times[indices[-1]] - times[indices[0]]),
        "audio_frame_mappings": mapping,
        "native20_edit_eligible": edit is not None,
        "missing_or_ambiguous_audio_frame_mappings": sum(
            len(m["audio_frame_candidates"]) != 1 for m in mapping
        ),
        "audio_conditioning_ready": False,
        "native20_edit": edit,
        "native20_unavailable_reason": reason,
        "scope": "Observed picture-center and native audio-frame clock planning. No display-tail inference, resampled PCM alignment, padding, 25-Hz conversion, neural generation or achieved phonetic-sync claim.",
    }


def identity():
    old = read(PRIOR / "plan.json")
    return {
        "schema": "chipmunk.av-sentence-support-freeze/104",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                PRIOR / "plan.json",
                NATIVE / "qualified-plan.json",
                NATIVE / "summary.json",
                ACQUIRED / "asset-plan.json",
            )
        },
        "history_sha256": old["crema_history_sha256"],
        "new_neural_calls": 0,
        "scope": "Metadata-only new sentence-connected study roles and rational support. Previous inventories are immutable. No detector results, source truth adjudication, new media or model execution.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def prepare():
    import shutil
    import time

    import psutil

    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged source procedure required")
    output = TRIAL / "preparation"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    observations = []

    def observe(initial=False):
        row = {
            "elapsed_s": time.monotonic() - started,
            "self_rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
            "free_disk_bytes": shutil.disk_usage(output).free,
        }
        observations.append(row)
        minimum = 3 * 1024**3 if initial else 1536 * 1024**2
        require(
            row["elapsed_s"] <= 900
            and row["self_rss_bytes"] <= 1536 * 1024**2
            and min(row["available_ram_bytes"], row["free_disk_bytes"]) >= minimum,
            "Metadata preparation resource bound exceeded",
        )

    observe(initial=True)
    prior = read(PRIOR / "plan.json")
    native = read(NATIVE / "qualified-plan.json")
    require(
        sha(NATIVE / "qualified-plan.json") == read(NATIVE / "summary.json")["qualified_private_plan_sha256"],
        "Native qualification changed",
    )
    history = set()
    for name, digest in lock["history_sha256"].items():
        path = BASE / name
        require(sha(path) == digest, "Recorded historical manifest changed")
        history.update(m[2] for m in HISTORY_NAME.finditer(path.read_text()))
    sources = {r["group"]: r for r in prior["groups"] if r["source_family"] == "recorded_audiovisual_speech"}
    records = []
    for row in native["assets"]:
        if row["source_family"] != "recorded_audiovisual_speech":
            continue
        source = sources[row["group"]]
        require(
            len(source["assets"]) == 1 and row["qualification"]["eligible"],
            "Qualified single-source AV input required",
        )
        match = NAME.fullmatch(Path(source["assets"][0]["source_path"]).name)
        require(
            match is not None and source["group"] == "CREMA-D:" + match[1],
            "Publisher filename/actor lineage differs",
        )
        records.append(
            {
                "id": row["group"],
                "actor": match[1],
                "sentence": match[2],
                "ancestry": row["qualification"]["component"],
                "prior_role": row["qualification"]["role"],
                "asset_index": row["index"],
                "input_sha256": row["input_sha256"],
                "clock_support": clock_support(row["frame_table"], row["audio_clocks"]),
            }
        )
        observe()
    require(
        len(records) == len(sources) == 48 and len({r["actor"] for r in records}) == 48,
        "Complete unchanged 48-actor inventory required",
    )
    records = assign_roles(records, history)
    plan = {
        "schema": "chipmunk.av-sentence-support-plan/104",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "records": records,
        "detector_study_ready": False,
        "new_neural_calls": 0,
    }
    write(output / "plan.json", plan)
    by_sentence = defaultdict(set)
    old_sentence = defaultdict(set)
    for row in records:
        by_sentence[row["sentence"]].add(row["study_role"])
        old_sentence[row["sentence"]].add(row["prior_role"])
    require(
        all(len(v) == 1 for v in by_sentence.values())
        and all(r["study_role"] == "fusion_fit" for r in records if r["historical_sentence"]),
        "Sentence or historical context crossed reserved roles",
    )
    summary = {
        "schema": "chipmunk.av-sentence-support-result/104",
        "status": "complete",
        "sources": 48,
        "known_sentence_types": len(by_sentence),
        "historical_sentence_types": len(history & by_sentence.keys()),
        "historical_sentence_sources": sum(r["historical_sentence"] for r in records),
        "prior_cross_role_sentence_types": sum(len(v) > 1 for v in old_sentence.values()),
        "new_cross_role_sentence_types": 0,
        "historical_sentence_sources_in_reserved_roles": 0,
        "roles": {
            role: {
                "sources": sum(r["study_role"] == role for r in records),
                "sentence_types": len({r["sentence"] for r in records if r["study_role"] == role}),
                "native20_edit_eligible": sum(
                    r["clock_support"]["native20_edit_eligible"] for r in records if r["study_role"] == role
                ),
            }
            for role in ("fusion_fit", "confidence_cal", "selection", "confirmation")
        },
        "native20_edit_eligible": sum(r["clock_support"]["native20_edit_eligible"] for r in records),
        "native20_edit_unavailable": sum(not r["clock_support"]["native20_edit_eligible"] for r in records),
        "native_frames_with_missing_or_ambiguous_audio_frame_mappings": sum(
            r["clock_support"]["missing_or_ambiguous_audio_frame_mappings"] for r in records
        ),
        "role_changed_sources": sum(r["study_role"] != r["prior_role"] for r in records),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "private_plan_sha256": sha(output / "plan.json"),
        "duration_s": time.monotonic() - started,
        "maximum_observed_self_rss_bytes": max(r["self_rss_bytes"] for r in observations),
        "minimum_available_ram_bytes": min(r["available_ram_bytes"] for r in observations),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "New metadata study view: sentence/known-actor/ancestry connections and historically observed sentences confined to fitting. No original role change, decoder pass, generated media, resampler parity, phonetic synchronization test, detector accuracy or population confidence. Only 12 known sentence contexts; unknown historical/pretraining links remain unknown.",
    }
    require(
        not {"torch", "numpy", "transformers", "scipy"}.intersection(sys.modules),
        "Metadata-only preparation required",
    )
    write(output / "result.json", summary)
    write(ROOT / "docs/evaluation-panels/av-sentence-support-result-v104.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare"))
    freeze() if parser.parse_args().command == "freeze" else prepare()
