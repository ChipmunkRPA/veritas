"""Standard-library review of public source connections, recipes and code pins.

This checks the published metadata, not actual audio decoding, learned outputs or
an outside reviewer's participation. It needs no model, credentials or network.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = (
    "clean",
    "noise_0",
    "noise_5",
    "noise_15",
    "music_5",
    "music_15",
    "speech_0",
    "speech_5",
    "speech_15",
)
ROLE_COUNTS = {"fit": 219, "cal": 99, "development": 158, "confirmation": 120}


def connections(sources, backgrounds):
    rows = sources + backgrounds
    if len({r["sha256"] for r in rows}) != len(rows) or len(
        {r["production_pcm_sha256"] for r in rows}
    ) != len(rows):
        raise ValueError("Repeated original bytes or full production audio")
    owners = {}
    for row in rows:
        if row.get("kind") in ("noise", "music"):
            continue
        keys = {"full:" + row["production_pcm_sha256"], "prefix:" + row["prefix_pcm_sha256"]}
        keys.update("text:" + key for key in row["text_connection_keys"])
        keys.update(key + ":" + row[key] for key in ("voice_key_sha256", "source_key_sha256") if row.get(key))
        for key in keys:
            identity = (row["role"], row["group"])
            if key in owners and owners[key] != identity:
                raise ValueError("Known connection crosses role/group boundaries")
            owners[key] = identity
    groups = defaultdict(list)
    for row in sources:
        groups[row["group"]].append(row)
    if any(len({r["role"] for r in members}) != 1 for members in groups.values()):
        raise ValueError("Connected foreground group crosses roles")
    return {role: len({r["group"] for r in sources if r["role"] == role}) for role in ROLE_COUNTS}


def review(body, project=None):
    sources, backgrounds = body["sources"], body["backgrounds"]
    if Counter(r["role"] for r in sources) != ROLE_COUNTS or len(sources) != 596 or len(backgrounds) != 66:
        raise ValueError("Every planned original and role required")
    if len(body["exclusions"]) != 64:
        raise ValueError("Every consumed-source exclusion must remain visible")
    group_counts = connections(sources, backgrounds)
    if group_counts != {"fit": 174, "cal": 84, "development": 106, "confirmation": 120}:
        raise ValueError("Known foreground groups changed")
    expected_backgrounds = {
        (role, kind): count
        for role in ROLE_COUNTS
        for kind, count in (("noise", 8), ("music", 4), ("speech", 6 if role == "confirmation" else 4))
    }
    if Counter((r["role"], r["kind"]) for r in backgrounds) != expected_backgrounds:
        raise ValueError("Every role-specific background required")
    foreground = {r["sha256"]: r for r in sources}
    background = {r["sha256"]: r for r in backgrounds}
    variants = body["variants"]["development"] + body["variants"]["confirmation"]
    if len(variants) != 5364 or len({r["id"] for r in variants}) != 5364:
        raise ValueError("Every exact variant required")
    by_source = defaultdict(list)
    for row in variants:
        original = foreground[row["source_sha256"]]
        if (
            row["base_id_sha256"] != original["id_sha256"]
            or any(row[k] != original[k] for k in ("role", "group", "label", "generator", "dataset"))
            or row["source_prefix_pcm_sha256"] != original["prefix_pcm_sha256"]
        ):
            raise ValueError("Variant original/group/label binding changed")
        if (
            row["source_interval_samples"] != [0, 64000]
            or row["samples"] != 64000
            or row["sample_rate"] != 16000
            or row["seconds"] != 4
        ):
            raise ValueError("Source clock changed")
        if not 0 < row["common_peak_gain"] <= 1:
            raise ValueError("Invalid common source gain")
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        if condition == "clean":
            if any(
                row[k] is not None
                for k in (
                    "background_sha256",
                    "background_role",
                    "background_kind",
                    "background_start_sample",
                    "snr_db",
                )
            ):
                raise ValueError("Clean condition has a background")
        else:
            bg = background[row["background_sha256"]]
            kind, snr = condition.split("_")
            if (
                bg["role"] != row["role"]
                or row["background_role"] != row["role"]
                or bg["kind"] != kind
                or row["background_kind"] != kind
                or row["snr_db"] != int(snr)
            ):
                raise ValueError("Background role/kind/SNR changed")
            if not 0 <= row["background_start_sample"] <= bg["full_samples"] - 64000:
                raise ValueError("Background interval outside original source")
            if row["measured_snr_db"] is None or not math.isfinite(row["measured_snr_db"]):
                raise ValueError("Measured SNR missing")
        by_source[row["source_sha256"]].append(row)
    if set(by_source) != set(foreground):
        raise ValueError("A planned foreground disappeared")
    for rows in by_source.values():
        if (
            Counter(r["condition"] for r in rows) != Counter(CONDITIONS)
            or len({r["common_peak_gain"] for r in rows}) != 1
        ):
            raise ValueError("Complete nine conditions and common gain required")
        for kind in ("noise", "music", "speech"):
            selected = [r for r in rows if r["background_kind"] == kind]
            if (
                len(
                    {
                        (
                            r["background_sha256"],
                            r["background_start_sample"],
                            r["background_interval_pcm_sha256"],
                        )
                        for r in selected
                    }
                )
                != 1
            ):
                raise ValueError("Background interval changed between SNR variants")
    if (
        len(body["variants"]["development"]) != 4284
        or any(r["role"] == "confirmation" for r in body["variants"]["development"])
        or len(body["variants"]["confirmation"]) != 1080
        or any(r["role"] != "confirmation" for r in body["variants"]["confirmation"])
    ):
        raise ValueError("Fresh inputs mixed with consumed panel")
    lock = body["lock"]
    if (
        lock["variants_sha256"] != body["variants_manifest_sha256"]["development"]
        or lock["reserved_confirmation_variants_sha256"] != body["variants_manifest_sha256"]["confirmation"]
        or lock["prepared_sources_sha256"] != body["prepared_sources_sha256"]
        or lock["planned"] != 4284
    ):
        raise ValueError("Frozen method/input pins changed")
    code_checked = 0
    if project is not None:
        for name, expected in lock["implementation_sha256"].items():
            path = project / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Current checkout differs from frozen implementation")
            code_checked += 1
    return {
        "passed": True,
        "original_sources": len(sources) + len(backgrounds),
        "foregrounds": len(sources),
        "inputs": len(variants),
        "role_foreground_groups": group_counts,
        "implementation_files_checked": code_checked,
        "detector_outcomes_reviewed": False,
        "outside_reviewer_participation": False,
        "scope": "Public metadata/connection/recipe/code-pin consistency only; no actual source decoding or learned inference.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    result = review(json.loads(gzip.decompress(args.capsule.read_bytes())), args.project)
    result["capsule_sha256"] = hashlib.sha256(args.capsule.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
