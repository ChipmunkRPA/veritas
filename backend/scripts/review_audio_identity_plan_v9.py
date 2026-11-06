"""Review public canonical identity/role/recipe consistency without ML packages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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
COUNTS = {"fit": 214, "cal": 97, "development": 161, "confirmation": 120}
GROUPS = {"fit": 169, "cal": 82, "development": 106, "confirmation": 118}


def check_components(sources, backgrounds):
    rows = sources + backgrounds
    parent = {r["sha256"]: r["sha256"] for r in rows}
    if len(parent) != len(rows):
        raise ValueError("Duplicate original source bytes")

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    owners = {}
    for row in rows:
        fingerprints = row["connection_fingerprints"]
        if row["canonical_speaker_sha256"]:
            fingerprints = fingerprints + ["canonical:" + row["canonical_speaker_sha256"]]
        for key in fingerprints:
            if key in owners:
                parent[find(row["sha256"])] = find(owners[key])
            else:
                owners[key] = row["sha256"]
    groups = defaultdict(list)
    foreground = {r["sha256"] for r in sources}
    for row in rows:
        groups[find(row["sha256"])].append(row)
    seen_group = {}
    for component, members in groups.items():
        if len({r["role"] for r in members}) != 1:
            raise ValueError("Canonical source/voice/text connection crosses roles")
        kinds = {r["sha256"] in foreground for r in members}
        if len(kinds) != 1:
            raise ValueError("Foreground and background share a known source/speaker connection")
        if members[0]["sha256"] in foreground:
            declared = {r["group"] for r in members}
            if len(declared) != 1:
                raise ValueError("Connected speakers/scripts split across statistical groups")
            group = next(iter(declared))
            if group in seen_group and seen_group[group] != component:
                raise ValueError("Declared group merges unrelated components")
            seen_group[group] = component
    return {role: len({r["group"] for r in sources if r["role"] == role}) for role in COUNTS}


def review(body, original):
    sources, backgrounds = body["sources"], body["backgrounds"]
    if Counter(r["role"] for r in sources) != COUNTS or len(backgrounds) != 66:
        raise ValueError("Every corrected source/role required")
    if check_components(sources, backgrounds) != GROUPS:
        raise ValueError("Canonical statistical group counts changed")
    old_sources = {r["sha256"]: r for r in original["sources"]}
    retired = {r["source_sha256"] for r in body["retired_consumed_sources"]}
    if (
        len(retired) != 4
        or {r["sha256"] for r in sources} | retired != set(old_sources)
        or retired & {r["sha256"] for r in sources}
    ):
        raise ValueError("Every original must be retained or explicitly retired")
    if {r["sha256"] for r in sources if r["role"] == "confirmation"} != {
        r["sha256"] for r in original["sources"] if r["role"] == "confirmation"
    }:
        raise ValueError("Fixed confirmation recordings changed")
    for row in sources:
        if row["prior_role"] != old_sources[row["sha256"]]["role"] or any(
            row[k] != old_sources[row["sha256"]][k]
            for k in ("label", "generator", "dataset", "prefix_pcm_sha256", "production_pcm_sha256")
        ):
            raise ValueError("Original class/source/audio lineage changed")
    background_map = {r["sha256"]: r for r in backgrounds}
    if set(background_map) != {r["sha256"] for r in original["backgrounds"]}:
        raise ValueError("Fixed original backgrounds changed")
    foreground = {r["sha256"]: r for r in sources}
    variants = body["variants"]["development"] + body["variants"]["confirmation"]
    if len(variants) != 5328 or len({r["id"] for r in variants}) != 5328:
        raise ValueError("Every corrected logical source/condition required")
    old_variants = {r["id"]: r for rows in original["variants"].values() for r in rows}
    by_source = defaultdict(list)
    retained = 0
    for row in variants:
        source = foreground[row["source_sha256"]]
        if (
            any(row[k] != source[k] for k in ("role", "group", "label", "generator", "dataset"))
            or row["base_id_sha256"] != source["id_sha256"]
        ):
            raise ValueError("Corrected recipe source/role/group/class binding changed")
        if (
            row["samples"] != 64000
            or row["source_interval_samples"] != [0, 64000]
            or row["sample_rate"] != 16000
        ):
            raise ValueError("Original source sample clock changed")
        prior = old_variants[row["id"]]
        if row["prior_input_sha256"] != prior["sha256"]:
            raise ValueError("Prior waveform binding changed")
        if row["retained_exact_v8_input"]:
            if source["role"] != source["prior_role"] or any(
                row[k] != prior[k]
                for k in (
                    "sha256",
                    "pcm_sha256",
                    "payload_pcm_sha256",
                    "background_sha256",
                    "background_start_sample",
                    "common_peak_gain",
                )
            ):
                raise ValueError("Retained exact input or original role changed")
            retained += 1
        elif source["role"] == source["prior_role"]:
            raise ValueError("An unchanged role must retain its exact original input")
        if row["background_sha256"] is not None:
            bg = background_map[row["background_sha256"]]
            kind, snr = row["condition"].split("_")
            if (
                row["background_role"] != row["role"]
                or bg["role"] != row["role"]
                or row["background_kind"] != bg["kind"]
                or kind != bg["kind"]
                or row["snr_db"] != int(snr)
            ):
                raise ValueError("Background no longer belongs to the source's role/kind/SNR")
            if not 0 <= row["background_start_sample"] <= bg["full_samples"] - 64000:
                raise ValueError("Background interval outside original")
        elif row["condition"] != "clean":
            raise ValueError("A background condition lost its source")
        by_source[source["sha256"]].append(row)
    if set(by_source) != set(foreground) or any(
        Counter(r["condition"] for r in rows) != Counter(CONDITIONS)
        or len({r["common_peak_gain"] for r in rows}) != 1
        for rows in by_source.values()
    ):
        raise ValueError("Complete nine conditions and common source gain required")
    if retained != 5274:
        raise ValueError("Only the54 moved-role inputs may be regenerated")
    if (
        len(body["variants"]["development"]) != 4248
        or len(body["variants"]["confirmation"]) != 1080
        or any(r["role"] == "confirmation" for r in body["variants"]["development"])
        or any(r["role"] != "confirmation" for r in body["variants"]["confirmation"])
    ):
        raise ValueError("Fixed confirmation is not separated from consumed data")
    return {
        "passed": True,
        "foregrounds": len(sources),
        "backgrounds": len(backgrounds),
        "inputs": len(variants),
        "canonical_role_groups": GROUPS,
        "retained_exact_inputs": retained,
        "regenerated_role_inputs": len(variants) - retained,
        "retired_consumed_sources": len(retired),
        "detector_outcomes_reviewed": False,
        "scope": "Public canonical-connection/role/recipe/lineage consistency; no source-media, neural inference or outside reviewer implied.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    body = json.loads(gzip.decompress(args.capsule.read_bytes()))
    raw = args.original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != body["original_source_catalogue_sha256"]:
        raise ValueError("Original public source catalogue changed")
    result = review(body, json.loads(gzip.decompress(raw)))
    result["capsule_sha256"] = hashlib.sha256(args.capsule.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
