"""Freeze a metadata-only, source-grouped roster for short ordinary audio edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

from scripts.audio_history_v58 import text_keys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
OUTPUT = BASE / "audio-short-edit-sources-v169"
PUBLIC = ROOT / "docs/evaluation-panels/audio-short-edit-source-plan-v169.json"
METHOD = ROOT / "docs/audio-short-edit-source-plan-v169.md"
SEED = "chipmunk.audio-short-edit-sources/169"
ROLES = ("holdout", "development")
LANES = ("genuine", "clone", "expressive", "stock")
SOURCE_SAMPLES = 47_648


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable(key):
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


class Metadata:
    def __init__(self):
        self.bindings = {}

    def read(self, name, path):
        require(name not in self.bindings, "Unique metadata binding name required")
        content = path.read_bytes()
        self.bindings[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def source(self, name, relative):
        return self.read(name, BASE / relative)


def exclude_history(meta):
    history = set(meta.source("historical_identity_catalogue", "audio-history-v58/catalogue.json")["keys"])
    v81 = meta.source("consumed_v81", "audio-complete-sources-v81/manifest.json")
    ids, contexts = set(), set()
    for row in [*v81["foregrounds"], *v81["backgrounds"]]:
        ids.update(x for key in ("id", "source_id") if (x := row.get(key)))
        contexts.update(x for key in ("source_group", "challenge_group") if (x := row.get(key)))
        if voice := row.get("canonical_speaker"):
            history.add(f"voice:{voice}")
        history.update(text_keys(row.get("transcript")))
        for key in ("raw_sha256", "sha256"):
            if digest := row.get(key):
                history.add(f"bytes:{digest}")
        for key in ("pcm_sha256", "prefix_pcm_sha256", "first4_pcm_sha256", "decoded_pcm_sha256"):
            if digest := row.get(key):
                history.add(f"pcm:{digest}")
    release = meta.read(
        "production_calibration", ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
    )
    meta.source("v3_historical_calibration", "audio-v3/calibration.json")
    require(
        release["source_sha256"] == meta.bindings["v3_historical_calibration"],
        "Serving calibration must bind the recorded v3 history",
    )
    grid = meta.source("grid_original_plan", "grid-transfer-plan-v143/bind-00/output/private-plan.json")
    grid_rows = [row["input"] for row in grid["sources"]]
    require(
        len(grid_rows) == 32 and len({row["pcm_mapping"]["pcm_sha256"] for row in grid_rows}) == 32,
        "All 32 exact GRID original audio identities are required",
    )
    for row in grid_rows:
        planned = row["planned"]
        ids.add(planned["member"]["name"])
        contexts.add(f"grid:{planned['source']['publisher_name']}")
        for key in ("source_sha256", "wav_file_sha256", "pcm_file_sha256"):
            history.add(f"bytes:{row[key]}")
        history.add(f"pcm:{row['pcm_mapping']['pcm_sha256']}")
    alignments = meta.source("grid_publisher_word_matches", "grid-phone-alignment-v162/private-inputs.json")
    words = [row["words"] for row in alignments if "words" in row]
    require(len(words) == 8, "Eight exact publisher word references are retained")
    for reference in words:
        history.update(text_keys(" ".join(row[0] for row in reference)))
    return history, ids, contexts


def assets(meta, label, directory, plan=None):
    receipt = meta.source(f"{label}_receipt", f"{directory}/receipt.json")
    planned = {entry["name"]: entry for entry in meta.source(f"{label}_plan", plan)["assets"]} if plan else {}
    result = {}
    for row in receipt["archives"]:
        name = row["name"]
        path = BASE / directory / name
        require(
            PurePosixPath(name).name == name
            and row["status"] == "verified"
            and (row["publisher_sha256_verified"] or row["publisher_md5_verified"])
            and path.is_file()
            and path.stat().st_size == row["bytes"],
            f"Publisher-verified local asset missing or size differs in {label}",
        )
        if planned:
            require(planned[name]["sha256"] == row["sha256"], f"Publisher and saved hashes differ in {label}")
        result[name] = {**planned.get(name, {}), **row, "file": str(path)}
    return result


def candidate(lane, identity, voice, text, locator, license_name, **extras):
    return {
        "lane": lane,
        "id": identity,
        "voice": voice,
        "texts": sorted(text_keys(text)),
        "license": license_name,
        "locator": locator,
        **extras,
    }


def eligible(row, history, ids, contexts, *, seen_voice=False):
    raw, group = row.get("raw_sha256"), row.get("context")
    return (
        bool(row["texts"])
        and row["id"] not in ids
        and group not in contexts
        and not (set(row["texts"]) & history)
        and f"bytes:{raw}" not in history
        and (seen_voice or f"voice:{row['voice']}" not in history)
    )


def parquet_locator(asset, row_group, offset):
    return {
        "format": "parquet",
        "file": asset["file"],
        "archive_sha256": asset["sha256"],
        "row_group": row_group,
        "row_offset": offset,
    }


def genuine(meta, history, ids, contexts):
    local = assets(
        meta,
        "libri100",
        "audio-v8/reserved-sources/libri-parquet",
        "audio-v8/source-planning/libri-acquisition-plan.json",
    )
    rows = []
    for item in local.values():
        require(item["license"] == "CC-BY-4.0", "Exact LibriSpeech CC BY source required")
        reader = pq.ParquetFile(item["file"])
        for index in range(reader.num_row_groups):
            table = reader.read_row_group(index, columns=["id", "speaker_id", "text"])
            for offset, row in enumerate(table.to_pylist()):
                if len(row["text"].split()) < 14:
                    continue
                rows.append(
                    candidate(
                        "genuine",
                        f"libri:{row['id']}",
                        f"librispeech:{row['speaker_id']}",
                        row["text"],
                        parquet_locator(item, index, offset),
                        item["license"],
                        corpus="librispeech_train_clean_100",
                    )
                )
    receipt = meta.source("libri360_receipt", "audio-genuine-sources-v67/receipt.json")
    external = {}
    for item in receipt["archives"]:
        path = ARCHIVE / "audio-calibration-sourcebank-v67/libri360" / item["name"]
        require(
            item["status"] == "publisher-verified"
            and path.is_file()
            and path.stat().st_size == item["bytes"],
            "Publisher-verified LibriSpeech train.360 asset required",
        )
        external[item["name"]] = {**item, "file": str(path)}
    for row in meta.source("libri360_candidates", "audio-source-inventory-v68/genuine-candidates.json"):
        if len(row["transcript"].split()) < 14:
            continue
        item = external[row["parquet"]]
        require(
            row["parquet_sha256"] == item["sha256"] and row["license"] == "CC-BY-4.0",
            "Exact train.360 source provenance required",
        )
        rows.append(
            candidate(
                "genuine",
                row["id"],
                row["canonical_speaker"],
                row["transcript"],
                parquet_locator(item, row["parquet_row_group"], row["parquet_group_row"]),
                row["license"],
                corpus="librispeech_train_360",
            )
        )
    return [row for row in rows if eligible(row, history, ids, contexts)]


def tar_locator(asset, row):
    return {
        "format": "tar",
        "file": asset["file"],
        "archive_sha256": asset["sha256"],
        "member": row["member"],
    }


def synthetic(meta, history, ids, contexts, genuine_voices):
    clone_assets = assets(meta, "clone", "audio-v7/fresh-sources/archives")
    v8 = assets(
        meta,
        "stock_expressive",
        "audio-v8/reserved-sources/archives",
        "audio-v8/source-planning/confirmation-acquisition-plan.json",
    )
    clones, expressive, stock = [], [], []
    for row in meta.source("clone_candidates", "audio-source-inventory-v68/clone-candidates.json"):
        if (
            row["audio_header_status"] != "ready"
            or row["header_frames"] * 16000 < SOURCE_SAMPLES * row["sample_rate"]
            or row["seconds"] > 4
            or row["novelty"]["archive_member_seen"]
            or row["canonical_speaker"] in genuine_voices
        ):
            continue
        item = clone_assets[row["archive"]]
        require(
            item["sha256"] == row["archive_sha256"]
            and row["license"] == "MIT (publisher PLN scenario declaration)",
            "Exact publisher-declared PLN clone archive required",
        )
        clones.append(
            candidate(
                "clone",
                row["id"],
                row["canonical_speaker"],
                row["transcript"],
                tar_locator(item, row),
                row["license"],
                context=row["source_group"],
                donor=row["canonical_speaker"],
                voice_seen_in_history=False,
            )
        )
    for row in meta.source("expressive_candidates", "audio-expressive-inventory-v66/candidates.json"):
        if row["duration_estimate_s"] < SOURCE_SAMPLES / 16000:
            continue
        item = v8[row["archive"]]
        require(item["license"] == row["license"] == "CC-BY-4.0", "Expressive source rights differ")
        expressive.append(
            candidate(
                "expressive",
                row["id"],
                row["canonical_speaker"],
                row["transcript"],
                tar_locator(item, row),
                row["license"],
                context=row["challenge_group"],
                raw_sha256=row["sha256"],
                voice_seen_in_history=f"voice:{row['canonical_speaker']}" in history,
            )
        )
    item = v8["kokoro-directed-test.parquet"]
    require(item["license"] == "CC-BY-SA-4.0", "Declared stock recording license required")
    reader = pq.ParquetFile(item["file"])
    fields = [
        "id",
        "rendition_id",
        "duration_s",
        "language",
        "transcript",
        "final_text",
        "tts_engine",
        "tts_voice",
        "license",
        "audio",
    ]
    for index in range(reader.num_row_groups):
        table = reader.read_row_group(index, columns=fields)
        for offset, row in enumerate(table.to_pylist()):
            if row["language"] != "en" or row["duration_s"] < SOURCE_SAMPLES / 16000:
                continue
            require(
                row["tts_engine"] == "kokoro-82m"
                and row["license"] == "cc-by-sa-4.0"
                and row["audio"]["bytes"],
                "Known local stock rendering and publisher data license required",
            )
            voice = f"kokoro-82m:{row['tts_voice']}"
            stock.append(
                candidate(
                    "stock",
                    row["rendition_id"],
                    voice,
                    row["transcript"],
                    parquet_locator(item, index, offset),
                    item["license"],
                    context=f"stock-script:{row['id']}",
                    raw_sha256=hashlib.sha256(row["audio"]["bytes"]).hexdigest(),
                    voice_seen_in_history=f"voice:{voice}" in history,
                )
            )
            stock[-1]["texts"] = sorted(set(stock[-1]["texts"]) | text_keys(row["final_text"]))
    return {
        lane: [row for row in rows if eligible(row, history, ids, contexts, seen_voice=lane != "clone")]
        for lane, rows in (("clone", clones), ("expressive", expressive), ("stock", stock))
    }


def identities(row):
    keys = {f"id:{row['id']}", f"voice:{row['voice']}", *row["texts"]}
    if context := row.get("context"):
        keys.add(f"context:{context}")
    if donor := row.get("donor"):
        keys.add(f"voice:{donor}")
    if raw := row.get("raw_sha256"):
        keys.add(f"bytes:{raw}")
    return keys


def connected(rows):
    parents = list(range(len(rows)))
    seen = {}

    def root(index):
        while parents[index] != index:
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        for key in identities(row):
            if key in seen:
                parents[root(index)] = root(seen[key])
            seen[key] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(root(index), []).append(row)
    return list(groups.values())


def freeze_groups(lanes):
    claimed, groups = set(), []
    for lane in LANES:
        candidates = [row for row in lanes[lane] if not identities(row) & claimed]
        components = connected(candidates)
        components.sort(key=lambda rows: stable(min(row["id"] for row in rows)))
        for index, rows in enumerate(components):
            required = 2 if lane == "genuine" else 1
            ordered = sorted(rows, key=lambda row: stable(row["id"]))
            shortlist, keys = [], set()
            for row in ordered:
                nonvoice = identities(row) - {f"voice:{row['voice']}", f"voice:{row.get('donor')}"}
                if nonvoice & keys:
                    continue
                shortlist.append(row)
                keys.update(nonvoice)
                if len(shortlist) == 6:
                    break
            groups.append(
                {
                    "lane": lane,
                    "group_id": f"{lane}-{index:02d}",
                    "role": ROLES[index % len(ROLES)],
                    "required": min(required, len(shortlist)),
                    "ordered_candidates": shortlist,
                    "status": "pending_exact_production_decode",
                }
            )
        # Every candidate edge in an assigned component remains in its lane.
        for row in candidates:
            claimed.update(identities(row))
    return groups


def noise(meta, history):
    values = assets(
        meta,
        "noise",
        "audio-v8/reserved-sources/additional-noise",
        "audio-v8/source-planning/additional-noise-acquisition-plan.json",
    )
    item = values["musan-noise-0000.parquet"]
    require(
        item["license"] == "Public-domain for selected noise/free-sound records only",
        "Known noise terms required",
    )
    inventory = meta.source("noise_inventory", "audio-v8/source-planning/additional-noise-inventory.json")
    rows = []
    for entry in inventory["rows"]:
        row = entry["ref"]
        if (
            entry["source"] != "free-sound"
            or row["decoded_samples"] < SOURCE_SAMPLES
            or f"bytes:{row['sha256']}" in history
            or f"pcm:{row['decoded_pcm_sha256']}" in history
        ):
            continue
        path = Path(row["source_path"])
        require(
            row["whole_parquet_sha256"] == item["sha256"]
            and row["publisher_whole_parquet_sha256_verified"]
            and path.is_file()
            and path.stat().st_size == row["bytes"]
            and sha(path) == row["sha256"],
            "Complete selected noise source and publisher hash required",
        )
        rows.append(
            {
                "id": entry["path"],
                "path": str(path),
                "license": item["license"],
                "raw_sha256": row["sha256"],
                "recorded_pcm_sha256": row["decoded_pcm_sha256"],
            }
        )
    rows.sort(key=lambda row: stable(row["id"]))
    require(
        len(rows) >= 8 and len({row["recorded_pcm_sha256"] for row in rows}) == len(rows),
        "Eight fresh distinct noise parents required",
    )
    return [{**row, "role": ROLES[index % len(ROLES)]} for index, row in enumerate(rows[:8])]


def validate_roles(groups, backgrounds):
    roles = {}
    for group in groups:
        for row in group["ordered_candidates"]:
            for key in identities(row):
                require(roles.setdefault(key, group["role"]) == group["role"], "Cross-role source identity")
    for row in backgrounds:
        for key in (f"bytes:{row['raw_sha256']}", f"pcm:{row['recorded_pcm_sha256']}"):
            require(roles.setdefault(key, row["role"]) == row["role"], "Cross-role noise identity")


def aggregate(groups, backgrounds):
    result = {}
    for role in ROLES:
        own = [group for group in groups if group["role"] == role]
        result[role] = {
            "components": dict(Counter(group["lane"] for group in own)),
            "planned_foregrounds": {
                lane: sum(group["required"] for group in own if group["lane"] == lane) for lane in LANES
            },
            "seen_provider_voices_in_pending_groups": {
                lane: sum(
                    any(row.get("voice_seen_in_history") for row in group["ordered_candidates"])
                    for group in own
                    if group["lane"] == lane
                )
                for lane in ("expressive", "stock")
            },
            "noise_parents": sum(row["role"] == role for row in backgrounds),
        }
    return result


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def freeze():
    require(not OUTPUT.exists() and not PUBLIC.exists(), "Preserve the initial source roster")
    meta = Metadata()
    history, ids, contexts = exclude_history(meta)
    real = genuine(meta, history, ids, contexts)
    lanes = {
        "genuine": real,
        **synthetic(meta, history, ids, contexts, {row["voice"] for row in real}),
    }
    groups = freeze_groups(lanes)
    backgrounds = noise(meta, history)
    validate_roles(groups, backgrounds)
    require(
        all(
            any(group["role"] == role and group["lane"] == lane for group in groups)
            for role in ROLES
            for lane in LANES
        ),
        "Each prospective role requires genuine speech and every planned synthesis family",
    )
    OUTPUT.mkdir(mode=0o700)
    private = OUTPUT / "private-roster.json"
    write(
        private,
        {
            "schema": "chipmunk.audio-short-edit-private-source-plan/169",
            "source_samples": SOURCE_SAMPLES,
            "sample_rate": 16000,
            "noise_snr_db": [0, 5],
            "groups": groups,
            "noise_parents": backgrounds,
            "input_metadata_sha256": meta.bindings,
            "excluded_keys": sorted(key for key in history if key.startswith(("bytes:", "pcm:"))),
            "exact_production_decode_complete": False,
            "holdout_scored": False,
        },
    )
    summary = {
        "schema": "chipmunk.audio-short-edit-source-plan/169",
        "status": "metadata_frozen_exact_decode_pending",
        "private_roster_sha256": sha(private),
        "implementation_sha256": sha(Path(__file__)),
        "method_sha256": sha(METHOD),
        "input_metadata_sha256": meta.bindings,
        "candidate_counts_before_cross_lane_connections": {lane: len(rows) for lane, rows in lanes.items()},
        "roles": aggregate(groups, backgrounds),
        "grid_originals_bound": 32,
        "grid_publisher_word_references_bound": 8,
        "source_samples": SOURCE_SAMPLES,
        "sample_rate": 16000,
        "noise_snr_db": [0, 5],
        "model_calls": 0,
        "holdout_scored": False,
    }
    write(PUBLIC, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze",))
    parser.parse_args()
    freeze()
