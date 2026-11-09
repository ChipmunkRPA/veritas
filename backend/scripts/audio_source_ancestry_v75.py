"""Repair declared dialogue ancestry before any new model call; retain original roles."""
from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_calibration_roles_v71 import ROLES, stable
from scripts.audio_expressive_inventory_v66 import sha, write
from scripts.audio_history_v58 import text_keys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-source-ancestry-v75"
QUALIFIED = BASE / "audio-source-qualification-v72"
INPUT_SHA = "c222df9ea7fc5e201258600547707a454ef73394b0288f38dd64cccec2165755"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def keys(row):
    found = set()
    for name in ("canonical_speaker", "source_group"):
        if row.get(name):
            found.add(name + ":" + row[name])
    for name in ("raw_sha256", "pcm_sha256", "first4_pcm_sha256", "prefix_pcm_sha256"):
        if row.get(name):
            found.add(("bytes:" if name == "raw_sha256" else "pcm:") + row[name])
    found.update(text_keys(row.get("transcript") or ""))
    return found


def components(rows):
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        for key in keys(row):
            if key in owners:
                parent[root(index)] = root(owners[key])
            else:
                owners[key] = index
    grouped = defaultdict(list)
    for index in range(len(rows)):
        grouped[root(index)].append(index)
    return list(grouped.values())


def component_id(rows):
    encoded = json.dumps(sorted(set().union(*(keys(row) for row in rows))), separators=(",", ":")).encode()
    return "connected:" + hashlib.sha256(encoded).hexdigest()


def clone_roles(rows):
    groups = components(rows)
    weighted = [(len({rows[index]["canonical_speaker"] for index in indices}),
                 component_id([rows[index] for index in indices]), indices) for indices in groups]
    assigned, loads = {}, dict.fromkeys(ROLES, 0)
    for weight, identifier, indices in sorted(weighted, key=lambda item: (-item[0], stable(item[1]))):
        role = min(ROLES, key=lambda name: (loads[name], ROLES.index(name)))
        require(loads[role] + weight <= 10, "No split or overflow of a connected clone component")
        loads[role] += weight
        for index in indices:
            assigned[index] = (role, identifier)
    require(set(loads.values()) == {10}, "Ten cloned parent voices per role required")
    return assigned, groups


def repair():
    TRIAL.mkdir(exist_ok=False)
    path = QUALIFIED / "manifest.json"
    require(sha(path) == INPUT_SHA, "Exact completed original-source qualification required")
    profile_path = QUALIFIED.parent / (QUALIFIED.name + "-profile/profile.json")
    profile = json.loads(profile_path.read_text())
    require(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0,
            "Original byte/PCM qualification profile must pass")
    old = json.loads(path.read_text())
    body = copy.deepcopy(old)
    clones = [row for row in body["foregrounds"] if row["kind"] == "clone"]
    old_owners = defaultdict(set)
    for row in clones:
        old_owners[row["source_group"]].add(row["role"])
    original_conflicts = sum(len(roles) > 1 for roles in old_owners.values())
    require(len(clones) == 80 and original_conflicts == 8, "Exact pre-model shared-dialogue defect required")
    assigned, clone_components = clone_roles(clones)
    moved = 0
    for index, row in enumerate(clones):
        role, identifier = assigned[index]
        row["prior_role"] = row["role"]
        moved += row["role"] != role
        row.update(role=role, study_group=identifier)
    archives = {}
    try:
        for row in body["foregrounds"]:
            require(sha(Path(row["source_path"])) == row["raw_sha256"], "Qualified original bytes changed")
            if row["kind"] == "genuine":
                row["source_group"] = "libri-chapter:" + row["id"].split(":", 1)[1].split("-")[1]
            elif row["kind"] == "expressive":
                name = row["metadata_archive"]
                if name not in archives:
                    archives[name] = tarfile.open(BASE / "audio-v8/reserved-sources/archives" / name, "r:")
                archive = archives[name]
                member = archive.getmember(row["metadata_member"])
                with archive.extractfile(member) as stream:
                    raw = stream.read()
                require(hashlib.sha256(raw).hexdigest() == row["metadata_sha256"], "Expressive generation sidecar changed")
                metadata = json.loads(raw)
                require(metadata["manifest"]["id"] == row["id"], "Expressive source sidecar rebound")
                row["source_group"] = "laion-challenge:" + str(metadata["manifest"]["challenge_id"])
        for row in body["backgrounds"]:
            require(sha(Path(row["source_path"])) == row["raw_sha256"], "Qualified background bytes changed")
            if row["kind"] == "speech":
                row["source_group"] = "libri-chapter:" + row["id"].split(":", 1)[1].split("-")[1]
            else:
                row["source_group"] = "background:" + row["source_id"]
    finally:
        for archive in archives.values():
            archive.close()
    planned = [*body["foregrounds"], *body["backgrounds"], *body["pending_matched_generation"]]
    all_groups = components(planned)
    counts = {role: {"foreground_groups": 0, "genuine_groups": 0, "ai_groups_including_pending": 0,
                     "background_groups": 0} for role in ROLES}
    for indices in all_groups:
        rows = [planned[index] for index in indices]
        roles = {row["role"] for row in rows}
        require(len(roles) == 1, "Known voice/context/text/byte/PCM component crosses roles")
        role = roles.pop()
        identifier = component_id(rows)
        fore = [row for row in rows if row.get("foreground") is True or row.get("generator") == "Kokoro-82M"]
        back = [row for row in rows if row.get("foreground") is False]
        require(not (fore and back), "Foreground/background source ancestry must stay separate")
        for row in rows:
            row["study_group"] = identifier
        if fore:
            counts[role]["foreground_groups"] += 1
            counts[role]["genuine_groups"] += any(row["label"] == 0 for row in fore)
            counts[role]["ai_groups_including_pending"] += any(row["label"] == 1 for row in fore)
        if back:
            counts[role]["background_groups"] += 1
    require(all(value["genuine_groups"] >= 9 and value["ai_groups_including_pending"] >= 9 for value in counts.values()),
            "Independent truth-class support must survive ancestry joining")
    require(Counter(row["role"] for row in body["foregrounds"] if row["kind"] == "clone") == dict.fromkeys(ROLES, 20),
            "All original clone sources and per-role source counts required")
    body.update(schema="chipmunk.audio-confidence-ancestry-corrected-originals/75",
                derived_from_original_manifest_sha256=INPUT_SHA, correction_code_sha256=sha(Path(__file__)),
                original_roles_superseded_for_shared_dialogues=True, new_model_calls=0)
    write(TRIAL / "manifest.json", body)
    result = {"schema": "chipmunk.audio-confidence-ancestry-correction-summary/75",
              "original_manifest_sha256": INPUT_SHA, "original_shared_dialogues_crossing_roles": original_conflicts,
              "new_cross_role_components": 0, "clone_voice_components": len(clone_components),
              "clone_component_voice_sizes": sorted([len({clones[index]["canonical_speaker"] for index in indices}) for indices in clone_components], reverse=True),
              "clone_records_moved_before_any_model_call": moved, "original_foregrounds_preserved": 400,
              "original_backgrounds_preserved": 48, "matched_synthesis_prompts_preserved": 64,
              "connected_role_support_including_pending": counts,
              "original_qualification_profile_sha256": sha(profile_path),
              "corrected_manifest_sha256": sha(TRIAL / "manifest.json"), "code_sha256": sha(Path(__file__)),
              "new_model_calls": 0, "defaults_changed": False,
              "scope": "Original decoding integrity remains valid; v71/v72 role independence is superseded for shared dialogue ancestry. All selected media/text retained. Corrected before any new generation, detector inference or confidence fitting. Pending synthesis still needs its own freeze and qualification."}
    write(TRIAL / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    repair()
