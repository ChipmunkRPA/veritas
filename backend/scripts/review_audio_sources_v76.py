"""Separate breadth-first ancestry/immutability review before new audio inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from scripts.audio_history_v58 import text_keys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
ORIGINAL_SHA = "c222df9ea7fc5e201258600547707a454ef73394b0288f38dd64cccec2165755"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def links(row):
    found = set(text_keys(row.get("transcript") or ""))
    for name in ("canonical_speaker", "source_group"):
        if row.get(name):
            found.add(f"{name}:{row[name]}")
    if row.get("raw_sha256"):
        found.add("bytes:" + row["raw_sha256"])
    for name in ("pcm_sha256", "first4_pcm_sha256", "prefix_pcm_sha256"):
        if row.get(name):
            found.add("pcm:" + row[name])
    return found


def grouped(rows):
    table = defaultdict(list)
    row_links = [links(row) for row in rows]
    for index, keys in enumerate(row_links):
        for key in keys:
            table[key].append(index)
    unseen, result = set(range(len(rows))), []
    while unseen:
        first = min(unseen)
        queue, visited, touched = deque([first]), set(), set()
        while queue:
            index = queue.popleft()
            if index in visited:
                continue
            visited.add(index)
            for key in row_links[index] - touched:
                touched.add(key)
                queue.extend(table[key])
        unseen -= visited
        result.append(sorted(visited))
    return result


def review(path):
    old_path = BASE / "audio-source-qualification-v72/manifest.json"
    require(sha(old_path) == ORIGINAL_SHA, "Original qualified manifest changed")
    old, body = json.loads(old_path.read_text()), json.loads(path.read_text())
    require(body["derived_from_original_manifest_sha256"] == ORIGINAL_SHA, "Corrected source lineage changed")
    originals = [*old["foregrounds"], *old["backgrounds"]]
    prepared = [*body["foregrounds"], *body["backgrounds"]]
    require(len(prepared) == len(originals) == 448, "All original files must be retained")
    before = {row["raw_sha256"]: row for row in originals}
    require(len(before) == len({row["raw_sha256"] for row in prepared}) == 448, "Unique original source bytes required")
    allowed = {"role", "prior_role", "study_group", "source_group"}
    for row in prepared:
        previous = before[row["raw_sha256"]]
        require({k: v for k, v in row.items() if k not in allowed}
                == {k: v for k, v in previous.items() if k not in allowed}, "Original source/label/content changed")
        if row["kind"] != "clone":
            require(row["role"] == previous["role"], "Only connected clone-component roles may move")
        require(sha(Path(row["source_path"])) == row["raw_sha256"], "Qualified source bytes changed")
    pending = body["pending_matched_generation"]
    require(len(pending) == len(old["pending_matched_generation"]) == 64, "All pending paired prompts required")
    for prior, current in zip(old["pending_matched_generation"], pending, strict=True):
        require({k: v for k, v in prior.items() if k != "study_group"}
                == {k: v for k, v in current.items() if k != "study_group"}, "Pending source prompt/profile/role changed")
    foreground = [*body["foregrounds"], *pending]
    rows = [*foreground, *body["backgrounds"]]
    groups = grouped(rows)
    counts = defaultdict(lambda: {"foreground_groups": 0, "genuine_groups": 0,
                                  "ai_groups_including_pending": 0, "background_groups": 0})
    for indices in groups:
        members = [rows[index] for index in indices]
        role_set = {row["role"] for row in members}
        require(len(role_set) == 1, "Known ancestry crosses roles")
        role = role_set.pop()
        encoded = json.dumps(sorted(set().union(*(links(row) for row in members))), separators=(",", ":")).encode()
        expected_group = "connected:" + hashlib.sha256(encoded).hexdigest()
        require({row["study_group"] for row in members} == {expected_group}, "Actual connected study group differs")
        fore = [rows[index] for index in indices if index < len(foreground)]
        back = [index for index in indices if index >= len(foreground)]
        require(not (fore and back), "Foreground/background origins overlap")
        if fore:
            counts[role]["foreground_groups"] += 1
            counts[role]["genuine_groups"] += any(row["label"] == 0 for row in fore)
            counts[role]["ai_groups_including_pending"] += any(row["label"] == 1 for row in fore)
        if back:
            counts[role]["background_groups"] += 1
    summary_path = BASE / "audio-source-ancestry-v75/summary.json"
    summary = json.loads(summary_path.read_text())
    require(dict(counts) == summary["connected_role_support_including_pending"], "Independent role support counts differ")
    return {"schema": "chipmunk.audio-confidence-source-bfs-review/76", "status": "complete",
            "corrected_manifest_sha256": sha(path), "original_manifest_sha256": ORIGINAL_SHA,
            "reviewer_sha256": sha(Path(__file__)), "original_media_files_reverified": len(prepared),
            "pending_matched_prompts_preserved": len(pending), "planned_source_nodes": len(rows),
            "connected_source_components": len(groups), "cross_role_components": 0,
            "role_support": dict(counts), "original_labels_content_and_bytes_preserved": True,
            "new_model_calls": 0,
            "scope": "Separate breadth-first source graph and immutable-content/physical-byte review, not a second decoding pass, biometric proof, model accuracy or outside human review.64 sources still require matched generation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = review(args.manifest)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value), flush=True)
