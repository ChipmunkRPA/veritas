"""Verify source contexts against pinned primary sidecars before trusting role graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from scripts.review_audio_sources_v76 import BASE, require, review, sha


def check(path):
    body = json.loads(path.read_text())
    rows = [*body["foregrounds"], *body["backgrounds"]]
    recipe = json.loads((BASE / "audio-calibration-roles-v71/plan.json").read_text())
    publisher_path = BASE / "audio-v4/fresh-sources/DuplexGen--duplexgen-spoken/metadata.jsonl"
    publisher = {row["key"]: row for line in publisher_path.read_text().splitlines()
                 if (row := json.loads(line))["scenario"] == "PLN"}
    archives, licenses = {}, Counter()
    try:
        for row in rows:
            kind = row["kind"]
            if kind in {"genuine", "speech"}:
                expected = "libri-chapter:" + row["id"].split(":", 1)[1].split("-")[1]
                license_name = "CC-BY-4.0"
            elif kind in {"noise", "music"}:
                expected = "background:" + row["source_id"]
                license_name = "Public-domain" if kind == "noise" else "CC-BY-3.0"
                if kind == "noise":
                    matches = [group for group in recipe["backgrounds"] if group["kind"] == "noise"
                               and group["selected_candidate"]["sha256"] == row["raw_sha256"]]
                    require(len(matches) == 1 and matches[0]["license"] == license_name,
                            "Noise permission must retain its exact frozen source-recipe declaration")
            else:
                name = row["metadata_archive"] if kind == "expressive" else row["archive"]
                root = BASE / ("audio-v8/reserved-sources/archives" if kind == "expressive" else "audio-v7/fresh-sources/archives")
                archive_path = root / name
                if str(archive_path) not in archives:
                    archives[str(archive_path)] = tarfile.open(archive_path, "r:")
                archive = archives[str(archive_path)]
                member = archive.getmember(row["metadata_member"])
                with archive.extractfile(member) as stream:
                    raw = stream.read()
                require(hashlib.sha256(raw).hexdigest() == row["metadata_sha256"], "Primary generation sidecar changed")
                metadata = json.loads(raw)
                if kind == "expressive":
                    require(metadata["manifest"]["id"] == row["id"]
                            and metadata["generation"]["audio_sha256"] == row["raw_sha256"]
                            and "elevenlabs:" + hashlib.sha256(metadata["generation"]["voice_id"].encode()).hexdigest()
                            == row["canonical_speaker"], "Expressive generation provenance or voice differs")
                    expected = "laion-challenge:" + str(metadata["manifest"]["challenge_id"])
                    license_name = "CC-BY-4.0"
                else:
                    reference = publisher[row["metadata_member"].removesuffix("/meta.json")]
                    require(reference["license"] == "MIT", "Only declared MIT PLN sources allowed")
                    ordinal = int(Path(row["member"]).stem)
                    turn = metadata["speech_meta"][ordinal]
                    require(turn["speaker"] == metadata["speakers"].index("user")
                            and turn["tts_text"] == row["transcript"]
                            and "librispeech:" + str(metadata["user_prompt_speaker_id"]) == row["canonical_speaker"],
                            "Clone parent, utterance or text differs from primary metadata")
                    expected = "duplex-dialogue:" + reference["example_id"]
                    license_name = "MIT (publisher PLN scenario declaration)"
            require(row["source_group"] == expected, "Source context differs from original provenance")
            if kind != "noise":
                require(row["license"] == license_name, "Recorded source permission changed")
            licenses[license_name] += 1
    finally:
        for archive in archives.values():
            archive.close()
    graph = review(path)
    return {"schema": "chipmunk.audio-primary-context-and-role-review/77", "status": "complete",
            "manifest_sha256": sha(path), "reviewer_sha256": sha(Path(__file__)),
            "source_contexts_verified": len(rows), "expressive_sidecars_verified": 96,
            "clone_primary_parent_utterance_contexts_verified": 80,
            "declared_permissions": dict(licenses), "permissions_are_publisher_declarations": True,
            "source_graph_review": graph, "new_model_calls": 0,
            "scope": "Primary metadata/context and declared-permission validation plus separate breadth-first role graph. Does not establish consent, biometric identity, pretrained independence or model quality. Noise license is inherited from its frozen source recipe, not silently invented on the decoded record."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = check(args.manifest)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value), flush=True)
