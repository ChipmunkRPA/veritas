"""Freeze a deterministic consumed-data development panel; never score fresh data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def order(value):
    return hashlib.sha256(("audio-v4-consumed-development:" + str(value)).encode()).hexdigest()


def connected_folds(bases, folds=5):
    parent = {row["id"]: row["id"] for row in bases}

    def find(value):
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    owners = {}
    for row in bases:
        keys = [("voice", row["cluster_id"])]
        if row.get("transcript_sha256"):
            keys.append(("text", row["transcript_sha256"]))
        if row["dataset"] == "isjwdu/DFADD" and row["label"] == 1:
            match = re.search(r"p\d+_(\d+)(?:_|$)", row["id"])
            if match:
                keys.append(("dfadd_prompt_index", match.group(1)))
        for key in keys:
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for row in bases:
        groups[find(row["id"])].append(row)
    target = [sum(row["label"] == label for row in bases) / folds for label in (0, 1)]
    counts = [[0, 0] for _ in range(folds)]
    assignment, records = {}, []
    for members in sorted(
        groups.values(), key=lambda rows: (-len(rows), order(sorted(row["id"] for row in rows)))
    ):
        sizes = [sum(row["label"] == label for row in members) for label in (0, 1)]
        fold = min(
            range(folds),
            key=lambda k: (
                sum(
                    (counts[k][c] + sizes[c] - target[c]) ** 2 - (counts[k][c] - target[c]) ** 2
                    for c in (0, 1)
                ),
                k,
            ),
        )
        counts[fold] = [counts[fold][c] + sizes[c] for c in (0, 1)]
        group = "connected:" + order(sorted(row["id"] for row in members))
        records.append(
            {
                "group": group,
                "fold": fold,
                "sources": sorted(row["id"] for row in members),
                "class_counts": sizes,
            }
        )
        for row in members:
            assignment[row["id"]] = {"group": group, "fold": fold}
    if any(min(row) == 0 for row in counts):
        raise ValueError("Connected groups cannot support the prespecified class-complete folds")
    return assignment, records, counts


def prepare(source: Path, output: Path):
    manifest_path, scores_path = source / "manifest.json", source / "predictions.json"
    m, scores = json.loads(manifest_path.read_text()), json.loads(scores_path.read_text())
    by_group = defaultdict(list)
    for row in m["base_files"]:
        if row["split"] == "test":
            by_group[row["cluster_id"]].append(row)
    generators = {"NaturalSpeech2", "PflowTTS", "StyleTTS2"}
    eligible = [
        group
        for group, rows in by_group.items()
        if any(row["label"] == 0 and row["dataset"] == "isjwdu/DFADD" for row in rows)
        and generators.issubset({row["generator"] for row in rows if row["label"] == 1})
    ]
    if len(eligible) < 20:
        raise ValueError("Insufficient complete consumed joint identities; do not silently replace the panel")
    selected = []
    for group in sorted(eligible, key=order)[:20]:
        rows = by_group[group]
        selected.append(min((r for r in rows if r["label"] == 0), key=lambda r: order(r["id"])))
        for generator in sorted(generators):
            selected.append(
                min(
                    (r for r in rows if r["label"] == 1 and r["generator"] == generator),
                    key=lambda r: order(r["id"]),
                )
            )
    libri = [
        group
        for group, rows in by_group.items()
        if all(r["label"] == 0 and r["dataset"] == "openslr/librispeech_asr" for r in rows)
    ]
    if len(libri) < 40:
        raise ValueError("Insufficient consumed authentic identities")
    selected.extend(
        min(by_group[group], key=lambda r: order(r["id"])) for group in sorted(libri, key=order)[:40]
    )
    assignment, groups, counts = connected_folds(selected)
    ids = {r["id"] for r in selected}
    expected = {r["id"]: r for r in scores}
    if len(expected) != len(scores):
        raise ValueError("Duplicate prior prediction identity")
    variants = []
    for row in m["variants"]:
        if row["base_id"] not in ids:
            continue
        reference = expected[row["id"]]
        if reference["sha256"] != row["sha256"] or reference["pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("Prior prediction differs from the source manifest")
        value = {
            k: row[k]
            for k in (
                "id",
                "base_id",
                "label",
                "cluster_id",
                "condition",
                "path",
                "sha256",
                "pcm_sha256",
                "samples",
                "seconds",
                "background_id",
                "background_sha256",
                "dataset",
                "generator",
            )
        }
        value.update(
            assignment[row["base_id"]],
            raw=reference["new_raw"],
            raw_status=reference["new_status"],
            original_windows=reference["new_windows"],
            serving_p=reference["new_p"],
            serving_decision=reference["new_guarded"],
        )
        variants.append(value)
    if len(variants) != 720 or Counter(row["label"] for row in selected) != {0: 60, 1: 60}:
        raise ValueError("The complete 120-source/720-input panel is required")
    project = Path(__file__).resolve().parents[2]
    pins = [
        "backend/scripts/prepare_audio_noise_v4.py",
        "backend/scripts/audio_noise_view_v4.py",
        "backend/scripts/score_audio_noise_v4.py",
        "backend/scripts/review_audio_noise_v4.py",
        "docs/audio-noise-view-v4-development.md",
        "backend/app/detectors/audio/upgraded.py",
        "backend/app/detectors/audio/decode.py",
        "backend/app/third_party/kaldi/fbank.py",
    ]
    body = {
        "schema": "chipmunk.audio-noise-development/4",
        "role": "consumed v3 development; not independent confirmation",
        "source_manifest_file_sha256": digest(manifest_path),
        "reference_predictions_file_sha256": digest(scores_path),
        "source_root": str(source.resolve()),
        "selection": "20 complete VCTK joint voices + 40 Libri voices, hash order, all 6 conditions",
        "sources": selected,
        "groups": groups,
        "fold_class_counts": counts,
        "variants": variants,
        "implementation_sha256": {name: digest(project / name) for name in pins},
        "limitations": [
            "All prior test responses are consumed.",
            "Unpublished DFADD transcript mapping; prompt-index grouping is conservative, not proof of text independence.",
            "Unverified pretrained overlap.",
            "Intervals conditional on the reused consumed background pool.",
            "Authentic source and generated-family channel confounding remain.",
        ],
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "sources": len(selected),
                "inputs": len(variants),
                "connected_groups": len(groups),
                "fold_class_counts": counts,
                "manifest_sha256": digest(output),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
