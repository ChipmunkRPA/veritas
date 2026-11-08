"""Private source/creator/near-duplicate graph before any regional learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def author_key(url):
    if url is None:
        return None
    parsed = urlsplit(url)
    pieces = parsed.path.strip("/").split("/")
    need(parsed.scheme == "https" and parsed.netloc == "www.flickr.com" and not parsed.query and not parsed.fragment and len(pieces) == 2 and pieces[0] == "photos" and pieces[1], "Canonical attributed Flickr profile required")
    return pieces[1].casefold()


def sketch(image):
    """256 horizontal comparisons; no neural embedding or statistical fitting."""
    small = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.BILINEAR))
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big")


def near(a, b):
    # Very low-information thumbnails are not sufficient for a near-duplicate link.
    informative = all(32 <= int(row["sketch"], 16).bit_count() <= 224 for row in (a, b))
    aw, ah = a["size"]
    bw, bh = b["size"]
    aspect_close = abs(aw * bh - bw * ah) * 100 <= 2 * max(aw * bh, bw * ah)
    return informative and aspect_close and (int(a["sketch"], 16) ^ int(b["sketch"], 16)).bit_count() <= 8


def components(rows):
    need(len(rows) == len({r["key"] for r in rows}), "Unique source graph keys required")
    ordered = sorted(rows, key=lambda r: r["key"])
    parent = list(range(len(ordered)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links = Counter()
    for i, a in enumerate(ordered):
        for j in range(i):
            b = ordered[j]
            reasons = []
            if a["source_id"] == b["source_id"]:
                reasons.append("same_source_id")
            if a["author"] is not None and a["author"] == b["author"]:
                reasons.append("same_attributed_profile")
            if a["size"] == b["size"] and a["rgb_sha256"] == b["rgb_sha256"]:
                reasons.append("exact_source_rgb")
            if near(a, b):
                reasons.append("near_thumbnail")
            if reasons:
                ri, rj = root(i), root(j)
                parent[max(ri, rj)] = min(ri, rj)
                links.update(reasons)
    groups = {}
    for i, row in enumerate(ordered):
        groups.setdefault(root(i), []).append(row["key"])
    return list(groups.values()), dict(sorted(links.items()))


def source_record(key, source_id, path, expected, profile, role, consumed):
    need(type(source_id) in (int, str) and int(source_id) > 0, "Positive integer source identity required")
    need(sha(path) == expected, "Committed original source bytes differ")
    with Image.open(path) as source:
        need(source.format == "PNG" and 0 < min(source.size) <= max(source.size) <= 2048, "Bounded original PNG required")
        rgb = source.convert("RGB").copy()
    return {"key": key, "source_id": int(source_id), "original": str(path.relative_to(ROOT)), "original_sha256": expected, "author": author_key(profile), "publisher_role": role, "previously_consumed": consumed, "size": list(rgb.size), "rgb_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(), "sketch": f"{sketch(rgb):064x}"}


def run(pairs_folder, output):
    output.mkdir(parents=True, exist_ok=False)
    freeze_path = ROOT / "docs/evaluation-panels/image-source-groups-v22-freeze.json"
    freeze = read(freeze_path)
    need(all(sha(ROOT / name) == expected for name, expected in {**freeze["code_sha256"], **freeze["protected_sha256"]}.items()), "Frozen source-group implementation or protected reference differs")
    need(freeze["runtime_versions"] == {"numpy": np.__version__, "Pillow": Image.__version__}, "Source-group decoder versions differ")
    need(sha(pairs_folder / "plan.json") == freeze["private_pair_plan_sha256"], "Source grouping refers to another pair plan")
    review = read(pairs_folder / "independent-review.json")
    need(review["passed"] is True and review["cases_reviewed"] == 397 and review["failed_case_count"] == 0, "All397 source cases need a successful complete review before grouping")
    need(review["summary_sha256"] == sha(pairs_folder / "qualification/summary.json"), "Reviewed qualification summary changed")
    qualification = read(pairs_folder / "qualification/summary.json")
    need(qualification["statuses"].get("qualified_source_relative_pair", 0) == review["qualified_truths_reconstructed"], "Qualified/unresolved pair counts differ")
    plan = read(pairs_folder / "plan.json")
    rows = []
    for case in plan["cases"]:
        item = source_record("new/" + str(case["source_id"]), case["source_id"], ROOT / case["original"], case["original_sha256"], case["author_profile"], case["publisher_role"], False)
        need(item["rgb_sha256"] == case["source_rgb_sha256"], "Source pixel-lineage binding differs")
        rows.append(item)
    manifests = freeze["previously_consumed_manifest_sha256"]
    original_by_source = {}
    for name, expected in manifests.items():
        path = ROOT / name
        need(sha(path) == expected, "Prior consumed-source manifest changed")
        for case in read(path)["items"]:
            if case["label"] != 0:
                continue
            image = path.parent / case["image"]
            identifier = int(case["source_id"])
            row = source_record("old/" + str(identifier), identifier, image, case["image_sha256"], None, "prior_consumed", True)
            if identifier in original_by_source:
                need(original_by_source[identifier]["rgb_sha256"] == row["rgb_sha256"], "A prior source has multiple incompatible originals")
            original_by_source[identifier] = row
    need(len(original_by_source) == 240 and len(rows) == 397 and not (set(original_by_source) & {r["source_id"] for r in rows}), "Complete397-new/240-prior source denominator required")
    rows.extend(original_by_source.values())
    groups, links = components(rows)
    by_key = {r["key"]: r for r in rows}
    groups_with_new = [g for g in groups if any(not by_key[k]["previously_consumed"] for k in g)]
    connected_to_prior = [g for g in groups_with_new if any(by_key[k]["previously_consumed"] for k in g)]
    nontraining_candidates = [g for g in groups_with_new if all(not by_key[k]["previously_consumed"] for k in g) and any(by_key[k]["publisher_role"] in {"validation", "testing"} for k in g)]
    private = {"schema": "chipmunk.image-source-connected-components/22", "sources": sorted(rows, key=lambda r: r["key"]), "components": groups, "roles_assigned": False}
    write(output / "private-groups.json", private)
    summary = {"schema": "chipmunk.image-source-group-summary/22", "new_sources": 397, "qualified_pairs": review["qualified_truths_reconstructed"], "unresolved_pairs": 397 - review["qualified_truths_reconstructed"], "prior_consumed_sources": 240, "all_components": len(groups), "components_with_new_sources": len(groups_with_new), "new_components_connected_to_prior_sources": len(connected_to_prior), "new_sources_connected_to_prior_sources": sum(not by_key[k]["previously_consumed"] for g in connected_to_prior for k in g), "components_containing_publisher_nontraining_sources_without_detected_prior_link": len(nontraining_candidates), "within_new_source_attributed_profiles": len({r["author"] for r in rows if not r["previously_consumed"]}), "prior_sources_without_current_author_check": 240, "link_counts": links, "private_groups_sha256": sha(output / "private-groups.json"), "freeze_sha256": sha(freeze_path), "pair_review_sha256": sha(pairs_folder / "independent-review.json"), "model_forwards": 0, "fitting_steps": 0, "roles_assigned": False, "outside_human_review": False, "scope": "Source IDs, exact RGB, attributed new-source profiles and a fixed conservative thumbnail heuristic. Prior-source photographer identity, aliases, burst/crop relationships and external model-pretraining overlap remain unverified. No source role, fresh-confirmation claim or training admission follows from grouping alone."}
    write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    run(args.pairs_folder.resolve(), args.output_folder.resolve())
